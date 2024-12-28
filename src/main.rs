use actix_web::{web, App, HttpServer};
use anyhow::Context;
use std::sync::Arc;

use coder_openapi::routes;
use coder_openapi::utils::init;
use coder_openapi::Locales;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // 初始化应用配置和日志系统
    let config = init::init()
        .await
        .context("初始化应用程序失败")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // 初始化本地化系统
    let mut locales = Locales::new(&config.locales.path.clone()).expect("加载本地化文件失败");
    locales.set_default(&config.locales.default.clone()).expect("设置默认语言失败");
    let locales = Arc::new(locales);

    // 创建带有优雅关闭功能的服务器
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(locales.clone()))
            .app_data(web::PayloadConfig::new(32768 * 1024)) // 32MB payload limit
            .wrap(coder_openapi::middleware::error_handler::error_handler())
            .configure(routes::route::configure)
    })
    .client_request_timeout(std::time::Duration::from_secs(30)) // 客户端请求超时30秒
    .bind((config.server.host.clone(), config.server.port))?
    .shutdown_timeout(30) // 优雅关闭等待时间30秒
    .run()
    .await
}
