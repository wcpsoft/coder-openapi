use actix_web::{web, App, HttpServer};
rust_i18n::i18n!("locales");
use anyhow::Context;
use coder_openapi::routes;
use coder_openapi::set_locale;
use coder_openapi::utils::init;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Set default locale to zh
    set_locale("zh");

    // 初始化应用配置和日志系统
    let config = init::init()
        .await
        .context("init failed")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let server_config = config.clone();
    let host = server_config.server.host.clone();
    let port = server_config.server.port;
    let shutdown_timeout = server_config.server.shutdown_timeout;

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(server_config.clone()))
            .app_data(web::PayloadConfig::new(32768 * 1024)) // 32MB payload limit
            .wrap(coder_openapi::middleware::error_handler::error_handler())
            .configure(routes::route::configure)
    })
    .client_request_timeout(std::time::Duration::from_secs(30)) // 客户端请求超时30秒
    .bind((host, port))?
    .shutdown_timeout(shutdown_timeout) // 优雅关闭等待时间
    .run()
    .await
}
