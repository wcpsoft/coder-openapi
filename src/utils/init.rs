use crate::utils::config::AppConfig;
use log::info;
use log4rs;
use std::sync::Arc;

pub async fn init() -> crate::error::Result<Arc<AppConfig>> {
    // 初始化日志系统
    log4rs::init_file("config/log4rs.yml", Default::default())?;

    // 加载应用配置
    let config = AppConfig::load("config/app.yml")?;
    info!("应用配置加载完成");

    // 初始化本地化系统
    info!("使用本地化文件路径: {}, 默认语言: {}", config.locales.path, config.locales.default);

    // 初始化模型配置
    for model_id in config.models.keys() {
        info!("已初始化模型配置: {}", model_id);
    }

    Ok(Arc::new(config))
}
