use crate::utils::config::AppConfig;
use actix_web::{App, HttpServer};
use log::info;
use log4rs;
use std::sync::Arc;

pub async fn init() -> crate::error::Result<Arc<AppConfig>> {
    // Initialize logging
    log4rs::init_file("config/log4rs.yml", Default::default())?;

    // Load application configuration
    let config = AppConfig::load("config/app.yml")?;
    info!("Application configuration loaded");

    // Initialize server
    let server_addr = format!("{}:{}", config.server.host, config.server.port);
    info!("Server will listen on: {}", server_addr);

    // Initialize localization
    info!(
        "Using localization from: {}, default language: {}",
        config.locales.path, config.locales.default
    );

    // Initialize models
    for (model_id, _) in &config.models {
        info!("Initialized model configuration for: {}", model_id);
    }

    Ok(Arc::new(config))
}

pub async fn run(host: &str, port: u16) -> crate::error::Result<()> {
    info!("Starting server at {}:{}", host, port);

    HttpServer::new(move || {
        App::new().wrap(crate::middleware::logging::Logging).configure(crate::routes::configure)
    })
    .bind((host, port))?
    .run()
    .await?;

    Ok(())
}
