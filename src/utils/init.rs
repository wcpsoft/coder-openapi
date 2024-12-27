use actix_web::{App, HttpServer};
use log::info;

pub async fn init() -> crate::error::Result<()> {
    // Initialize logging
    log4rs::init_file("config/log4rs.yml", Default::default())?;
    info!("Application initialized");
    Ok(())
}

pub async fn run() -> crate::error::Result<()> {
    info!("Starting server at 127.0.0.1:8080");

    HttpServer::new(move || {
        App::new()
            .wrap(crate::middleware::logging::Logging)
            .configure(crate::routes::configure)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await?;

    Ok(())
}
