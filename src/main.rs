use actix_web::{web, App, HttpServer};
use anyhow::Context;
use std::sync::Arc;

use coder_openapi::routes;
use coder_openapi::utils::init;
use coder_openapi::Locales;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize application
    let config = init::init()
        .await
        .context("Failed to initialize application")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // Initialize locale system
    let mut locales = Locales::new(&config.locales.path.clone()).expect("Failed to load locales");
    locales.set_default(&config.locales.default.clone()).expect("Failed to set default locale");
    let locales = Arc::new(locales);

    HttpServer::new(move || {
        App::new().app_data(web::Data::new(locales.clone())).configure(routes::route::configure)
    })
    .bind((config.server.host.clone(), config.server.port))?
    .run()
    .await
}
