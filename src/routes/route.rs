use actix_web::web;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct RouteConfig {
    routes: V1Routes,
}

#[derive(Debug, Deserialize)]
struct V1Routes {
    chat: String,
    models: String,
    download: String,
}

pub fn chat_routes() -> actix_web::Scope {
    let config = load_route_config();
    web::scope(&config.routes.chat)
        .service(web::resource("").route(web::get().to(|| async move { "Chat API" })))
        .service(
            web::resource("/completions")
                .route(web::post().to(crate::controller::chat::chat_completion::chat_completion)),
        )
}

pub fn model_routes() -> actix_web::Scope {
    let config = load_route_config();
    actix_web::web::scope(&config.routes.models)
        .configure(crate::controller::models::models::routes)
}

pub fn download_routes() -> actix_web::Scope {
    let config = load_route_config();
    web::scope(&config.routes.download).route("", web::get().to(|| async move { "Download API" }))
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    let chat_service = crate::service::chat::ChatService::new();
    let model_manager = crate::service::models::ModelManager::new();

    cfg.service(
        web::scope("/v1")
            .app_data(web::Data::new(chat_service))
            .app_data(web::Data::new(model_manager))
            .service(chat_routes())
            .service(model_routes())
            .service(download_routes()),
    );
}

fn load_route_config() -> RouteConfig {
    let config_str = std::fs::read_to_string("config/route.yml").expect("Failed to read route.yml");
    serde_yaml::from_str(&config_str).expect("Failed to parse route.yml")
}
