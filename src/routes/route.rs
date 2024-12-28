use crate::utils::config::load_route_config;
use actix_web::web;

pub fn chat_routes() -> actix_web::Scope {
    let config = load_route_config();
    web::scope(&config.routes.v1.chat)
        .service(
            web::resource("").route(web::get().to(|| async move { "Chat API" })).name("chat_root"),
        )
        .service(
            web::resource("/completions")
                .route(web::post().to(crate::controller::chat::chat_completion::chat_completion))
                .name("chat_completions"),
        )
}

pub fn model_routes() -> actix_web::Scope {
    let config = load_route_config();
    actix_web::web::scope(&config.routes.v1.models)
        .configure(crate::controller::models::models::routes)
}

pub fn download_routes() -> actix_web::Scope {
    let config = load_route_config();
    web::scope(&config.routes.v1.download)
        .route("", web::get().to(|| async move { "Download API" }))
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
