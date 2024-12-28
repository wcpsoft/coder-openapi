use crate::Locales;
use actix_web::web;
use std::sync::Arc;

pub fn chat_routes() -> actix_web::Scope {
    web::scope("/chat")
        .service(
            web::resource("")
                .route(web::get().to(|locales: web::Data<Arc<Locales>>| async move {
                    locales.t("routes.chat")
                })),
        )
        .service(
            web::resource("/completions")
                .route(web::post().to(crate::controller::chat::chat_completion::chat_completion)),
        )
}

pub fn model_routes() -> actix_web::Scope {
    crate::controller::models::models::routes()
}

pub fn download_routes() -> actix_web::Scope {
    web::scope("/download").route(
        "",
        web::get()
            .to(|locales: web::Data<Arc<Locales>>| async move { locales.t("routes.download") }),
    )
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
