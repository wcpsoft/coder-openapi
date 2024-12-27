use actix_web::web;

pub fn chat_routes() -> actix_web::Scope {
    web::scope("/chat").route("", web::get().to(|| async { "Chat routes" }))
}

pub fn model_routes() -> actix_web::Scope {
    web::scope("/models").route("", web::get().to(|| async { "Model routes" }))
}

pub fn download_routes() -> actix_web::Scope {
    web::scope("/download").route("", web::get().to(|| async { "Download routes" }))
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
