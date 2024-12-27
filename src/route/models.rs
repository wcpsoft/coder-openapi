use actix_web::web;

pub fn routes() -> actix_web::Scope {
    web::scope("/models")
        // Add model routes here
        .route("", web::get().to(|| async { "Model routes" }))
}

pub fn download_routes() -> actix_web::Scope {
    web::scope("/download")
        // Add download routes here
        .route("", web::get().to(|| async { "Download routes" }))
}
