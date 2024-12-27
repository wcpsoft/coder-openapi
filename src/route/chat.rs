use actix_web::web;

pub fn routes() -> actix_web::Scope {
    web::scope("/chat")
        // Add chat routes here
        .route("", web::get().to(|| async { "Chat routes" }))
}
