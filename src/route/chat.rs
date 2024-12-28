use actix_web::web;

use crate::controller::chat::chat_completion;

pub fn routes() -> actix_web::Scope {
    web::scope("/chat")
        .service(web::resource("/completions").route(web::post().to(chat_completion)))
}
