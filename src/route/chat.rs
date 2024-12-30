use actix_web::web;

use crate::controller::chat::chat_completion;

pub fn routes(cfg: &mut actix_web::web::ServiceConfig) {
    cfg.service(
        web::scope("/chat")
            .service(web::resource("/completions").route(web::post().to(chat_completion))),
    );
}
