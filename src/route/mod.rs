pub mod chat;
pub mod models;

pub use crate::controller::models::routes as model_routes;
pub use chat::routes as chat_routes;

use actix_web::web;

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(web::scope("/v1").configure(chat_routes).configure(model_routes));
}
