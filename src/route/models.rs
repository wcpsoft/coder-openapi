use crate::controller::models::routes;
use crate::utils::config::CONFIG;
use actix_web::web;

pub fn models_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope(&CONFIG.get().expect("Failed to get config").server.model_route)
            .configure(routes),
    );
}
