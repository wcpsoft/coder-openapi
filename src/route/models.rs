use crate::controller::models::routes;
use crate::utils::config::load_route_config;
use actix_web::web;

pub fn models_routes(cfg: &mut web::ServiceConfig) {
    let route_config = load_route_config().expect("Failed to load route config");
    cfg.service(web::scope(&route_config.routes.v1.models).configure(routes));
}
