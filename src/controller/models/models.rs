use crate::error::AppError;
use crate::service::models::{ModelManager, ModelStatus};
pub use crate::Locales;
use actix_web::{get, post, web, HttpResponse};
use anyhow::Result;
use log::info;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[get("/models")]
pub async fn list_models(
    manager: web::Data<ModelManager>,
    locales: web::Data<Arc<Locales>>,
) -> HttpResponse {
    let status = manager.get_all_model_status().await;
    let models = vec![
        ("yi-coder", locales.t("models.yi_coder"), locales.t("models.yi_coder_description")),
        (
            "deepseek-coder",
            locales.t("models.deepseek_coder"),
            locales.t("models.deepseek_coder_description"),
        ),
    ];

    let response = models
        .into_iter()
        .map(|(id, name, description)| {
            let status = status
                .get(id)
                .cloned()
                .unwrap_or(ModelStatus { is_cached: false, is_enabled: false });
            json!({
                "id": id,
                "name": name,
                "description": description,
                "is_cached": status.is_cached,
                "is_enabled": status.is_enabled
            })
        })
        .collect::<Vec<_>>();

    HttpResponse::Ok().json(json!({ "models": response }))
}

#[post("")]
pub async fn download_model(
    manager: web::Data<ModelManager>,
    _locales: web::Data<Arc<Locales>>,
    req: web::Json<DownloadRequest>,
) -> Result<HttpResponse, AppError> {
    info!("Received download request: {}", req.model_id);
    let model_id = &req.model_id;
    manager.download_model(model_id, "config/app.yml").await?;
    info!("Successfully downloaded model: {}", model_id);
    Ok(HttpResponse::Ok().json(json!({
        "status": "success",
        "model_id": model_id
    })))
}

#[derive(Deserialize)]
struct DownloadRequest {
    model_id: String,
}

pub fn routes() -> actix_web::Scope {
    web::scope("/models").service(list_models).service(download_model)
}
