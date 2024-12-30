use crate::error::AppError;
use crate::service::models::yi_coder::loader::ModelLoader;
use crate::service::models::{ModelManager, ModelStatus};
use actix_web::{get, post, web, HttpResponse};
use anyhow::Result;
use log::{debug, info};
use serde::Deserialize;
use serde_json::json;

#[get("")]
pub async fn list_models(manager: web::Data<ModelManager>) -> HttpResponse {
    debug!("{}", t!("logs.handling_request"));
    let status = manager.get_all_model_status().await;
    let models = vec![
        ("yi-coder", t!("models.yi_coder"), t!("models.yi_coder_description")),
        ("deepseek-coder", t!("models.deepseek_coder"), t!("models.deepseek_coder_description")),
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

#[post("/download")]
pub async fn download_model(
    _manager: web::Data<ModelManager>,
    req: web::Json<DownloadRequest>,
) -> Result<HttpResponse, AppError> {
    debug!("{}", t!("download.request", "model_id" => req.model_id));
    let model_id = &req.model_id;
    let config_path = "config/app.yml";

    // Initialize model loader which will download all required files
    let _loader = ModelLoader::new(model_id, config_path).await?;

    info!("{}", t!("download.success", "model_id" => model_id));
    Ok(HttpResponse::Ok().json(json!({
        "status": "success",
        "model_id": model_id
    })))
}

#[derive(Deserialize)]
struct DownloadRequest {
    model_id: String,
}

pub fn routes(cfg: &mut actix_web::web::ServiceConfig) {
    cfg.service(list_models).service(download_model);
}
