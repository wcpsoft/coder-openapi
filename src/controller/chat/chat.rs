use crate::entities::models::{DeepseekCoderModel, YiCoderModel};
use crate::service::chat::ChatService;
use crate::service::models::ModelManager;
use actix_web::{post, web, HttpResponse, ResponseError};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;

pub trait ModelResponseGenerator {
    fn generate_response(
        self,
        input: &str,
    ) -> impl std::future::Future<Output = anyhow::Result<String>> + Send;
}

impl ModelResponseGenerator for YiCoderModel {
    async fn generate_response(self, input: &str) -> anyhow::Result<String> {
        // TODO: Implement actual response generation
        Ok(format!("Yi-Coder response for: {}", input))
    }
}

impl ModelResponseGenerator for DeepseekCoderModel {
    async fn generate_response(self, input: &str) -> anyhow::Result<String> {
        // TODO: Implement actual response generation
        Ok(format!("Deepseek-Coder response for: {}", input))
    }
}

pub mod error {
    use super::*;

    #[derive(Debug)]
    pub enum ChatError {
        ModelNotAvailable,
        ModelNotFound,
        ModelNotLoaded(String),
        OutputProcessingFailed(String),
    }

    impl fmt::Display for ChatError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ChatError::ModelNotAvailable => write!(f, "Model not available"),
                ChatError::ModelNotFound => write!(f, "Model not found"),
                ChatError::ModelNotLoaded(e) => write!(f, "Model not loaded: {}", e),
                ChatError::OutputProcessingFailed(e) => {
                    write!(f, "Output processing failed: {}", e)
                }
            }
        }
    }

    impl ResponseError for ChatError {
        fn error_response(&self) -> HttpResponse {
            match self {
                ChatError::ModelNotAvailable => HttpResponse::BadRequest().json(json!({
                    "error": "Model not available",
                    "message": "Please download the model first"
                })),
                ChatError::ModelNotFound => HttpResponse::NotFound().json(json!({
                    "error": "Model not found"
                })),
                ChatError::ModelNotLoaded(e) => HttpResponse::InternalServerError().json(json!({
                    "error": "Model not loaded",
                    "message": e
                })),
                ChatError::OutputProcessingFailed(e) => {
                    HttpResponse::InternalServerError().json(json!({
                        "error": "Output processing failed",
                        "message": e
                    }))
                }
            }
        }
    }
}

use error::ChatError;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[post("/chat/completions")]
pub async fn chat_completions(
    manager: web::Data<ModelManager>,
    _chat_service: web::Data<ChatService>,
    req: web::Json<ChatRequest>,
) -> Result<HttpResponse, ChatError> {
    // Check if model is cached and enabled
    let model_status = manager.get_model_status(&req.model).await;
    if let Some(status) = model_status {
        if !status.is_cached || !status.is_enabled {
            return Err(ChatError::ModelNotAvailable);
        }
    } else {
        return Err(ChatError::ModelNotFound);
    }

    // Get and use the appropriate model based on the request
    let response = match req.model.as_str() {
        "yi-coder" => manager
            .get_yi_coder()
            .await
            .ok_or(ChatError::ModelNotAvailable)?
            .generate_response(&req.messages[0].content)
            .await
            .map_err(|e| ChatError::OutputProcessingFailed(e.to_string()))?,
        "deepseek-coder" => manager
            .get_deepseek_coder()
            .await
            .ok_or(ChatError::ModelNotAvailable)?
            .generate_response(&req.messages[0].content)
            .await
            .map_err(|e| ChatError::OutputProcessingFailed(e.to_string()))?,
        _ => return Err(ChatError::ModelNotFound),
    };

    Ok(HttpResponse::Ok().json(json!({
        "model": req.model,
        "response": response
    })))
}

pub fn routes() -> actix_web::Scope {
    web::scope("/chat").service(chat_completions)
}
