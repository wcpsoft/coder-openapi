use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::service::chat::chat_completion::{ChatCompletionParams, ChatCompletionService};
use crate::utils::config::get_config;
use actix_web::{web, HttpResponse};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<usize>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: DateTime<Utc>,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub async fn chat_completion(req: web::Json<ChatCompletionRequest>) -> HttpResponse {
    log::info!("Received chat completion request for model: {}", req.model);
    log::debug!("Request details: {:?}", req);

    // Validate required fields
    if req.model.is_empty() {
        log::warn!("Empty model field in request");
        return HttpResponse::BadRequest().json("model field is required");
    }
    if req.messages.is_empty() {
        log::warn!("Empty messages field in request");
        return HttpResponse::BadRequest().json("messages field cannot be empty");
    }

    log::debug!("Request validation passed");

    let service = ChatCompletionService::new();
    let config = get_config();
    let chat_config = &config.chat;

    let params = ChatCompletionParams {
        temperature: req.temperature.or(Some(chat_config.defaults.temperature)),
        top_p: req.top_p.or(Some(chat_config.defaults.top_p)),
        n: req.n.or(Some(chat_config.defaults.n)),
        max_tokens: req.max_tokens.or(Some(chat_config.defaults.max_tokens)),
        stream: req.stream.or(Some(chat_config.defaults.stream)),
    };

    log::debug!("Using completion parameters: {:?}", params);

    match service.complete(&req.model, req.messages.clone(), params).await {
        Ok(messages) => {
            log::info!("Successfully completed chat request for model: {}", req.model);
            let response = ChatCompletionResponse {
                id: Uuid::new_v4().to_string(),
                object: "chat.completion".to_string(),
                created: Utc::now(),
                model: req.model.clone(),
                choices: messages
                    .into_iter()
                    .map(|message| Choice { message, finish_reason: "stop".to_string() })
                    .collect(),
                usage: Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
            };
            HttpResponse::Ok().json(response)
        }
        Err(e) => {
            log::error!("Error completing chat request: {}", e);
            HttpResponse::InternalServerError().json(e.to_string())
        }
    }
}
