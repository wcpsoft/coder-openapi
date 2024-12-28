use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::service::chat::chat_completion::ChatCompletionService;
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
    let service = ChatCompletionService::new();
    let config = get_config();
    let chat_config = &config.chat;

    match service
        .complete(
            &req.model,
            req.messages.clone(),
            req.temperature.or(Some(chat_config.defaults.temperature)),
            req.top_p.or(Some(chat_config.defaults.top_p)),
            req.n.or(Some(chat_config.defaults.n)),
            req.max_tokens.or(Some(chat_config.defaults.max_tokens)),
            req.stream.or(Some(chat_config.defaults.stream)),
        )
        .await
    {
        Ok(messages) => {
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
        Err(e) => HttpResponse::InternalServerError().json(e.to_string()),
    }
}
