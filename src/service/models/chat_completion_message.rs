use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub sender: String,
    pub role: String,
    pub content: String,
    pub timestamp: u64,
    pub model: Option<String>,
    pub temperature: Option<f32>,
}

impl ChatCompletionMessage {
    pub fn new_chat(sender: String, role: String, content: String, model: Option<String>) -> Self {
        Self {
            sender,
            role,
            content,
            timestamp: 0, // Will be set when sent
            model,
            temperature: None,
        }
    }

    pub fn new_completion(
        sender: String,
        role: String,
        content: String,
        temperature: Option<f32>,
    ) -> Self {
        Self {
            sender,
            role,
            content,
            timestamp: 0, // Will be set when sent
            model: None,
            temperature,
        }
    }

    pub fn new_message(sender: String, content: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            sender,
            role: "user".to_string(),
            content,
            timestamp,
            model: None,
            temperature: None,
        }
    }
}
