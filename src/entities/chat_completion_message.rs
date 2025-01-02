use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}
