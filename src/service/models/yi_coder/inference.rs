use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use candle_core::Device;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub struct YiCoderInference {
    _device: Device,
    sender: Arc<Mutex<Option<mpsc::Sender<ChatCompletionMessage>>>>,
}

impl YiCoderInference {
    pub fn new(_config: &super::config::ModelConfig) -> Self {
        Self { _device: Device::Cpu, sender: Arc::new(Mutex::new(None)) }
    }

    pub fn set_stream_sender(&self, sender: mpsc::Sender<ChatCompletionMessage>) {
        if let Ok(mut guard) = self.sender.lock() {
            *guard = Some(sender);
        }
    }

    pub async fn infer(
        &self,
        messages: Vec<ChatCompletionMessage>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        n: Option<usize>,
        max_tokens: Option<usize>,
        stream: Option<bool>,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        // Validate parameters
        let temperature = temperature.unwrap_or(0.7);
        if temperature <= 0.0 || temperature > 2.0 {
            return Err(AppError::InvalidParameter(
                "temperature must be between 0 and 2".to_string(),
            ));
        }

        let top_p = top_p.unwrap_or(0.9);
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(AppError::InvalidParameter("top_p must be between 0 and 1".to_string()));
        }

        let n = n.unwrap_or(1);
        if n == 0 {
            return Err(AppError::InvalidParameter("n must be greater than 0".to_string()));
        }

        let max_tokens = max_tokens.unwrap_or(100);
        if max_tokens == 0 {
            return Err(AppError::InvalidParameter(
                "max_tokens must be greater than 0".to_string(),
            ));
        }

        // Process input messages
        let prompt = messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<String>>()
            .join("\n");

        // Generate response
        if let Some(true) = stream {
            // Initialize stream sender if not already set
            let sender = self
                .sender
                .lock()
                .map_err(|_| AppError::Generic("Failed to lock sender".to_string()))?
                .clone()
                .ok_or_else(|| {
                    AppError::Generic(
                        "Stream sender not initialized. Call set_stream_sender() first".to_string(),
                    )
                })?;

            let message = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: format!(
                    "Streaming response (temp: {:.2}, top_p: {:.2}, n: {}, max_tokens: {})...",
                    temperature, top_p, n, max_tokens
                ),
            };

            sender
                .send(message.clone())
                .await
                .map_err(|e| AppError::Generic(format!("Failed to send stream message: {}", e)))?;

            Ok(vec![message])
        } else {
            // Handle single response
            Ok(vec![ChatCompletionMessage {
                role: "assistant".to_string(),
                content: format!(
                    "Processed prompt (temp: {:.2}, top_p: {:.2}, n: {}, max_tokens: {}):\n{}",
                    temperature, top_p, n, max_tokens, prompt
                ),
            }])
        }
    }

    pub fn send_stream_response(&self, message: &ChatCompletionMessage) -> Result<(), AppError> {
        if let Some(sender) = self
            .sender
            .lock()
            .map_err(|_| AppError::Generic("Failed to lock sender".to_string()))?
            .as_ref()
        {
            sender.try_send(message.clone()).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => {
                    AppError::Generic("Stream buffer full".to_string())
                }
                mpsc::error::TrySendError::Closed(_) => {
                    AppError::Generic("Stream channel closed".to_string())
                }
            })?;
            Ok(())
        } else {
            Err(AppError::Generic("Stream sender not initialized".to_string()))
        }
    }
}
