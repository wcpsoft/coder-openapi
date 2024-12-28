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
        // Process input messages
        let prompt = messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<String>>()
            .join("\n");

        // Apply generation parameters
        let temperature = temperature.unwrap_or(0.7);
        let top_p = top_p.unwrap_or(0.9);
        let n = n.unwrap_or(1);
        let max_tokens = max_tokens.unwrap_or(100);

        // Generate response
        let response = if let Some(true) = stream {
            // Handle streaming response
            if let Some(sender) = self.sender.lock().unwrap().as_ref() {
                let message = ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: "Streaming response...".to_string(),
                };
                sender.send(message.clone()).await.map_err(|e| {
                    AppError::Generic(format!("Failed to send stream message: {}", e))
                })?;
                vec![message]
            } else {
                return Err(AppError::Generic("Stream sender not initialized".to_string()));
            }
        } else {
            // Handle single response
            vec![ChatCompletionMessage {
                role: "assistant".to_string(),
                content: format!("Processed prompt:\n{}", prompt),
            }]
        };

        Ok(response)
    }

    pub fn send_stream_response(&self, message: &ChatCompletionMessage) -> Result<(), AppError> {
        if let Some(sender) = self.sender.lock().unwrap().as_ref() {
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
