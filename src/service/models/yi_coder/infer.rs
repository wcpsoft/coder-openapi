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
    pub fn new(_config: &crate::service::models::deepseek_coder::config::ModelConfig) -> Self {
        log::info!("Initializing Yi Coder with CPU device");
        Self { _device: Device::Cpu, sender: Arc::new(Mutex::new(None)) }
    }

    pub fn set_stream_sender(&self, sender: mpsc::Sender<ChatCompletionMessage>) {
        log::debug!("Setting up stream sender for Yi Coder");
        if let Ok(mut guard) = self.sender.lock() {
            *guard = Some(sender);
            log::debug!("Stream sender successfully initialized");
        } else {
            log::error!("Failed to acquire lock for stream sender initialization");
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
        log::debug!("Starting Yi Coder inference");
        log::debug!("Input messages count: {}", messages.len());
        log::debug!("Inference parameters - temperature: {:?}, top_p: {:?}, n: {:?}, max_tokens: {:?}, stream: {:?}",
            temperature, top_p, n, max_tokens, stream);
        // Validate parameters
        log::debug!("Validating inference parameters");
        let temperature = temperature.unwrap_or(0.7);
        log::debug!("Using temperature: {:.2}", temperature);
        if temperature <= 0.0 || temperature > 2.0 {
            return Err(AppError::InvalidParameter(
                t!("errors.validation.temperature_range").to_string(),
            ));
        }

        let top_p = top_p.unwrap_or(0.9);
        log::debug!("Using top_p: {:.2}", top_p);
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(AppError::InvalidParameter(
                t!("errors.validation.top_p_range").to_string(),
            ));
        }

        let n = n.unwrap_or(1);
        log::debug!("Using n: {}", n);
        if n == 0 {
            return Err(AppError::InvalidParameter(t!("errors.validation.n_range").to_string()));
        }

        let max_tokens = max_tokens.unwrap_or(100);
        log::debug!("Using max_tokens: {}", max_tokens);
        if max_tokens == 0 {
            return Err(AppError::InvalidParameter(
                t!("errors.validation.max_tokens_range").to_string(),
            ));
        }

        // Process input messages
        log::debug!("Processing input messages");
        let prompt = messages
            .iter()
            .map(|msg| format!("{}: {}", msg.role, msg.content))
            .collect::<Vec<String>>()
            .join("\n");

        // Generate response
        log::debug!("Generating response");
        if let Some(true) = stream {
            log::info!("Streaming response requested");
            // Initialize stream sender if not already set
            let sender = self
                .sender
                .lock()
                .map_err(|_| AppError::Generic(t!("errors.stream.lock_failed").to_string()))?
                .clone()
                .ok_or_else(|| {
                    AppError::Generic(t!("errors.stream.sender_not_initialized").to_string())
                })?;

            log::debug!("Creating streaming response message");
            let message = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: format!(
                    "Streaming response (temp: {:.2}, top_p: {:.2}, n: {}, max_tokens: {})...",
                    temperature, top_p, n, max_tokens
                ),
            };

            log::debug!("Sending streaming response");
            sender.send(message.clone()).await.map_err(|e| {
                log::error!("Failed to send streaming response: {}", e);
                AppError::Generic(format!(
                    "{}: {}",
                    t!("errors.stream_response.failed").to_string(),
                    e
                ))
            })?;

            log::debug!("Streaming response sent successfully");
            Ok(vec![message])
        } else {
            // Handle single response
            log::info!("Generating single response");
            let response = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: format!(
                    "Processed prompt (temp: {:.2}, top_p: {:.2}, n: {}, max_tokens: {}):\n{}",
                    temperature, top_p, n, max_tokens, prompt
                ),
            };
            log::debug!("Generated response: {:?}", response);
            Ok(vec![response])
        }
    }

    pub fn send_stream_response(&self, message: &ChatCompletionMessage) -> Result<(), AppError> {
        log::debug!("Attempting to send stream response");
        if let Some(sender) = self
            .sender
            .lock()
            .map_err(|_| AppError::Generic(t!("errors.stream.lock_failed").to_string()))?
            .as_ref()
        {
            log::debug!("Sending stream response message");
            match sender.try_send(message.clone()) {
                Ok(_) => Ok(()),
                Err(e) => {
                    log::error!("Failed to send stream response: {:?}", e);
                    match e {
                        mpsc::error::TrySendError::Full(_) => {
                            Err(AppError::Generic(t!("errors.stream.buffer_full").to_string()))
                        }
                        mpsc::error::TrySendError::Closed(_) => {
                            Err(AppError::Generic(t!("errors.stream.channel_closed").to_string()))
                        }
                    }
                }
            }
        } else {
            Err(AppError::Generic(t!("errors.stream.sender_not_initialized").to_string()))
        }
    }
}
