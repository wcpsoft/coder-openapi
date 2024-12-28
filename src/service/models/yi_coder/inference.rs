use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use candle_core::Device;

pub struct YiCoderInference {
    _device: Device,
}

impl YiCoderInference {
    pub fn new(_config: &super::config::ModelConfig) -> Self {
        Self { _device: Device::Cpu }
    }

    pub async fn infer(
        &self,
        _messages: Vec<ChatCompletionMessage>,
        _temperature: Option<f32>,
        _top_p: Option<f32>,
        _n: Option<usize>,
        _max_tokens: Option<usize>,
        _stream: Option<bool>,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        // TODO: Implement actual inference logic
        Ok(vec![ChatCompletionMessage {
            role: "assistant".to_string(),
            content: "Yi Coder response".to_string(),
        }])
    }
}
