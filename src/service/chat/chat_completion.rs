use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use crate::service::models::deepseek_coder::DeepseekCoder;
use crate::service::models::yi_coder::YiCoder;

pub struct ChatCompletionService;

impl ChatCompletionService {
    pub fn new() -> Self {
        Self
    }

    pub async fn complete(
        &self,
        model: &str,
        messages: Vec<ChatCompletionMessage>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        n: Option<usize>,
        max_tokens: Option<usize>,
        stream: Option<bool>,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        // TODO: Implement actual model selection and inference
        match model {
            "deepseek-coder" => {
                let model = DeepseekCoder::new().await?;
                model.infer(messages, temperature, top_p, n, max_tokens, stream).await
            }
            "yi-coder" => {
                let model = YiCoder::new().await?;
                model.infer(messages, temperature, top_p, n, max_tokens, stream).await
            }
            _ => Err(AppError::InvalidModel(model.to_string())),
        }
    }
}
