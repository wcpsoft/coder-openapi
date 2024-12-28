use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use crate::service::models::deepseek_coder::DeepseekCoder;
use crate::service::models::yi_coder::YiCoder;

#[derive(Debug)]
pub struct ChatCompletionParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<usize>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
}

pub struct ChatCompletionService;

impl Default for ChatCompletionService {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatCompletionService {
    pub fn new() -> Self {
        Self
    }

    pub async fn complete(
        &self,
        model: &str,
        messages: Vec<ChatCompletionMessage>,
        params: ChatCompletionParams,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        // TODO: Implement actual model selection and inference
        match model {
            "deepseek-coder" => {
                let model = DeepseekCoder::new().await?;
                model.infer(messages, params).await
            }
            "yi-coder" => {
                let model = YiCoder::new().await?;
                model.infer(messages, params).await
            }
            _ => Err(AppError::InvalidModel(model.to_string())),
        }
    }
}
