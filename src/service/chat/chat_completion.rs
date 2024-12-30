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
        log::debug!("Starting completion for model: {}", model);
        log::debug!("Input messages count: {}", messages.len());
        log::debug!("Completion params: {:?}", params);

        let result = match model {
            "deepseek-coder" => {
                log::info!("Initializing Deepseek Coder model");
                let model = DeepseekCoder::new().await?;
                log::info!("Starting Deepseek Coder inference");
                model.infer(messages, params).await
            }
            "yi-coder" => {
                log::info!("Initializing Yi Coder model");
                let model = YiCoder::new().await?;
                log::info!("Starting Yi Coder inference");
                model.infer(messages, params).await
            }
            _ => {
                log::error!("Invalid model requested: {}", model);
                Err(AppError::InvalidModel(model.to_string()))
            }
        };

        match &result {
            Ok(messages) => log::debug!("Successfully generated {} messages", messages.len()),
            Err(e) => log::error!("Error during completion: {}", e),
        }

        result
    }
}
