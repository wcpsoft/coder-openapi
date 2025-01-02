pub mod infer;
pub mod loader;
pub mod transformer;

use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use crate::service::chat::chat_completion::ChatCompletionParams;
use anyhow::Result;

pub use self::loader::ModelLoader;
pub use self::transformer::{TransformerError, YiCoderTransformer};

pub struct YiCoder {
    _transformer: YiCoderTransformer,
}

impl YiCoder {
    pub fn new() -> Result<Self> {
        let model_key = "yi-coder";
        let config_path = std::env::current_dir()?.join("config/app.yml");
        let config_path_str = config_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Failed to convert config path to string"))?;
        log::info!("正在加载模型");
        let loader = ModelLoader::new(model_key, config_path_str)?;
        log::info!("加载模型完成");
        let transformer = loader.load_transformer()?;
        Ok(Self { _transformer: transformer })
    }

    pub fn infer(
        &self,
        messages: Vec<ChatCompletionMessage>,
        params: ChatCompletionParams,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        // Convert messages to model input format
        let input =
            messages.iter().map(|msg| msg.content.clone()).collect::<Vec<String>>().join("\n");
        log::debug!("messages: {}", input);
        // Get max_tokens with default value if None
        let max_tokens = params.max_tokens.unwrap_or(100);
        log::debug!("max_tokens: {}", max_tokens);
        // Process input through transformer
        let output = self._transformer.process(&input, max_tokens)?;
        log::debug!("output: {}", output);
        // Convert output to chat completion messages
        let response = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: output,
            ..Default::default()
        };

        Ok(vec![response])
    }
}
