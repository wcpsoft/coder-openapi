use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use crate::service::models::deepseek_coder::{
    config::ModelConfig as DeepSeekConfig, DeepSeekCoder,
};
use crate::service::models::yi_coder::YiCoder;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// 聊天完成参数
#[derive(Debug)]
pub struct ChatCompletionParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<usize>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
}

/// 聊天完成服务
pub struct ChatCompletionService;

impl Default for ChatCompletionService {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatCompletionService {
    /// 创建新的聊天完成服务实例
    pub fn new() -> Self {
        Self
    }

    /// 完成聊天请求
    ///
    /// # 参数
    /// - model: 模型名称
    /// - messages: 聊天消息列表
    /// - params: 完成参数
    ///
    /// # 返回
    /// 生成的聊天消息列表
    pub fn complete(
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
                let config =
                    crate::utils::config::get_config().get_model_config("deepseek-coder")?;
                let deepseek_config = DeepSeekConfig {
                    hidden_size: config.hidden_size.unwrap_or(4096) as usize,
                    num_attention_heads: config.num_attention_heads.unwrap_or(32) as usize,
                    num_hidden_layers: config.num_hidden_layers.unwrap_or(32) as usize,
                    intermediate_size: config.intermediate_size.unwrap_or(11008) as usize,
                    vocab_size: config.vocab_size.unwrap_or(32000) as usize,
                    num_layers: config.num_hidden_layers.unwrap_or(32) as usize,
                    bos_token_id: 1,
                    eos_token_id: 2,
                    pad_token_id: 0,
                    temperature: 0.7,
                    top_p: 0.9,
                    max_tokens: 2048,
                    layer_norm_eps: 1e-5,
                    tokenizer_path: format!(
                        "{}/{}/{}",
                        crate::utils::config::get_config().models_cache_dir,
                        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                        "tokenizer.json"
                    ),
                };
                let vb = VarBuilder::from_tensors(HashMap::new(), DType::F32, &Device::Cpu);
                let model = DeepSeekCoder::new(vb, &deepseek_config)
                    .map_err(|e| AppError::Transformer(e.to_string()))?;
                log::info!("Starting Deepseek Coder inference");
                model.infer(messages, params)
            }
            "yi-coder" => {
                log::info!("Initializing Yi Coder model");
                let model = YiCoder::new()?;
                log::info!("Starting Yi Coder inference");
                model.infer(messages, params)
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
