use super::config::ModelConfig;
use crate::error::AppError;
use candle_core::DType;
use candle_core::{Device, Tensor};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

pub struct DeepseekCoderLoader {
    config: ModelConfig,
    device: Device,
    tokenizer: Option<Tokenizer>,
}

impl DeepseekCoderLoader {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            tokenizer: None,
        }
    }

    pub fn get_var_builder(&self) -> Result<candle_nn::VarBuilder, AppError> {
        let mut tensors = std::collections::HashMap::new();
        let _zeros_data = vec![0.0f32; self.config.hidden_size];
        let shape = vec![self.config.hidden_size];
        let zeros = Tensor::zeros(shape, candle_core::DType::F32, &self.device)?;
        tensors.insert("zeros".to_string(), zeros);
        Ok(candle_nn::VarBuilder::from_tensors(tensors, DType::F32, &self.device))
    }

    pub async fn get_tokenizer(&self) -> Result<Tokenizer, AppError> {
        let tokenizer_path = format!(
            "{}/{}/{}",
            self.config.models_cache_dir, self.config.hf_hub_id, self.config.model_files.tokenizer
        );
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| AppError::TokenizerError(e.to_string()))?;
        Ok(tokenizer)
    }

    pub async fn load_weights(&self) -> Result<Vec<Tensor>, AppError> {
        let weights_path = format!(
            "{}/{}/{}",
            self.config.models_cache_dir, self.config.hf_hub_id, self.config.model_files.weights[0]
        );
        let data = tokio::fs::read(weights_path).await?;
        let safetensors = SafeTensors::deserialize(&data)?;

        let mut tensors = Vec::new();
        for (_name, tensor_view) in safetensors.tensors() {
            let tensor = Tensor::from_slice(tensor_view.data(), tensor_view.shape(), &self.device)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    pub async fn load_tokenizer(&mut self) -> Result<(), AppError> {
        let tokenizer_path = format!(
            "{}/{}/{}",
            self.config.models_cache_dir, self.config.hf_hub_id, self.config.model_files.tokenizer
        );
        let tokenizer_data = tokio::fs::read(tokenizer_path).await?;
        self.tokenizer = Some(Tokenizer::from_bytes(&tokenizer_data).map_err(|e| {
            AppError::TokenizerError(format!("Failed to initialize tokenizer: {}", e))
        })?);
        Ok(())
    }

    pub async fn load_config(&self) -> Result<ModelConfig, AppError> {
        let config_path =
            "models_cache/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/config.json".to_string();
        let config_str = tokio::fs::read_to_string(config_path).await?;
        let config: ModelConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    }

    pub async fn initialize(&self) -> Result<(), AppError> {
        let _config = self.load_config().await?;
        Ok(())
    }
}
