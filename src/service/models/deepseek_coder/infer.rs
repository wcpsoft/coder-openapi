use super::transformer::DeepseekCoderTransformer;
use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::sync::Arc;

pub struct DeepSeekCoderInference {
    _device: Device,
    _config: super::config::ModelConfig,
    _transformer: Option<Arc<DeepseekCoderTransformer>>,
}

impl DeepSeekCoderInference {
    pub fn new(config: super::config::ModelConfig) -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        log::info!("Initializing Deepseek Coder with device: {:?}", device);
        Self {
            _device: device,
            _config: config,
            _transformer: None,
        }
    }

    pub fn prepare_input(&self, messages: &[ChatCompletionMessage]) -> Result<Tensor, AppError> {
        // Convert messages to token IDs
        let token_ids: Vec<u32> = messages
            .iter()
            .flat_map(|msg| msg.content.chars().map(|c| c as u32))
            .collect();
        
        Tensor::new(&token_ids, &self._device)
            .map_err(|e| AppError::new(format!("Failed to create input tensor: {}", e)))
    }

    fn load_transformer(&mut self) -> Result<Arc<DeepseekCoderTransformer>, AppError> {
        if self._transformer.is_none() {
            let vb = VarBuilder::zeros(DType::F32, &self._device);
            let transformer = DeepseekCoderTransformer::new(&self._config, vb)?;
            self._transformer = Some(Arc::new(transformer));
        }
        Ok(self._transformer.as_ref().unwrap().clone())
    }

    fn process_output(&self, output: Tensor) -> Result<ChatCompletionMessage, AppError> {
        // Convert output tensor to text
        let tokens: Vec<u32> = output.to_vec1()?;
        let content = tokens
            .into_iter()
            .map(|t| char::from_u32(t).unwrap_or(''))
            .collect::<String>();
        
        Ok(ChatCompletionMessage {
            role: "assistant".to_string(),
            content,
        })
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
        log::debug!("Starting Deepseek Coder inference");
        log::debug!("Input messages count: {}", messages.len());
        log::debug!("Inference parameters - temperature: {:?}, top_p: {:?}, n: {:?}, max_tokens: {:?}, stream: {:?}",
            temperature, top_p, n, max_tokens, stream);

        // Prepare input tensor from messages
        let input_ids = self.prepare_input(&messages)?;

        // Get transformer instance
        let transformer = self.load_transformer()?;

        // Run transformer forward pass
        let output = transformer.forward(&input_ids, None)?;

        // Process output to generate response
        let response = self.process_output(output)?;

        log::debug!("Generated response: {:?}", response);
        Ok(vec![response])
    }
}
