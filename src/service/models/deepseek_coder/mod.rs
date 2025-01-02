use crate::error::AppError;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::path::Path;

pub mod transformer;

use crate::service::models::deepseek_coder::transformer::{
    DeepSeekCoderDecoder, DeepSeekCoderEncoder, TransformerError,
};

#[derive(Debug, Deserialize, Clone)]
pub struct DeepSeekCoderConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub layer_norm_eps: f64,
    pub vocab_size: usize,
}

pub struct DeepSeekCoder {
    encoder: DeepSeekCoderEncoder,
    decoder: DeepSeekCoderDecoder,
    device: Device,
}

impl DeepSeekCoder {
    pub fn new(config: &DeepSeekCoderConfig, vb: VarBuilder) -> Result<Self, TransformerError> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let encoder = DeepSeekCoderEncoder::new(config, vb.pp("encoder"))?;
        let decoder = DeepSeekCoderDecoder::new(config, vb.pp("decoder"))?;

        Ok(Self { encoder, decoder, device })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        encoder_output: &Tensor,
        self_attention_mask: Option<&Tensor>,
        cross_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, TransformerError> {
        let encoder_output = self.encoder.forward(input, None)?;
        self.decoder.forward(input, &encoder_output, self_attention_mask, cross_attention_mask)
    }
}

impl DeepSeekCoderConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, AppError> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&config_str)?;
        Ok(config)
    }
}
