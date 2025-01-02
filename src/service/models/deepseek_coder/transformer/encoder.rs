use crate::service::models::deepseek_coder::ModelConfig;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

use super::attention::MultiHeadAttention;
use super::feed_forward::PositionWiseFeedForward;

/// DeepSeekCoder Transformer Encoder
/// Implements the encoder part of the Transformer architecture
#[derive(Debug)]
pub struct DeepSeekCoderEncoder {
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
    _device: Device,
}

/// Single Encoder Layer
#[derive(Debug)]
struct EncoderLayer {
    attention: MultiHeadAttention,
    feed_forward: PositionWiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl DeepSeekCoderEncoder {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = EncoderLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                vb.pp(format!("layer_{}", i)),
            )?;
            layers.push(layer);
        }

        let norm = LayerNorm::new(
            vb.get((config.hidden_size,), "model.norm.weight")?,
            vb.get((config.hidden_size,), "model.norm.bias")?,
            config.layer_norm_eps,
        );

        Ok(Self { layers, norm, _device: device })
    }

    pub fn forward(&self, input: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = input.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        self.norm.forward(&hidden_states)
    }
}

impl EncoderLayer {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attention = MultiHeadAttention::new(hidden_size, num_heads, vb.pp("attention"))?;
        let feed_forward =
            PositionWiseFeedForward::new(hidden_size, intermediate_size, vb.pp("ffn"))?;

        let norm1 = LayerNorm::new(
            vb.get((hidden_size,), "input_layernorm.weight")?,
            vb.get((hidden_size,), "input_layernorm.bias")?,
            1e-5,
        );

        let norm2 = LayerNorm::new(
            vb.get((hidden_size,), "post_attention_layernorm.weight")?,
            vb.get((hidden_size,), "post_attention_layernorm.bias")?,
            1e-5,
        );

        Ok(Self { attention, feed_forward, norm1, norm2 })
    }

    fn forward(&self, input: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let attention_output = self.attention.forward(input, input, input, attention_mask)?;
        let attention_output = self.norm1.forward(&(input + &attention_output)?)?;

        let feed_forward_output = self.feed_forward.forward(&attention_output)?;
        self.norm2.forward(&(attention_output + &feed_forward_output)?)
    }
}
