use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, LayerNorm, VarBuilder};

use super::attention::MultiHeadAttention;
use super::feed_forward::PositionWiseFeedForward;

/// DeepSeekCoder Transformer Decoder
/// Implements the decoder part of the Transformer architecture
#[derive(Debug)]
pub struct DeepSeekCoderDecoder {
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    device: Device,
}

/// Single Decoder Layer
#[derive(Debug)]
struct DecoderLayer {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feed_forward: PositionWiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl DeepSeekCoderDecoder {
    pub fn new(config: &super::config::ModelConfig, vb: VarBuilder) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = DecoderLayer::new(
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

        Ok(Self { layers, norm, device })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        encoder_output: &Tensor,
        self_attention_mask: Option<&Tensor>,
        cross_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = input.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                encoder_output,
                self_attention_mask,
                cross_attention_mask,
            )?;
        }

        self.norm.forward(&hidden_states)
    }
}

impl DecoderLayer {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attention =
            MultiHeadAttention::new(hidden_size, num_heads, vb.pp("self_attention"))?;
        let cross_attention =
            MultiHeadAttention::new(hidden_size, num_heads, vb.pp("cross_attention"))?;
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

        let norm3 = LayerNorm::new(
            vb.get((hidden_size,), "post_cross_attention_layernorm.weight")?,
            vb.get((hidden_size,), "post_cross_attention_layernorm.bias")?,
            1e-5,
        );

        Ok(Self { self_attention, cross_attention, feed_forward, norm1, norm2, norm3 })
    }

    fn forward(
        &self,
        input: &Tensor,
        encoder_output: &Tensor,
        self_attention_mask: Option<&Tensor>,
        cross_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self attention
        let self_attention_output =
            self.self_attention.forward(input, input, input, self_attention_mask)?;
        let self_attention_output = self.norm1.forward(&(input + &self_attention_output)?)?;

        // Cross attention
        let cross_attention_output = self.cross_attention.forward(
            &self_attention_output,
            encoder_output,
            encoder_output,
            cross_attention_mask,
        )?;
        let cross_attention_output =
            self.norm2.forward(&(self_attention_output + &cross_attention_output)?)?;

        // Feed forward
        let feed_forward_output = self.feed_forward.forward(&cross_attention_output)?;
        self.norm3.forward(&(cross_attention_output + &feed_forward_output)?)
    }
}
