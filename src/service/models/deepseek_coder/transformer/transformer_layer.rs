use candle_core::{Module, Result, Tensor};
use candle_nn::LayerNorm;

use super::attention::MultiHeadAttention;
use super::feed_forward::PositionWiseFeedForward;

/// Single Transformer Layer
#[derive(Debug)]
pub struct TransformerLayer {
    attention: MultiHeadAttention,
    feed_forward: PositionWiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerLayer {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        vb: candle_nn::VarBuilder,
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

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(input, input, input, None)?;
        let attention_output = self.norm1.forward(&(input + &attention_output)?)?;

        let feed_forward_output = self.feed_forward.forward(&attention_output)?;
        self.norm2.forward(&(attention_output + &feed_forward_output)?)
    }
}
