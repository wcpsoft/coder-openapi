use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

use super::{attention::MultiHeadAttention, feed_forward::PositionWiseFeedForward};

pub struct YiCoderTransformer {
    layers: Vec<TransformerLayer>,
}

pub struct TransformerLayer {
use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

use super::{attention::MultiHeadAttention, feed_forward::PositionWiseFeedForward};

pub struct TransformerLayer {
    attention: MultiHeadAttention,
    feed_forward: PositionWiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerLayer {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attention =
            MultiHeadAttention::new(hidden_size, num_attention_heads, vb.pp("attention"))?;
        let feed_forward =
            PositionWiseFeedForward::new(hidden_size, intermediate_size, vb.pp("ffn"))?;

        let weight1 = vb.get((hidden_size,), "norm1.weight")?;
        let bias1 = vb.get((hidden_size,), "norm1.bias")?;
        let norm1 = LayerNorm::new(weight1, bias1, 1e-5);

        let weight2 = vb.get((hidden_size,), "norm2.weight")?;
        let bias2 = vb.get((hidden_size,), "norm2.bias")?;
        let norm2 = LayerNorm::new(weight2, bias2, 1e-5);

        Ok(Self { attention, feed_forward, norm1, norm2 })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Self-attention
        let attention_output = self.attention.forward(input, input, input)?;

        // Add & Norm
        let attention_output = input.add(&attention_output)?;
        let attention_output = Module::forward(&self.norm1, &attention_output)?;

        // Feed forward
        let feed_forward_output = self.feed_forward.forward(&attention_output)?;

        // Add & Norm
        let output = attention_output.add(&feed_forward_output)?;
        let output = Module::forward(&self.norm2, &output)?;

        Ok(output)
    }
}
