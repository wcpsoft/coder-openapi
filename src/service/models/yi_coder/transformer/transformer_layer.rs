use candle_core::{DType, Device, Module, Result, Tensor, WithDType};
use candle_nn::{LayerNorm, VarBuilder};

use super::{attention::MultiHeadAttention, feed_forward::PositionWiseFeedForward};

pub struct YiCoderTransformer {
    layers: Vec<TransformerLayer>,
}

impl YiCoderTransformer {
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = TransformerLayer::new(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                1e-5, // TODO: 从config中获取
                vb.pp(&format!("model.layers.{}", i)),
            )?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    pub fn process(&self, input: &str, max_tokens: usize) -> Result<String> {
        // Convert input to tensor with proper shape [1, seq_len, hidden_size]
        let device = Device::cuda_if_available(0)?;
        let bytes = input.as_bytes();
        let seq_len = bytes.len();
        // Create embedding matrix (simple byte to float mapping)
        let mut embeddings = Vec::with_capacity(seq_len * 2048);
        for &byte in bytes {
            let mut embedding = vec![0.0; 2048];
            embedding[byte as usize] = 1.0;
            embeddings.extend(embedding);
        }

        let input_tensor = Tensor::from_slice(&embeddings, (1, seq_len, 2048), &device)?;

        // Process through transformer layers
        let mut output = self.forward(&input_tensor)?;
        // Generate tokens up to max_tokens
        let mut generated = String::new();
        for _ in 0..max_tokens {
            // Get next token from last position in sequence
            let last_token = output.narrow(1, output.dim(1)? - 1, 1)?;
            let logits = last_token.squeeze(1)?; // Remove sequence dimension
            let next_token =
                logits.argmax(1)?.squeeze(0)?.to_dtype(DType::U8)?.to_scalar::<u8>()?;
            generated.push(next_token as char);

            // Update input for next iteration
            let mut embedding = vec![0.0; 2048];
            embedding[next_token as usize] = 1.0;
            let new_token = Tensor::from_slice(&embedding, (1, 1, 2048), output.device())?;
            output = Tensor::cat(&[output, new_token], 1)?;
        }

        Ok(generated)
    }
}

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
        layer_norm_eps: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attention =
            MultiHeadAttention::new(hidden_size, num_attention_heads, vb.pp("self_attn"))?;
        let feed_forward =
            PositionWiseFeedForward::new(hidden_size, intermediate_size, vb.pp("mlp"))?;

        let weight1 = vb.get((hidden_size,), "input_layernorm.weight")?;
        let norm1 = LayerNorm::new_no_bias(weight1, layer_norm_eps.to_f64());

        let weight2 = vb.get((hidden_size,), "post_attention_layernorm.weight")?;
        let norm2 = LayerNorm::new_no_bias(weight2, layer_norm_eps.to_f64());

        Ok(Self { attention, feed_forward, norm1, norm2 })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Convert input to F32
        let input = input.to_dtype(candle_core::DType::F32)?;

        // Self-attention
        let attention_output =
            self.attention.forward(&input, &input, &input)?.to_dtype(candle_core::DType::F32)?;

        // Add & Norm
        let attention_output = input.add(&attention_output)?.to_dtype(candle_core::DType::F32)?;
        let attention_output =
            Module::forward(&self.norm1, &attention_output)?.to_dtype(candle_core::DType::F32)?;

        // Feed forward
        let feed_forward_output =
            self.feed_forward.forward(&attention_output)?.to_dtype(candle_core::DType::F32)?;

        // Add & Norm
        let output =
            attention_output.add(&feed_forward_output)?.to_dtype(candle_core::DType::F32)?;
        let output = Module::forward(&self.norm2, &output)?.to_dtype(candle_core::DType::F32)?;
        Ok(output)
    }
}
