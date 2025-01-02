use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, ops::softmax, VarBuilder};

/// Multi-head attention implementation
#[derive(Debug)]
pub struct MultiHeadAttention {
    query: linear::Linear,
    key: linear::Linear,
    value: linear::Linear,
    out: linear::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Create new MultiHeadAttention instance
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        let query = linear(hidden_size, hidden_size, vb.pp("query"))?;
        let key = linear(hidden_size, hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, hidden_size, vb.pp("value"))?;
        let out = linear(hidden_size, hidden_size, vb.pp("out"))?;

        Ok(Self { query, key, value, out, num_heads, head_dim })
    }

    /// Forward pass implementation
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = query.dims3()?;

        // Linear transformations with optimized dtype handling
        let query = self.query.forward(query)?;
        let key = self.key.forward(key)?;
        let value = self.value.forward(value)?;

        // Reshape for multi-head attention
        let query = query.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let key = key.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let value = value.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Compute attention scores with optimized scaling
        let scale = Tensor::new((self.head_dim as f64).sqrt(), query.device())?;
        let mut attention_scores = query.matmul(&key.t()?)?.broadcast_div(&scale)?;

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            let mask = mask.to_dtype(candle_core::DType::F32)?;
            attention_scores = attention_scores.broadcast_add(&mask)?;
        }

        // Softmax normalization
        let attention_probs = softmax(&attention_scores, attention_scores.dims().len() - 1)?;

        // Compute context
        let context = attention_probs.matmul(&value)?;
        let context = context.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Final linear transformation
        self.out.forward(&context)
    }
}
