use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

pub struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_size: usize,
}

impl MultiHeadAttention {
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_size = hidden_size / num_heads;
        let query = linear(hidden_size, hidden_size, vb.pp("query"))?;
        let key = linear(hidden_size, hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, hidden_size, vb.pp("value"))?;
        let output = linear(hidden_size, hidden_size, vb.pp("output"))?;

        Ok(Self { query, key, value, output, num_heads, head_size })
    }

    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let batch_size = query.dim(0)?;
        let seq_len = query.dim(1)?;

        // Project inputs
        let query = self.query.forward(query)?;
        let key = self.key.forward(key)?;
        let value = self.value.forward(value)?;

        // Reshape for multi-head attention
        let query = query.reshape((batch_size, seq_len, self.num_heads, self.head_size))?;
        let key = key.reshape((batch_size, seq_len, self.num_heads, self.head_size))?;
        let value = value.reshape((batch_size, seq_len, self.num_heads, self.head_size))?;

        // Compute attention scores
        let scores = query.matmul(&key.transpose(2, 3)?)?;
        let scores = scores / (self.head_size as f64).sqrt();

        // Apply softmax
        let attention_weights = scores.softmax(3)?;

        // Apply attention to values
        let context = attention_weights.matmul(&value)?;

        // Reshape back to original dimensions
        let context = context.reshape((batch_size, seq_len, self.num_heads * self.head_size))?;

        // Project output
        self.output.forward(&context)
    }
}
