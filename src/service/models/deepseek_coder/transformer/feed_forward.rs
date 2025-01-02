use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, VarBuilder};

/// Position-wise feed forward network implementation
#[derive(Debug)]
pub struct PositionWiseFeedForward {
    fc1: linear::Linear,
    fc2: linear::Linear,
}

impl PositionWiseFeedForward {
    /// Create new PositionWiseFeedForward instance
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(intermediate_size, hidden_size, vb.pp("fc2"))?;

        Ok(Self { fc1, fc2 })
    }

    /// Forward pass implementation
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(input)?;
        let hidden = hidden.gelu()?;
        self.fc2.forward(&hidden)
    }
}
