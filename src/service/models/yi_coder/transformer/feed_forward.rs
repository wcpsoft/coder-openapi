use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

fn gelu(x: &Tensor) -> Result<Tensor> {
    // Precompute constants for GELU approximation
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // sqrt(2.0 / PI)
    const COEFF: f64 = 0.044715;
    const HALF: f64 = 0.5;

    // Convert constants to tensors
    let sqrt_2_over_pi = Tensor::new(SQRT_2_OVER_PI, x.device())?;
    let coeff = Tensor::new(COEFF, x.device())?;
    let half = Tensor::new(HALF, x.device())?;
    let one = Tensor::new(1.0, x.device())?;

    // Compute GELU using tensor operations
    let x_cubed = x.powf(3.0)?;
    let inner = x.add(&x_cubed.mul(&coeff)?)?;
    let tanh = inner.mul(&sqrt_2_over_pi)?.tanh()?;
    let result = x.mul(&half)?.mul(&tanh.add(&one)?)?;

    Ok(result)
}

pub struct PositionWiseFeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl PositionWiseFeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(intermediate_size, hidden_size, vb.pp("fc2"))?;

        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(input)?;
        let hidden = gelu(&hidden)?;
        self.fc2.forward(&hidden)
    }
}
