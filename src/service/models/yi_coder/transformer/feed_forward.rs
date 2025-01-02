use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

fn gelu(x: &Tensor) -> Result<Tensor> {
    // GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2.0 / PI) * (x + 0.044715 * x.powf(3.0))))
    let sqrt_2_over_pi = Tensor::new(&[(2.0 / std::f64::consts::PI).sqrt()], x.device())?;
    let coeff = Tensor::new(&[0.044715], x.device())?;
    let half = Tensor::new(&[0.5], x.device())?;
    let one = Tensor::new(&[1.0], x.device())?;

    let x_cubed = x.powf(3.0)?;
    let inner = x.add(&x_cubed.mul(&coeff)?)?;
    let tanh_arg = inner.mul(&sqrt_2_over_pi)?;
    let tanh = tanh_arg.tanh()?;

    let gelu = x.mul(&half)?.mul(&tanh.add(&one)?)?;

    Ok(gelu)
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
