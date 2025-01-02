use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

fn gelu(x: &Tensor) -> Result<Tensor> {
    // Precompute constants for GELU approximation
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // sqrt(2.0 / PI)
    const COEFF: f64 = 0.044715;
    const HALF: f64 = 0.5;

    // Compute GELU using tensor operations with broadcasting
    let device = x.device();
    let dtype = x.dtype();

    let x_cubed = x.powf(3.0)?;

    // Get input tensor shape for broadcasting
    let shape = x.shape().dims();

    // Create and expand tensors to match input shape
    let coeff_tensor = Tensor::new(&[COEFF], device)?.to_dtype(dtype)?.expand(shape)?;
    let inner = x.add(&x_cubed.mul(&coeff_tensor)?)?;

    let sqrt_2_over_pi_tensor =
        Tensor::new(&[SQRT_2_OVER_PI], device)?.to_dtype(dtype)?.expand(shape)?;
    let tanh = inner.mul(&sqrt_2_over_pi_tensor)?.tanh()?;

    let half_tensor = Tensor::new(&[HALF], device)?.to_dtype(dtype)?.expand(shape)?;
    let one_tensor = Tensor::new(&[1.0], device)?.to_dtype(dtype)?.expand(shape)?;
    let tanh_plus_one = tanh.add(&one_tensor)?;
    let result = x.mul(&half_tensor)?.mul(&tanh_plus_one)?;

    Ok(result)
}

pub struct PositionWiseFeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl PositionWiseFeedForward {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let fc2 = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(input)?;
        let hidden = gelu(&hidden)?;
        self.fc2.forward(&hidden)
    }
}
