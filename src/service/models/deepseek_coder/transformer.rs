use crate::error::AppError;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, LayerNorm, Module, VarBuilder};

pub struct DeepseekCoderTransformer {
    device: Device,
    layers: Vec<TransformerLayer>,
    norm: LayerNorm,
}

impl Module for DeepseekCoderTransformer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.norm.forward(&x)
    }
}

impl DeepseekCoderTransformer {
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl Module for TransformerLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x = self._attention.forward(input)?;
        let x = self._norm1.forward(&x)?;
        let x = self._feed_forward.forward(&x)?;
        self._norm2.forward(&x)
    }
}

struct TransformerLayer {
    _attention: MultiHeadAttention,
    _feed_forward: PositionWiseFeedForward,
    _norm1: LayerNorm,
    _norm2: LayerNorm,
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let query = self._query.forward(input)?;
        let key = self._key.forward(input)?;
        let value = self._value.forward(input)?;

        let attention = query.matmul(&key.t()?)?;
        let dim = attention.rank() as usize - 1;
        let max = attention.max(dim)?;
        let exp = (attention - max)?.exp()?;
        let sum = exp.sum(dim)?;
        let attention = exp.broadcast_div(&sum)?;
        let context = attention.matmul(&value)?;

        self._out.forward(&context)
    }
}

struct MultiHeadAttention {
    _query: linear::Linear,
    _key: linear::Linear,
    _value: linear::Linear,
    _out: linear::Linear,
    _num_heads: usize,
    _head_dim: usize,
}

impl Module for PositionWiseFeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x = self._fc1.forward(input)?;
        let x = x.relu()?;
        self._fc2.forward(&x)
    }
}

struct PositionWiseFeedForward {
    _fc1: linear::Linear,
    _fc2: linear::Linear,
}

impl DeepseekCoderTransformer {
    pub fn new(config: &super::config::ModelConfig, vb: VarBuilder) -> Result<Self, AppError> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = TransformerLayer::new(
                config.num_attention_heads,
                config.hidden_size,
                config.intermediate_size,
                vb.pp(format!("layer_{}", i)),
            )?;
            layers.push(layer);
        }

        let norm = LayerNorm::new(
            Tensor::ones(config.hidden_size, DType::F32, &device)?,
            Tensor::zeros(config.hidden_size, DType::F32, &device)?,
            config.layer_norm_eps,
        );

        Ok(Self { device, layers, norm })
    }
}

impl TransformerLayer {
    fn new(
        num_heads: usize,
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self, AppError> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let _attention = MultiHeadAttention::new(num_heads, hidden_size, vb.pp("attention"))?;
        let _feed_forward =
            PositionWiseFeedForward::new(hidden_size, intermediate_size, vb.pp("ffn"))?;
        let _norm1 = LayerNorm::new(
            Tensor::ones(hidden_size, DType::F32, &device)?,
            Tensor::zeros(hidden_size, DType::F32, &device)?,
            1e-5,
        );
        let _norm2 = LayerNorm::new(
            Tensor::ones(hidden_size, DType::F32, &device)?,
            Tensor::zeros(hidden_size, DType::F32, &device)?,
            1e-5,
        );

        Ok(Self {
            _attention: _attention,
            _feed_forward: _feed_forward,
            _norm1: _norm1,
            _norm2: _norm2,
        })
    }
}

impl MultiHeadAttention {
    fn new(num_heads: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self, AppError> {
        let head_dim = hidden_size / num_heads;
        let query = linear(hidden_size, hidden_size, vb.pp("query"))?;
        let key = linear(hidden_size, hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, hidden_size, vb.pp("value"))?;
        let out = linear(hidden_size, hidden_size, vb.pp("out"))?;

        Ok(Self {
            _query: query,
            _key: key,
            _value: value,
            _out: out,
            _num_heads: num_heads,
            _head_dim: head_dim,
        })
    }
}

impl PositionWiseFeedForward {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self, AppError> {
        Ok(Self {
            _fc1: linear(hidden_size, intermediate_size, vb.pp("fc1"))?,
            _fc2: linear(intermediate_size, hidden_size, vb.pp("fc2"))?,
        })
    }
}
