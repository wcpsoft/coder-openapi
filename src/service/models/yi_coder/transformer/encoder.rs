use candle_core::{Module, Result, Tensor, WithDType};
use candle_nn::{LayerNorm, VarBuilder};

use super::{config::ModelConfig, transformer_layer::TransformerLayer};

pub struct YiCoderEncoder {
    layers: Vec<TransformerLayer>,
    norm: LayerNorm,
}

impl YiCoderEncoder {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let layer = TransformerLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.layer_norm_eps as f32,
                vb.pp(format!("layer_{}", i)),
            )?;
            layers.push(layer);
        }

        let weight = vb.get((config.hidden_size,), "norm.weight")?;
        let bias = vb.get((config.hidden_size,), "norm.bias")?;
        let norm = LayerNorm::new(weight, bias, config.layer_norm_eps as f64);

        Ok(Self { layers, norm })
    }

    pub fn forward(&self, input: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = input.to_dtype(candle_core::DType::F32)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        let output = Module::forward(&self.norm, &hidden_states)?;
        output.to_dtype(candle_core::DType::F32)
    }
}
