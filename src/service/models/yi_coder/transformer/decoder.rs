use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

use super::{config::ModelConfig, transformer_layer::TransformerLayer};

pub struct YiCoderDecoder {
    layers: Vec<TransformerLayer>,
    norm: LayerNorm,
}

impl YiCoderDecoder {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let layer = TransformerLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                vb.pp(format!("layer_{}", i)),
            )?;
            layers.push(layer);
        }

        let weight = vb.get((config.hidden_size,), "norm.weight")?;
        let bias = vb.get((config.hidden_size,), "norm.bias")?;
        let norm = LayerNorm::new(weight, bias, config.layer_norm_eps as f64);

        Ok(Self { layers, norm })
    }

    pub fn forward(&self, input: &Tensor, _encoder_output: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = input.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        Module::forward(&self.norm, &hidden_states)
    }
}
