use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    #[serde(default)]
    pub bos_token_id: usize,
    #[serde(default)]
    pub eos_token_id: usize,
    #[serde(default)]
    pub pad_token_id: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: f32,
    #[serde(default)]
    pub max_tokens: usize,
    #[serde(default)]
    pub layer_norm_eps: f64,
    pub tokenizer_path: String,
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&config_str)?;
        Ok(config)
    }
}

pub type DeepSeekCoderConfig = ModelConfig;
pub type TransformerModelConfig = ModelConfig;
