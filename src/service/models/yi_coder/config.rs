use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
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
    pub hidden_size: usize,
    #[serde(default)]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub num_layers: usize,
    #[serde(default)]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub vocab_size: usize,
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&config_str)?;
        Ok(config)
    }
}
