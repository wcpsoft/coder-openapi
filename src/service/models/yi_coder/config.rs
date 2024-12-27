use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&config_str)?;
        Ok(config)
    }
}
