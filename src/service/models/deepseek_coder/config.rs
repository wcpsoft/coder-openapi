use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct ModelFiles {
    pub weights: Vec<String>,
    pub config: String,
    pub tokenizer: String,
    pub tokenizer_config: String,
    pub generation_config: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub models_cache_dir: String,
    pub hf_hub_id: String,
    pub model_files: ModelFiles,
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
    pub attention_dropout: f32,
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub initializer_range: f32,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_theta: f64,
    #[serde(default)]
    pub torch_dtype: String,
    #[serde(default)]
    pub transformers_version: String,
    #[serde(default)]
    pub use_cache: bool,
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
