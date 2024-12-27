use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize)]
pub struct LocalesConfig {
    pub path: String,
    pub default: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub hf_hub_id: String,
    pub model_files: ModelFiles,
}

#[derive(Debug, Deserialize)]
pub struct ModelFiles {
    pub weights: Vec<String>,
    pub config: String,
    pub tokenizer: String,
    pub tokenizer_config: String,
    pub generation_config: String,
}

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub locales: LocalesConfig,
    pub models: HashMap<String, ModelConfig>,
}

impl AppConfig {
    pub fn load(config_path: &str) -> anyhow::Result<Self> {
        let config_file = std::fs::File::open(config_path)?;
        let config: Self = serde_yaml::from_reader(config_file)?;
        Ok(config)
    }

    pub fn get_model_config(&self, model_id: &str) -> anyhow::Result<&ModelConfig> {
        self.models
            .get(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model config not found for {}", model_id))
    }
}
