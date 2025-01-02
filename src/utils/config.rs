use anyhow::anyhow;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Deserialize)]
pub struct RouteConfig {
    pub routes: RoutePaths,
}

#[derive(Debug, Deserialize)]
pub struct RoutePaths {
    pub v1: V1Routes,
}

#[derive(Debug, Deserialize)]
pub struct V1Routes {
    pub chat: String,
    pub models: String,
    pub download: String,
}

#[derive(Debug, Deserialize)]
pub struct Chat {
    pub defaults: ChatDefaults,
}

#[derive(Debug, Deserialize)]
pub struct ChatDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub n: usize,
    pub max_tokens: usize,
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub shutdown_timeout: u64,
}

#[derive(Debug, Deserialize)]
pub struct LocalesConfig {
    pub path: String,
    pub default: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub hf_hub_id: String,
    pub model_files: ModelFiles,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ModelFiles {
    pub weights: Vec<String>,
    pub config: String,
    pub tokenizer: String,
    pub tokenizer_config: String,
    pub generation_config: String,
}

impl Clone for ModelFiles {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            config: self.config.clone(),
            tokenizer: self.tokenizer.clone(),
            tokenizer_config: self.tokenizer_config.clone(),
            generation_config: self.generation_config.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub locales: LocalesConfig,
    pub models: HashMap<String, ModelConfig>,
    pub models_cache_dir: String,
    pub chat: Chat,
}

pub static CONFIG: OnceLock<AppConfig> = OnceLock::new();

pub fn get_config() -> &'static AppConfig {
    CONFIG.get_or_init(|| {
        AppConfig::load("config/app.yml").expect("Failed to load application configuration")
    })
}

impl AppConfig {
    pub fn load(config_path: &str) -> anyhow::Result<Self> {
        let config_file = std::fs::File::open(config_path)?;
        let config: Self = serde_yaml::from_reader(config_file)?;
        Ok(config)
    }

    pub fn get_model_config(&self, model_id: &str) -> anyhow::Result<ModelConfig> {
        let mut config = self
            .models
            .get(model_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Model config not found for {}", model_id))?;

        // Load parameters from config.json if they're not set
        if config.hidden_size.is_none()
            || config.num_attention_heads.is_none()
            || config.num_hidden_layers.is_none()
            || config.intermediate_size.is_none()
            || config.vocab_size.is_none()
        {
            let config_path =
                format!("{}/{}/{}", self.models_cache_dir, model_id, config.model_files.config);
            let config_file = std::fs::File::open(config_path)?;
            let model_config: serde_json::Value = serde_json::from_reader(config_file)?;

            if config.hidden_size.is_none() {
                config.hidden_size = model_config["hidden_size"].as_u64().map(|v| v as usize);
            }
            if config.num_attention_heads.is_none() {
                config.num_attention_heads =
                    model_config["num_attention_heads"].as_u64().map(|v| v as usize);
            }
            if config.num_hidden_layers.is_none() {
                config.num_hidden_layers =
                    model_config["num_hidden_layers"].as_u64().map(|v| v as usize);
            }
            if config.intermediate_size.is_none() {
                config.intermediate_size =
                    model_config["intermediate_size"].as_u64().map(|v| v as usize);
            }
            if config.vocab_size.is_none() {
                config.vocab_size = model_config["vocab_size"].as_u64().map(|v| v as usize);
            }
        }

        Ok(config)
    }
}
