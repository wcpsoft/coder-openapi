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
    pub models_cache_dir: String,
    pub chat: Chat,
}

pub static CONFIG: OnceLock<AppConfig> = OnceLock::new();

pub fn get_config() -> &'static AppConfig {
    CONFIG.get_or_init(|| {
        AppConfig::load("config/app.yml").expect("Failed to load application configuration")
    })
}

pub fn load_route_config() -> RouteConfig {
    let config_file =
        std::fs::File::open("config/route.yml").expect("Failed to open route configuration file");
    serde_yaml::from_reader(config_file).expect("Failed to parse route configuration")
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
