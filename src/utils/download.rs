use anyhow::Result;
use hf_hub::api::sync::Api;
use std::path::PathBuf;

pub struct ModelDownloader;

impl ModelDownloader {
    pub fn download_model(model_name: &str) -> Result<PathBuf> {
        let api = Api::new()?;
        let repo = api.model(model_name.to_string());
        let model_path = repo.get("model.safetensors")?;
        Ok(model_path)
    }
}
