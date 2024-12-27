use anyhow::Result;
use hf_hub::api::sync::Api;
use std::path::PathBuf;

pub struct ModelDownloader;

impl ModelDownloader {
    pub fn download_model(hub_id: &str, file_name: &str) -> Result<PathBuf> {
        let api = Api::new()?;
        let repo = api.model(hub_id.to_string());
        let model_path = repo.get(file_name)?;
        Ok(model_path)
    }
}
