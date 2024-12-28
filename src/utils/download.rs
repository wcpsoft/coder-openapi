use crate::utils::config::AppConfig;
use anyhow::{Context, Result};
use hf_hub::api::tokio::Api;
use std::path::PathBuf;
use tokio::fs;

pub struct ModelDownloader;

impl ModelDownloader {
    pub async fn download_all_model_files(
        config_path: &str,
        hub_id: &str,
        files: &[&str],
    ) -> Result<Vec<PathBuf>> {
        let config = AppConfig::load(config_path)?;
        let cache_dir = PathBuf::from(&config.models_cache_dir).join(hub_id);
        fs::create_dir_all(&cache_dir).await.context("Failed to create models cache directory")?;

        let mut paths = Vec::new();
        for file in files {
            let local_path = cache_dir.join(file);
            if !local_path.exists() {
                // Only download if file doesn't exist
                let api = Api::new()?;
                let repo = api.model(hub_id.to_string());
                let remote_path = repo.get(file).await?;
                fs::copy(&remote_path, &local_path)
                    .await
                    .context("Failed to copy model file to cache")?;
            }
            paths.push(local_path);
        }

        Ok(paths)
    }
}
