use crate::utils::config::{AppConfig, ModelConfig};
use anyhow::{Context, Result};
use hf_hub::api::tokio::Api;
use std::path::PathBuf;
use tokio::fs;

pub struct ModelDownloader;

impl ModelDownloader {
    /// 下载指定模型的所有必要文件
    pub async fn download_model_files(model_id: &str, config_path: &str) -> Result<Vec<PathBuf>> {
        let config = AppConfig::load(config_path)?;
        let model_config = config.get_model_config(model_id)?;

        // 创建缓存目录
        let cache_dir = PathBuf::from(&config.models_cache_dir).join(&model_config.hf_hub_id);
        fs::create_dir_all(&cache_dir).await.context("Failed to create models cache directory")?;

        // 构建需要下载的文件列表
        let mut files_to_download = Vec::new();
        let mut model_paths = Vec::new();

        // 检查权重文件
        for weight_file in &model_config.model_files.weights {
            if !weight_file.ends_with(".safetensors") {
                continue;
            }
            let file_path = cache_dir.join(weight_file);
            if !file_path.exists() {
                files_to_download.push(weight_file.as_str());
            }
            model_paths.push(file_path);
        }

        // 检查tokenizer文件
        let tokenizer_file = &model_config.model_files.tokenizer;
        let tokenizer_path = cache_dir.join(tokenizer_file);
        if !tokenizer_path.exists() {
            files_to_download.push(tokenizer_file.as_str());
        }
        model_paths.push(tokenizer_path);

        // 检查配置文件
        let config_files = [
            &model_config.model_files.config,
            &model_config.model_files.tokenizer_config,
            &model_config.model_files.generation_config,
        ];

        for file in config_files {
            let file_path = cache_dir.join(file);
            if !file_path.exists() {
                files_to_download.push(file.as_str());
            }
            model_paths.push(file_path);
        }

        // 下载缺失的文件
        if !files_to_download.is_empty() {
            Self::download_all_model_files(
                config_path,
                &model_config.hf_hub_id,
                &files_to_download,
            )
            .await?;
        }

        Ok(model_paths)
    }

    /// 下载指定文件列表
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

    /// 获取指定模型的配置文件路径
    pub fn get_model_config_path(model_id: &str, config_path: &str) -> Result<PathBuf> {
        let config = AppConfig::load(config_path)?;
        let model_config = config.get_model_config(model_id)?;
        Ok(PathBuf::from(&config.models_cache_dir)
            .join(&model_config.hf_hub_id)
            .join(&model_config.model_files.config))
    }
}
