use crate::utils::config::AppConfig;
use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use std::fs;
use std::path::PathBuf;

/// 模型下载器，负责从Hugging Face Hub下载模型文件
pub struct ModelDownloader;

impl ModelDownloader {
    /// 下载指定模型的所有必要文件
    ///
    /// # 参数
    /// - model_id: 模型ID
    /// - config_path: 配置文件路径
    ///
    /// # 返回
    /// 下载的文件路径列表
    pub fn download_model_files(model_id: &str, config_path: &str) -> Result<Vec<PathBuf>> {
        log::debug!("Starting model files download for model_id: {}", model_id);
        log::debug!("Using config path: {}", config_path);

        let config = AppConfig::load(config_path)?;
        log::debug!("config {:?}", config);
        let model_config = config.get_model_config(model_id)?;
        log::debug!("Loaded model config for: {}", model_id);

        // 创建缓存目录
        let cache_dir = PathBuf::from(&config.models_cache_dir).join(&model_config.hf_hub_id);
        log::debug!("Cache directory: {}", cache_dir.display());
        fs::create_dir_all(&cache_dir).context("Failed to create models cache directory")?;
        log::debug!("Created cache directory");

        // 构建需要下载的文件列表
        let mut files_to_download = Vec::new();
        let mut model_paths = Vec::new();
        log::debug!("Checking for required model files");

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
            )?;
        }

        Ok(model_paths)
    }

    /// 下载指定文件列表
    ///
    /// # 参数
    /// - config_path: 配置文件路径
    /// - hub_id: Hugging Face Hub模型ID
    /// - files: 需要下载的文件列表
    ///
    /// # 返回
    /// 下载的文件路径列表
    pub fn download_all_model_files(
        config_path: &str,
        hub_id: &str,
        files: &[&str],
    ) -> Result<Vec<PathBuf>> {
        let config = AppConfig::load(config_path)?;
        let cache_dir = PathBuf::from(&config.models_cache_dir).join(hub_id);
        fs::create_dir_all(&cache_dir).context("Failed to create models cache directory")?;

        let mut paths = Vec::new();
        for file in files {
            let local_path = cache_dir.join(file);
            if !local_path.exists() {
                let api = Api::new()?;
                let repo = api.model(hub_id.to_string());
                let remote_path = repo.get(file)?;
                fs::copy(&remote_path, &local_path)
                    .context("Failed to copy model file to cache")?;
            }
            paths.push(local_path);
        }

        Ok(paths)
    }

    /// 获取指定模型的配置文件路径
    ///
    /// # 参数
    /// - model_id: 模型ID
    /// - config_path: 配置文件路径
    ///
    /// # 返回
    /// 模型配置文件路径
    pub fn get_model_config_path(model_id: &str, config_path: &str) -> Result<PathBuf> {
        let config = AppConfig::load(config_path)?;
        let model_config = config.get_model_config(model_id)?;
        Ok(PathBuf::from(&config.models_cache_dir)
            .join(&model_config.hf_hub_id)
            .join(&model_config.model_files.config))
    }
}
