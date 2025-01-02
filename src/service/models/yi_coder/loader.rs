use crate::error::AppError;
use crate::utils::{config::AppConfig, download::ModelDownloader};
use anyhow;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use log;
use safetensors::SafeTensors;
use std::path::PathBuf;
use tokenizers::Tokenizer;

// Byte size conversion constants
const BYTES_PER_MB: f64 = 1024.0 * 1024.0;
const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

pub struct ModelLoader {
    model_paths: Vec<PathBuf>,
    device: Device,
    config_path: PathBuf,
}

impl ModelLoader {
    pub async fn new(model_id: &str, config_path: &str) -> anyhow::Result<Self> {
        // 下载模型文件
        let model_paths = ModelDownloader::download_model_files(model_id, config_path).await?;

        Ok(Self {
            model_paths,
            device: Device::cuda_if_available(0)
                .map_err(|e| AppError::Generic(format!("Failed to get CUDA device: {}", e)))?,
            config_path: PathBuf::from(config_path),
        })
    }

    pub fn load(&self) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
        let mut model_tensors = std::collections::HashMap::new();

        // 只加载.safetensors文件
        for model_path in &self.model_paths {
            if !model_path.to_string_lossy().ends_with(".safetensors") {
                continue;
            }

            let mmap =
                unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(model_path)?)? };
            let tensors = SafeTensors::deserialize(&mmap)?;

            let mut total_bytes = 0;
            for (name, _tensor_info) in tensors.tensors() {
                let data = tensors.tensor(&name)?;
                let tensor = Tensor::from_raw_buffer(
                    data.data(),
                    data.dtype().try_into()?,
                    data.shape(),
                    &self.device,
                )?;

                // Calculate tensor size in bytes
                let tensor_size = data.data().len();
                total_bytes += tensor_size;

                // Debug log tensor info with proper unit conversion
                log::debug!(
                    "Loaded tensor: {}, shape: {:?}, dtype: {:?}, size: {:.2} MB ({:.2} GB)",
                    name,
                    data.shape(),
                    data.dtype(),
                    tensor_size as f64 / BYTES_PER_MB,
                    tensor_size as f64 / BYTES_PER_GB
                );

                model_tensors.insert(name.to_string(), tensor);
            }

            // Log total size for this file in GB and MB
            log::debug!(
                "Total loaded size for {}: {:.2} GB ({:.2} MB)",
                model_path.display(),
                total_bytes as f64 / BYTES_PER_GB,
                total_bytes as f64 / BYTES_PER_MB
            );
        }

        Ok(model_tensors)
    }

    pub fn get_config_path(&self) -> &PathBuf {
        &self.config_path
    }

    pub fn get_model_config(
        &self,
        model_id: &str,
    ) -> anyhow::Result<crate::utils::config::ModelConfig> {
        let config_str = self
            .config_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Failed to convert PathBuf to str"))?;
        let config = AppConfig::load(config_str)?;
        Ok(config.get_model_config(model_id)?.clone())
    }

    pub fn get_var_builder(&self) -> anyhow::Result<VarBuilder> {
        let model_tensors = self.load()?;
        Ok(VarBuilder::from_tensors(model_tensors, DType::F32, &self.device))
    }

    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        // 查找tokenizer文件
        let tokenizer_path = self
            .model_paths
            .iter()
            .find(|p| {
                let path = p.to_string_lossy().to_lowercase();
                path.ends_with("tokenizer.json")
            })
            .ok_or_else(|| {
                anyhow::anyhow!("Tokenizer file not found. Expected format: tokenizer.json")
            })?;
        log::debug!("Loading tokenizer from: {:?}", tokenizer_path);
        if !tokenizer_path.exists() {
            log::error!("Tokenizer file does not exist at path: {:?}", tokenizer_path);
            return Err(anyhow::anyhow!("Tokenizer file not found at path: {:?}", tokenizer_path));
        }

        // 使用tokenizers::Tokenizer加载tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
        log::debug!("Tokenizer loaded successfully");

        Ok(tokenizer)
    }
}
