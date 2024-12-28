use crate::utils::{config::AppConfig, download::ModelDownloader};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use memmap2;
use safetensors::SafeTensors;
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct ModelLoader {
    model_paths: Vec<PathBuf>,
    device: Device,
    config_path: PathBuf,
}

impl ModelLoader {
    pub async fn new(model_id: &str, config_path: &str) -> anyhow::Result<Self> {
        let config = AppConfig::load(config_path)?;
        let model_config = config.get_model_config(model_id)?;

        // Create a Vec<String> of files to download
        // Create cache directory if it doesn't exist
        let cache_dir = format!("models_cache/{}", model_config.hf_hub_id);
        std::fs::create_dir_all(&cache_dir)?;

        // Check which files need to be downloaded
        let mut files_to_download = Vec::new();
        let mut model_paths = Vec::new();

        // Check weights files
        for weight_file in &model_config.model_files.weights {
            let file_path = format!("{}/{}", cache_dir, weight_file);
            if !std::path::Path::new(&file_path).exists() {
                files_to_download.push(weight_file.as_str());
            }
            model_paths.push(PathBuf::from(file_path));
        }

        // Check other required files
        let other_files = [
            &model_config.model_files.config,
            &model_config.model_files.tokenizer,
            &model_config.model_files.tokenizer_config,
            &model_config.model_files.generation_config,
        ];

        for file in other_files {
            let file_path = format!("{}/{}", cache_dir, file);
            if !std::path::Path::new(&file_path).exists() {
                files_to_download.push(file.as_str());
            }
            model_paths.push(PathBuf::from(file_path));
        }

        // Only download missing files
        if !files_to_download.is_empty() {
            ModelDownloader::download_all_model_files(
                config_path,
                &model_config.hf_hub_id,
                &files_to_download,
            )
            .await?;
        }

        Ok(Self {
            model_paths,
            device: Device::cuda_if_available(0).unwrap(),
            config_path: PathBuf::from(config_path),
        })
    }

    pub fn load(&self) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
        let mut model_tensors = std::collections::HashMap::new();

        // Load all model shards
        for model_path in &self.model_paths {
            let mmap =
                unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(model_path)?)? };
            let tensors = SafeTensors::deserialize(&mmap)?;

            for (name, _tensor_info) in tensors.tensors() {
                let data = tensors.tensor(&name)?;
                let tensor = Tensor::from_raw_buffer(
                    data.data(),
                    data.dtype().try_into()?,
                    data.shape(),
                    &self.device,
                )?;
                model_tensors.insert(name.to_string(), tensor);
            }
        }

        Ok(model_tensors)
    }

    pub fn get_config_path(&self) -> &PathBuf {
        &self.config_path
    }

    pub fn get_var_builder(&self) -> anyhow::Result<VarBuilder> {
        let model_tensors = self.load()?;
        Ok(VarBuilder::from_tensors(model_tensors, DType::F32, &self.device))
    }

    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        // The tokenizer file should be in the model paths
        let tokenizer_path = self
            .model_paths
            .iter()
            .find(|p| p.to_string_lossy().ends_with("tokenizer.json"))
            .ok_or_else(|| anyhow::anyhow!("Tokenizer file not found"))?;

        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
    }
}
