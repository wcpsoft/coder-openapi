use crate::utils::{config::AppConfig, download::ModelDownloader};
use candle_core::{Device, Tensor};
use memmap2;
use safetensors::SafeTensors;
use std::path::PathBuf;
pub struct ModelLoader {
    model_paths: Vec<PathBuf>,
    device: Device,
    config_path: PathBuf,
}

impl ModelLoader {
    pub fn new(model_id: &str, config_path: &str) -> anyhow::Result<Self> {
        let config = AppConfig::load(config_path)?;
        let model_config = config.get_model_config(model_id)?;

        // 下载所有模型文件
        let model_paths = model_config
            .model_files
            .weights
            .iter()
            .map(|file| ModelDownloader::download_model(&model_config.hf_hub_id, file))
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            model_paths,
            device: Device::cuda_if_available(0).unwrap(),
            config_path: PathBuf::from(config_path),
        })
    }

    pub fn load(&self) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
        let mut model_tensors = std::collections::HashMap::new();

        // 加载所有分片模型文件
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
}
