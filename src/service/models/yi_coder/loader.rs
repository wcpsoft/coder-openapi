use crate::utils::download::ModelDownloader;
use candle_core::{Device, Tensor};
use memmap2;
use safetensors::SafeTensors;
use std::path::PathBuf;

pub struct ModelLoader {
    model_path: PathBuf,
    device: Device,
}

impl ModelLoader {
    pub fn new(model_name: &str) -> anyhow::Result<Self> {
        let model_path = ModelDownloader::download_model(model_name)?;

        Ok(Self {
            model_path,
            device: Device::cuda_if_available(0).unwrap(),
        })
    }

    pub fn load(&self) -> anyhow::Result<std::collections::HashMap<String, Tensor>> {
        // 使用内存映射优化大模型加载
        let mmap =
            unsafe { memmap2::MmapOptions::new().map(&std::fs::File::open(&self.model_path)?)? };
        let tensors = SafeTensors::deserialize(&mmap)?;

        // 加载所有张量
        let mut model_tensors = std::collections::HashMap::new();
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

        Ok(model_tensors)
    }
}
