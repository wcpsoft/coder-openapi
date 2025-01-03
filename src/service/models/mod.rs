//! 模型服务实现
//!
//! 本模块提供管理AI模型的功能，包括：
//! - 模型下载和加载
//! - 模型状态跟踪
//! - 模型配置管理
//!
//! # 示例
//! ```rust
//! use coder_openapi::service::models::ModelManager;
//!
//! #[tokio::main]
//! async fn main() {
//!     let manager = ModelManager::new().await;
//!     manager.download_model("yi-coder", "config/app.yml").await.unwrap();
//! }
//! ```

pub mod deepseek_coder;
pub mod yi_coder;

use crate::entities::models::{DeepSeekCoderModel, YiCoderModel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

// Model weights file path
#[allow(dead_code)]
const MODEL_WEIGHTS_PATH: &str = "models_cache/weights.safetensors";

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("Unknown model: {0}")]
    UnknownModel(String),
    #[error("Model initialization failed: {0}")]
    InitializationFailed(String),
}

#[derive(Clone)]
pub struct ModelManager {
    yi_coder: Arc<RwLock<Option<YiCoderModel>>>,
    deepseek_coder: Arc<RwLock<Option<DeepSeekCoderModel>>>,
    model_status: Arc<RwLock<HashMap<String, ModelStatus>>>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct ModelStatus {
    pub is_cached: bool,
    pub is_enabled: bool,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self {
            yi_coder: Arc::new(RwLock::new(None)),
            deepseek_coder: Arc::new(RwLock::new(None)),
            model_status: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl ModelManager {
    /// 创建一个新的ModelManager实例
    pub async fn new() -> Self {
        let manager = Self {
            yi_coder: Arc::new(RwLock::new(None)),
            deepseek_coder: Arc::new(RwLock::new(None)),
            model_status: Arc::new(RwLock::new(HashMap::new())),
        };
        // Initialize status from disk
        let _ = manager.refresh_status_from_disk().await;
        // Initialize default models
        manager
            .model_status
            .write()
            .await
            .insert("yi-coder".to_string(), ModelStatus { is_cached: false, is_enabled: false });
        manager.model_status.write().await.insert(
            "deepseek-coder".to_string(),
            ModelStatus { is_cached: false, is_enabled: false },
        );
        manager
    }

    /// Refresh model status from disk
    async fn refresh_status_from_disk(&self) -> Result<(), ModelError> {
        let mut status = self.model_status.write().await;

        // Check yi-coder files
        let yi_coder_dir = "models_cache/01-ai/Yi-Coder-1.5B-Chat".to_string();
        let yi_coder_files = [
            "model.safetensors",
            "config.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "tokenizer.json",
            "generation_config.json",
        ];
        let yi_coder_any_exists = yi_coder_files
            .iter()
            .any(|file| std::path::Path::new(&format!("{}/{}", yi_coder_dir, file)).exists());
        let yi_coder_all_exists = yi_coder_files
            .iter()
            .all(|file| std::path::Path::new(&format!("{}/{}", yi_coder_dir, file)).exists());
        status.insert(
            "yi-coder".to_string(),
            ModelStatus { is_cached: yi_coder_any_exists, is_enabled: yi_coder_all_exists },
        );

        // Check deepseek-coder files
        let deepseek_coder_dir =
            "models_cache/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct".to_string();
        let deepseek_coder_files = [
            "model-00001-of-000004.safetensors",
            "model-00002-of-000004.safetensors",
            "model-00003-of-000004.safetensors",
            "model-00004-of-000004.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ];
        let deepseek_coder_any_exists = deepseek_coder_files
            .iter()
            .any(|file| std::path::Path::new(&format!("{}/{}", deepseek_coder_dir, file)).exists());
        let deepseek_coder_all_exists = deepseek_coder_files
            .iter()
            .all(|file| std::path::Path::new(&format!("{}/{}", deepseek_coder_dir, file)).exists());
        status.insert(
            "deepseek-coder".to_string(),
            ModelStatus {
                is_cached: deepseek_coder_any_exists,
                is_enabled: deepseek_coder_all_exists,
            },
        );

        Ok(())
    }

    /// 下载并初始化模型
    ///
    /// # 参数
    /// * `model_id` - 模型ID，目前支持"yi-coder"和"deepseek-coder"
    /// * `config_path` - 配置文件路径
    ///
    /// # 返回值
    /// * `Ok(())` - 模型下载并初始化成功
    /// * `Err(ModelError)` - 模型下载或初始化失败
    ///
    /// # 示例
    /// ```rust
    /// use coder_openapi::service::models::ModelManager;
    ///
    /// #[tokio::test]
    /// async fn test_download_model() {
    ///     let manager = ModelManager::new().await;
    ///     assert!(manager.download_model("yi-coder", "config/app.yml").await.is_ok());
    /// }
    /// ```
    pub async fn download_model(
        &self,
        model_id: &str,
        config_path: &str,
    ) -> Result<(), ModelError> {
        let mut status = self.model_status.write().await;
        if let Some(model_status) = status.get_mut(model_id) {
            match model_id {
                "yi-coder" => {
                    let mut model = self.yi_coder.write().await;
                    *model = Some(YiCoderModel::new(config_path).map_err(|e| {
                        ModelError::InitializationFailed(format!("Yi-Coder: {}", e))
                    })?);
                    model_status.is_cached = true;
                    model_status.is_enabled = true;
                }
                "deepseek-coder" => {
                    let mut model = self.deepseek_coder.write().await;
                    *model =
                        Some(DeepSeekCoderModel::new(&config_path.to_string()).await.map_err(
                            |e| ModelError::InitializationFailed(format!("DeepSeek-Coder: {}", e)),
                        )?);
                    model_status.is_cached = true;
                    model_status.is_enabled = true;
                }
                _ => return Err(ModelError::UnsupportedModel(model_id.to_string())),
            }
            Ok(())
        } else {
            Err(ModelError::UnknownModel(model_id.to_string()))
        }
    }

    /// 检查模型是否可用
    pub async fn is_model_available(&self, model_id: &str) -> bool {
        let status = self.model_status.read().await;
        status.get(model_id).map(|s| s.is_cached && s.is_enabled).unwrap_or(false)
    }

    /// 获取特定模型的状态
    ///
    /// # 参数
    /// * `model_id` - 要查询的模型ID
    ///
    /// # 返回值
    /// * `Some(ModelStatus)` - 如果模型存在
    /// * `None` - 如果模型不存在
    ///
    /// # 示例
    /// ```rust
    /// use coder_openapi::service::models::ModelManager;
    ///
    /// #[tokio::test]
    /// async fn test_get_model_status() {
    ///     let manager = ModelManager::new().await;
    ///     manager.download_model("yi-coder").await.unwrap();
    ///     assert!(manager.get_model_status("yi-coder").await.is_some());
    /// }
    /// ```
    pub async fn get_model_status(&self, model_id: &str) -> Option<ModelStatus> {
        let status = self.model_status.read().await;
        status.get(model_id).cloned()
    }

    /// 获取Yi-Coder模型实例
    ///
    /// # 返回值
    /// * `Some(YiCoderModel)` - 如果模型已加载
    /// * `None` - 如果模型未加载
    pub async fn get_yi_coder(&self) -> Option<YiCoderModel> {
        let model = self.yi_coder.read().await;
        model.clone()
    }

    /// 获取DeepSeek-Coder模型实例
    ///
    /// # 返回值
    /// * `Some(DeepSeekCoderModel)` - 如果模型已加载
    /// * `None` - 如果模型未加载
    pub async fn get_deepseek_coder(&self) -> Option<DeepSeekCoderModel> {
        let model = self.deepseek_coder.read().await;
        model.clone()
    }

    /// 获取所有模型的状态
    ///
    /// # 返回值
    /// * `HashMap<String, ModelStatus>` - 包含所有模型状态的映射
    pub async fn get_all_model_status(&self) -> HashMap<String, ModelStatus> {
        // Refresh status from disk before returning
        let _ = self.refresh_status_from_disk().await;
        let status = self.model_status.read().await;
        status.clone()
    }
}
