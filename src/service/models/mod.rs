//! 模型服务实现
//!
//! 本模块提供管理AI模型的功能，包括：
//! - 模型下载和加载
//! - 模型状态跟踪
//! - 模型配置管理
//!
//! # 示例
//! ```rust
//! use yi_coder_openapi::service::models::ModelManager;
//!
//! #[tokio::main]
//! async fn main() {
//!     let manager = ModelManager::new();
//!     manager.download_model("yi-coder").await.unwrap();
//! }

//! */

use crate::entities::models::{DeepseekCoderModel, YiCoderModel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

// 模型权重文件路径
#[allow(dead_code)]
const MODEL_WEIGHTS_PATH: &str = "models_cache/weights.safetensors";

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("不支持的模型: {0}")]
    UnsupportedModel(String),
    #[error("未知的模型: {0}")]
    UnknownModel(String),
    #[error("模型初始化失败: {0}")]
    InitializationFailed(String),
}

#[derive(Clone)]
pub struct ModelManager {
    yi_coder: Arc<RwLock<Option<YiCoderModel>>>,
    deepseek_coder: Arc<RwLock<Option<DeepseekCoderModel>>>,
    model_status: Arc<RwLock<HashMap<String, ModelStatus>>>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct ModelStatus {
    pub is_cached: bool,  // 模型是否已缓存
    pub is_enabled: bool, // 模型是否已启用
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    /// 创建一个新的ModelManager实例
    pub fn new() -> Self {
        Self {
            yi_coder: Arc::new(RwLock::new(None)),
            deepseek_coder: Arc::new(RwLock::new(None)),
            model_status: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 下载并初始化模型
    ///
    /// # 参数
    /// * `model_id` - 模型ID，目前支持 "yi-coder" 和 "deepseek-coder"
    ///
    /// # 返回值
    /// * `Ok(())` - 模型下载并初始化成功
    /// * `Err(ModelError)` - 模型下载或初始化失败
    ///
    /// # 示例
    /// ```rust
    /// use yi_coder_openapi::service::models::ModelManager;
    ///
    /// #[tokio::test]
    /// async fn test_download_model() {
    ///     let manager = ModelManager::new();
    ///     assert!(manager.download_model("yi-coder").await.is_ok());
    /// }
    /// */
    pub async fn download_model(&self, model_id: &str) -> Result<(), ModelError> {
        let mut status = self.model_status.write().await;
        if let Some(model_status) = status.get_mut(model_id) {
            match model_id {
                "yi-coder" => {
                    let mut model = self.yi_coder.write().await;
                    *model = Some(YiCoderModel::new().map_err(|e| {
                        ModelError::InitializationFailed(format!("Yi-Coder: {}", e))
                    })?);
                    model_status.is_cached = true;
                    model_status.is_enabled = true;
                }
                "deepseek-coder" => {
                    let mut model = self.deepseek_coder.write().await;
                    *model = Some(DeepseekCoderModel::new().map_err(|e| {
                        ModelError::InitializationFailed(format!("Deepseek-Coder: {}", e))
                    })?);
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
        status
            .get(model_id)
            .map(|s| s.is_cached && s.is_enabled)
            .unwrap_or(false)
    }

    /// 获取指定模型的状态
    ///
    /// # 参数
    /// * `model_id` - 要查询的模型ID
    ///
    /// # 返回值
    /// * `Some(ModelStatus)` - 如果模型存在则返回其状态
    /// * `None` - 如果模型不存在
    ///
    /// # 示例
    /// ```rust
    /// use yi_coder_openapi::service::models::ModelManager;
    ///
    /// #[tokio::test]
    /// async fn test_get_model_status() {
    ///     let manager = ModelManager::new();
    ///     manager.download_model("yi-coder").await.unwrap();
    ///     assert!(manager.get_model_status("yi-coder").await.is_some());
    /// }
    /// */
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

    /// 获取Deepseek-Coder模型实例
    ///
    /// # 返回值
    /// * `Some(DeepseekCoderModel)` - 如果模型已加载
    /// * `None` - 如果模型未加载
    pub async fn get_deepseek_coder(&self) -> Option<DeepseekCoderModel> {
        let model = self.deepseek_coder.read().await;
        model.clone()
    }

    /// 获取所有模型状态
    ///
    /// # 返回值
    /// * `HashMap<String, ModelStatus>` - 包含所有模型状态的映射
    pub async fn get_all_model_status(&self) -> HashMap<String, ModelStatus> {
        let status = self.model_status.read().await;
        status.clone()
    }
}
