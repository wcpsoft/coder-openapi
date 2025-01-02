use super::transformer::{self, TransformerError, YiCoderTransformer};
use candle_core::{Device, Result, Tensor, VarBuilder};
use candle_nn::{Embedding, LayerNorm};
use std::fmt;

impl fmt::Display for TransformerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransformerError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            TransformerError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
            TransformerError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            TransformerError::InvalidTensorValues(msg) => {
                write!(f, "Invalid tensor values: {}", msg)
            }
            TransformerError::LayerError(msg) => write!(f, "Layer error: {}", msg),
        }
    }
}

impl std::error::Error for TransformerError {}

impl From<TransformerError> for candle_core::Error {
    fn from(err: TransformerError) -> Self {
        candle_core::Error::Msg(err.to_string())
    }
}

/// 验证张量值是否为NaN/Infinity
fn validate_tensor(tensor: &Tensor, context: &str) -> Result<()> {
    // Convert tensor to f32 and check for invalid values
    let tensor_f32 = tensor.to_dtype(candle_core::DType::F32)?;
    let values = tensor_f32.flatten_all()?.to_vec1::<f32>()?;

    let nan_count = values.iter().filter(|&x| x.is_nan()).count();
    let inf_count = values.iter().filter(|&x| x.is_infinite()).count();

    if nan_count > 0 || inf_count > 0 {
        log::error!(
            "{}: Invalid tensor values detected - NaN: {}, Infinite: {}",
            context,
            nan_count,
            inf_count
        );
        return Err(TransformerError::InvalidTensorValues(format!(
            "{}: Contains NaN/Inf values",
            context
        ))
        .into());
    }
    Ok(())
}

/// 验证张量形状
fn validate_shape(tensor: &Tensor, expected: &[usize], context: &str) -> Result<()> {
    let actual = tensor.dims();
    if actual != expected {
        log::error!("{}: Shape mismatch. Expected {:?}, got {:?}", context, expected, actual);
        return Err(TransformerError::ShapeMismatch(format!(
            "{}: Expected shape {:?}, got {:?}",
            context, expected, actual
        ))
        .into());
    }
    Ok(())
}

/// YiCoder Transformer模型
/// 实现用于代码生成的Transformer架构
#[derive(Debug)]
pub struct YiCoderTransformer {
    encoder: transformer::YiCoderEncoder,
    decoder: transformer::YiCoderDecoder,
    embeddings: Embedding,
    device: Device,
    _config: super::config::ModelConfig,
}

impl YiCoderTransformer {
    /// 创建新的YiCoderTransformer实例
    ///
    /// # 参数
    /// * `config` - 模型配置
    /// * `vb` - 变量构建器
    ///
    /// # 返回
    /// Result<Self> - 新的transformer实例
    pub fn new(config: &super::config::ModelConfig, vb: VarBuilder) -> Result<Self> {
        let config = config.clone();
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        log::debug!("Selected computation device: {:?}", device);

        // Initialize encoder and decoder
        let encoder = transformer::YiCoderEncoder::new(&config, vb.pp("encoder"))?;
        let decoder = transformer::YiCoderDecoder::new(&config, vb.pp("decoder"))?;

        // Initialize embeddings
        log::debug!("Initializing embeddings");
        let embeddings = Embedding::new(
            vb.get((config.vocab_size, config.hidden_size), "model.embeddings.word_embeddings")
                .unwrap_or_else(|_| {
                    log::warn!("model.embeddings.word_embeddings not found, using zero tensor");
                    Tensor::zeros(
                        (config.vocab_size, config.hidden_size),
                        candle_core::DType::F32,
                        &device,
                    )
                    .unwrap()
                }),
            config.hidden_size,
        );

        Ok(Self { encoder, decoder, embeddings, device, _config: config.clone() })
    }

    /// 执行Transformer前向传播
    /// 参数:
    /// - input: 输入张量
    /// - attention_mask: 注意力掩码（可选）
    /// 返回: Result<Tensor>
    pub fn transform(&self, input: Tensor, attention_mask: Option<Tensor>) -> Result<Tensor> {
        self.forward(&input, attention_mask.as_ref())
    }

    /// 获取当前设备 (CPU/GPU)
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 执行Transformer前向传播
    /// 参数:
    /// - input: 输入张量
    /// 返回: Result<Tensor>
    pub fn forward(&self, input: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        log::debug!("[Transformer] Starting forward pass");
        log::debug!(
            "[Transformer] Initial input - shape: {:?}, dtype: {:?}",
            input.shape(),
            input.dtype()
        );

        // Validate and convert input tensor
        log::debug!("[Transformer] Validating input tensor");

        // Convert input to i64 for Embedding layer
        let input_i64 = if input.dtype() != candle_core::DType::I64 {
            log::warn!("[Transformer] Converting input dtype from {:?} to I64", input.dtype());
            input.to_dtype(candle_core::DType::I64)?
        } else {
            input.clone()
        };

        // Validate integer values
        let min_value = input_i64.min(&Tensor::new(0, &self.device)?)?.to_scalar::<i64>()?;
        if min_value < 0 {
            log::error!("[Transformer] Input contains negative values");
            return Err(candle_core::Error::msg(AppError::new(
                "Input tensor contains negative values which are invalid for embeddings"
                    .to_string(),
            )));
        }

        // Apply embeddings with i64 input
        log::debug!("[Transformer] Applying embeddings");
        let mut hidden_states = self.embeddings.forward(&input_i64)?;
        hidden_states = hidden_states.clamp(-1e4, 1e4)?;

        // Convert embeddings output to F32 for subsequent layers
        hidden_states = hidden_states.to_dtype(candle_core::DType::F32)?;
        log::debug!(
            "[Transformer] Embeddings output - shape: {:?}, dtype: {:?}",
            hidden_states.shape(),
            hidden_states.dtype()
        );

        // Validate embeddings output
        validate_tensor(&hidden_states, "Embeddings output")?;

        if hidden_states.dtype() != candle_core::DType::F32 {
            log::error!(
                "[Transformer] Invalid embeddings output dtype: {:?}, expected F32",
                hidden_states.dtype()
            );
            return Err(candle_core::Error::msg(AppError::new(format!(
                "Expected F32 embeddings output, got {:?}",
                hidden_states.dtype()
            ))));
        }

        // Handle tensor dimensions
        log::debug!("[Transformer] Handling tensor dimensions");
        match hidden_states.dims() {
            &[1] => {
                // Special case for single-element tensor
                log::debug!("[Transformer] Handling single-element tensor");
                hidden_states = hidden_states.unsqueeze(0)?.unsqueeze(0)?;
                log::debug!(
                    "[Transformer] After adding dimensions - shape: {:?}",
                    hidden_states.shape()
                );
            }
            &[_batch_size] => {
                // General rank 1 case
                log::debug!("[Transformer] Adding batch dimension");
                hidden_states = hidden_states.unsqueeze(0)?;
                log::debug!(
                    "[Transformer] After adding batch dimension - shape: {:?}",
                    hidden_states.shape()
                );
            }
            &[_batch_size, _seq_len] => {
                log::debug!("[Transformer] Input has correct dimensions");
            }
            &[1, _batch_size, _seq_len] => {
                log::debug!("[Transformer] Removing extra dimension");
                hidden_states = hidden_states.squeeze(0)?;
                log::debug!(
                    "[Transformer] After removing extra dimension - shape: {:?}",
                    hidden_states.shape()
                );
            }
            _ => {
                log::error!("[Transformer] Unexpected tensor shape: {:?}", hidden_states.dims());
                return Err(candle_core::Error::msg(AppError::new(format!(
                    "Unexpected tensor shape: {:?}",
                    hidden_states.dims()
                ))));
            }
        }
        // Process through encoder
        log::debug!("[Transformer] Processing through encoder");
        hidden_states = self.encoder.forward(&hidden_states, None)?;
        log::debug!(
            "[Transformer] Encoder output - shape: {:?}, dtype: {:?}",
            hidden_states.shape(),
            hidden_states.dtype()
        );

        // Process through decoder
        log::debug!("[Transformer] Processing through decoder");
        hidden_states = self.decoder.forward(&hidden_states, None)?;
        log::debug!(
            "[Transformer] Decoder output - shape: {:?}, dtype: {:?}",
            hidden_states.shape(),
            hidden_states.dtype()
        );

        // Ensure proper input type (F32)
        let hidden_states = if hidden_states.dtype() != candle_core::DType::F32 {
            log::warn!("Converting hidden states from {:?} to F32", hidden_states.dtype());
            hidden_states.to_dtype(candle_core::DType::F32)?
        } else {
            hidden_states
        };

        // Add more aggressive numerical stability checks
        let mut hidden_states = hidden_states.clamp(-1e3, 1e3)?;

        // Check for NaN/Inf values before layer norm
        let values = hidden_states.flatten_all()?.to_vec1::<f32>()?;
        let nan_count = values.iter().filter(|&x| x.is_nan()).count();
        let inf_count = values.iter().filter(|&x| x.is_infinite()).count();

        if nan_count > 0 || inf_count > 0 {
            log::error!(
                "NaN/Inf detected before final layer norm - NaN: {}, Inf: {}",
                nan_count,
                inf_count
            );
            return Err(candle_core::Error::msg(AppError::new(
                "NaN/Inf values detected before final layer norm".to_string(),
            )));
        }

        // Add robust variance stability check with more aggressive stabilization
        let variance = hidden_states.var(1)?;
        let min_variance = variance.min(0)?.to_scalar::<f32>()?;
        let stability_factor = if min_variance < 1e-20 {
            log::warn!(
                "Extremely low variance detected: {}. Adding larger stability factor.",
                min_variance
            );
            1e-5f32 // Increased stability factor for very low variance
        } else if min_variance < 1e-8 {
            log::warn!("Low variance detected: {}. Adding stability factor.", min_variance);
            1e-6f32 // Increased stability factor
        } else {
            1e-8f32 // Always add small stability factor
        };

        // Add stability factor and clamp values
        hidden_states = hidden_states
            .broadcast_add(&Tensor::new(stability_factor, &self.device)?)?
            .clamp(-1e3, 1e3)?;

        // Recompute variance after stabilization
        let _variance = hidden_states.var(1)?.clamp(1e-10, f32::MAX)?;

        log::debug!("[Transformer] Forward pass completed successfully");
        Ok(hidden_states)
    }
}
