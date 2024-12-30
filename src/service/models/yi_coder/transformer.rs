use crate::error::AppError;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, ops::softmax, Embedding, LayerNorm, VarBuilder};
use std::fmt;

#[derive(Debug)]
pub enum TransformerError {
    InvalidInput(String),
    NumericalInstability(String),
    ShapeMismatch(String),
    InvalidTensorValues(String),
    LayerError(String),
}

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
/// 包含多个Transformer层和最终的LayerNorm
#[derive(Debug)]
pub struct YiCoderTransformer {
    /// Word embeddings layer
    embeddings: Embedding,
    /// List of Transformer layers
    layers: Vec<TransformerLayer>,
    /// Final LayerNorm layer
    norm: LayerNorm,
    /// Computation device (CPU/GPU)
    device: Device,
    /// Configuration parameters
    _config: super::config::ModelConfig,
}

/// 单个Transformer层结构
/// 包含多头注意力机制和前馈网络
#[derive(Debug)]
struct TransformerLayer {
    /// 多头注意力机制
    attention: MultiHeadAttention,
    /// 位置前馈网络
    feed_forward: PositionWiseFeedForward,
    /// 第一个LayerNorm层
    norm1: LayerNorm,
    /// 第二个LayerNorm层
    norm2: LayerNorm,
}

/// 多头注意力机制结构
/// 实现公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V
#[derive(Debug)]
struct MultiHeadAttention {
    /// 查询矩阵线性变换
    query: linear::Linear,
    /// 键矩阵线性变换
    key: linear::Linear,
    /// 值矩阵线性变换
    value: linear::Linear,
    /// 输出线性变换
    out: linear::Linear,
    /// 注意力头数量
    num_heads: usize,
    /// 每个注意力头的维度
    head_dim: usize,
}

/// 位置前馈网络结构
/// 实现公式：FFN(x) = max(0, xW1 + b1)W2 + b2
#[derive(Debug)]
struct PositionWiseFeedForward {
    /// 第一个全连接层
    fc1: linear::Linear,
    /// 第二个全连接层
    fc2: linear::Linear,
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

        // Initialize Transformer layers
        log::debug!("Initializing {} Transformer layers", config.num_layers);
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            log::debug!("Initializing layer {}", i);
            let layer = TransformerLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                vb.pp(format!("layer_{}", i)),
            )
            .map_err(|e| {
                log::error!("Failed to initialize layer {}: {}", i, e);
                TransformerError::LayerError(format!("Failed to initialize layer {}: {}", i, e))
            })?;
            layers.push(layer);
        }

        // Initialize final LayerNorm
        log::debug!("Initializing final LayerNorm");
        let weight = vb.get((config.hidden_size,), "model.norm.weight")?;
        let bias = vb.get((config.hidden_size,), "model.norm.bias").unwrap_or_else(|_| {
            log::warn!("model.norm.bias not found, using zero tensor");
            Tensor::zeros((config.hidden_size,), weight.dtype(), &weight.device()).unwrap()
        });

        validate_tensor(&weight, "Final layer norm weight")?;
        validate_tensor(&bias, "Final layer norm bias")?;

        let norm = LayerNorm::new(weight, bias, config.layer_norm_eps);
        log::debug!("Final LayerNorm initialized");

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

        Ok(Self { embeddings, layers, norm, device, _config: config.clone() })
    }

    /// 执行Transformer前向传播
    /// 参数:
    /// - input: 输入张量
    /// - attention_mask: 注意力掩码（可选）
    /// 返回: Result<Tensor>
    pub async fn transform(&self, input: Tensor, attention_mask: Option<Tensor>) -> Result<Tensor> {
        let mut hidden_states = input;

        // 逐层处理
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask.as_ref())?;
        }

        // 应用最后的LayerNorm
        hidden_states = self.norm.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    /// 获取当前设备 (CPU/GPU)
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 执行Transformer前向传播
    /// 参数:
    /// - input: 输入张量
    /// 返回: Result<Tensor>
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
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
        let min_value = input_i64.min(0)?.to_scalar::<i64>()?;
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
        // Process through transformer layers
        log::debug!("[Transformer] Processing through layers");
        for (i, layer) in self.layers.iter().enumerate() {
            log::debug!("[Transformer] Processing layer {}", i);
            hidden_states = layer.forward(&hidden_states, None)?;
            log::debug!(
                "[Transformer] Layer {} output - shape: {:?}, dtype: {:?}",
                i,
                hidden_states.shape(),
                hidden_states.dtype()
            );

            // Validate layer output
            validate_tensor(&hidden_states, &format!("Layer {} output", i))?;
        }
        // Validate input to final layer norm
        validate_tensor(&hidden_states, "Final layer norm input")?;

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

        // Apply layer norm with additional stability
        log::debug!("[Transformer] Applying final layer norm");
        log::debug!("[Transformer] self.norm.forward xs input hidden_states values {:?}", values);
        log::debug!(
            "[Transformer] Input mean: {:?}, variance: {:?}",
            hidden_states.mean(1)?.to_vec1::<f32>()?,
            hidden_states.var(1)?.to_vec1::<f32>()?
        );

        // Apply layer norm in smaller chunks with additional validation
        let chunk_size = 256; // Reduced chunk size for better numerical stability
        let mut output = Vec::new();
        for chunk in hidden_states.chunk(chunk_size, 0)? {
            // Add additional clamping and validation before layer norm
            let chunk = chunk.clamp(-1e3, 1e3)?;
            validate_tensor(&chunk, "Layer norm input chunk")?;

            // Apply layer norm with additional stability
            let normed_chunk = self.norm.forward(&chunk)?;

            // Validate and clamp output
            validate_tensor(&normed_chunk, "Layer norm chunk output")?;
            let normed_chunk = normed_chunk.clamp(-1e3, 1e3)?;

            output.push(normed_chunk);
        }
        let hidden_states = Tensor::cat(&output, 0)?.clamp(-1e3, 1e3)?;

        log::debug!(
            "[Transformer] Output mean: {:?}, variance: {:?}",
            hidden_states.mean(1)?.to_vec1::<f32>()?,
            hidden_states.var(1)?.to_vec1::<f32>()?
        );

        log::debug!(
            "[Transformer] self.norm.forward output values {:?}",
            hidden_states.flatten_all()?.to_vec1::<f32>()?
        );

        // Validate output
        validate_tensor(&hidden_states, "Final layer norm output")?;

        log::debug!(
            "[Transformer] Final output - shape: {:?}, dtype: {:?}",
            hidden_states.shape(),
            hidden_states.dtype()
        );
        // Validate final output
        validate_tensor(&hidden_states, "Final output")?;

        log::debug!("[Transformer] Forward pass completed successfully");
        Ok(hidden_states)
    }
}

impl TransformerLayer {
    /// 创建新的TransformerLayer实例
    /// 参数:
    /// - hidden_size: 隐藏层大小
    /// - num_heads: 注意力头数量
    /// - intermediate_size: 前馈网络中间层大小
    /// - vb: 变量构建器
    /// 返回: Result<Self>
    fn new(
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // 初始化多头注意力机制
        let attention = MultiHeadAttention::new(hidden_size, num_heads, vb.pp("attention"))?;

        // 初始化前馈网络
        let feed_forward =
            PositionWiseFeedForward::new(hidden_size, intermediate_size, vb.pp("ffn"))?;

        // 初始化LayerNorm层
        let weight1 = vb.get((hidden_size,), "input_layernorm.weight")?;
        let bias1 = vb.get((hidden_size,), "input_layernorm.bias")?;
        log::debug!(
            "Loaded input layer norm: weight shape {:?}, bias shape {:?}",
            weight1.shape(),
            bias1.shape()
        );
        let norm1 = LayerNorm::new(weight1, bias1, 1e-5);

        let weight2 = vb.get((hidden_size,), "post_attention_layernorm.weight")?;
        let bias2 = vb.get((hidden_size,), "post_attention_layernorm.bias")?;
        log::debug!(
            "Loaded post attention layer norm: weight shape {:?}, bias shape {:?}",
            weight2.shape(),
            bias2.shape()
        );
        let norm2 = LayerNorm::new(weight2, bias2, 1e-5);

        Ok(Self { attention, feed_forward, norm1, norm2 })
    }

    /// Transformer层前向传播
    /// 实现公式: Layer(x) = LayerNorm(x + Attention(x))
    ///           Layer(x) = LayerNorm(x + FFN(x))
    /// 参数:
    /// - input: 输入张量
    /// - attention_mask: 注意力掩码（可选）
    /// 返回: Result<Tensor>
    fn forward(&self, input: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // 多头注意力机制
        let attention_output = self.attention.forward(input, input, input, attention_mask)?;
        // 残差连接 + LayerNorm
        let attention_output = self.norm1.forward(&(input + &attention_output)?)?;

        // 前馈网络
        let feed_forward_output = self.feed_forward.forward(&attention_output)?;
        // 残差连接 + LayerNorm
        let output = self.norm2.forward(&(attention_output + &feed_forward_output)?)?;

        Ok(output)
    }
}

impl MultiHeadAttention {
    /// 创建新的MultiHeadAttention实例
    /// 参数:
    /// - hidden_size: 隐藏层大小
    /// - num_heads: 注意力头数量
    /// - vb: 变量构建器
    /// 返回: Result<Self>
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        // 初始化线性变换层
        let query = linear(hidden_size, hidden_size, vb.pp("query"))?;
        let key = linear(hidden_size, hidden_size, vb.pp("key"))?;
        let value = linear(hidden_size, hidden_size, vb.pp("value"))?;
        let out = linear(hidden_size, hidden_size, vb.pp("out"))?;

        Ok(Self { query, key, value, out, num_heads, head_dim })
    }

    /// 多头注意力机制前向传播
    /// 实现公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    /// 参数:
    /// - query: 查询张量
    /// - key: 键张量
    /// - value: 值张量
    /// - attention_mask: 注意力掩码（可选）
    /// 返回: Result<Tensor>
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        log::debug!(
            "MultiHeadAttention input shapes - query: {:?}, key: {:?}, value: {:?}",
            query.shape(),
            key.shape(),
            value.shape()
        );
        let (batch_size, seq_len, _) = query.dims3()?;

        // 线性变换并转换为F32
        let query = self.query.forward(query)?.to_dtype(candle_core::DType::F32)?;
        let key = self.key.forward(key)?.to_dtype(candle_core::DType::F32)?;
        let value = self.value.forward(value)?.to_dtype(candle_core::DType::F32)?;

        // 打印调试信息
        log::debug!("Query shape before reshape: {:?}", query.shape());
        log::debug!("Key shape before reshape: {:?}", key.shape());
        log::debug!("Value shape before reshape: {:?}", value.shape());

        // 重塑为多头形式
        let query = query.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let key = key.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let value = value.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // 打印调试信息
        log::debug!("Query shape after reshape: {:?}", query.shape());
        log::debug!("Key shape after reshape: {:?}", key.shape());
        log::debug!("Value shape after reshape: {:?}", value.shape());

        // 计算注意力分数 QK^T/√d_k
        log::debug!("Query shape before matmul: {:?}", query.shape());
        log::debug!("Key shape before matmul: {:?}", key.shape());

        // Add numerical stability checks
        let query = query.clamp(-1e4, 1e4)?;
        let key = key.clamp(-1e4, 1e4)?;

        let mut attention_scores = query.matmul(&key.t()?)?.to_dtype(candle_core::DType::F32)?;
        log::debug!("Attention scores shape: {:?}", attention_scores.shape());
        log::debug!("Attention scores dtype: {:?}", attention_scores.dtype());

        // Add validation for attention scores
        let flattened_scores = attention_scores.flatten_all()?;
        let scores_min = flattened_scores.min(0)?.to_scalar::<f32>()?;
        let scores_max = flattened_scores.max(0)?.to_scalar::<f32>()?;
        log::debug!("Attention scores range: [{}, {}]", scores_min, scores_max);

        if scores_min.is_nan() || scores_max.is_nan() {
            log::error!("Attention scores contain NaN values");
            return Err(candle_core::Error::msg(AppError::new(
                "Attention scores contain NaN values".to_string(),
            )));
        }

        let scale_factor = Tensor::new((self.head_dim as f32).sqrt(), attention_scores.device())?
            .to_dtype(candle_core::DType::F32)?;
        attention_scores = attention_scores.broadcast_div(&scale_factor)?;

        // Clamp attention scores to prevent overflow
        attention_scores = attention_scores.clamp(-50.0, 50.0)?;

        // 应用注意力掩码
        if let Some(mask) = attention_mask {
            log::debug!("Applying attention mask with shape: {:?}", mask.shape());
            let mask = mask.to_dtype(candle_core::DType::F32)?;
            let mask = mask.broadcast_as(attention_scores.shape())?;
            if mask.shape() != attention_scores.shape() {
                log::error!(
                    "Attention mask shape mismatch. Expected: {:?}, Got: {:?}",
                    attention_scores.shape(),
                    mask.shape()
                );
                return Err(candle_core::Error::msg(AppError::new(format!(
                    "Attention mask shape mismatch. Expected: {:?}, Got: {:?}",
                    attention_scores.shape(),
                    mask.shape()
                ))));
            }
            attention_scores = attention_scores.broadcast_add(&mask)?;
        }

        // Validate tensor rank before softmax
        let cloned_scores = attention_scores.clone();
        let dims = cloned_scores.dims();
        if dims.len() < 2 {
            log::error!("Invalid attention scores rank: {:?}, expected at least rank 2", dims);
            return Err(candle_core::Error::msg(AppError::new(format!(
                "Invalid attention scores rank: {:?}, expected at least rank 2",
                dims
            ))));
        }

        // Softmax normalization
        let dim = dims.len() - 1;

        // Add numerical stability to softmax
        let max_values = attention_scores.max(dim)?;

        // Ensure max_values is rank 0
        let max_scalar = if max_values.dims().is_empty() {
            max_values.to_scalar::<f32>()?
        } else {
            log::warn!(
                "Max values tensor has rank {}, converting to scalar",
                max_values.dims().len()
            );
            max_values.flatten_all()?.to_scalar::<f32>()?
        };

        let max_tensor = Tensor::new(max_scalar, attention_scores.device())?;
        let stable_scores = (attention_scores.clone() - max_tensor)?;

        // Validate stable scores shape
        validate_shape(&stable_scores, dims, "Stable attention scores")?;

        let attention_probs = softmax(&stable_scores, dim)?;

        // Validate softmax output
        let flattened_probs = attention_probs.flatten_all()?;
        let probs_min = flattened_probs.min(0)?.to_scalar::<f32>()?;
        let probs_max = flattened_probs.max(0)?.to_scalar::<f32>()?;
        log::debug!("Attention probabilities range: [{}, {}]", probs_min, probs_max);

        if probs_min.is_nan() || probs_max.is_nan() {
            log::error!("Attention probabilities contain NaN values");
            return Err(candle_core::Error::msg(AppError::new(
                "Attention probabilities contain NaN values".to_string(),
            )));
        }

        // 计算加权和
        log::debug!("Attention probs shape: {:?}", attention_probs.shape());
        log::debug!("Value shape: {:?}", value.shape());
        let context = attention_probs.matmul(&value)?;
        log::debug!("Context shape before reshape: {:?}", context.shape());
        log::debug!("Context dtype: {:?}", context.dtype());
        // 重塑回原始形状
        let context = context.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        log::debug!("Context shape after reshape: {:?}", context.shape());

        // 输出线性变换
        let output = self.out.forward(&context)?;
        Ok(output)
    }
}

impl PositionWiseFeedForward {
    /// 创建新的PositionWiseFeedForward实例
    /// 参数:
    /// - hidden_size: 隐藏层大小
    /// - intermediate_size: 中间层大小
    /// - vb: 变量构建器
    /// 返回: Result<Self>
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // 初始化全连接层
        let fc1 = linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(intermediate_size, hidden_size, vb.pp("fc2"))?;

        Ok(Self { fc1, fc2 })
    }

    /// 前馈网络前向传播
    /// 实现公式: FFN(x) = GELU(xW1 + b1)W2 + b2
    /// 参数:
    /// - input: 输入张量
    /// 返回: Result<Tensor>
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // 第一层全连接 + GELU激活
        let hidden = self.fc1.forward(input)?;
        let hidden = hidden.gelu()?;
        // 第二层全连接
        let output = self.fc2.forward(&hidden)?;
        Ok(output)
    }
}
