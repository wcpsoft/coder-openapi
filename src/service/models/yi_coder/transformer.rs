use crate::error::AppError;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, ops::softmax, Embedding, LayerNorm, VarBuilder};

/// YiCoder Transformer 模型结构
/// 实现基于Transformer架构的代码生成模型
/// 包含多个Transformer层和最后的LayerNorm
pub struct YiCoderTransformer {
    /// 词嵌入层
    embeddings: Embedding,
    /// Transformer层列表
    layers: Vec<TransformerLayer>,
    /// 最后的LayerNorm层
    norm: LayerNorm,
    /// 计算设备 (CPU/GPU)
    device: Device,
}

/// 单个Transformer层结构
/// 包含多头注意力机制和前馈网络
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
struct PositionWiseFeedForward {
    /// 第一个全连接层
    fc1: linear::Linear,
    /// 第二个全连接层
    fc2: linear::Linear,
}

impl YiCoderTransformer {
    /// 创建新的YiCoderTransformer实例
    /// 参数:
    /// - config: 模型配置
    /// - vb: 变量构建器
    /// 返回: Result<Self>
    pub fn new(config: &super::config::ModelConfig, vb: VarBuilder) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        log::debug!("完成计算资源类型选择：{:?}", device.clone());
        // 初始化Transformer层
        log::debug!("初始化Transformer层开始");
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = TransformerLayer::new(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                vb.pp(format!("layer_{}", i)),
            )?;
            layers.push(layer);
        }
        log::debug!("初始化Transformer层结束");

        log::debug!("初始化LayerNorm层开始");
        log::debug!("读取hidden_size:{}", config.hidden_size);
        // 初始化LayerNorm层
        let weight = vb.get((config.hidden_size,), "model.norm.weight")?;
        let bias = vb.get((config.hidden_size,), "model.norm.bias").unwrap_or_else(|_| {
            log::warn!("model.norm.bias not found, using zero tensor");
            Tensor::zeros((config.hidden_size,), weight.dtype(), &weight.device()).unwrap()
        });
        log::debug!(
            "Loaded final layer norm: weight shape {:?}, bias shape {:?}",
            weight.shape(),
            bias.shape()
        );
        let norm = LayerNorm::new(weight, bias, config.layer_norm_eps);
        log::debug!("初始化LayerNorm层开始");

        // 初始化词嵌入层
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

        Ok(Self { embeddings, layers, norm, device })
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

        // Validate input tensor
        log::debug!("[Transformer] Validating input tensor");
        let values = input.to_vec1::<f32>()?;
        if values.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            log::error!("[Transformer] Input contains NaN/infinite values");
            return Err(candle_core::Error::msg(AppError::new(
                "Input tensor contains NaN or infinite values".to_string(),
            )));
        }

        // Convert input to F32
        log::debug!("[Transformer] Converting input to F32");
        let input = input.to_dtype(candle_core::DType::F32)?;
        log::debug!(
            "[Transformer] Converted input - shape: {:?}, dtype: {:?}",
            input.shape(),
            input.dtype()
        );
        if input.dtype() != candle_core::DType::F32 {
            log::error!("[Transformer] Invalid input dtype: {:?}, expected F32", input.dtype());
            return Err(candle_core::Error::msg(AppError::new(format!(
                "Expected F32 input, got {:?}",
                input.dtype()
            ))));
        }

        // Apply embeddings
        log::debug!("[Transformer] Applying embeddings");
        let mut hidden_states = self.embeddings.forward(&input)?;
        log::debug!(
            "[Transformer] Embeddings output - shape: {:?}, dtype: {:?}",
            hidden_states.shape(),
            hidden_states.dtype()
        );
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
        match hidden_states.dims().len() {
            1 => {
                log::debug!("[Transformer] Adding batch dimension");
                hidden_states = hidden_states.unsqueeze(0)?;
                log::debug!(
                    "[Transformer] After adding batch dimension - shape: {:?}",
                    hidden_states.shape()
                );
            }
            2 => {
                log::debug!("[Transformer] Input has correct dimensions");
            }
            3 => {
                log::debug!("[Transformer] Removing extra dimension");
                hidden_states = hidden_states.squeeze(0)?;
                log::debug!(
                    "[Transformer] After removing extra dimension - shape: {:?}",
                    hidden_states.shape()
                );
            }
            _ => {
                log::error!("[Transformer] Unexpected tensor rank: {}", hidden_states.dims().len());
                return Err(candle_core::Error::msg(AppError::new(format!(
                    "Unexpected tensor rank: {}",
                    hidden_states.dims().len()
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
            let values = hidden_states.to_vec1::<f32>()?;
            if values.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                log::error!("[Transformer] Layer {} output contains NaN/infinite values", i);
                return Err(candle_core::Error::msg(AppError::new(format!(
                    "Layer {} output contains invalid values",
                    i
                ))));
            }
        }

        // Apply final layer norm
        log::debug!("[Transformer] Applying final layer norm");
        hidden_states = self.norm.forward(&hidden_states)?;
        log::debug!(
            "[Transformer] Final output - shape: {:?}, dtype: {:?}",
            hidden_states.shape(),
            hidden_states.dtype()
        );

        // Validate final output
        log::debug!("[Transformer] Validating final output");
        let values = hidden_states.to_vec1::<f32>()?;
        if values.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            log::error!("[Transformer] Final output contains NaN/infinite values");
            return Err(candle_core::Error::msg(AppError::new(
                "Output tensor contains NaN or infinite values".to_string(),
            )));
        }

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
        let mut attention_scores = query.matmul(&key.t()?)?.to_dtype(candle_core::DType::F32)?;
        log::debug!("Attention scores shape: {:?}", attention_scores.shape());
        log::debug!("Attention scores dtype: {:?}", attention_scores.dtype());
        let scale_factor = Tensor::new((self.head_dim as f32).sqrt(), attention_scores.device())?
            .to_dtype(candle_core::DType::F32)?;
        attention_scores = attention_scores.broadcast_div(&scale_factor)?;

        // 应用注意力掩码
        if let Some(mask) = attention_mask {
            let mask = mask.broadcast_as(attention_scores.shape())?;
            attention_scores = attention_scores.broadcast_add(&mask)?;
        }

        // Softmax归一化
        let dim = attention_scores.dims().len() - 1;
        let attention_probs = softmax(&attention_scores, dim)?;

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
