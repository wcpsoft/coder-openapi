use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, ops::softmax, LayerNorm, VarBuilder};

/// YiCoder Transformer 模型结构
/// 实现基于Transformer架构的代码生成模型
/// 包含多个Transformer层和最后的LayerNorm
pub struct YiCoderTransformer {
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

        // 初始化Transformer层
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

        // 初始化LayerNorm层
        let weight = vb.get((config.hidden_size,), "weight")?;
        let bias = vb.get((config.hidden_size,), "bias")?;
        let norm = LayerNorm::new(weight, bias, config.layer_norm_eps);

        Ok(Self { layers, norm, device })
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
        let mut hidden_states = input.clone();

        // 逐层处理
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, None)?;
        }

        // 应用最后的LayerNorm
        hidden_states = self.norm.forward(&hidden_states)?;
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
        let weight1 = vb.get((hidden_size,), "weight")?;
        let bias1 = vb.get((hidden_size,), "bias")?;
        let norm1 = LayerNorm::new(weight1, bias1, 1e-5);

        let weight2 = vb.get((hidden_size,), "weight")?;
        let bias2 = vb.get((hidden_size,), "bias")?;
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
        let (batch_size, seq_len, _) = query.dims3()?;

        // 线性变换
        let query = self.query.forward(query)?;
        let key = self.key.forward(key)?;
        let value = self.value.forward(value)?;

        // 重塑为多头形式
        let query = query.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let key = key.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let value = value.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // 计算注意力分数 QK^T/√d_k
        let mut attention_scores = query.matmul(&key.t()?)?;
        attention_scores = (attention_scores / (self.head_dim as f64).sqrt())?;

        // 应用注意力掩码
        if let Some(mask) = attention_mask {
            let mask = mask.broadcast_as(attention_scores.shape())?;
            attention_scores = attention_scores.broadcast_add(&mask)?;
        }

        // Softmax归一化
        let dim = attention_scores.dims().len() - 1;
        let attention_probs = softmax(&attention_scores, dim)?;

        // 计算加权和
        let context = attention_probs.matmul(&value)?;
        // 重塑回原始形状
        let context = context.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

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
