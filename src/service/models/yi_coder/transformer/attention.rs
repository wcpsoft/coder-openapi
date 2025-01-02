use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// 多头注意力机制实现
pub struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    num_heads: usize,
    head_size: usize,
}

impl MultiHeadAttention {
    /// 创建新的多头注意力层
    ///
    /// # 参数
    /// - hidden_size: 隐藏层大小
    /// - num_heads: 注意力头数量
    /// - vb: 变量构建器
    ///
    /// # 返回
    /// 初始化后的MultiHeadAttention实例
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_size = hidden_size / num_heads;
        let query = linear_no_bias(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let key = linear_no_bias(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let value = linear_no_bias(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let output = linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self { query, key, value, output, num_heads, head_size })
    }

    /// 前向传播
    ///
    /// # 参数
    /// - query: 查询向量 [batch_size, seq_len, hidden_size]
    /// - key: 键向量 [batch_size, seq_len, hidden_size]
    /// - value: 值向量 [batch_size, seq_len, hidden_size]
    ///
    /// # 返回
    /// 注意力计算结果 [batch_size, seq_len, hidden_size]
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let batch_size = query.dim(0)?;
        let seq_len = query.dim(1)?;

        // 投影输入
        let query = self.query.forward(query)?;
        let key = self.key.forward(key)?;
        let value = self.value.forward(value)?;

        // 重塑为多头注意力形状
        let query = query.reshape((batch_size, seq_len, self.num_heads, self.head_size))?;
        let key = key.reshape((batch_size, seq_len, self.num_heads, self.head_size))?;
        let value = value.reshape((batch_size, seq_len, self.num_heads, self.head_size))?;

        // 计算注意力分数
        // 公式: Q * K^T / sqrt(d_k)
        // 其中:
        // Q: 查询矩阵 [batch_size, seq_len, num_heads, head_size]
        // K: 键矩阵 [batch_size, seq_len, num_heads, head_size]
        // d_k: 每个注意力头的维度大小 (head_size)
        let scores = query.matmul(&key.transpose(2, 3)?)?;
        let scores = scores / (self.head_size as f64).sqrt();
        let scores = scores?; // Ensure scores is a Tensor, not Result<Tensor>

        // 应用softmax
        let attention_weights = candle_nn::ops::softmax(&scores, 3)?;

        // 将注意力应用到值向量
        let context = attention_weights.matmul(&value)?;

        // 重塑回原始维度
        let context = context.reshape((batch_size, seq_len, self.num_heads * self.head_size))?;

        // 投影输出
        self.output.forward(&context)
    }
}
