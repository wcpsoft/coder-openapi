pub mod config;
pub mod infer;
pub mod loader;
pub mod transformer;

use self::config::ModelConfig;
use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use crate::service::chat::chat_completion::ChatCompletionParams;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use self::transformer::{DeepSeekCoderDecoder, DeepSeekCoderEncoder, TransformerError};

/// DeepSeekCoder 模型结构体
///
/// 包含编码器、解码器和设备信息
pub struct DeepSeekCoder {
    encoder: DeepSeekCoderEncoder,
    _decoder: DeepSeekCoderDecoder,
    _device: Device,
}

impl DeepSeekCoder {
    /// 执行推理
    ///
    /// # 参数
    /// - messages: 聊天消息列表
    /// - params: 完成参数
    ///
    /// # 返回
    /// 生成的聊天消息列表
    pub fn infer(
        &self,
        messages: Vec<ChatCompletionMessage>,
        _params: ChatCompletionParams,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        // Implement inference logic here
        Ok(messages)
    }
    /// 创建新的 DeepSeekCoder 实例
    ///
    /// # 参数
    /// - config: 模型配置
    /// - vb: 变量构建器
    ///
    /// # 返回
    /// 初始化后的 DeepSeekCoder 实例
    pub fn new(
        vb: VarBuilder,
        config: &crate::service::models::deepseek_coder::config::ModelConfig,
    ) -> Result<Self, TransformerError> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let encoder = DeepSeekCoderEncoder::new(config, vb.pp("encoder"))?;
        let decoder = DeepSeekCoderDecoder::new(config, vb.pp("decoder"))?;

        Ok(Self { encoder, _decoder: decoder, _device: device })
    }

    /// 执行前向传播
    ///
    /// # 参数
    /// - input: 输入张量
    /// - _encoder_output: 编码器输出（未使用）
    /// - self_attention_mask: 自注意力掩码
    /// - cross_attention_mask: 交叉注意力掩码
    ///
    /// # 返回
    /// 计算结果张量
    pub fn forward(
        &self,
        input: &Tensor,
        _encoder_output: &Tensor,
        self_attention_mask: Option<&Tensor>,
        _cross_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, TransformerError> {
        let encoder_output = self.encoder.forward(input, self_attention_mask)?;
        Ok(encoder_output)
    }
}
