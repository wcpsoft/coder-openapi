use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use crate::service::chat::chat_completion::ChatCompletionParams;
use crate::service::models::deepseek_coder::DeepSeekCoder;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use rand::{self, Rng};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct DeepSeekCoderInference {
    device: Device,
    var_builder: VarBuilder<'static>,
    tokenizer: Tokenizer,
    transformer: Option<Arc<DeepSeekCoder>>,
    config: crate::service::models::deepseek_coder::config::ModelConfig,
}

impl DeepSeekCoderInference {
    /// 创建新的推理模块实例
    ///
    /// # 参数
    /// - config: 模型配置
    ///
    /// # 返回
    /// 初始化后的推理模块
    pub fn new(
        config: crate::service::models::deepseek_coder::config::ModelConfig,
    ) -> Result<Self, AppError> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        log::info!("Initializing Deepseek Coder with device: {:?}", device);

        let var_builder = VarBuilder::from_tensors(
            std::collections::HashMap::new(),
            candle_core::DType::F32,
            &device,
        );

        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| AppError::TokenizerError(e.to_string()))?;

        Ok(Self { device, var_builder, tokenizer, transformer: None, config })
    }

    fn load_transformer(&mut self) -> Result<Arc<DeepSeekCoder>, AppError> {
        if self.transformer.is_none() {
            let transformer = DeepSeekCoder::new(self.var_builder.clone(), &self.config)?;
            self.transformer = Some(Arc::new(transformer));
        }
        Ok(self.transformer.as_ref().unwrap().clone())
    }

    /// 执行推理
    ///
    /// # 参数
    /// - messages: 聊天消息列表
    /// - params: 完成参数
    ///
    /// # 返回
    /// 生成的聊天消息列表
    pub async fn infer(
        &mut self,
        messages: Vec<ChatCompletionMessage>,
        params: ChatCompletionParams,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        log::debug!("Starting Deepseek Coder inference");
        log::debug!("Input messages count: {}", messages.len());

        // 1. 将输入消息转换为模型输入格式
        let input_tokens = self.preprocess_messages(&messages)?;

        // 2. 执行模型推理
        let output_tokens = self.run_model(&input_tokens, &params)?;

        // 3. 将输出转换为聊天消息
        let responses = self.postprocess_output(&output_tokens)?;

        log::debug!("Generated response: {:?}", responses);
        Ok(responses)
    }

    fn preprocess_messages(
        &self,
        messages: &[ChatCompletionMessage],
    ) -> Result<Vec<u32>, AppError> {
        let mut tokens = Vec::new();
        for message in messages {
            let message_tokens = self
                .tokenizer
                .encode(message.content.clone(), false)
                .map_err(|e| AppError::TokenizerError(e.to_string()))?
                .get_ids()
                .to_vec();
            tokens.extend(message_tokens);
        }
        Ok(tokens)
    }

    fn run_model(
        &mut self,
        input_tokens: &[u32],
        params: &ChatCompletionParams,
    ) -> Result<Vec<u32>, AppError> {
        let input_tensor = Tensor::from_slice(input_tokens, &[input_tokens.len()], &self.device)?;

        let transformer = self.load_transformer()?;
        let logits = transformer.forward(
            &input_tensor,
            &Tensor::zeros(&[0], candle_core::DType::F32, &self.device)?,
            None,
            None,
        )?;

        self.sample_tokens(&logits, params)
    }

    fn postprocess_output(
        &self,
        output_tokens: &[u32],
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        let content =
            output_tokens.iter().map(|&t| char::from_u32(t).unwrap_or(' ')).collect::<String>();

        Ok(vec![ChatCompletionMessage { role: "assistant".to_string(), content }])
    }

    /// 根据logits进行token采样
    ///
    /// 使用温度采样算法，公式为：
    /// p_i = exp(logit_i / T) / sum(exp(logit_j / T))
    /// 其中 T 为温度参数，控制采样分布的平滑程度
    ///
    /// # 参数
    /// - logits: 模型输出的logits张量
    /// - params: 包含温度等采样参数的配置
    ///
    /// # 返回
    /// 采样得到的token序列
    fn sample_tokens(
        &self,
        logits: &Tensor,
        params: &ChatCompletionParams,
    ) -> Result<Vec<u32>, AppError> {
        if let Some(temp) = params.temperature {
            let scaled_logits = logits.div(&Tensor::new(&[temp], &self.device)?)?;
            let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
            let mut rng = rand::thread_rng();
            let random_val: f32 = rng.gen();
            let mut cumulative = 0.0;
            let probs_vec: Vec<f32> = probs.to_vec1()?;
            for (i, &p) in probs_vec.iter().enumerate() {
                cumulative += p;
                if cumulative >= random_val {
                    return Ok(vec![i as u32]);
                }
            }
        }

        Ok(vec![logits.argmax(0)?.to_scalar::<u32>()?])
    }
}
