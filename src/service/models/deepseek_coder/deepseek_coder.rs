use super::config::ModelConfig;
use super::inference::DeepSeekCoderInference;
use super::loader::DeepseekCoderLoader;
use super::transformer::DeepseekCoderTransformer;
use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use crate::service::chat::chat_completion::ChatCompletionParams;
use candle_core::Tensor;
use candle_nn::Module;
use rand::{thread_rng, Rng};

/// DeepseekCoder 代码生成模型
/// 基于 DeepSeek AI 的代码生成模型实现
/// 包含配置、加载器、转换器和推理模块
pub struct DeepseekCoder {
    _config: ModelConfig,                   // 模型配置
    _loader: DeepseekCoderLoader,           // 模型加载器
    _transformer: DeepseekCoderTransformer, // 转换器模块
    _inference: DeepSeekCoderInference,     // 推理模块
}

impl DeepseekCoder {
    /// 创建新的 DeepseekCoder 实例
    /// 返回 Result<Self, AppError>
    pub async fn new() -> Result<Self, AppError> {
        // 从配置文件加载模型配置
        let config = ModelConfig::from_file("config/deepseek_coder.json")?;
        // 初始化模型加载器
        let loader = DeepseekCoderLoader::new(config.clone());
        // 初始化转换器
        let transformer = DeepseekCoderTransformer::new(&config, loader.get_var_builder()?)?;
        // 初始化推理模块
        let inference = DeepSeekCoderInference::new(&config);

        Ok(Self {
            _config: config,
            _loader: loader,
            _transformer: transformer,
            _inference: inference,
        })
    }
    /// 执行推理
    /// 参数:
    ///   - messages: 聊天消息列表
    ///   - temperature: 采样温度
    ///   - top_p: top-p 采样参数
    ///   - n: 生成结果数量
    ///   - max_tokens: 最大token数
    ///   - stream: 是否流式输出
    /// 返回 Result<Vec<ChatCompletionMessage>, AppError>
    pub async fn infer(
        &self,
        messages: Vec<ChatCompletionMessage>,
        params: ChatCompletionParams,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        // 1. 使用tokenizer将输入消息转换为token序列
        let tokenizer = self._loader.get_tokenizer().await?;
        let mut input_ids: Vec<u32> = Vec::with_capacity(1024);
        for message in &messages {
            let encoding = tokenizer
                .encode(message.content.clone(), false)
                .map_err(|e| AppError::TokenizerError(e.to_string()))?;
            input_ids.extend(encoding.get_ids().iter().copied());
        }

        // 2. 将token序列输入transformer模型进行处理
        let input_tensor =
            Tensor::from_slice(&input_ids, &[input_ids.len()], self._transformer.device())?;

        // 使用contiguous()确保内存布局优化
        let input_tensor = input_tensor.contiguous()?;

        // 处理输入序列，添加batch维度
        let input_tensor = input_tensor.unsqueeze(0)?;

        let logits = self._transformer.forward(&input_tensor)?;

        // 移除batch维度
        let mut logits = logits.squeeze(0)?;

        // 3. 根据temperature和top_p参数进行采样
        let next_token = if let Some(temp) = params.temperature {
            let logits = logits.squeeze(0)?;
            let temp_tensor: Tensor =
                Tensor::from_slice(&[temp], &[1], self._transformer.device())?;
            let scaled_logits: Tensor = logits.div(&temp_tensor)?;
            let probs: Tensor = candle_nn::ops::softmax(&scaled_logits, 0)?;
            let mut rng = thread_rng();
            let random_val: f32 = rng.gen();
            let mut cumulative = 0.0;
            let probs_vec: Vec<f32> = probs.to_vec1()?;
            for (i, &p) in probs_vec.iter().enumerate() {
                cumulative += p;
                if cumulative >= random_val {
                    return Ok(vec![ChatCompletionMessage {
                        role: "assistant".to_string(),
                        content: tokenizer.decode(&[i as u32], true)?,
                    }]);
                }
            }

            probs.argmax(0)?.to_scalar::<u32>()?
        } else {
            logits.argmax(1)?.to_scalar::<u32>()?
        };

        // 4. 将生成的token序列转换回文本
        let output_text = tokenizer.decode(&[next_token], true)?;

        // 5. 处理流式输出（如果stream参数为true）
        if params.stream.unwrap_or(false) {
            let mut stream_output = String::new();
            let mut generated_tokens = 0;
            let max_tokens = params.max_tokens.unwrap_or(2048);

            while generated_tokens < max_tokens {
                // 生成下一个token
                let next_token = if let Some(temp) = params.temperature {
                    let logits = logits.squeeze(0)?;
                    let _shape = [1];
                    let temp_tensor =
                        Tensor::from_slice(&[temp], &[1], self._transformer.device())?;
                    let scaled_logits = logits.div(&temp_tensor)?;
                    let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
                    let mut rng = thread_rng();
                    let random_val: f32 = rng.gen();
                    let mut cumulative = 0.0;
                    let probs_vec: Vec<f32> = probs.to_vec1()?;
                    for (i, &p) in probs_vec.iter().enumerate() {
                        cumulative += p;
                        if cumulative >= random_val {
                            return Ok(vec![ChatCompletionMessage {
                                role: "assistant".to_string(),
                                content: tokenizer.decode(&[i as u32], true)?,
                            }]);
                        }
                    }

                    probs.argmax(0)?.to_scalar::<u32>()?
                } else {
                    logits.argmax(1)?.to_scalar::<u32>()?
                };

                // 解码token并添加到输出
                let token_text = tokenizer.decode(&[next_token], true)?;
                stream_output.push_str(&token_text);
                generated_tokens += 1;

                // 发送部分响应
                let _message =
                    ChatCompletionMessage { role: "assistant".to_string(), content: token_text };
                // TODO: 实现实际的流式输出机制

                // 更新输入序列
                input_ids.push(next_token);
                let input_tensor =
                    Tensor::from_slice(&input_ids, (input_ids.len(),), self._transformer.device())?;
                logits = self._transformer.forward(&input_tensor)?;
            }

            return Ok(vec![]);
        }

        // 6. 返回生成的聊天消息列表
        Ok(vec![ChatCompletionMessage { role: "assistant".to_string(), content: output_text }])
    }
}
