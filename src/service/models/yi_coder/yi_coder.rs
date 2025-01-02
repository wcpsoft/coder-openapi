use super::config::ModelConfig;
use super::inference::YiCoderInference;
use super::loader::ModelLoader;
use super::transformer::YiCoderTransformer;
use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use crate::service::chat::chat_completion::ChatCompletionParams;
use candle_core::{DType, IndexOp, Tensor};
use rand::distributions::{Distribution, WeightedIndex};
use rust_i18n::t;
/// softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x)))
fn softmax(tensor: &Tensor, dim: usize) -> Result<Tensor, candle_core::Error> {
    log::debug!("Softmax input tensor shape: {:?}", tensor.shape());

    // Check tensor dimensions
    if tensor.shape().dims().is_empty() {
        return Err(candle_core::Error::msg(AppError::new(
            "Empty tensor provided to softmax".to_string(),
        )));
    }

    // Validate input tensor before conversion
    let values = tensor.to_vec1::<f32>()?;
    if values.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return Err(candle_core::Error::msg(AppError::new(
            "Input tensor contains NaN or infinite values before conversion".to_string(),
        )));
    }

    // Convert to f64 for better numerical stability
    let tensor = tensor.to_dtype(DType::F64)?;
    log::debug!("After dtype conversion tensor shape: {:?}", tensor.shape());

    // Validate input tensor after conversion
    let values = tensor.to_vec1::<f64>()?;
    if values.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return Err(candle_core::Error::msg(AppError::new(
            "Input tensor contains NaN or infinite values".to_string(),
        )));
    }

    // Subtract max for numerical stability
    let max = tensor.max_keepdim(dim)?;
    log::debug!("Max tensor shape: {:?}", max.shape());

    // Safely convert max tensor to scalar
    let max = max.squeeze(0)?; // Remove the extra dimension
    let max_value = if max.dtype() == DType::F64 {
        max.to_scalar::<f64>()?
    } else {
        max.to_dtype(DType::F64)?.to_scalar::<f64>()?
    };
    log::debug!("Max tensor value: {:?}", max_value);

    // Ensure proper broadcasting by reshaping max to match tensor dimensions
    let max = max.broadcast_as(tensor.shape())?;
    log::debug!("Broadcasted max tensor shape: {:?}", max.shape());

    let diff = tensor.sub(&max)?;
    log::debug!("Diff tensor values: {:?}", diff.to_vec1::<f64>()?);
    log::debug!("Diff tensor shape: {:?}", diff.shape());

    // Clip values to prevent overflow in exp calculation
    let diff = diff.clamp(-100.0, 100.0)?;
    log::debug!("Clipped diff tensor values: {:?}", diff.to_vec1::<f64>()?);

    // Compute exp with validation
    let exp = diff.exp()?;
    log::debug!("Exp tensor shape: {:?}", exp.shape());
    log::debug!("Exp tensor values: {:?}", exp.to_vec1::<f64>()?);

    // Compute sum with larger epsilon to prevent division by zero
    let sum = exp.sum_keepdim(dim)?;
    log::debug!("Sum tensor shape: {:?}", sum.shape());

    // Use larger epsilon value (1e-6) for better stability
    let epsilon = Tensor::new(1e-6, tensor.device())?.broadcast_as(sum.shape())?;
    log::debug!("Epsilon tensor shape: {:?}", epsilon.shape());

    let sum = sum.add(&epsilon)?;
    log::debug!("Sum with epsilon tensor shape: {:?}", sum.shape());

    // Compute probabilities
    log::debug!("Exp shape before division: {:?}", exp.shape());
    log::debug!("Sum shape before division: {:?}", sum.shape());

    // Broadcast sum to match exp shape
    let sum = sum.broadcast_as(exp.shape())?;
    log::debug!("Broadcasted sum shape: {:?}", sum.shape());

    let probs = exp.div(&sum)?;
    log::debug!("Probs tensor shape: {:?}", probs.shape());

    // Convert back to f32
    let result = probs.to_dtype(DType::F32)?;
    log::debug!("Final softmax result shape: {:?}", result.shape());

    // Validate probabilities
    let values = result.to_vec1::<f32>()?;
    if values.iter().any(|&x| x.is_nan()) {
        return Err(candle_core::Error::msg(AppError::new(
            "NaN values detected in softmax output".to_string(),
        )));
    }
    if values.iter().any(|&x| x.is_infinite()) {
        return Err(candle_core::Error::msg(AppError::new(
            "Infinite values detected in softmax output".to_string(),
        )));
    }
    if values.iter().any(|&x| x < 0.0) {
        return Err(candle_core::Error::msg(AppError::new(
            "Negative values detected in softmax output".to_string(),
        )));
    }

    // Normalization with proper broadcasting
    let sum = result.sum_keepdim(0)?;
    let sum = sum.broadcast_as(result.shape())?;
    let normalized = result.div(&sum)?;

    // Final validation
    let normalized_values = normalized.to_vec1::<f32>()?;
    if normalized_values.iter().any(|&x| x.is_nan() || x.is_infinite() || x < 0.0) {
        return Err(candle_core::Error::msg(AppError::new(
            "Invalid values detected after normalization".to_string(),
        )));
    }

    Ok(normalized)
}

pub struct YiCoder {
    generation_config: Box<ModelConfig>,
    _loader: ModelLoader,
    _transformer: YiCoderTransformer,
    _inference: YiCoderInference,
}

impl YiCoder {
    pub async fn new() -> Result<Self, AppError> {
        log::debug!("进入Yi-1.5B");
        let loader = ModelLoader::new("yi-coder", "config/app.yml").await?;
        let model_config = loader.get_model_config("yi-coder")?;
        let model_dir = format!("{}/{}", "models_cache", model_config.hf_hub_id);
        let config_path = format!("{}/{}", model_dir, "config.json");
        let generation_config = Box::new(ModelConfig::from_file(config_path)?);
        log::debug!("完成generation_config");
        let loader = ModelLoader::new("yi-coder", "config/app.yml").await?;
        log::debug!("完成loader");
        let transformer = YiCoderTransformer::new(&generation_config, loader.get_var_builder()?);
        log::debug!("完成transformer");
        let inference = YiCoderInference::new(&generation_config);
        log::debug!("完成inference");
        Ok(Self {
            generation_config,
            _loader: loader,
            _transformer: transformer?,
            _inference: inference,
        })
    }

    pub async fn infer(
        &self,
        messages: Vec<ChatCompletionMessage>,
        params: ChatCompletionParams,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        let temp = params.temperature.unwrap_or(self.generation_config.temperature) as f64;
        let top_p = params.top_p.unwrap_or(self.generation_config.top_p);
        let max_tokens = params.max_tokens.unwrap_or(self.generation_config.max_tokens);

        log::debug!("{}", t!("logs.handling_request"));
        log::debug!(
            "Inference parameters - temperature: {}, top_p: {}, max_tokens: {}",
            temp,
            top_p,
            max_tokens
        );
        log::debug!("Input messages count: {}", messages.len());
        let tokenizer = self._loader.get_tokenizer().await?;
        let mut input_ids = Vec::new();
        for message in &messages {
            log::debug!(
                "Processing message - role: {}, content length: {}",
                message.role,
                message.content.len()
            );
            let encoding = tokenizer.encode(message.content.clone(), true)?;
            input_ids.extend(encoding.get_ids().to_vec());
            log::debug!("encoding tokens: {:?}", encoding);
            log::debug!("Encoded tokens count: {}", encoding.len());
        }
        log::debug!("input_ids tokens: {:?}", input_ids);
        log::debug!("Total input tokens: {}", input_ids.len());

        // Create input tensor with batch dimension, keep original BF16 type
        let input_tensor = Tensor::from_slice(
            &input_ids[..],
            (1, input_ids.len()), // Add batch dimension
            self._transformer.device(),
        )?;
        log::debug!(
            "Input tensor shape: {:?}, dtype: {:?}",
            input_tensor.shape(),
            input_tensor.dtype()
        );
        // Remove batch dimension before passing to transformer
        let input_tensor = input_tensor.squeeze(0)?;
        log::debug!("Transformer input tensor shape: {:?}", input_tensor.shape());
        let logits = self._transformer.forward(&input_tensor)?;
        log::debug!("Logits shape: {:?}, dtype: {:?}", logits.shape(), logits.dtype());
        // Add batch dimension back for consistency
        let logits = logits.unsqueeze(0)?;
        log::debug!("Logits with batch dimension: {:?}", logits.shape());

        log::debug!("Generating next token...");
        let mut messages: Result<Vec<ChatCompletionMessage>, AppError> = if let Some(temp) =
            params.temperature
        {
            log::debug!("Before squeeze - logits shape: {:?}", logits.shape());
            // Remove batch dimension
            let logits = logits.squeeze(0)?;
            log::debug!(
                "After first squeeze - logits shape: {:?}, dtype: {:?}",
                logits.shape(),
                logits.dtype()
            );

            // Convert to U32 for index-select operation
            let logits_u32 = logits.to_dtype(DType::U32)?;
            log::debug!(
                "After U32 conversion - logits shape: {:?}, dtype: {:?}",
                logits_u32.shape(),
                logits_u32.dtype()
            );

            // Select last token's logits
            let selected_logits = logits_u32.i((logits_u32.dim(0)? - 1, ..))?;
            log::debug!(
                "After index select - selected_logits shape: {:?}, dtype: {:?}",
                selected_logits.shape(),
                selected_logits.dtype()
            );

            // Convert back to F32 for computation
            let logits_f32 = selected_logits.to_dtype(DType::F32)?;
            log::debug!(
                "After F32 conversion - logits_f32 shape: {:?}, dtype: {:?}",
                logits_f32.shape(),
                logits_f32.dtype()
            );

            // Validate final dtype
            if logits_f32.dtype() != DType::F32 {
                log::error!("Final tensor has incorrect dtype: {:?}", logits_f32.dtype());
                return Err(AppError::new(format!(
                    "Final tensor has incorrect dtype: {:?}, expected F32",
                    logits_f32.dtype()
                )));
            }

            let scaled_logits = logits_f32;
            log::debug!("Scaled logits shape: {:?}", scaled_logits.shape());
            // Validate temperature value
            if temp.is_nan() || temp.is_infinite() || temp <= 0.0 {
                return Err(AppError::new(format!(
                    "Invalid temperature value: {} (must be positive finite number)",
                    temp
                )));
            }

            log::debug!("Creating temperature tensor with value: {}", temp);
            log::debug!("Scaled logits shape before broadcast: {:?}", scaled_logits.shape());

            // Create temperature tensor with proper shape
            let temp_tensor =
                Tensor::new(temp, self._transformer.device())?.to_dtype(DType::F32)?;
            log::debug!("Initial temperature tensor shape: {:?}", temp_tensor.shape());

            // Broadcast to match logits shape
            let temp_tensor = temp_tensor.broadcast_as(scaled_logits.shape())?;
            log::debug!("Broadcast temperature tensor shape: {:?}", temp_tensor.shape());
            log::debug!("Scaled logits shape before division: {:?}", scaled_logits.shape());

            // Explicit reshape to ensure shape compatibility
            let temp_tensor = temp_tensor.reshape(scaled_logits.shape())?;
            log::debug!("Reshaped temperature tensor shape: {:?}", temp_tensor.shape());
            log::debug!("Scaled logits shape before division: {:?}", scaled_logits.shape());
            log::debug!("Temp tensor shape before division: {:?}", temp_tensor.shape());

            let scaled_logits = scaled_logits.div(&temp_tensor)?;
            log::debug!("Scaled logits shape after division: {:?}", scaled_logits.shape());
            log::debug!("After broadcast_div - scaled_logits shape: {:?}", scaled_logits.shape());
            log::debug!("Temp tensor shape after division: {:?}", temp_tensor.shape());
            let probs = softmax(&scaled_logits, 0)?;
            log::debug!("Softmax probs shape: {:?}", probs.shape());

            // Convert to vector with validation
            let probs_vec: Vec<f32> = probs.to_vec1()?;
            log::debug!("Raw probabilities vector: {:?}", probs_vec);

            // If any invalid values remain, use uniform distribution as fallback
            if probs_vec.iter().any(|&x| x.is_nan() || x.is_infinite() || x < 0.0) {
                log::warn!("Invalid probabilities detected, using uniform distribution");
                let uniform_prob = 1.0 / probs_vec.len() as f32;
                let probs_vec = vec![uniform_prob; probs_vec.len()];
                let dist = WeightedIndex::new(&probs_vec)
                    .map_err(|e| AppError::new(format!("WeightedIndex error: {}", e)))?;
                let next_token = dist.sample(&mut rand::thread_rng()) as u32;
                let output_text = tokenizer.decode(&[next_token], true)?;
                return Ok(vec![ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: output_text,
                }]);
            }

            log::debug!("Normalized probabilities: {:?}", probs_vec);
            let dist = WeightedIndex::new(&probs_vec)
                .map_err(|e| AppError::new(format!("WeightedIndex error: {}", e)))?;
            let next_token = dist.sample(&mut rand::thread_rng()) as u32;
            let output_text = tokenizer.decode(&[next_token], true)?;
            Ok(vec![ChatCompletionMessage { role: "assistant".to_string(), content: output_text }])
        } else {
            let next_token = logits.argmax(1)?.to_scalar::<u32>()?;
            let output_text = tokenizer.decode(&[next_token], true)?;
            Ok(vec![ChatCompletionMessage { role: "assistant".to_string(), content: output_text }])
        };

        if params.stream.unwrap_or(false) {
            let _messages = Ok::<Vec<ChatCompletionMessage>, AppError>(vec![]);
            log::debug!("Starting streaming response...");
            let mut stream_output = String::new();
            let mut generated_tokens = 0;
            let max_tokens = params.max_tokens.unwrap_or(self.generation_config.max_tokens);
            let mut input_ids = input_ids;
            log::debug!("Max tokens for streaming: {}", max_tokens);

            while generated_tokens < max_tokens {
                // Generate next token
                let next_token = if let Some(temp) = params.temperature {
                    let logits = logits.squeeze(0)?;
                    log::debug!("Creating temperature tensor with value: {}", temp);
                    let temp_tensor = Tensor::new(temp, self._transformer.device())?
                        .to_dtype(DType::F32)?
                        .broadcast_as(logits.shape())?;
                    log::debug!("Temperature tensor shape: {:?}", temp_tensor.shape());
                    log::debug!("Logits shape before division: {:?}", logits.shape());
                    let scaled_logits = logits.to_dtype(DType::F32)?.div(&temp_tensor)?;
                    log::debug!("Scaled logits shape after division: {:?}", scaled_logits.shape());
                    let probs = softmax(&scaled_logits, 0)?;

                    let probs_vec: Vec<f32> = probs.to_vec1()?;
                    let dist = WeightedIndex::new(&probs_vec)
                        .map_err(|e| AppError::new(format!("WeightedIndex error: {}", e)))?;
                    dist.sample(&mut rand::thread_rng()) as u32
                } else {
                    logits.argmax(1)?.to_scalar::<u32>()?
                };

                // Decode token and add to output
                let token_text = tokenizer.decode(&[next_token], true)?;
                log::debug!("Stream token {}: {}", generated_tokens + 1, token_text);
                stream_output.push_str(&token_text);
                let _output_text = stream_output.clone();
                generated_tokens += 1;
                log::debug!("Total generated tokens: {}", generated_tokens);

                // Send partial response
                let message =
                    ChatCompletionMessage { role: "assistant".to_string(), content: token_text };
                log::debug!("{}", t!("logs.chat_request_received"));
                if let Err(e) = self._inference.send_stream_response(&message) {
                    log::warn!("{} {}", t!("errors.stream_response.failed"), e);
                    break;
                }

                // Update input sequence
                input_ids.push(next_token);
                let input_tensor =
                    Tensor::from_slice(&input_ids, (input_ids.len(),), self._transformer.device())?;
                log::debug!("Streaming input tensor shape: {:?}", input_tensor.shape());
                let logits = self._transformer.forward(&input_tensor)?;
                log::debug!("Streaming logits shape: {:?}", logits.shape());
                // Add batch dimension for consistency
                let logits = logits.unsqueeze(0)?;
                log::debug!("Streaming logits with batch dimension: {:?}", logits.shape());
            }

            messages = Ok::<Vec<ChatCompletionMessage>, AppError>(vec![ChatCompletionMessage {
                role: "assistant".to_string(),
                content: stream_output,
            }]);
        }

        messages
    }
}
