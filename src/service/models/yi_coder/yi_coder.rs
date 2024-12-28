use super::config::ModelConfig;
use super::inference::YiCoderInference;
use super::loader::ModelLoader;
use super::transformer::YiCoderTransformer;
use crate::entities::chat_completion_message::ChatCompletionMessage;
use crate::error::AppError;
use candle_core::{DType, Tensor};
use rand::distributions::{Distribution, WeightedIndex};

fn softmax(tensor: &Tensor, dim: usize) -> Result<Tensor, candle_core::Error> {
    let max = tensor.max_keepdim(dim)?;
    let diff = tensor.broadcast_sub(&max)?;
    let exp = diff.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    exp.broadcast_div(&sum)
}

pub struct YiCoder {
    generation_config: ModelConfig,
    _loader: ModelLoader,
    _transformer: YiCoderTransformer,
    _inference: YiCoderInference,
}

impl YiCoder {
    pub async fn new() -> Result<Self, AppError> {
        let generation_config =
            ModelConfig::from_file("models_cache/01-ai/Yi-Coder-1.5B-Chat/generation_config.json")?;
        let loader = ModelLoader::new("yi_coder", "config/app.yml").await?;
        let transformer = YiCoderTransformer::new(&generation_config, loader.get_var_builder()?);
        let inference = YiCoderInference::new(&generation_config);

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
        temperature: Option<f32>,
        top_p: Option<f32>,
        _n: Option<usize>,
        max_tokens: Option<usize>,
        stream: Option<bool>,
    ) -> Result<Vec<ChatCompletionMessage>, AppError> {
        let _temp = temperature.unwrap_or(self.generation_config.temperature) as f64;
        let _top_p = top_p.unwrap_or(self.generation_config.top_p);
        let _max_tokens = max_tokens.unwrap_or(self.generation_config.max_tokens);

        let tokenizer = self._loader.get_tokenizer().await?;
        let mut input_ids = Vec::new();
        for message in &messages {
            let tokens = tokenizer.encode(message.content.clone(), false)?;
            input_ids.extend(tokens.get_ids().to_vec());
        }

        let input_tensor = Tensor::new(&input_ids[..], &self._transformer.device())?;
        let mut logits = self._transformer.forward(&input_tensor)?;

        let next_token = if let Some(temp) = temperature {
            let logits = logits.squeeze(0)?;
            let scaled_logits = logits
                .to_dtype(DType::F64)?
                .broadcast_div(&Tensor::new(temp, &self._transformer.device())?)?;
            let probs = softmax(&scaled_logits, 0)?.to_dtype(DType::F32)?;

            let probs_vec: Vec<f32> = probs.to_vec1()?;
            let dist = WeightedIndex::new(&probs_vec)
                .map_err(|e| AppError::new(format!("WeightedIndex error: {}", e)))?;
            dist.sample(&mut rand::thread_rng()) as u32
        } else {
            logits.argmax(1)?.to_scalar::<u32>()?
        };

        let output_text = tokenizer.decode(&[next_token], true)?;

        if stream.unwrap_or(false) {
            let mut stream_output = String::new();
            let mut generated_tokens = 0;
            let max_tokens = max_tokens.unwrap_or(self.generation_config.max_tokens);
            let mut input_ids = input_ids;

            while generated_tokens < max_tokens {
                // Generate next token
                let next_token = if let Some(temp) = temperature {
                    let logits = logits.squeeze(0)?;
                    let scaled_logits = logits
                        .to_dtype(DType::F64)?
                        .broadcast_div(&Tensor::new(temp, &self._transformer.device())?)?;
                    let probs = softmax(&scaled_logits, 0)?.to_dtype(DType::F32)?;

                    let probs_vec: Vec<f32> = probs.to_vec1()?;
                    let dist = WeightedIndex::new(&probs_vec)
                        .map_err(|e| AppError::new(format!("WeightedIndex error: {}", e)))?;
                    dist.sample(&mut rand::thread_rng()) as u32
                } else {
                    logits.argmax(1)?.to_scalar::<u32>()?
                };

                // Decode token and add to output
                let token_text = tokenizer.decode(&[next_token], true)?;
                stream_output.push_str(&token_text);
                generated_tokens += 1;

                // Send partial response
                let message =
                    ChatCompletionMessage { role: "assistant".to_string(), content: token_text };
                if let Err(e) = self._inference.send_stream_response(&message) {
                    log::warn!("Failed to send stream response: {}", e);
                    break;
                }

                // Update input sequence
                input_ids.push(next_token);
                let input_tensor = Tensor::from_slice(
                    &input_ids,
                    (input_ids.len(),),
                    &self._transformer.device(),
                )?;
                logits = self._transformer.forward(&input_tensor)?;
            }

            return Ok(vec![]);
        }

        Ok(vec![ChatCompletionMessage { role: "assistant".to_string(), content: output_text }])
    }
}
