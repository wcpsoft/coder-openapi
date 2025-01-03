use super::*;
use crate::service::chat::chat_completion::ChatCompletionParams;
use approx::assert_relative_eq;
use candle_core::{Device, Tensor};

#[test]
fn test_sample_tokens_temperature_zero() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0, 2.0, 3.0], &device).unwrap();
    let params = ChatCompletionParams { temperature: Some(0.0), ..Default::default() };

    let infer = DeepSeekCoderInference {
        device,
        var_builder: VarBuilder::from_tensors(
            std::collections::HashMap::new(),
            candle_core::DType::F32,
            &device,
        ),
        tokenizer: Tokenizer::from_pretrained("gpt2", None).unwrap(),
        transformer: None,
        config: Default::default(),
    };

    let result = infer.sample_tokens(&logits, &params).unwrap();
    assert_eq!(result, vec![2]); // 温度=0时总是选择最大logit
}

#[test]
fn test_sample_tokens_temperature_one() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0, 2.0, 3.0], &device).unwrap();
    let params = ChatCompletionParams { temperature: Some(1.0), ..Default::default() };

    let infer = DeepSeekCoderInference {
        device,
        var_builder: VarBuilder::from_tensors(
            std::collections::HashMap::new(),
            candle_core::DType::F32,
            &device,
        ),
        tokenizer: Tokenizer::from_pretrained("gpt2", None).unwrap(),
        transformer: None,
        config: Default::default(),
    };

    // 运行多次以确保概率分布正确
    let mut counts = [0; 3];
    for _ in 0..1000 {
        let result = infer.sample_tokens(&logits, &params).unwrap();
        counts[result[0] as usize] += 1;
    }

    // 验证概率分布与理论值接近
    let total: i32 = counts.iter().sum();
    let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
    let expected_probs = vec![0.09, 0.24, 0.67]; // softmax([1,2,3])

    for (actual, expected) in probs.iter().zip(expected_probs) {
        assert_relative_eq!(actual, expected, epsilon = 0.05);
    }
}

#[test]
fn test_sample_tokens_invalid_temperature() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0, 2.0, 3.0], &device).unwrap();
    let params = ChatCompletionParams {
        temperature: Some(-1.0), // 无效温度
        ..Default::default()
    };

    let infer = DeepSeekCoderInference {
        device,
        var_builder: VarBuilder::from_tensors(
            std::collections::HashMap::new(),
            candle_core::DType::F32,
            &device,
        ),
        tokenizer: Tokenizer::from_pretrained("gpt2", None).unwrap(),
        transformer: None,
        config: Default::default(),
    };

    let result = infer.sample_tokens(&logits, &params);
    assert!(result.is_err());
}
