use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use coder_openapi::service::models::yi_coder::transformer::attention::MultiHeadAttention;

#[test]
fn test_multi_head_attention_forward() -> Result<()> {
    let device = &Device::cuda_if_available(0)?;
    let vb = VarBuilder::zeros(DType::F32, device);
    let hidden_size = 64;
    let num_heads = 8;
    let seq_len = 10;
    let batch_size = 2;

    let mha = MultiHeadAttention::new(hidden_size, num_heads, vb)?;

    let query = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, hidden_size], device)?;
    let key = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, hidden_size], device)?;
    let value = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, hidden_size], device)?;

    let output = mha.forward(&query, &key, &value)?;

    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);

    let min = output.min_all()?.to_scalar::<f32>()?;
    let max = output.max_all()?.to_scalar::<f32>()?;
    assert!(min >= -10.0 && max <= 10.0);
    Ok(())
}

#[test]
fn test_attention_scores_calculation() -> Result<()> {
    let device = &Device::cuda_if_available(0)?;
    let vb = VarBuilder::zeros(DType::F32, device);
    let hidden_size = 64;
    let num_heads = 8;
    let seq_len = 10;
    let batch_size = 2;

    let mha = MultiHeadAttention::new(hidden_size, num_heads, vb)?;

    let query = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, hidden_size], device)?;
    let key = query.clone();
    let value = query.clone();

    let output = mha.forward(&query, &key, &value)?;

    let output_f32 = output.to_dtype(DType::F32)?;
    let query_f32 = query.to_dtype(DType::F32)?;
    let output_minus_input = output_f32.sub(&query_f32)?;
    let squared = output_minus_input.sqr()?;
    let diff_norm = squared.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    assert!(diff_norm > 0.0);
    assert!(diff_norm < 10.0);
    Ok(())
}

#[test]
fn test_attention_edge_cases() {
    let device = Device::cuda_if_available(0).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &device);
    let hidden_size = 64;
    let num_heads = 8;

    let mha = MultiHeadAttention::new(hidden_size, num_heads, vb).unwrap();

    let empty_input = Tensor::zeros(&[0, 0, hidden_size], DType::F32, &device).unwrap();
    let result = mha.forward(&empty_input, &empty_input, &empty_input);
    assert!(result.is_err());

    let max_seq_len = 4096;
    let input = Tensor::randn(0.0, 1.0, &[1, max_seq_len, hidden_size], &device).unwrap();
    let output = mha.forward(&input, &input, &input).unwrap();
    assert_eq!(output.dims(), &[1, max_seq_len, hidden_size]);
}

#[test]
fn test_attention_error_handling() {
    let device = Device::cuda_if_available(0).unwrap();
    let vb = VarBuilder::zeros(DType::F32, &device);
    let hidden_size = 64;
    let num_heads = 8;

    let mha = MultiHeadAttention::new(hidden_size, num_heads, vb).unwrap();

    let query = Tensor::randn(0.0, 1.0, &[1, 10, hidden_size], &device).unwrap();
    let key = Tensor::randn(0.0, 1.0, &[1, 20, hidden_size], &device).unwrap();
    let value = Tensor::randn(0.0, 1.0, &[1, 10, hidden_size], &device).unwrap();

    let result = mha.forward(&query, &key, &value);
    assert!(result.is_err());

    let invalid_num_heads = 7;
    let new_vb = VarBuilder::zeros(DType::F32, &device);
    let result = MultiHeadAttention::new(hidden_size, invalid_num_heads, new_vb);
    assert!(result.is_err());
}
