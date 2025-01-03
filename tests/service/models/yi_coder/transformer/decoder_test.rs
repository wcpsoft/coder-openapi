use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use coder_openapi::service::models::yi_coder::transformer::{
    config::ModelConfig, decoder::YiCoderDecoder,
};

#[test]
fn test_decoder_forward() -> Result<()> {
    // 初始化测试数据
    let device = &Device::cuda_if_available(0)?;
    let vb = VarBuilder::zeros(DType::F32, device);
    let config = ModelConfig {
        num_layers: 6,
        hidden_size: 64,
        num_attention_heads: 8,
        intermediate_size: 256,
        layer_norm_eps: 1e-5,
        vocab_size: 32000,
    };
    let seq_len = 10;
    let batch_size = 2;

    // 创建解码器
    let decoder = YiCoderDecoder::new(&config, vb)?;

    // 创建随机输入张量
    let input = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, config.hidden_size], device)?;

    // 执行前向传播
    let output = decoder.forward(&input, None)?;

    // 验证输出形状
    assert_eq!(output.dims(), &[batch_size, seq_len, config.hidden_size]);

    // 验证输出值在合理范围内
    let min = output.min(0)?.to_scalar::<f32>()?;
    let max = output.max(0)?.to_scalar::<f32>()?;
    assert!(min >= -10.0 && max <= 10.0, "Output values should be within reasonable range");
    Ok(())
}

#[test]
fn test_decoder_layer_norm() -> Result<()> {
    // 初始化测试数据
    let device = &Device::cuda_if_available(0)?;
    let vb = VarBuilder::zeros(DType::F32, device);
    let config = ModelConfig {
        num_layers: 6,
        hidden_size: 64,
        num_attention_heads: 8,
        intermediate_size: 256,
        layer_norm_eps: 1e-5,
        vocab_size: 32000,
    };
    let seq_len = 10;
    let batch_size = 2;

    // 创建解码器
    let decoder = YiCoderDecoder::new(&config, vb)?;

    // 创建全1输入张量
    let input = Tensor::ones(&[batch_size, seq_len, config.hidden_size], DType::F32, device)?;

    // 执行前向传播
    let output = decoder.forward(&input, None)?;

    // 验证层归一化效果
    let mean = output.mean_all()?.to_scalar::<f32>()?;
    let mean_tensor = Tensor::from_slice(&[mean], &[1], device)?;
    let squared_diff = output.sub(&mean_tensor)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let std = squared_diff.sqrt();
    assert!(mean.abs() < 1e-5, "Mean should be close to zero after layer norm");
    assert!((std - 1.0).abs() < 1e-5, "Std should be close to 1 after layer norm");
    Ok(())
}
