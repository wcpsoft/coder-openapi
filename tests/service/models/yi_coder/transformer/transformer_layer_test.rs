use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use coder_openapi::service::models::yi_coder::transformer::transformer_layer::{
    TransformerLayer, YiCoderTransformer,
};

#[test]
fn test_transformer_layer_forward() {
    // 初始化测试数据
    let device = &Device::cuda_if_available(0).unwrap();
    let vb = VarBuilder::zeros(DType::F64, device);
    let hidden_size = 64;
    let num_attention_heads = 8;
    let intermediate_size = 256;
    let seq_len = 10;
    let batch_size = 2;

    // 创建transformer层
    let layer =
        TransformerLayer::new(hidden_size, num_attention_heads, intermediate_size, 1e-5, vb)
            .unwrap();

    // 创建随机输入张量
    let input = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

    // 执行前向传播
    let output = layer.forward(&input).unwrap();

    // 验证输出形状
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);

    // 验证输出值在合理范围内
    let min = output.min_all().unwrap().to_scalar::<f32>().unwrap();
    let max = output.max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(min >= -10.0 && max <= 10.0, "Output values should be within reasonable range");
}

#[test]
fn test_yi_coder_transformer_forward() {
    // 初始化测试数据
    let device = &Device::cuda_if_available(0).unwrap();
    let vb = VarBuilder::zeros(DType::F64, device);
    let num_layers = 6;
    let hidden_size = 64;
    let num_attention_heads = 8;
    let intermediate_size = 256;
    let seq_len = 10;
    let batch_size = 2;

    // 创建transformer模型
    let transformer = YiCoderTransformer::new(
        num_layers,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        vb,
    )
    .unwrap();

    // 创建随机输入张量
    let input = Tensor::randn(0.0, 1.0, &[batch_size, seq_len, hidden_size], &device).unwrap();

    // 执行前向传播
    let output = transformer.forward(&input).unwrap();

    // 验证输出形状
    assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);

    // 验证输出值在合理范围内
    let min = output.min_all().unwrap().to_scalar::<f32>().unwrap();
    let max = output.max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(min >= -10.0 && max <= 10.0, "Output values should be within reasonable range");
}
