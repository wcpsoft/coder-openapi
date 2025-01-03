use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use coder_openapi::service::models::yi_coder::transformer::{
    config::ModelConfig, encoder::YiCoderEncoder,
};

/// 测试上下文，封装测试所需的共享资源
struct TestContext<'a> {
    device: Device,
    vb: VarBuilder<'a>,
    config: ModelConfig,
    seq_len: usize,
    batch_size: usize,
}

impl<'a> TestContext<'a> {
    /// 创建新的测试上下文
    fn new() -> Self {
        let device = Device::cuda_if_available(0).unwrap();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = ModelConfig {
            num_layers: 6,
            hidden_size: 64,
            num_attention_heads: 8,
            intermediate_size: 256,
            layer_norm_eps: 1e-5,
            vocab_size: 32000,
        };

        Self { device, vb, config, seq_len: 10, batch_size: 2 }
    }

    /// 创建编码器实例
    fn create_encoder(&self) -> YiCoderEncoder {
        YiCoderEncoder::new(&self.config, self.vb.clone()).unwrap()
    }

    /// 创建随机输入张量
    fn create_random_input(&self) -> Tensor {
        Tensor::randn(
            0.0,
            1.0,
            &[self.batch_size, self.seq_len, self.config.hidden_size],
            &self.device,
        )
        .unwrap()
    }

    /// 创建全1输入张量
    fn create_ones_input(&self) -> Tensor {
        Tensor::ones(
            &[self.batch_size, self.seq_len, self.config.hidden_size],
            DType::F32,
            &self.device,
        )
        .unwrap()
    }
}

#[test]
fn test_encoder_forward() {
    let ctx = TestContext::new();
    let encoder = ctx.create_encoder();
    let input = ctx.create_random_input();

    // 执行前向传播
    let output = encoder.forward(&input, None).unwrap();

    // 验证输出形状
    assert_eq!(output.dims(), &[ctx.batch_size, ctx.seq_len, ctx.config.hidden_size]);

    // 验证输出值在合理范围内
    let min = output.min_all().unwrap().to_scalar::<f32>().unwrap();
    let max = output.max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(min >= -10.0 && max <= 10.0, "Output values should be within reasonable range");
}

#[test]
fn test_encoder_layer_norm() {
    let ctx = TestContext::new();
    let encoder = ctx.create_encoder();
    let input = ctx.create_ones_input();

    // 执行前向传播
    let output = encoder.forward(&input, None).unwrap();

    // 验证层归一化效果
    let mean = output.mean_all().unwrap().to_scalar::<f32>().unwrap();
    let var = output.var(0).unwrap().to_scalar::<f32>().unwrap();
    let std = var.sqrt();
    assert!(mean.abs() < 1e-5, "Mean should be close to zero after layer norm");
    assert!((std - 1.0).abs() < 1e-5, "Std should be close to 1 after layer norm");
}

#[test]
fn test_encoder_empty_input() {
    let ctx = TestContext::new();
    let encoder = ctx.create_encoder();

    // 创建空输入张量
    let input = Tensor::zeros(&[0, 0, ctx.config.hidden_size], DType::F32, &ctx.device).unwrap();

    // 执行前向传播
    let output = encoder.forward(&input, None);

    // 验证空输入处理
    assert!(output.is_err(), "Encoder should return error for empty input");
}

#[test]
fn test_encoder_invalid_shape() {
    let ctx = TestContext::new();
    let encoder = ctx.create_encoder();

    // 创建形状不匹配的输入张量
    let input = Tensor::randn(
        0.0,
        1.0,
        &[ctx.batch_size, ctx.seq_len, ctx.config.hidden_size + 1], // 不匹配的hidden_size
        &ctx.device,
    )
    .unwrap();

    // 执行前向传播
    let output = encoder.forward(&input, None);

    // 验证形状不匹配处理
    assert!(output.is_err(), "Encoder should return error for invalid input shape");
}

#[test]
fn test_encoder_attention_mask() {
    let ctx = TestContext::new();
    let encoder = ctx.create_encoder();
    let input = ctx.create_random_input();

    // 创建attention mask
    let mask = Tensor::ones(&[ctx.batch_size, ctx.seq_len], DType::F32, &ctx.device).unwrap();

    // 执行带mask的前向传播
    let output = encoder.forward(&input, Some(&mask)).unwrap();

    // 验证输出形状
    assert_eq!(output.dims(), &[ctx.batch_size, ctx.seq_len, ctx.config.hidden_size]);
}
