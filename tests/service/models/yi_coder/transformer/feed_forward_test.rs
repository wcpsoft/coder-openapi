use super::gelu;
use candle_core::{Device, Tensor};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::cuda_if_available(0)?;

        // 测试正数输入
        let input = Tensor::new(&[1.0f32, 2.0f32, 3.0f32], &device)?;
        let output = gelu(&input)?;
        let expected = Tensor::new(&[0.8413f32, 1.9546f32, 2.9960f32], &device)?;
        assert!(output
            .to_vec1::<f32>()?
            .iter()
            .zip(expected.to_vec1::<f32>()?)
            .all(|(a, b)| (a - b).abs() < 1e-3));

        // 测试负数输入
        let input = Tensor::new(&[-1.0f32, -2.0f32, -3.0f32], &device)?;
        let output = gelu(&input)?;
        let expected = Tensor::new(&[-0.1587f32, -0.0454f32, -0.0040f32], &device)?;
        assert!(output
            .to_vec1::<f32>()?
            .iter()
            .zip(expected.to_vec1::<f32>()?)
            .all(|(a, b)| (a - b).abs() < 1e-3));

        // 测试零输入
        let input = Tensor::new(&[0.0f32], &device)?;
        let output = gelu(&input)?;
        let expected = Tensor::new(&[0.0f32], &device)?;
        assert!(output
            .to_vec1::<f32>()?
            .iter()
            .zip(expected.to_vec1::<f32>()?)
            .all(|(a, b)| (a - b).abs() < 1e-3));

        Ok(())
    }

    #[test]
    fn test_gelu_dtype() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::cuda_if_available(0)?;

        // 测试F32输入
        let input = Tensor::new(&[1.0f32, 2.0f32, 3.0f32], &device)?;
        let output = gelu(&input)?;
        assert_eq!(output.dtype(), candle_core::DType::F32);

        // 测试F64输入
        let input = Tensor::new(&[1.0f64, 2.0f64, 3.0f64], &device)?;
        let output = gelu(&input)?;
        assert_eq!(output.dtype(), candle_core::DType::F64);

        Ok(())
    }
}
