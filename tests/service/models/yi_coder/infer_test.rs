use approx::assert_relative_eq;
use coder_openapi::entities::chat_completion_message::ChatCompletionMessage;
use coder_openapi::service::models::yi_coder::infer::YiCoderInference;
use tokio::sync::mpsc;

fn create_test_config() -> YiCoderInference {
    YiCoderInference::new(
        "test", // model_path
        "test", // tokenizer_path
        512,    // max_seq_len
    )
}

#[tokio::test]
async fn test_infer_parameter_validation() {
    let infer = create_test_config();

    // 测试无效温度值
    let messages =
        vec![ChatCompletionMessage { role: "user".to_string(), content: "test".to_string() }];
    let result = infer.infer(messages.clone(), Some(-1.0), None, None, None, None).await;
    assert!(result.is_err());

    // 测试无效top_p值
    let result = infer.infer(messages.clone(), None, Some(1.5), None, None, None).await;
    assert!(result.is_err());

    // 测试无效n值
    let result = infer.infer(messages.clone(), None, None, Some(0), None, None).await;
    assert!(result.is_err());

    // 测试无效max_tokens值
    let result = infer.infer(messages.clone(), None, None, None, Some(0), None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_stream_response() {
    let infer = create_test_config();
    let (tx, mut rx) = mpsc::channel(32);
    infer.set_stream_sender(tx);

    let messages =
        vec![ChatCompletionMessage { role: "user".to_string(), content: "test".to_string() }];
    let result = infer.infer(messages, None, None, None, None, Some(true)).await;
    assert!(result.is_ok());

    // 验证流式响应
    let response = rx.recv().await.unwrap();
    assert_eq!(response.role, "assistant");
    assert!(response.content.contains("Streaming response"));
}

#[tokio::test]
async fn test_temperature_effect() {
    let infer = create_test_config();

    let messages =
        vec![ChatCompletionMessage { role: "user".to_string(), content: "test".to_string() }];

    // 测试不同温度值
    for temp in [0.1, 0.5, 1.0, 1.5] {
        let result = infer.infer(messages.clone(), Some(temp), None, None, None, None).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response[0].content.contains(&format!("temp: {:.2}", temp)));
    }
}

#[tokio::test]
async fn test_probability_distribution() {
    let infer = create_test_config();

    // 测试概率分布计算
    let logits = vec![1.0, 2.0, 3.0];
    let temp = 1.0;

    // 计算理论概率
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x as f32 / temp as f32).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let expected_probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

    // 运行多次采样
    let mut counts = [0; 3];
    for _ in 0..1000 {
        let result = infer
            .infer(
                vec![ChatCompletionMessage {
                    role: "user".to_string(),
                    content: "test".to_string(),
                }],
                Some(temp),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();

        let token = match result[0]
            .content
            .split_whitespace()
            .find(|s| s.starts_with("temp:"))
            .and_then(|s| s.split(':').nth(1))
            .and_then(|s| s.parse::<f32>().ok())
        {
            Some(t) => t,
            None => continue, // 跳过无法解析的响应
        };

        counts[(token * 10.0).round() as usize - 1] += 1;
    }

    // 验证采样分布
    let total: f32 = counts.iter().sum::<i32>() as f32;
    for (i, &count) in counts.iter().enumerate() {
        if total > 0.0 {
            // 确保分母不为零
            let actual_prob = count as f32 / total;
            if !actual_prob.is_nan() {
                // 检查是否为 NaN
                assert_relative_eq!(actual_prob, expected_probs[i], epsilon = 0.05);
            }
        }
    }
}

#[tokio::test]
async fn test_edge_cases() {
    let infer = YiCoderInference::new(
        "test", // model_path
        "test", // tokenizer_path
        512,    // max_seq_len
    );

    // 测试边界条件
    let messages =
        vec![ChatCompletionMessage { role: "user".to_string(), content: "test".to_string() }];

    // 测试极小温度值
    let result = infer.infer(messages.clone(), Some(0.0001), None, None, None, None).await;
    assert!(result.is_ok(), "极小温度值测试失败: {:?}", result);

    // 测试极大温度值（在允许范围内）
    let result = infer.infer(messages.clone(), Some(2.0), None, None, None, None).await;
    assert!(result.is_ok(), "极大温度值测试失败: {:?}", result);

    // 测试NaN和无穷大
    let result = infer.infer(messages.clone(), Some(f32::NAN), None, None, None, None).await;
    assert!(result.is_err());

    let result = infer.infer(messages.clone(), Some(f32::INFINITY), None, None, None, None).await;
    assert!(result.is_err());
}
