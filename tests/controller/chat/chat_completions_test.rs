use actix_web::{test, web, App};
use coder_openapi::controller::chat::chat_completions;
use coder_openapi::controller::models::list_models;
use coder_openapi::routes::config;
use serde_json::json;

#[actix_web::test]
async fn test_chat_completions() {
    let app = test::init_service(App::new().configure(config)).await;

    // Test basic chat completion
    let req = test::TestRequest::post()
        .uri("/api/v1/chat/completions")
        .set_json(&json!({
            "model": "yi-coder",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        }))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    // Test C language program request
    let req = test::TestRequest::post()
        .uri("/api/v1/chat/completions")
        .set_json(&json!({
            "model": "yi-coder",
            "messages": [{
                "role": "user",
                "content": "编写一个1+1=2的C语言程序"
            }]
        }))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    // Verify response structure
    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body.get("id").is_some());
    assert!(body.get("object").is_some());
    assert!(body.get("created").is_some());
    assert!(body.get("model").is_some());
    assert!(body.get("choices").is_some());
    assert!(body.get("usage").is_some());
}

#[actix_web::test]
async fn test_invalid_model() {
    let app = test::init_service(App::new().configure(config)).await;

    let req = test::TestRequest::post()
        .uri("/api/v1/chat/completions")
        .set_json(&json!({
            "model": "invalid-model",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        }))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 400);
}

#[actix_web::test]
async fn test_empty_messages() {
    let app = test::init_service(App::new().configure(config)).await;

    let req = test::TestRequest::post()
        .uri("/api/v1/chat/completions")
        .set_json(&json!({
            "model": "yi-coder",
            "messages": []
        }))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status().as_u16(), 400);
}

#[actix_web::test]
async fn test_list_models() {
    let app = test::init_service(App::new().configure(config)).await;

    let req = test::TestRequest::get().uri("/api/v1/models").to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}
