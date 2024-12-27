use actix_web::{test, web, App};
use coder_openapi::controller::chat::chat_completions;
use coder_openapi::controller::models::list_models;
use coder_openapi::routes::config;

#[actix_web::test]
async fn test_chat_completions() {
    let app = test::init_service(App::new().configure(config)).await;

    let req = test::TestRequest::post()
        .uri("/api/v1/chat/completions")
        .set_json(&serde_json::json!({
            "model": "yi-coder",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        }))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}

#[actix_web::test]
async fn test_list_models() {
    let app = test::init_service(App::new().configure(config)).await;

    let req = test::TestRequest::get().uri("/api/v1/models").to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}
