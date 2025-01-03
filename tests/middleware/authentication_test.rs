use actix_web::{http::header, test, web, App, Error};
use coder_openapi::middleware::Authentication;
use coder_openapi::routes::config;
use std::env;

#[actix_web::test]
async fn test_authentication_valid_key() -> Result<(), Error> {
    env::set_var("API_KEY", "test-key");

    let app = test::init_service(App::new().wrap(Authentication).configure(config)).await;

    let req = test::TestRequest::get()
        .uri("/api/v1/models")
        .insert_header((header::AUTHORIZATION, "Bearer test-key"))
        .to_request();

    let resp = test::call_service(&app, req).await;
    if !resp.status().is_success() {
        let body = test::read_body(resp).await;
        eprintln!("Test failed with response body: {:?}", body);
    }
    assert!(resp.status().is_success());
    Ok(())
}

#[actix_web::test]
async fn test_authentication_missing_key() -> Result<(), Error> {
    env::set_var("API_KEY", "test-key");

    let app = test::init_service(App::new().wrap(Authentication).configure(config)).await;

    let req = test::TestRequest::get().uri("/api/v1/models").to_request();

    let resp = test::call_service(&app, req).await;
    if resp.status() != 401 {
        let body = test::read_body(resp).await;
        eprintln!("Unexpected response for missing key: {:?}", body);
    }
    assert_eq!(resp.status().as_u16(), 401);
    Ok(())
}

#[actix_web::test]
async fn test_authentication_invalid_key() -> Result<(), Error> {
    env::set_var("API_KEY", "test-key");

    let app = test::init_service(App::new().wrap(Authentication).configure(config)).await;

    let req = test::TestRequest::get()
        .uri("/api/v1/models")
        .insert_header((header::AUTHORIZATION, "Bearer wrong-key"))
        .to_request();

    let resp = test::call_service(&app, req).await;
    if resp.status() != 401 {
        let body = test::read_body(resp).await;
        eprintln!("Unexpected response for invalid key: {:?}", body);
    }
    assert_eq!(resp.status().as_u16(), 401);
    Ok(())
}

#[actix_web::test]
async fn test_authentication_missing_env_key() -> Result<(), Error> {
    env::remove_var("API_KEY");

    let app = test::init_service(App::new().wrap(Authentication).configure(config)).await;

    let req = test::TestRequest::get()
        .uri("/api/v1/models")
        .insert_header((header::AUTHORIZATION, "Bearer test-key"))
        .to_request();

    let resp = test::call_service(&app, req).await;
    if resp.status() != 500 {
        let body = test::read_body(resp).await;
        eprintln!("Unexpected response for missing env key: {:?}", body);
    }
    assert_eq!(resp.status().as_u16(), 500);
    Ok(())
}
