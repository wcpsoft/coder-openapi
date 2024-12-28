use actix_web::{
    dev::{Service, ServiceRequest, ServiceResponse, Transform},
    error::ResponseError,
    http::{header::ContentType, StatusCode},
    Error, HttpResponse,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, future::Future, pin::Pin};
use thiserror::Error;

// Centralized error handling middleware for the application
//
// This module provides:
// - Custom error types using `thiserror`
// - Error response formatting
// - Middleware for handling errors in Actix-web services

// Application error types
#[derive(Debug, Error, Serialize, Deserialize)]
pub enum AppError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Anyhow error: {0}")]
    Anyhow(String),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Candle error: {0}")]
    Candle(String),
    #[error("Chat error: {0}")]
    Chat(String),
    #[error("SafeTensor error: {0}")]
    SafeTensor(String),
    #[error("Invalid model: {0}")]
    InvalidModel(String),
    #[error("Config error: {0}")]
    ConfigError(String),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("Validation error: {0}")]
    ValidationError(ValidationDetails),
    #[error("Not Found")]
    NotFound,
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Forbidden")]
    Forbidden,
    #[error("Generic error: {0}")]
    Generic(String),
}

impl From<actix_web::Error> for AppError {
    fn from(err: actix_web::Error) -> Self {
        match err.as_response_error().status_code() {
            StatusCode::NOT_FOUND => AppError::NotFound,
            StatusCode::UNAUTHORIZED => AppError::Unauthorized,
            StatusCode::FORBIDDEN => AppError::Forbidden,
            StatusCode::BAD_REQUEST => AppError::ValidationError(ValidationDetails {
                field: "request".to_string(),
                message: err.to_string(),
            }),
            _ => AppError::Generic(err.to_string()),
        }
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let response = ErrorResponse::from_error(self);
        HttpResponse::build(
            StatusCode::from_u16(response.code as u16).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        )
        .content_type(ContentType::json())
        .json(response)
    }
}

// Validation error details
#[derive(Serialize, Deserialize, Debug)]
pub struct ValidationDetails {
    field: String,
    message: String,
}

impl std::fmt::Display for ValidationDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Field: {}, Message: {}", self.field, self.message)
    }
}

// Standardized error response format
#[derive(Serialize, Deserialize, Debug)]
pub struct ErrorResponse {
    code: u32,
    status: String,
    message: String,
    data: Option<serde_json::Value>,
}

impl ErrorResponse {
    // Creates an ErrorResponse from an AppError
    pub fn from_error(error: &AppError) -> Self {
        let (code, status) = match error {
            AppError::Io(_) => (500, "Internal Server Error"),
            AppError::Anyhow(_) => (500, "Internal Server Error"),
            AppError::Model(_) => (400, "Bad Request"),
            AppError::Candle(_) => (500, "Internal Server Error"),
            AppError::Chat(_) => (400, "Bad Request"),
            AppError::SafeTensor(_) => (500, "Internal Server Error"),
            AppError::InvalidModel(_) => (400, "Bad Request"),
            AppError::ConfigError(_) => (500, "Internal Server Error"),
            AppError::TokenizerError(_) => (500, "Internal Server Error"),
            AppError::ValidationError(_) => (400, "Bad Request"),
            AppError::NotFound => (404, "Not Found"),
            AppError::Unauthorized => (401, "Unauthorized"),
            AppError::Forbidden => (403, "Forbidden"),
            AppError::Generic(_) => (500, "Internal Server Error"),
        };

        let mut response = ErrorResponse {
            code,
            status: status.to_string(),
            message: error.to_string(),
            data: None,
        };

        if let AppError::ValidationError(details) = error {
            response.data = Some(serde_json::to_value(details).unwrap_or_default());
        }

        response
    }
}

// Error handling middleware implementation
pub struct ErrorHandlerMiddleware;

impl<S, B> Transform<S, ServiceRequest> for ErrorHandlerMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: actix_web::body::MessageBody + 'static,
    B::Error: Into<Error> + ResponseError + 'static,
{
    type Response = ServiceResponse<actix_web::body::BoxBody>;
    type Error = Error;
    type Transform = ErrorHandlerService<S>;
    type InitError = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Transform, Self::InitError>>>>;

    fn new_transform(&self, service: S) -> Self::Future {
        Box::pin(async move { Ok(ErrorHandlerService { service }) })
    }
}

// Inner service implementation for error handling
pub struct ErrorHandlerService<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for ErrorHandlerService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: actix_web::body::MessageBody + 'static,
    B::Error: Into<Error>,
{
    type Response = ServiceResponse<actix_web::body::BoxBody>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(
        &self,
        ctx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Result<(), Self::Error>> {
        self.service.poll_ready(ctx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let req_parts = req.parts().0.clone();
        let fut = self.service.call(req);

        Box::pin(async move {
            match fut.await {
                Ok(res) => Ok(res.map_into_boxed_body()),
                Err(err) => {
                    let app_error: AppError = err.into();
                    let response = ErrorResponse::from_error(&app_error);

                    log::error!(
                        "Error occurred - URI: {}, Method: {}, Error: {:?}",
                        req_parts.uri(),
                        req_parts.method(),
                        app_error
                    );

                    let status_code = StatusCode::from_u16(response.code as u16)
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                    let http_response = HttpResponse::build(status_code).json(response);
                    Ok(ServiceResponse::new(req_parts, http_response.map_into_boxed_body()))
                }
            }
        })
    }
}

// Creates a new error handler middleware instance
pub fn error_handler() -> ErrorHandlerMiddleware {
    ErrorHandlerMiddleware
}
