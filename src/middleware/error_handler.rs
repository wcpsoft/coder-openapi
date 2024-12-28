use actix_web::{
    dev::{Service, ServiceRequest, ServiceResponse, Transform},
    error::ResponseError,
    http::{header::ContentType, StatusCode},
    Error, HttpResponse,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, future::Future, pin::Pin};
use thiserror::Error;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum AppError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Anyhow error: {0}")]
    Anyhow(String),
    #[error("Model not available: {0}")]
    Model(String),
    #[error("Candle error: {0}")]
    Candle(String),
    #[error("Chat error: {0}")]
    Chat(String),
    #[error("SafeTensor error: {0}")]
    SafeTensor(String),
    #[error("Invalid model: {0}")]
    InvalidModel(String),
    #[error("Configuration error: {0}")]
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
    #[error("Service Unavailable")]
    ServiceUnavailable,
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

#[derive(Serialize, Deserialize, Debug, derive_more::Display)]
#[display("code: {}, status: {}, message: {}", code, status, message)]
pub struct ErrorResponse {
    code: u32,
    status: String,
    message: String,
    data: Option<serde_json::Value>,
}

impl ErrorResponse {
    pub fn from_error(error: &AppError) -> Self {
        let (code, status) = match error {
            AppError::Io(_) => (500, t!("errors.http.internal_server_error").to_string()),
            AppError::Anyhow(_) => (500, t!("errors.http.internal_server_error").to_string()),
            AppError::Model(_) => (400, t!("errors.http.bad_request").to_string()),
            AppError::Candle(_) => (500, t!("errors.http.internal_server_error").to_string()),
            AppError::Chat(_) => (400, t!("errors.http.bad_request").to_string()),
            AppError::SafeTensor(_) => (500, t!("errors.http.internal_server_error").to_string()),
            AppError::InvalidModel(_) => (400, t!("errors.http.bad_request").to_string()),
            AppError::ConfigError(_) => (500, t!("errors.http.internal_server_error").to_string()),
            AppError::TokenizerError(_) => {
                (500, t!("errors.http.internal_server_error").to_string())
            }
            AppError::ValidationError(_) => (400, t!("errors.http.bad_request").to_string()),
            AppError::NotFound => (404, t!("errors.http.not_found").to_string()),
            AppError::Unauthorized => (401, t!("errors.http.unauthorized").to_string()),
            AppError::Forbidden => (403, t!("errors.http.forbidden").to_string()),
            AppError::ServiceUnavailable => {
                (503, t!("errors.http.service_unavailable").to_string())
            }
            AppError::Generic(_) => (500, t!("errors.http.internal_server_error").to_string()),
        };

        let log_message = format!("{}: {:?}", t!("logs.creating_error_response"), error);
        log::debug!("{}", log_message); // 记录错误上下文
        let mut response = ErrorResponse {
            code,
            status: status.to_string(),
            message: error.to_string(),
            data: None,
        };
        log::debug!("Initial error response: {:?}", response); // 记录初始响应

        if let AppError::ValidationError(details) = error {
            response.data = match serde_json::to_value(details) {
                Ok(value) => Some(value),
                Err(err) => {
                    let msg = format!("Serialization failed: {}", err);
                    log::error!("{}", msg);
                    log::debug!("Validation details that failed serialization: {:?}", details); // 记录序列化失败的验证详情
                    None
                }
            };
        }

        response
    }
}

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
        let (req_parts, _payload) = req.parts();
        let req_parts = req_parts.clone();

        match self
            .service
            .poll_ready(&mut core::task::Context::from_waker(futures::task::noop_waker_ref()))
        {
            core::task::Poll::Ready(Ok(())) => {
                log::debug!(
                    "Service ready to handle request: {} {}",
                    req_parts.method(),
                    req_parts.uri()
                ); // 服务准备处理请求
            }
            core::task::Poll::Ready(Err(err)) => {
                let msg = format!(
                    "Service unavailable: {} {} {}",
                    err,
                    req_parts.uri(),
                    req_parts.method()
                );
                log::error!("{}", msg);
                log::debug!("Service unavailable details: {:?}", req_parts); // 记录服务不可用详情
                let response = ErrorResponse::from_error(&AppError::ServiceUnavailable);
                let http_response = HttpResponse::ServiceUnavailable().json(response);
                return Box::pin(async move {
                    Ok(ServiceResponse::new(req_parts, http_response.map_into_boxed_body()))
                });
            }
            core::task::Poll::Pending => {
                let msg = format!("Service not ready: {} {}", req_parts.uri(), req_parts.method());
                log::warn!("{}", msg);
                log::debug!("Service not ready details: {:?}", req_parts); // 记录服务未准备详情
                let response = ErrorResponse::from_error(&AppError::ServiceUnavailable);
                let http_response = HttpResponse::ServiceUnavailable().json(response);
                return Box::pin(async move {
                    Ok(ServiceResponse::new(req_parts, http_response.map_into_boxed_body()))
                });
            }
        }

        // Log request details for debugging
        log::debug!(
            "Handling request: method={}, uri={}, headers={:?}",
            req_parts.method(),
            req_parts.uri(),
            req_parts.headers()
        );

        // Log potential issues (these are warnings, not errors)
        if req_parts.headers().is_empty() {
            log::warn!("Request has no headers");
        }
        if req_parts.uri().path().is_empty() {
            log::warn!("Request path is empty");
        }

        // 继续处理请求
        let fut = self.service.call(req);

        Box::pin(async move {
            match fut.await {
                Ok(res) => Ok(res.map_into_boxed_body()),
                Err(err) => {
                    let app_error: AppError = match err.try_into() {
                        Ok(err) => err,
                        Err(conv_err) => {
                            let msg = format!(
                                "Error conversion failed: {} {} {}",
                                conv_err,
                                req_parts.uri(),
                                req_parts.method()
                            );
                            log::error!("{}", msg);
                            AppError::Generic("Internal server error".to_string())
                        }
                    };

                    let response = ErrorResponse::from_error(&app_error);
                    let status_code = StatusCode::from_u16(response.code as u16)
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                    let http_response = HttpResponse::build(status_code).json(response);
                    Ok(ServiceResponse::new(req_parts, http_response.map_into_boxed_body()))
                }
            }
        })
    }
}

pub fn error_handler() -> ErrorHandlerMiddleware {
    ErrorHandlerMiddleware
}
