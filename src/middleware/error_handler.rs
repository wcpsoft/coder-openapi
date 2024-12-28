use actix_web::{
    dev::{Service, ServiceRequest, ServiceResponse, Transform},
    error::ResponseError,
    http::{header::ContentType, StatusCode},
    Error as ActixError, HttpResponse,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, future::Future, pin::Pin};
use thiserror::Error;

// Remove any duplicate ValidationDetails definitions

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum AppError {
    #[error("IO error: {0}")]
    Io(String),
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
    #[error("Database error: {0}")]
    Database(String),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    #[error("Not Found")]
    NotFound,
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Forbidden")]
    Forbidden,
    #[error("Service Unavailable")]
    ServiceUnavailable,
    #[error("Timeout error: {0}")]
    Timeout(String),
    #[error("Generic error: {0}")]
    Generic(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ValidationDetails {
    pub field: String,
    pub message: String,
}

impl std::fmt::Display for ValidationDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Field: {}, Message: {}", self.field, self.message)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ErrorResponse {
    pub code: u32,
    pub status: String,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

impl AppError {
    fn error_context(&self) -> String {
        match self {
            AppError::Io(msg) => format!("IO Error: {}", msg),
            AppError::Model(msg) => format!("Model Error: {}", msg),
            AppError::Candle(msg) => format!("Candle Error: {}", msg),
            AppError::Chat(msg) => format!("Chat Error: {}", msg),
            AppError::SafeTensor(msg) => format!("SafeTensor Error: {}", msg),
            AppError::InvalidModel(msg) => format!("Invalid Model Error: {}", msg),
            AppError::ConfigError(msg) => format!("Config Error: {}", msg),
            AppError::TokenizerError(msg) => format!("Tokenizer Error: {}", msg),
            AppError::ValidationError(details) => format!("Validation Error: {}", details),
            AppError::Database(msg) => format!("Database Error: {}", msg),
            AppError::Network(msg) => format!("Network Error: {}", msg),
            AppError::RateLimitExceeded => "Rate Limit Exceeded".to_string(),
            AppError::NotFound => "Resource Not Found".to_string(),
            AppError::Unauthorized => "Unauthorized Access".to_string(),
            AppError::Forbidden => "Forbidden Access".to_string(),
            AppError::ServiceUnavailable => "Service Unavailable".to_string(),
            AppError::Timeout(msg) => format!("Timeout Error: {}", msg),
            AppError::Generic(msg) => format!("Generic Error: {}", msg),
        }
    }

    fn status_code(&self) -> (u32, String) {
        match self {
            AppError::Io(_) => (500, t!("errors.http.internal_server_error").to_string()),
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
            AppError::Database(_) => (500, t!("errors.http.internal_server_error").to_string()),
            AppError::Network(_) => (503, t!("errors.http.service_unavailable").to_string()),
            AppError::RateLimitExceeded => (429, t!("errors.http.too_many_requests").to_string()),
            AppError::NotFound => (404, t!("errors.http.not_found").to_string()),
            AppError::Unauthorized => (401, t!("errors.http.unauthorized").to_string()),
            AppError::Forbidden => (403, t!("errors.http.forbidden").to_string()),
            AppError::ServiceUnavailable => {
                (503, t!("errors.http.service_unavailable").to_string())
            }
            AppError::Timeout(_) => (504, t!("errors.http.gateway_timeout").to_string()),
            AppError::Generic(_) => (500, t!("errors.http.internal_server_error").to_string()),
        }
    }
}

impl From<ActixError> for AppError {
    fn from(err: ActixError) -> Self {
        let status = err.as_response_error().status_code();
        let error_str = err.to_string();
        let context = format!("Status: {}, Error: {}", status, error_str);

        log::error!("{}", t!("logs.error_occurred", context = context));

        // Handle known status codes specifically
        let app_error = match status {
            StatusCode::NOT_FOUND => AppError::NotFound,
            StatusCode::UNAUTHORIZED => AppError::Unauthorized,
            StatusCode::FORBIDDEN => AppError::Forbidden,
            StatusCode::BAD_REQUEST => {
                if error_str.contains("route") || error_str.contains("match") {
                    AppError::NotFound
                } else {
                    AppError::ValidationError(ValidationDetails {
                        field: "request".to_string(),
                        message: format!("{} (context: {})", error_str, context),
                    })
                }
            }
            StatusCode::TOO_MANY_REQUESTS => AppError::RateLimitExceeded,
            StatusCode::GATEWAY_TIMEOUT => AppError::Timeout(error_str),
            _ => {
                // Log details about the unmatched error
                log::debug!("Unmatched error occurred. Status: {}, Error: {}", status, error_str);
                // Handle all other errors generically with detailed context
                AppError::Generic(format!(
                    "Unexpected error occurred: {} (context: {})",
                    error_str, context
                ))
            }
        };

        // Log the final error type being returned
        log::debug!("Converted error to: {:?}", app_error);
        app_error
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let response = ErrorResponse::from(self);
        let status_code = match StatusCode::from_u16(response.code as u16) {
            Ok(code) => code,
            Err(_) => {
                log::error!("{}", t!("logs.invalid_status_code", code = response.code));
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        HttpResponse::build(status_code).content_type(ContentType::json()).json(response)
    }
}

impl From<&AppError> for ErrorResponse {
    fn from(error: &AppError) -> Self {
        let (code, status) = error.status_code();
        let error_context = error.error_context();

        let mut response = ErrorResponse {
            code,
            status: status.to_string(),
            message: format!("{} (context: {})", error.to_string(), error_context),
            data: None,
        };

        if let AppError::ValidationError(details) = error {
            response.data = serde_json::to_value(details)
                .map_err(|err| {
                    log::error!("{}", t!("logs.serialization_failed", msg = err.to_string()));
                })
                .ok()
                .or_else(|| {
                    log::error!("{}", t!("logs.json_creation_failed"));
                    Some(serde_json::json!({
                        "field": details.field.clone(),
                        "message": details.message.clone()
                    }))
                });
        }

        log::debug!("Final error response: {:?}", response);
        response
    }
}

pub struct ErrorHandlerMiddleware;

impl<S, B> Transform<S, ServiceRequest> for ErrorHandlerMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = ActixError> + 'static,
    S::Future: 'static,
    B: actix_web::body::MessageBody + 'static,
    B::Error: Into<ActixError> + ResponseError + 'static,
{
    type Response = ServiceResponse<actix_web::body::BoxBody>;
    type Error = ActixError;
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
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = ActixError>,
    S::Future: 'static,
    B: actix_web::body::MessageBody + 'static,
    B::Error: Into<ActixError>,
{
    type Response = ServiceResponse<actix_web::body::BoxBody>;
    type Error = ActixError;
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

        log::debug!(
            "Handling request: method={}, uri={}, headers={:?}",
            req_parts.method(),
            req_parts.uri(),
            req_parts.headers()
        );

        let fut = self.service.call(req);

        Box::pin(async move {
            match fut.await {
                Ok(res) => {
                    log::debug!(
                        "Successfully processed request: {} {}",
                        req_parts.method(),
                        req_parts.uri()
                    );
                    Ok(res.map_into_boxed_body())
                }
                Err(err) => {
                    log::error!("Request failed: {} {}", req_parts.method(), req_parts.uri());
                    log::debug!("Error details: {:?}", err);

                    let app_error: AppError = match err.try_into() {
                        Ok(err) => err,
                        Err(conv_err) => {
                            log::error!("Error conversion failed: {:?}", conv_err);
                            AppError::Generic(format!("Internal server error: {:?}", conv_err))
                        }
                    };

                    Ok(ServiceResponse::new(
                        req_parts,
                        app_error.error_response().map_into_boxed_body(),
                    ))
                }
            }
        })
    }
}

pub fn error_handler() -> ErrorHandlerMiddleware {
    ErrorHandlerMiddleware
}
