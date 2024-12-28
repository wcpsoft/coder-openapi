use actix_web::ResponseError;
use safetensors::SafeTensorError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MiddlewareError {
    #[error("Authentication failed")]
    #[allow(dead_code)]
    AuthenticationError,
}

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Invalid parameter: {0}")]
    ValidationError(String),
    #[error("Not Found")]
    NotFound,
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Forbidden")]
    Forbidden,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Generic error: {0}")]
    Anyhow(#[from] anyhow::Error),
    #[error("Model not available: {0}")]
    Model(String),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Chat error: {0}")]
    Chat(String),
    #[error("SafeTensor error: {0}")]
    SafeTensor(#[from] SafeTensorError),
    #[error("Model not found: {0}")]
    InvalidModel(String),
    #[error("Config error: {0}")]
    ConfigError(String),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Generic error: {0}")]
    Generic(String),
}

impl AppError {
    pub fn new(message: String) -> Self {
        AppError::Generic(message)
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for AppError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        AppError::Generic(err.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        AppError::ConfigError(err.to_string())
    }
}

#[derive(serde::Serialize)]
pub struct ErrorResponse {
    pub code: u32,
    pub status: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl From<actix_web::Error> for AppError {
    fn from(err: actix_web::Error) -> Self {
        AppError::Generic(err.to_string())
    }
}

impl ResponseError for AppError {
    fn status_code(&self) -> actix_web::http::StatusCode {
        match self {
            AppError::Io(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            AppError::Anyhow(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            AppError::Model(_) => actix_web::http::StatusCode::BAD_REQUEST,
            AppError::Candle(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            AppError::Chat(_) => actix_web::http::StatusCode::BAD_REQUEST,
            AppError::SafeTensor(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            AppError::InvalidModel(_) => actix_web::http::StatusCode::BAD_REQUEST,
            AppError::ConfigError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            AppError::TokenizerError(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            AppError::ValidationError(_) => actix_web::http::StatusCode::BAD_REQUEST,
            AppError::InvalidParameter(_) => actix_web::http::StatusCode::BAD_REQUEST,
            AppError::NotFound => actix_web::http::StatusCode::NOT_FOUND,
            AppError::Unauthorized => actix_web::http::StatusCode::UNAUTHORIZED,
            AppError::Forbidden => actix_web::http::StatusCode::FORBIDDEN,
            AppError::Generic(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_response(&self) -> actix_web::HttpResponse {
        let (code, status) = match self {
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
            AppError::InvalidParameter(_) => (400, "Bad Request"),
            AppError::NotFound => (404, "Not Found"),
            AppError::Unauthorized => (401, "Unauthorized"),
            AppError::Forbidden => (403, "Forbidden"),
            AppError::Generic(_) => (500, "Internal Server Error"),
        };

        let response = ErrorResponse {
            code: code as u32,
            status: status.to_string(),
            message: self.to_string(),
            data: None,
        };

        actix_web::HttpResponse::build(self.status_code()).json(response)
    }
}

impl From<AppError> for std::io::Error {
    fn from(err: AppError) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::Other, err.to_string())
    }
}

impl From<crate::service::models::ModelError> for AppError {
    fn from(err: crate::service::models::ModelError) -> Self {
        AppError::Model(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, AppError>;
