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
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Chat error: {0}")]
    Chat(String),
    #[error("SafeTensor error: {0}")]
    SafeTensor(#[from] SafeTensorError),
}

impl ResponseError for AppError {}

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
