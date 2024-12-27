use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Invalid configuration")]
    ConfigError,
}

use anyhow::Error as AnyhowError;

#[derive(Error, Debug)]
pub enum UtilsError {
    #[error("System time error")]
    TimeError(#[from] std::time::SystemTimeError),
    #[error("Anyhow error: {0}")]
    AnyhowError(#[from] AnyhowError),
}
