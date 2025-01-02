use candle_core::Error as CandleError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransformerError {
    #[error("Tensor operation failed: {0}")]
    TensorError(String),

    #[error("Model configuration error: {0}")]
    ConfigError(String),

    #[error("Invalid input dimensions")]
    InvalidInputDimensions,

    #[error("Attention mask error")]
    AttentionMaskError,

    #[error("Layer normalization error")]
    LayerNormError,

    #[error("Linear transformation error")]
    LinearError,

    #[error("Feed forward network error")]
    FeedForwardError,

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<CandleError> for TransformerError {
    fn from(err: CandleError) -> Self {
        TransformerError::TensorError(err.to_string())
    }
}
