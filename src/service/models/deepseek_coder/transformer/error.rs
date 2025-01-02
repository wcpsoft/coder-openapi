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
