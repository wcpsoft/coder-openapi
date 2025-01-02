use std::fmt;

#[derive(Debug)]
pub enum TransformerError {
    InvalidInput(String),
    NumericalInstability(String),
    ShapeMismatch(String),
    InvalidTensorValues(String),
    LayerError(String),
}

impl fmt::Display for TransformerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransformerError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            TransformerError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
            TransformerError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            TransformerError::InvalidTensorValues(msg) => {
                write!(f, "Invalid tensor values: {}", msg)
            }
            TransformerError::LayerError(msg) => write!(f, "Layer error: {}", msg),
        }
    }
}

impl std::error::Error for TransformerError {}

impl From<TransformerError> for candle_core::Error {
    fn from(err: TransformerError) -> Self {
        candle_core::Error::Msg(err.to_string())
    }
}
