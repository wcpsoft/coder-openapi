pub mod attention;
pub mod config;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod feed_forward;
pub mod transformer_layer;

pub use self::attention::MultiHeadAttention;
pub use self::config::ModelConfig;
pub use self::decoder::YiCoderDecoder;
pub use self::encoder::YiCoderEncoder;
pub use self::error::TransformerError;
pub use self::feed_forward::PositionWiseFeedForward;
pub use self::transformer_layer::{TransformerLayer, YiCoderTransformer};

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};
