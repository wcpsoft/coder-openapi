use crate::error::AppError;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder};
use std::fmt;

pub mod attention;
pub mod config;
pub use crate::service::models::deepseek_coder::transformer::config::ModelConfig;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod feed_forward;
pub mod transformer_layer;

use self::attention::MultiHeadAttention;
use self::decoder::DeepSeekCoderDecoder;
use self::encoder::DeepSeekCoderEncoder;
use self::error::TransformerError;
use self::feed_forward::PositionWiseFeedForward;
use self::transformer_layer::TransformerLayer;
