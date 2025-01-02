pub mod infer;
pub mod loader;
pub mod transformer;

use crate::error::AppError;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder};
use std::fmt;

pub use self::loader::ModelLoader;
pub use self::transformer::{TransformerError, YiCoderTransformer};
