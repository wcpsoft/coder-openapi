//! Coder OpenAPI
//!
//! 本crate提供了一个与OpenAI兼容的AI模型API服务。
//!
//! # 模块
//! - `controller`: 处理HTTP请求和路由
//! - `entities`: 定义核心数据结构和模型
//! - `error`: 提供错误处理和自定义错误类型
//! - `routes`: 定义API端点和路由
//! - `service`: 实现业务逻辑和服务
//! - `utils`: 包含实用函数和辅助工具
//!
//! # 示例
//! ```rust
//! use coder_openapi::controller::chat::chat_completions;
//! use actix_web::{web, App, HttpServer};
//!
//! #[actix_web::main]
//! async fn main() -> std::io::Result<()> {
//!     HttpServer::new(|| {
//!         App::new()
//!             .service(chat_completions)
//!     })
//!     .bind("127.0.0.1:8080")?
//!     .run()
//!     .await
//! }
//! ```

pub mod controller;
pub mod entities;
pub mod error;
pub mod middleware;
pub mod route;
pub mod routes;
pub mod service;
pub mod utils;

pub use controller::{chat, models};
pub use entities::*;
pub use error::*;
pub use routes::*;
pub use utils::*;
