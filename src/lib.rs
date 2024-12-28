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
#[macro_use]
extern crate rust_i18n;

// Initialize i18n with locales directory and fallback to English
i18n!("locales", fallback = "en");

// Re-export i18n functions
pub use rust_i18n::{i18n, set_locale, t};
extern crate log;
pub mod controller;
pub mod entities;
pub mod error;
pub mod middleware;
pub mod route;
pub mod routes;
pub mod service;
pub mod utils {
    pub mod config;
    pub mod download;
    pub mod init;
}

pub use controller::{chat, models};
pub use entities::*;
pub use error::*;
pub use routes::*;
pub use utils::*;
