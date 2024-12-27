//! Yi-Coder OpenAPI
//!
//! This crate provides an OpenAI-compatible API service for Yi-Coder models.
//!
//! # Modules
//! - `controller`: Handles HTTP requests and routes
//! - `entities`: Defines core data structures and models
//! - `error`: Provides error handling and custom error types
//! - `routes`: Defines API endpoints and routing
//! - `service`: Implements business logic and services
//! - `utils`: Contains utility functions and helpers
//!
//! # Examples
//! ```rust
//! use yi_coder_openapi::controller::chat::chat_completions;
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
//! */

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
