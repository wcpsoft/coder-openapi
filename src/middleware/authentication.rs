use actix_web::dev::{Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::Error;
use futures::future::{ok, Ready};
use std::future::Future;
use std::pin::Pin;

/// 身份验证中间件
///
/// # 示例
/// ```rust
/// use actix_web::{web, App, HttpServer};
/// use coder_openapi::middleware::authentication::Authentication;
/// use coder_openapi::routes::route::configure;
///
/// #[actix_web::main]
/// async fn main() -> std::io::Result<()> {
///     std::env::set_var("API_KEY", "test-api-key");
///     
///     let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
///     let port = listener.local_addr()?.port();
///     HttpServer::new(|| {
///         App::new()
///             .wrap(Authentication)
///             .configure(configure)
///     })
///     .listen(listener)?
///     .run()
///     .await
/// }
/// ```
pub struct Authentication;

impl<S, B> Transform<S, ServiceRequest> for Authentication
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Transform = AuthenticationMiddleware<S>;
    type InitError = ();
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ok(AuthenticationMiddleware { service })
    }
}

pub struct AuthenticationMiddleware<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for AuthenticationMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(
        &self,
        ctx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Result<(), Self::Error>> {
        self.service.poll_ready(ctx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        // Extract API key from Authorization header
        let api_key = req
            .headers()
            .get("Authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "));

        // Validate API key
        match (api_key, std::env::var("API_KEY")) {
            (Some(key), Ok(env_key)) if key == env_key => {
                let fut = self.service.call(req);
                Box::pin(async move {
                    let res = fut.await?;
                    Ok(res)
                })
            }
            (None, _) => {
                // Missing API key
                Box::pin(async move { Err(actix_web::error::ErrorUnauthorized("Missing API key")) })
            }
            (_, Err(_)) => {
                // API key not configured
                Box::pin(async move {
                    Err(actix_web::error::ErrorInternalServerError("Server configuration error"))
                })
            }
            _ => {
                // Invalid API key
                Box::pin(async move { Err(actix_web::error::ErrorUnauthorized("Invalid API key")) })
            }
        }
    }
}
