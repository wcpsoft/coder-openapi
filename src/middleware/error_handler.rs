use actix_web::{
    dev::{Service, ServiceRequest, ServiceResponse, Transform},
    error::ResponseError,
    http::{header::ContentType, StatusCode},
    Error as ActixError, HttpResponse,
};
use serde::Serialize;
use std::future::{Future, Ready};
use std::pin::Pin;

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

#[derive(Debug)]
pub enum AppError {
    NotFound(String),
    BadRequest(String),
    InternalServerError(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::NotFound(msg) => write!(f, "Not Found: {}", msg),
            AppError::BadRequest(msg) => write!(f, "Bad Request: {}", msg),
            AppError::InternalServerError(msg) => write!(f, "Internal Server Error: {}", msg),
        }
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let (status, error) = match self {
            AppError::NotFound(_msg) => (StatusCode::NOT_FOUND, "Not Found"),
            AppError::BadRequest(_msg) => (StatusCode::BAD_REQUEST, "Bad Request"),
            AppError::InternalServerError(_msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error")
            }
        };

        HttpResponse::build(status)
            .content_type(ContentType::json())
            .json(ErrorResponse { error: error.to_string(), message: self.to_string() })
    }
}

pub struct ErrorHandler;

impl<S, B> Transform<S, ServiceRequest> for ErrorHandler
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = ActixError>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = ActixError;
    type Transform = ErrorHandlerMiddleware<S>;
    type InitError = ();
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        std::future::ready(Ok(ErrorHandlerMiddleware { service }))
    }
}

pub struct ErrorHandlerMiddleware<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for ErrorHandlerMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = ActixError>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = ActixError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(
        &self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let fut = self.service.call(req);

        Box::pin(async move {
            match fut.await {
                Ok(res) => Ok(res),
                Err(err) => {
                    let app_error = match err.error_response().status() {
                        StatusCode::NOT_FOUND => AppError::NotFound(err.to_string()),
                        StatusCode::BAD_REQUEST => AppError::BadRequest(err.to_string()),
                        _ => AppError::InternalServerError(err.to_string()),
                    };
                    Err(app_error.into())
                }
            }
        })
    }
}

pub fn error_handler() -> ErrorHandler {
    ErrorHandler
}
