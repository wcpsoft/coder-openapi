use crate::locales::{LocaleError, Locales};
use actix_web::{
    dev::{Service, ServiceRequest, ServiceResponse, Transform},
    error::ResponseError,
    http::{header::ContentType, StatusCode},
    Error, HttpResponse,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, future::Future, pin::Pin};
use thiserror::Error;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum AppError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Anyhow error: {0}")]
    Anyhow(String),
    #[error("Model error: {0}")]
    Model(String),
    #[error("Candle error: {0}")]
    Candle(String),
    #[error("Chat error: {0}")]
    Chat(String),
    #[error("SafeTensor error: {0}")]
    SafeTensor(String),
    #[error("Invalid model: {0}")]
    InvalidModel(String),
    #[error("Config error: {0}")]
    ConfigError(String),
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("Validation error: {0}")]
    ValidationError(ValidationDetails),
    #[error("Not found")]
    NotFound,
    #[error("Unauthorized")]
    Unauthorized,
    #[error("Forbidden")]
    Forbidden,
    #[error("Service unavailable")]
    ServiceUnavailable,
    #[error("Generic error: {0}")]
    Generic(String),
    #[error("Locale error: {0}")]
    Locale(#[from] LocaleError),
}

impl From<actix_web::Error> for AppError {
    fn from(err: actix_web::Error) -> Self {
        match err.as_response_error().status_code() {
            StatusCode::NOT_FOUND => AppError::NotFound,
            StatusCode::UNAUTHORIZED => AppError::Unauthorized,
            StatusCode::FORBIDDEN => AppError::Forbidden,
            StatusCode::BAD_REQUEST => AppError::ValidationError(ValidationDetails {
                field: "request".to_string(),
                message: err.to_string(),
            }),
            _ => AppError::Generic(err.to_string()),
        }
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let response = ErrorResponse::from_error(self);
        HttpResponse::build(
            StatusCode::from_u16(response.code as u16).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        )
        .content_type(ContentType::json())
        .json(response)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ValidationDetails {
    field: String,
    message: String,
}

impl std::fmt::Display for ValidationDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Field: {}, Message: {}", self.field, self.message)
    }
}

#[derive(Serialize, Deserialize, Debug, derive_more::Display)]
#[display("code: {}, status: {}, message: {}", code, status, message)]
pub struct ErrorResponse {
    code: u32,
    status: String,
    message: String,
    data: Option<serde_json::Value>,
}

impl ErrorResponse {
    pub fn from_error(error: &AppError) -> Self {
        let locales = match Locales::new("locales") {
            Ok(locales) => locales,
            Err(err) => {
                log::error!("Failed to load locales: {}", err);
                return Self {
                    code: 500,
                    status: "Internal Server Error".to_string(),
                    message: "Failed to load locales".to_string(),
                    data: None,
                };
            }
        };

        let (code, status) = match error {
            AppError::Io(_) => (500, locales.t("errors.internal_server_error")),
            AppError::Anyhow(_) => (500, locales.t("errors.internal_server_error")),
            AppError::Model(_) => (400, locales.t("errors.bad_request")),
            AppError::Candle(_) => (500, locales.t("errors.internal_server_error")),
            AppError::Chat(_) => (400, locales.t("errors.bad_request")),
            AppError::SafeTensor(_) => (500, locales.t("errors.internal_server_error")),
            AppError::InvalidModel(_) => (400, locales.t("errors.bad_request")),
            AppError::ConfigError(_) => (500, locales.t("errors.internal_server_error")),
            AppError::TokenizerError(_) => (500, locales.t("errors.internal_server_error")),
            AppError::ValidationError(_) => (400, locales.t("errors.bad_request")),
            AppError::NotFound => (404, locales.t("errors.not_found")),
            AppError::Unauthorized => (401, locales.t("errors.unauthorized")),
            AppError::Forbidden => (403, locales.t("errors.forbidden")),
            AppError::ServiceUnavailable => (503, locales.t("errors.service_unavailable")),
            AppError::Generic(_) => (500, locales.t("errors.internal_server_error")),
            AppError::Locale(_) => (500, "Locale error".to_string()),
        };

        let mut response = ErrorResponse {
            code,
            status: status.to_string(),
            message: error.to_string(),
            data: None,
        };

        if let AppError::ValidationError(details) = error {
            response.data = match serde_json::to_value(details) {
                Ok(value) => Some(value),
                Err(err) => {
                    let msg = format!("Serialization failed: {}", err.to_string());
                    log::error!("{}", msg);
                    None
                }
            };
        }

        response
    }
}

pub struct ErrorHandlerMiddleware;

impl<S, B> Transform<S, ServiceRequest> for ErrorHandlerMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: actix_web::body::MessageBody + 'static,
    B::Error: Into<Error> + ResponseError + 'static,
{
    type Response = ServiceResponse<actix_web::body::BoxBody>;
    type Error = Error;
    type Transform = ErrorHandlerService<S>;
    type InitError = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Transform, Self::InitError>>>>;

    fn new_transform(&self, service: S) -> Self::Future {
        Box::pin(async move { Ok(ErrorHandlerService { service }) })
    }
}

pub struct ErrorHandlerService<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for ErrorHandlerService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: actix_web::body::MessageBody + 'static,
    B::Error: Into<Error>,
{
    type Response = ServiceResponse<actix_web::body::BoxBody>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(
        &self,
        ctx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Result<(), Self::Error>> {
        self.service.poll_ready(ctx)
    }

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let req_parts = req.parts().0.clone();
        match self
            .service
            .poll_ready(&mut core::task::Context::from_waker(futures::task::noop_waker_ref()))
        {
            core::task::Poll::Ready(Ok(())) => {
                log::debug!("Service ready to handle request");
            }
            core::task::Poll::Ready(Err(_err)) => {
                let msg = format!(
                    "Service unavailable: {} {} {}",
                    _err.to_string(),
                    req_parts.uri().to_string(),
                    req_parts.method().to_string()
                );
                log::error!("{}", msg);
                let response = ErrorResponse::from_error(&AppError::ServiceUnavailable);
                let http_response =
                    HttpResponse::build(StatusCode::SERVICE_UNAVAILABLE).json(response);
                return Box::pin(async move {
                    Ok(ServiceResponse::new(req_parts, http_response.map_into_boxed_body()))
                });
            }
            core::task::Poll::Pending => {
                let msg = format!(
                    "Service not ready: {} {}",
                    req_parts.uri().to_string(),
                    req_parts.method().to_string()
                );
                log::warn!("{}", msg);
                let response = ErrorResponse::from_error(&AppError::ServiceUnavailable);
                let http_response =
                    HttpResponse::build(StatusCode::SERVICE_UNAVAILABLE).json(response);
                return Box::pin(async move {
                    Ok(ServiceResponse::new(req_parts, http_response.map_into_boxed_body()))
                });
            }
        }

        let fut = self.service.call(req);

        Box::pin(async move {
            let result = fut.await;

            match result {
                Ok(res) => Ok(res.map_into_boxed_body()),
                Err(err) => {
                    let app_error: AppError = match err.try_into() {
                        Ok(err) => err,
                        Err(conv_err) => {
                            let msg = format!(
                                "Error conversion failed: {} {} {}",
                                conv_err.to_string(),
                                req_parts.uri().to_string(),
                                req_parts.method().to_string()
                            );
                            log::error!("{}", msg);
                            AppError::Generic("Internal server error".to_string())
                        }
                    };

                    let response = ErrorResponse::from_error(&app_error);

                    let msg = format!(
                        "Error occurred: {} {} {} {}",
                        req_parts.uri().to_string(),
                        req_parts.method().to_string(),
                        app_error.to_string(),
                        response.to_string()
                    );
                    log::error!("{}", msg);

                    let status_code = match StatusCode::from_u16(response.code as u16) {
                        Ok(code) => code,
                        Err(_) => {
                            let msg = format!("Invalid status code: {}", response.code.to_string());
                            log::error!("{}", msg);
                            StatusCode::INTERNAL_SERVER_ERROR
                        }
                    };

                    let http_response = HttpResponse::build(status_code).json(response);
                    Ok(ServiceResponse::new(req_parts, http_response.map_into_boxed_body()))
                }
            }
        })
    }
}

pub fn error_handler() -> ErrorHandlerMiddleware {
    ErrorHandlerMiddleware
}
