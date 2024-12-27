use thiserror::Error;

#[derive(Error, Debug)]
pub enum TimeError {
    #[error("System time error")]
    SystemTimeError(#[from] std::time::SystemTimeError),
}
