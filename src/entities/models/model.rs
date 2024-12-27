use crate::error::Result;

pub trait Model: Send + Sync {
    #[allow(dead_code)]
    fn generate_response(&self, input: &str) -> Result<String>;
}
