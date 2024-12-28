pub mod chat_completion;

pub struct ChatService;

impl Default for ChatService {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatService {
    pub fn new() -> Self {
        Self
    }
}
