#[derive(Debug, Clone)]
pub struct ChatService {
    // 为未来聊天服务实现预留
}

impl Default for ChatService {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatService {
    /// 创建一个新的ChatService实例
    pub fn new() -> Self {
        ChatService {}
    }
}
