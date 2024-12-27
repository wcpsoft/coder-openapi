# Coder OpenAPI

一个兼容OpenAI API的代码模型服务，支持Yi-Coder和Deepseek-Coder模型。

## 功能特性

- 兼容OpenAI API接口
- 支持多种代码模型：
  - Yi-Coder
  - Deepseek-Coder
- 模型管理功能：
  - 模型状态跟踪
  - 模型下载
  - 模型启用/禁用
- RESTful API接口
- 基于Actix-web构建，性能优异
- 使用Tokio实现异步I/O

## 安装指南

1. 克隆仓库：
   ```bash
   git clone https://github.com/wcpsoft/coder-openapi.git
   cd coder-openapi
   ```

2. 安装依赖：
   ```bash
   cargo build --release
   ```

3. 配置服务：
   - 编辑`config/log4rs.yml`配置日志
   - 根据需要设置环境变量

4. 启动服务：
   ```bash
   cargo run --release
   ```

## 使用说明

### API接口

- `GET /models` - 获取可用模型列表
- `POST /models/{model_id}/download` - 下载指定模型
- `POST /chat/completions` - 生成代码补全

示例请求：
```bash
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yi-coder",
    "messages": [
      {"role": "user", "content": "编写一个计算阶乘的Python函数"}
    ]
  }'
```

## 开发指南

### 代码质量检查

本项目使用pre-commit hooks来强制执行代码质量标准。设置方法：

1. 安装pre-commit：
   ```bash
   pip install pre-commit
   ```

2. 安装hooks：
   ```bash
   pre-commit install
   ```

以下检查将在每次提交前自动运行：
- Rustfmt: 代码格式化
- Clippy: 代码质量检查

手动运行所有检查：
```bash
pre-commit run --all-files
```

## 贡献指南

欢迎贡献代码！请提交issue或pull request。

## 许可证

Apache 2.0
