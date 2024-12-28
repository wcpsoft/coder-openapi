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

## API 文档

### 模型管理

#### 获取模型列表
`GET /v1/models`

**响应示例：**
```json
{
  "models": [
    {
      "id": "yi-coder",
      "name": "Yi Coder",
      "description": "Yi 1.5B 代码模型",
      "is_cached": true,
      "is_enabled": true
    },
    {
      "id": "deepseek-coder",
      "name": "Deepseek Coder",
      "description": "Deepseek 代码模型",
      "is_cached": false,
      "is_enabled": false
    }
  ]
}
```

#### 下载模型
`POST /v1/download`

**请求参数：**
```json
{
  "model_id": "yi-coder"
}
```

**响应示例：**
```json
{
  "status": "success",
  "model_id": "yi-coder"
}
```

### 代码补全

#### 生成代码补全
`POST /v1/chat/completions`

**请求参数：**
```json
{
  "model": "yi-coder",
  "messages": [
    {
      "role": "user",
      "content": "编写一个计算阶乘的Python函数"
    }
  ]
}
```

**响应示例：**
```json
{
  "model": "yi-coder",
  "response": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
}
```

### 错误响应

所有错误响应遵循以下格式：
```json
{
  "error": "错误类型",
  "message": "错误描述"
}
```

常见错误：
- 400 Bad Request: 请求参数无效
- 404 Not Found: 请求的资源不存在
- 500 Internal Server Error: 服务器内部错误

### 示例请求

获取模型列表：
```bash
curl -X GET http://localhost:8080/v1/models
```

下载模型：
```bash
curl -X POST http://localhost:8080/v1/download \
  -H "Content-Type: application/json" \
  -d '{"model_id": "yi-coder"}'
```

生成代码补全：
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
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
