# Yi-Coder OpenAPI

## Overview
Yi-Coder OpenAPI provides RESTful APIs for code generation using advanced language models. It supports multiple models and handles model caching and management.

## Features
- Chat completions API
- Model management API
- Model caching system
- Automatic model download
- Model status tracking

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-repo/yi-coder-openapi.git
```
2. Install dependencies:
```bash
cargo build
```

## Configuration
Create a `.env` file with the following variables:
```env
MODEL_CACHE_DIR=models_cache
```

## API Endpoints

### Model Management
- **GET /models**
  - Returns list of available models with status
  - Example response:
    ```json
    {
      "models": [
        {
          "id": "yi-coder",
          "name": "Yi-Coder",
          "is_cached": true,
          "is_enabled": true
        }
      ]
    }
    ```

- **POST /download**
  - Downloads specified model
  - Request body:
    ```json
    {
      "model_id": "yi-coder"
    }
    ```

### Chat Completions
- **POST /chat/completions**
  - Generates code completions
  - Request body:
    ```json
    {
      "model": "yi-coder",
      "messages": [
        {
          "role": "user",
          "content": "Write a Python function to calculate factorial"
        }
      ]
    }
    ```

## Usage Example
```bash
# Download model
curl -X POST http://localhost:8080/download \
  -H "Content-Type: application/json" \
  -d '{"model_id": "yi-coder"}'

# Get model list
curl http://localhost:8080/models

# Generate completion
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yi-coder",
    "messages": [
      {
        "role": "user",
        "content": "Write a Rust function to reverse a string"
      }
    ]
  }'
```

## Development
### Running the server
```bash
cargo run
```

### Testing
```bash
cargo test
```

## License
MIT License
