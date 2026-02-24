# 合并后的 Main 程序使用说明

## 概述

`main` 程序现在合并了原来的 `main`（命令行对话）和 `main_api`（HTTP 服务）的功能，通过一个统一的命令行接口提供两种运行模式。

## 使用方法

### 命令格式

```bash
./main <mode> <model_path> [options]
```

- `mode`: 运行模式，`run` 或 `serve`
- `model_path`: 模型目录路径，包含 `config.json` 和模型文件

### 1. 命令行对话模式 (run)

启动交互式命令行对话：

```bash
./main run /path/to/model
```

带实时输出行：

```bash
./main run /path/to/model --live-print
```

#### 交互命令

- `q` - 退出程序
- `reset` - 重置 KV Cache 和对话历史
- `dd` - 删除上一轮对话
- `pp` - 打印对话历史

#### 中文输入支持

程序已优化 UTF-8 中文输入处理，支持正确的中文退格删除。

### 2. HTTP API 服务模式 (serve)

启动 OpenAI API 兼容的 HTTP 服务：

```bash
./main serve /path/to/model --port 8080
```

如果不指定端口，默认使用 `config.json` 中的 `port` 配置，或 8080。

## 模型目录结构

模型目录应包含以下文件：

```
model_directory/
├── config.json              # 模型配置文件（必需）
├── tokenizer.model          # Tokenizer 模型文件
├── post_config.json         # 后处理配置（可选）
├── *.axmodel               # AXera 模型文件
└── *.bin                   # Embedding 权重文件
```

## 配置文件 (config.json)

示例配置：

```json
{
    "model_name": "Qwen2.5-1.5B",
    "system_prompt": "你的名字叫lisa，你是一个智能助手。",
    
    "template_filename_axmodel": "qwen2_p128_l%d_together.axmodel",
    "filename_post_axmodel": "qwen2_post.axmodel",
    "url_tokenizer_model": "http://127.0.0.1:12345",
    "filename_tokens_embed": "model.embed_tokens.weight.bfloat16.bin",
    "post_config_path": "post_config.json",
    
    "axmodel_num": 28,
    "tokens_embed_num": 151936,
    "tokens_embed_size": 1536,
    
    "b_use_mmap_load_embed": true,
    "b_use_mmap_load_layer": true,
    
    "port": 8080
}
```

### 配置参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_name` | 模型名称（API 服务使用） | "AXERA-LLM" |
| `system_prompt` | 系统提示词 | "You are a helpful assistant." |
| `template_filename_axmodel` | AXModel 路径模板 | - |
| `filename_post_axmodel` | 后处理模型路径 | - |
| `url_tokenizer_model` | Tokenizer 路径或 URL | - |
| `filename_tokens_embed` | Embedding 权重文件路径 | - |
| `post_config_path` | 后处理配置文件路径 | "post_config.json" |
| `axmodel_num` | 模型层数 | 22 |
| `tokens_embed_num` | Token 嵌入数量 | 32000 |
| `tokens_embed_size` | Token 嵌入维度 | 2048 |
| `b_use_mmap_load_embed` | 使用 mmap 加载 embedding | false |
| `b_use_mmap_load_layer` | 使用 mmap 加载层 | true |
| `port` | HTTP 服务默认端口 | 8080 |

## 从旧版本迁移

### 原 main 命令行参数

旧命令：
```bash
./main \
  --template_filename_axmodel "path/model_l%d.axmodel" \
  --axmodel_num 28 \
  --url_tokenizer_model "http://127.0.0.1:12345" \
  --filename_post_axmodel path/post.axmodel \
  --filename_tokens_embed path/embed.bin \
  --tokens_embed_num 151936 \
  --tokens_embed_size 1536 \
  --live_print 1
```

新命令：
1. 将参数保存到 `model_directory/config.json`
2. 运行：`./main run model_directory --live-print`

### 原 main_api 命令行参数

旧命令：
```bash
./main_api \
  --template_filename_axmodel "path/model_l%d.axmodel" \
  --axmodel_num 28 \
  --port 8080
```

新命令：
1. 将参数保存到 `model_directory/config.json`
2. 运行：`./main serve model_directory --port 8080`

## 示例

### 示例 1: 运行 Qwen2.5-1.5B 命令行对话

```bash
# 1. 创建模型目录和配置
mkdir -p models/qwen2.5-1.5b
cp axmodel_qwen2.5_1.5b_chunked_nogptq_bf16/* models/qwen2.5-1.5b/
cp scripts/config.json.example models/qwen2.5-1.5b/config.json

# 2. 启动对话
./main run models/qwen2.5-1.5b --live-print
```

### 示例 2: 启动 HTTP API 服务

```bash
./main serve models/qwen2.5-1.5b --port 8080
```

然后可以通过 curl 测试：
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 注意事项

1. 模型路径可以是相对路径或绝对路径
2. 配置文件中的相对路径会相对于模型目录解析
3. 命令行参数 `--port` 会覆盖配置文件中的 `port` 设置
4. `--live-print` 仅在 `run` 模式下有效
