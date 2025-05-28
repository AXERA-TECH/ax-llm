# 🧠 LLM 推理服务 API 文档

> 本文档介绍了通过 HTTP 协议与本地 LLM 服务进行交互的接口，包含生成、重置、状态查询等操作。

---

## 📌 基本信息

* **服务地址**：`http://<your_ip>:8000`
* **内容类型**：`application/json`
* **编码方式**：UTF-8

---

## 📤 接口列表

| 接口路径            | 方法   | 描述             |
| --------------- | ---- | -------------- |
| `/api/generate` | POST | 发送 Prompt 开始推理 |
| `/api/reset`    | POST | 重置 LLM 上下文     |
| `/api/stop`     | GET  | 停止当前推理任务       |
| `/api/content`  | GET  | 获取推理结果         |
| `/api/upload`   | POST | 上传文件到服务器       |

---

## 🔮 1. /api/generate

**描述**：发送 prompt 请求，异步启动生成流程

### 请求

* **URL**：`/api/generate`
* **方法**：`POST`
* **请求头**：

  ```http
  Content-Type: application/json
  ```
* **请求体参数**：

| 参数名                  | 类型     | 是否必须 | 默认值 | 描述                  |
| -------------------- | ------ | ---- | --- | ------------------- |
| `prompt`             | string | ✅    | -   | 用户输入的文本 prompt      |
| `file_path`          | string | ❌    | -   | 上传图片的路径（由 `/api/upload` 返回） |
| `temperature`        | float  | ❌    | -   | 温度参数，控制生成的随机性       |
| `repetition_penalty` | float  | ❌    | -   | 重复惩罚系数，大于 1 可降低复读概率 |
| `top-p`              | float  | ❌    | -   | Top-P 采样策略阈值        |
| `top-k`              | int    | ❌    | -   | Top-K 采样限制，值越小越保守   |

### 示例请求

```json
{
  "prompt": "今天天气怎么样？",
  "temperature": 0.8,
  "repetition_penalty": 1.1,
  "top-p": 0.9,
  "top-k": 40
}
```

### 响应

```json
{
  "status": "ok"
}
```

---

## 🔁 2. /api/reset

**描述**：重置上下文环境，一般用于长对话或上下文溢出前清理缓存。

### 请求

* **URL**：`/api/reset`
* **方法**：`POST`
* **请求头**：

  ```http
  Content-Type: application/json
  ```
* **请求体参数**：

| 参数名             | 类型     | 是否必须 | 描述                |
| --------------- | ------ | ---- | ----------------- |
| `system_prompt` | string | ❌    | 系统提示词，用于设定模型人格/行为 |

### 示例请求

```json
{
  "system_prompt": "你是一个专业的中文助理"
}
```

### 响应

```json
{
  "status": "ok"
}
```

---

## ⛔ 3. /api/stop

**描述**：中断当前推理任务（如长文本生成时希望提前停止）

### 请求

* **URL**：`/api/stop`
* **方法**：`GET`

### 响应

```json
{
  "status": "ok"
}
```

---

## 📥 4. /api/content

**描述**：从内部消息队列获取当前/最新生成的内容，建议轮询调用

### 请求

* **URL**：`/api/content`
* **方法**：`GET`

### 响应

| 字段名        | 类型      | 描述                |
| ---------- | ------- | ----------------- |
| `response` | string  | 模型输出内容            |
| `done`     | boolean | 是否生成完成（true 表示结束） |

### 示例返回

```json
{
  "response": "你好，今天的天气晴朗，适合出行。",
  "done": true
}
```

---

## 🖼️ 5. /api/upload

**描述**：上传一张图片至服务器指定目录，服务器保存文件并返回绝对路径。

### 请求

* **URL**：`/api/upload`
* **方法**：`POST`
* **请求类型**：`multipart/form-data`
* **表单字段**：

| 字段名   | 类型   | 是否必须 | 描述       |
| ----- | ---- | ---- | -------- |
| image | File | ✅    | 要上传的图片文件 |

### 示例请求（使用 curl）

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "image=@test.jpg"
```

### 成功响应

```json
{
  "message": "File uploaded successfully",
  "file_path": "/abs/path/to/uploaded/test_20240528_150212.jpg"
}
```

| 字段名        | 类型     | 描述             |
| ---------- | ------ | -------------- |
| message    | string | 上传结果消息         |
| file\_path | string | 文件保存的绝对路径（全路径） |

### 失败响应

```json
{
  "error": "Failed to save file"
}
```

---

### 📁 说明

* 图片保存路径为：`UPLOAD_DIR`（由服务端定义的上传目录） + 自动生成的唯一文件名（通常带时间戳或随机串）。
* 文件名由服务端通过 `generate_filename()` 自动生成，避免重名覆盖。
* 当前接口仅支持单张图片上传。
* 返回的 `file_path` 为服务器上的**绝对路径**，如需提供公网访问链接，请做路径映射或另行处理。

---

## ⚠️ 错误响应格式

若请求格式不合法或模型未初始化，统一返回：

```json
{
  "error": "错误描述"
}
```

常见错误如下：

| 错误码 | 描述          |
| --- | ----------- |
| 400 | 模型未初始化或正在运行 |
| 400 | 请求体无效       |

---

## 🧑‍💻 示例用法 (curl)

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好呀！"}'
```

---
