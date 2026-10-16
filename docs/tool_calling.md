# 工具调用（OpenAI 兼容）

`serve` 模式支持 OpenAI 的 tool / function calling：客户端传 `tools`，模型返回 `tool_calls`，客户端执行工具后把结果用 `role: "tool"` 发回，模型再给出最终回答。用法和 vLLM 开启 `--enable-auto-tool-choice` 时一致，官方 OpenAI SDK 可以直接用。

## 支持的模型

使用 Qwen3 的 `<tool_call>{...}</tool_call>` 约定，支持：
- Qwen3 tokenizer 覆盖的模型：Qwen2.5 / Qwen3 / Qwen3-VL / Qwen3.5 / Qwen3-Omni；
- MiniCPM-V-4.6。

图文请求可以同时带工具。其他模型如果收到 `tools`（且 `tool_choice` 不是 `none`），会返回 400：`This model does not support tool calling`。

## 请求

```json
{
  "model": "AXERA-TECH/Qwen3-1.7B",
  "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询城市当前天气",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
  }]
}
```

`tool_choice`：

| 值 | 行为 |
|---|---|
| `"auto"`（默认） | 模型自己决定是否调用 |
| `"none"` | 不提供工具，直接文字回答 |
| `"required"` | 要求模型必须调用某个工具 |
| `{"type":"function","function":{"name":"X"}}` | 要求模型调用函数 `X` |

`required` 和指定函数是通过提示词实现的，对小模型是强提示而不是硬约束。

## 响应

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_sD8QsjFXVbjqtGPhU16XFstc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}
      }]
    }
  }]
}
```

- `function.arguments` 是 JSON **字符串**（OpenAI 约定）；
- 模型在调用前写的文字保留在 `content`，没有则为 null；
- 开启思考时，思考过程在 `reasoning_content`，不会被当成工具调用解析。

## 回传工具结果

把模型返回的 assistant 消息（带 `tool_calls`）原样放回 `messages`，再追加 `role: "tool"` 消息。连续多条 tool 消息会合并成同一轮，与 Qwen3 官方模板一致；`content` 可以是字符串，也可以是 text 部件数组。

```json
"messages": [
  {"role": "user", "content": "北京和上海天气怎么样？"},
  {"role": "assistant", "content": null, "tool_calls": [
    {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}},
    {"id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"上海\"}"}}]},
  {"role": "tool", "tool_call_id": "call_a", "content": "北京：晴，25度"},
  {"role": "tool", "tool_call_id": "call_b", "content": "上海：小雨，19度"}
]
```

## 流式

`"stream": true` 时，正文逐 token 推送；一旦模型开始输出 `<tool_call>`，后面的内容先截留，生成结束后以一个 `delta.tool_calls` chunk 发出，结束 chunk 的 `finish_reason` 为 `"tool_calls"`。工具参数不是逐 token 推送的。

## 小模型建议

≤4B 模型偶尔会不调用工具而直接回答，必须调用时用 `tool_choice: "required"`。工具调用用 greedy 解码最稳（不传 temperature / top_p / top_k / min_p 即为 greedy）。
