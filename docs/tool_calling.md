# Tool / Function Calling (OpenAI-compatible)

`serve` mode supports OpenAI-style tool (function) calling: the client sends a `tools`
list, the model may respond with `tool_calls`, the client runs the tool and feeds the
result back, and the model produces a final answer.

## Supported models

> **Supported: the Qwen3 family and MiniCPM-V-4.6.**
> Tool calling uses Qwen3's `<tool_call>{...}</tool_call>` prompt format, so it works for
> `tokenizer_type` = `Qwen3`, `Qwen3VL`, `Qwen2_5`, `Qwen3_5`, `Qwen3_5VL`, `Qwen3Omni`
> (any model handled by the Qwen3 tokenizer), plus `MiniCPMV46` / `MiniCPMV46VL`
> (MiniCPM-V-4.6, whose text backbone is Qwen3.5 and shares the same tool convention).
> Both text and VLM (image + tools) work.
>
> **Other model families are not supported.** If you send `tools` to an unsupported model,
> the tool section is not rendered and the model will simply answer in text (no
> `tool_calls` will be returned).
>
> Practical note for small models: on ≤4B models the model occasionally answers directly
> instead of calling a tool. Use `tool_choice: "required"` to force a call. Tool calling is
> also most reliable with **greedy decoding** (sampling disabled) -- under sampling a small
> model is likelier to chatter instead of emitting a clean `<tool_call>` (e.g. MiniCPM-V-4.6
> tests: 6/6 greedy vs 2/6 at temperature 0.7).

## Request

Add `tools` (and optionally `tool_choice`) to a `/v1/chat/completions` request:

```json
{
  "model": "AXERA-TECH/Qwen3-1.7B",
  "messages": [{"role": "user", "content": "What's the weather in Beijing?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ]
}
```

`tool_choice` (default `auto`):

| value | behavior |
|-------|----------|
| `"auto"` | model decides whether to call a tool (default) |
| `"none"` | tools are not offered; model answers in text |
| `"required"` | model is instructed to call one of the tools |
| `{"type":"function","function":{"name":"X"}}` | model is instructed to call function `X` |

`tool_choice` is a prompt-level instruction, so on small models it is a strong hint, not a
hard guarantee (a model may still refuse to call a clearly-inappropriate forced tool).

## Response

When the model calls a tool, the response has `finish_reason: "tool_calls"` and a
`message.tool_calls` array (`content` is null):

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_0_get_weather",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
      }]
    }
  }]
}
```

`function.arguments` is a JSON **string** (OpenAI convention).

## Feeding the tool result back (multi-turn)

Run the tool, then send the conversation back including the assistant's `tool_calls` and a
`role: "tool"` message with the result:

```json
{
  "model": "AXERA-TECH/Qwen3-1.7B",
  "messages": [
    {"role": "user", "content": "What's the weather in Beijing?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_0_get_weather", "type": "function",
       "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}}]},
    {"role": "tool", "tool_call_id": "call_0_get_weather",
     "content": "{\"temperature\":\"32C\",\"condition\":\"sunny\"}"}
  ],
  "tools": [ ... same tools ... ]
}
```

The model then returns a normal text answer (`finish_reason: "stop"`), e.g.
*"The current weather in Beijing is 32°C and sunny."*

## Streaming

With `"stream": true`, when tools are active the tool call is emitted as a single
`delta.tool_calls` chunk followed by a finish chunk with `finish_reason: "tool_calls"`.
When the model answers in text instead, normal text deltas are streamed. (Streaming with
tools is buffered internally rather than token-incremental for the tool-call arguments.)

## Vision + tools

VLM models (e.g. `Qwen3-VL-2B-Instruct`) can be given an image and tools in the same
request; tool calling behaves the same as text.

## Limitations

- Qwen3 `<tool_call>` format only (Qwen3 family + MiniCPM-V-4.6); other model families are not supported.
- Streaming tool calls are buffered (not token-incremental).
- Reliability on ≤4B models is model-dependent (~95%); use `tool_choice: "required"` when a
  tool call is mandatory.
