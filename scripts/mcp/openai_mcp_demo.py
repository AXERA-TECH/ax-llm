#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def tool_to_prompt(tool: Any) -> str:
    if hasattr(tool, "model_dump"):
        data = tool.model_dump(mode="json", exclude_none=True)
    else:
        data = dict(tool)

    schema = data.get("inputSchema") or data.get("input_schema") or {}
    return compact_json(
        {
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "input_schema": schema,
        }
    )


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def display_text(text: str, show_thinking: bool) -> str:
    if show_thinking:
        return text
    if "<think>" in text and "</think>" not in text:
        return ""
    cleaned = strip_think(text)
    if cleaned:
        return cleaned
    return text


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = strip_think(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(cleaned)

    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        closed = False
        for index, char in enumerate(cleaned[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(cleaned[start : index + 1])
                    closed = True
                    break
        if not closed and depth > 0:
            candidates.append(cleaned[start:] + ("}" * depth))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    obj = extract_json_object(text)
    if not obj:
        return None

    if isinstance(obj.get("tool"), str):
        arguments = obj.get("arguments", {})
        return obj["tool"], arguments if isinstance(arguments, dict) else {}

    if isinstance(obj.get("tool"), dict):
        tool_call = obj["tool"]
        name = tool_call.get("name") or tool_call.get("tool") or tool_call.get("arg")
        arguments = tool_call.get("arguments") or obj.get("arguments", {})
        if isinstance(name, str):
            return name, arguments if isinstance(arguments, dict) else {}

    if isinstance(obj.get("tool_call"), dict):
        tool_call = obj["tool_call"]
        name = tool_call.get("name") or tool_call.get("tool")
        arguments = tool_call.get("arguments", {})
        if isinstance(name, str):
            return name, arguments if isinstance(arguments, dict) else {}

    return None


def mcp_result_to_text(result: Any, max_chars: int) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
            continue
        if hasattr(item, "model_dump"):
            parts.append(compact_json(item.model_dump(mode="json", exclude_none=True)))
        else:
            parts.append(str(item))

    if not parts:
        if hasattr(result, "model_dump"):
            parts.append(compact_json(result.model_dump(mode="json", exclude_none=True)))
        else:
            parts.append(str(result))

    text = "\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated, total_chars={len(text)}]"
    return text


def chat(client: OpenAI, model: str, messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    return response.choices[0].message.content or ""


async def run_demo(args: argparse.Namespace) -> None:
    server_script = Path(args.mcp_server).resolve()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script), "--workspace", args.workspace],
        env={**os.environ, "MCP_DEMO_WORKSPACE": str(Path(args.workspace).resolve())},
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            tool_names = {tool.name for tool in tools}
            tool_lines = "\n".join(f"- {tool_to_prompt(tool)}" for tool in tools)

            system_prompt = (
                "You are a concise assistant with access to MCP tools.\n"
                "When a tool is needed, reply with exactly one JSON object and no extra text:\n"
                '{"tool":"tool_name","arguments":{"arg":"value"}}\n'
                'For example: {"tool":"calculator","arguments":{"expression":"(12 + 8) * 3"}}\n'
                'Do not wrap the tool name inside another object, such as {"tool":{"arg":"calculator"}}.\n'
                "When no tool is needed, answer normally.\n"
                "After an MCP tool result is provided, use it to answer the user directly.\n"
                "Available MCP tools:\n"
                f"{tool_lines}"
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": args.prompt},
            ]

            client = OpenAI(api_key=args.api_key, base_url=args.api_url, timeout=args.timeout)

            original_prompt = args.prompt

            for round_index in range(args.max_tool_rounds + 1):
                assistant_text = chat(client, args.model, messages, args.temperature, args.max_tokens)
                tool_call = parse_tool_call(assistant_text)

                if not tool_call:
                    print(display_text(assistant_text, args.show_thinking) or assistant_text)
                    return

                name, arguments = tool_call
                if name not in tool_names:
                    print(f"assistant requested unknown tool: {name}", file=sys.stderr)
                    print(assistant_text)
                    return

                print(f"[mcp] call {name} {compact_json(arguments)}", file=sys.stderr)
                result = await session.call_tool(name, arguments)
                result_text = mcp_result_to_text(result, args.max_tool_result_chars)
                print(f"[mcp] result {result_text}", file=sys.stderr)

                if not args.allow_chained_tools:
                    final_messages = [
                        {
                            "role": "system",
                            "content": (
                                "Tool calling is now disabled. Answer the user directly in natural language. "
                                "Do not output JSON. Do not include hidden reasoning or <think> blocks. "
                                "Use the same language as the user when possible."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Original user request:\n{original_prompt}\n\n"
                                f"MCP tool result from {name}:\n{result_text}\n\n"
                                "Give the final answer."
                            ),
                        },
                    ]
                    final_text = chat(client, args.model, final_messages, args.temperature, args.final_max_tokens)
                    print(display_text(final_text, args.show_thinking) or f"MCP tool result: {result_text}")
                    return

                messages.append({"role": "assistant", "content": assistant_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"MCP tool result for {name}:\n{result_text}\n\n"
                            "Now answer the original user request. Do not call more tools unless absolutely necessary."
                        ),
                    }
                )

            print("max tool rounds reached before final answer", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Lightweight MCP tool-use demo for axllm OpenAI-compatible serve mode.")
    parser.add_argument("--api_url", default="http://127.0.0.1:8000/v1", help="axllm OpenAI API base URL.")
    parser.add_argument("--api_key", default="not-needed", help="API key placeholder.")
    parser.add_argument("--model", required=True, help="Model name exposed by axllm serve.")
    parser.add_argument("--prompt", default="现在几点？再计算 (12 + 8) * 3。", help="User prompt.")
    parser.add_argument("--mcp_server", default=str(Path(__file__).with_name("mcp_simple_server.py")), help="MCP stdio server script.")
    parser.add_argument("--workspace", default=os.getcwd(), help="Workspace for MCP demo file tools.")
    parser.add_argument("--max_tool_rounds", type=int, default=3, help="Maximum model-tool loop count.")
    parser.add_argument("--max_tool_result_chars", type=int, default=2000, help="Truncate long MCP tool results.")
    parser.add_argument("--allow_chained_tools", action="store_true", help="Allow the model to request more tools after a tool result.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens for model tool planning or direct answers.")
    parser.add_argument("--final_max_tokens", type=int, default=256, help="Max tokens for final answer after a tool result.")
    parser.add_argument("--show_thinking", action="store_true", help="Print model <think> blocks instead of hiding them.")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
