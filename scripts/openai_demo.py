import subprocess
import time
from openai import OpenAI
import requests
import json
import re

# Base URL of your API server; adjust host and port as needed
API_URL = "http://10.126.33.252:8000/v1"
MODEL = "AXERA-TECH/Qwen3-1.7B"

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "your name is Lisa,you are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": "你好"
    }
]

# 调后端（流式）
client = OpenAI(api_key="not-needed", base_url=API_URL.strip())
stream = client.chat.completions.create(
    model=MODEL.strip(),
    messages=messages,
    stream=True,
)

print("assistant:", end="\n")
# 逐 chunk 更新 assistant 气泡（Markdown）
for ev in stream:
    delta = getattr(ev.choices[0], "delta", None)
    if delta and getattr(delta, "content", None):
        ctx = delta.content
        print(ctx, end="", flush=True)
print("\n")

