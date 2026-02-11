import subprocess
import time
from openai import OpenAI
import requests
import json
import re
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='OpenAI Demo')
    parser.add_argument('--model', type=str, default=MODEL, help='Model name')
    parser.add_argument('--api_url', type=str, default=API_URL, help='API URL', default="http://10.126.33.252:8000/v1")
    args = parser.parse_args()

    # Base URL of your API server; adjust host and port as needed
    API_URL = args.api_url
    MODEL = args.model

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "you are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": "hello"
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

