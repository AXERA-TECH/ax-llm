import subprocess
import time
from openai import OpenAI
import requests
import json
import re
import argparse
import base64
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='OpenAI Demo')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--api_url', type=str, help='API URL', default="http://127.0.0.1:8000/v1")
    parser.add_argument('--image', type=str, default=None,
                        help='Optional image path for VLM. The image will be base64-encoded and sent as image_url.')
    parser.add_argument('--prompt', type=str, default="hello",
                        help='User prompt text (default: "hello")')
    args = parser.parse_args()

    # Base URL of your API server; adjust host and port as needed
    API_URL = args.api_url
    MODEL = args.model

    # Build user content
    if args.image:
        # VLM mode: send image + text
        image_path = args.image
        if not os.path.isfile(image_path):
            print(f"Error: image file not found: {image_path}")
            exit(1)

        # Detect MIME type from extension
        ext = os.path.splitext(image_path)[1].lower().lstrip('.')
        mime_map = {
            'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png',
            'gif': 'gif', 'bmp': 'bmp', 'webp': 'webp', 'tiff': 'tiff',
        }
        mime_ext = mime_map.get(ext, 'jpeg')

        with open(image_path, 'rb') as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8')

        data_uri = f"data:image/{mime_ext};base64,{b64_data}"

        user_content = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": args.prompt},
        ]
    else:
        user_content = args.prompt

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "you are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": user_content
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

