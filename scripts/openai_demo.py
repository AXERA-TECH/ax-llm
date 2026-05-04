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
    parser.add_argument('--audio', type=str, default=None,
                        help='Optional audio path for VLM. The audio will be base64-encoded and sent as audio_url.')
    parser.add_argument('--prompt', type=str, default="hello",
                        help='User prompt text (default: "hello")')
    args = parser.parse_args()

    # Base URL of your API server; adjust host and port as needed
    API_URL = args.api_url
    MODEL = args.model

    if args.image and args.audio:
        print("Error: this demo supports either --image or --audio, not both in the same request.")
        exit(1)

    def build_data_uri(path: str, kind: str) -> str:
        if not os.path.isfile(path):
            print(f"Error: {kind} file not found: {path}")
            exit(1)

        ext = os.path.splitext(path)[1].lower().lstrip('.')
        if kind == "image":
            mime_map = {
                'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png',
                'gif': 'gif', 'bmp': 'bmp', 'webp': 'webp', 'tiff': 'tiff',
            }
            mime_ext = mime_map.get(ext, 'jpeg')
            mime = f"image/{mime_ext}"
        else:
            mime_map = {
                'wav': 'wav',
                'mp3': 'mpeg',
                'flac': 'flac',
                'ogg': 'ogg',
                'm4a': 'mp4',
            }
            mime_ext = mime_map.get(ext, 'wav')
            mime = f"audio/{mime_ext}"

        with open(path, 'rb') as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8')

        return f"data:{mime};base64,{b64_data}"

    # Build user content
    if args.image:
        user_content = [
            {"type": "image_url", "image_url": {"url": build_data_uri(args.image, "image")}},
            {"type": "text", "text": args.prompt},
        ]
    elif args.audio:
        user_content = [
            {"type": "audio_url", "audio_url": {"url": build_data_uri(args.audio, "audio")}},
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
