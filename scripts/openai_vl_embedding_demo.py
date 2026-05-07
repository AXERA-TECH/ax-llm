import argparse
import base64
import glob
import math
import os

import requests


def list_images(path: str) -> list[str]:
    if os.path.isdir(path):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files: list[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
        return sorted(files)
    return [path]


def to_data_uri(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime_map = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "gif": "gif",
        "bmp": "bmp",
        "webp": "webp",
        "tiff": "tiff",
    }
    mime_ext = mime_map.get(ext, "jpeg")
    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime_ext};base64,{b64_data}"


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI /v1/embeddings demo (messages, multimodal) for axllm serve")
    parser.add_argument("--model", type=str, required=True, help="Embedding model name (e.g. AXERA-TECH/Qwen3-VL-Embedding-2B)")
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="API base url, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--text", type=str, default="", help="Optional text paired with image(s)")
    parser.add_argument("--instruction", type=str, default="Represent the user's input.", help="System instruction")
    parser.add_argument("--image", type=str, default="", help="Image path or directory (optional)")
    parser.add_argument(
        "--mode",
        type=str,
        default="dir",
        choices=["dir", "base64"],
        help="dir: pass local path(s) visible to server; base64: upload image(s) as data URI(s)",
    )
    args = parser.parse_args()

    model = args.model.strip()
    api_url = args.api_url.strip().rstrip("/")
    images: list[str] = []
    if args.image:
        images = list_images(args.image.strip())
        for p in images:
            if not os.path.exists(p):
                raise SystemExit(f"Error: image not found: {p}")

    content_parts = []
    if images:
        if args.mode == "dir":
            for p in images:
                content_parts.append({"type": "image_url", "image_url": {"url": p}})
        else:
            for p in images:
                content_parts.append({"type": "image_url", "image_url": {"url": to_data_uri(p)}})

    if args.text:
        content_parts.append({"type": "text", "text": args.text})
    if not content_parts:
        content_parts.append({"type": "text", "text": "hello"})

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": args.instruction}]},
            {"role": "user", "content": content_parts},
        ],
        "encoding_format": "float",
    }

    resp = requests.post(
        f"{api_url}/embeddings",
        json=body,
        headers={"Authorization": "Bearer not-needed"},
        timeout=600,
    )
    resp.raise_for_status()
    js = resp.json()
    emb = js["data"][0]["embedding"]
    print(f"dim={len(emb)} norm={l2_norm(emb):.6f}")
    print("head:", " ".join(f"{x:.5f}" for x in emb[:8]))
