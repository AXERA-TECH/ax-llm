import argparse
import base64
import os

import requests


def to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="llama.cpp-compatible /embedding demo for axllm serve (supports multimodal content)"
    )
    parser.add_argument("--model", type=str, required=True, help="Embedding model name")
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Server base url (no /v1), e.g. http://127.0.0.1:8000",
    )
    parser.add_argument("--text", type=str, default="Describe this image.", help="Text prompt")
    parser.add_argument("--image", type=str, default="", help="Optional image path")
    args = parser.parse_args()

    model = args.model.strip()
    base = args.api_url.strip().rstrip("/")

    headers = {"Authorization": "Bearer not-needed"}

    if args.image:
        img_path = args.image.strip()
        if not os.path.exists(img_path):
            raise SystemExit(f"Error: image not found: {img_path}")

        b64 = to_base64(img_path)
        # llama.cpp native multimodal prompt format:
        # content can be a JSON object that contains:
        # - prompt_string: a string containing MTMD media markers (e.g. "<__media__>")
        # - multimodal_data: array of base64 strings substituted into markers in order
        body = {
            "model": model,
            "content": {
                "prompt_string": f"<__media__> {args.text}",
                "multimodal_data": [b64],
            },
        }
    else:
        # text-only embedding via llama.cpp-compatible endpoint
        body = {"model": model, "content": args.text}

    r = requests.post(f"{base}/embedding", json=body, headers=headers, timeout=600)
    r.raise_for_status()
    j = r.json()

    emb = j.get("embedding", [])
    if not isinstance(emb, list) or not emb:
        raise SystemExit(f"Invalid response: {j}")

    print(f"model={j.get('model','')} dim={len(emb)} head={emb[:8]}")


if __name__ == "__main__":
    main()

