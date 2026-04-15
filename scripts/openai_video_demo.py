import argparse
import os
import glob
import base64

from openai import OpenAI


def list_frame_files(frames_dir: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: list[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frames_dir, e)))
    return sorted(files)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI Video (frames_dir) Demo for axllm serve")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="API base url, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help='Directory containing video frames. In "dir" mode, this path must be visible to the server. In "base64" mode, frames are uploaded.',
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dir",
        choices=["dir", "base64"],
        help='dir: send "video:<frames_dir>" (server-side frames dir); base64: upload frames as "video:data:image/...;base64,..."',
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=32,
        help="For base64 mode: max number of frames to upload (default: 32).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="For base64 mode: take every Nth frame (default: 1).",
    )
    parser.add_argument("--prompt", type=str, default="Describe this video.", help="User prompt text")
    args = parser.parse_args()

    model = args.model.strip()
    api_url = args.api_url.strip()
    frames_dir = args.frames_dir.strip()

    if not os.path.isdir(frames_dir):
        raise SystemExit(f"Error: frames_dir is not a directory: {frames_dir}")
    frame_files = list_frame_files(frames_dir)
    if not frame_files:
        raise SystemExit(f"Error: frames_dir contains no image frames: {frames_dir}")

    user_parts = []
    if args.mode == "dir":
        user_parts.append({"type": "image_url", "image_url": {"url": f"video:{frames_dir}"}})
    else:
        stride = max(1, int(args.stride))
        max_frames = max(1, int(args.max_frames))
        selected = frame_files[::stride][:max_frames]
        if not selected:
            raise SystemExit("Error: no frames selected (check --stride/--max_frames).")
        for p in selected:
            user_parts.append({"type": "image_url", "image_url": {"url": f"video:{to_data_uri(p)}"}})
        print(f"upload frames: {len(selected)} / total {len(frame_files)}  (stride={stride})")

    user_parts.append({"type": "text", "text": args.prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "you are a helpful assistant."}]},
        {
            "role": "user",
            "content": user_parts,
        },
    ]

    client = OpenAI(api_key="not-needed", base_url=api_url)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    print("assistant:", end="\n")
    for ev in stream:
        delta = getattr(ev.choices[0], "delta", None)
        if delta and getattr(delta, "content", None):
            print(delta.content, end="", flush=True)
    print("\n")
