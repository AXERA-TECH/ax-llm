#!/usr/bin/env python3
"""Run an AX-LLM prefill profiler A/B smoke test.

The script is intentionally board-side friendly: it only uses Python's
standard library, starts a fresh axllm process for each arm, sends the same
OpenAI-compatible request, and stores the response plus the complete runtime
log in separate directories.  A failed engine request is reported as a
failure; two failures are never treated as an equivalent model result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


def request_json(url: str, payload: dict, timeout: float) -> tuple[int, object]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except Exception as exc:  # connection/timeout errors are test evidence
        return 0, {"error": {"type": type(exc).__name__, "message": str(exc)}}


def wait_ready(url: str, process: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def run_arm(args: argparse.Namespace, name: str, profile_value: str, root: Path) -> dict:
    arm_dir = root / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    log_path = arm_dir / "axllm.log"
    response_path = arm_dir / "response.json"
    request_path = arm_dir / "request.json"
    port = args.port + (0 if name == "profile_on" else 1)
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")

    env = os.environ.copy()
    env["AXLLM_PROFILE_PREFILL_CHUNKS"] = profile_value
    if args.profile_decode:
        env["AXLLM_PROFILE_DECODE"] = "1"
    else:
        env["AXLLM_PROFILE_DECODE"] = "0"
    if args.ld_library_path:
        env["LD_LIBRARY_PATH"] = args.ld_library_path

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [args.binary, "serve", args.model_dir, "--port", str(port)],
            cwd=args.cwd or None,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            ready = wait_ready(f"http://127.0.0.1:{port}/health", process, args.start_timeout)
            if ready:
                status, response = request_json(url, request, args.request_timeout)
            else:
                status, response = 0, {
                    "error": {"type": "server_start_failed", "message": "axllm did not become ready"}
                }
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)

    result = {"arm": name, "profile_value": profile_value, "http_status": status, "response": response}
    response_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="aarch64 axllm executable")
    parser.add_argument("--model-dir", required=True, help="ax-llm package directory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="who are you")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--start-timeout", type=float, default=90)
    parser.add_argument("--request-timeout", type=float, default=90)
    parser.add_argument("--cwd")
    parser.add_argument("--ld-library-path", default="")
    parser.add_argument("--profile-decode", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    on = run_arm(args, "profile_on", "1", root)
    off = run_arm(args, "profile_off", "0", root)
    both_http_success = on["http_status"] == 200 and off["http_status"] == 200
    on_content = None
    off_content = None
    if isinstance(on["response"], dict) and on["response"].get("choices"):
        on_content = on["response"]["choices"][0].get("message", {}).get("content")
    if isinstance(off["response"], dict) and off["response"].get("choices"):
        off_content = off["response"]["choices"][0].get("message", {}).get("content")
    summary = {
        "profile_on": on,
        "profile_off": off,
        "same_success": both_http_success and on_content == off_content,
        "profile_on_content": on_content,
        "profile_off_content": off_content,
        "note": "HTTP/model errors are preserved as failures; they are not an equivalence result.",
    }
    summary["both_http_success"] = both_http_success
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["both_http_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
