from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import argparse
import json

from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "/data/tmp/yongqiang/nfs/lhj/Qwen/Qwen3.5-2B"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _build_user_content(content, num_img=1, img_token_num=256, video_prompt=False):
    pad_token = "<|video_pad|>" if video_prompt else "<|image_pad|>"
    repeat_num = int(num_img) * int(img_token_num)
    if repeat_num < 0:
        repeat_num = 0
    vision_tokens = "<|vision_start|>" + pad_token * repeat_num + "<|vision_end|>"
    return f"{vision_tokens}{content}"


class TokenizerHttp:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, enable_thinking=False):
        self.token_ids_cache = []
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

    def _chat_text(self, content):
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking:
            kwargs["enable_thinking"] = True
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def encode(self, content):
        text = self._chat_text(content)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_vpm(
        self,
        content="Describe this image.",
        num_img=1,
        img_token_num=256,
        video_prompt=False,
    ):
        user_content = _build_user_content(
            content=content,
            num_img=num_img,
            img_token_num=img_token_num,
            video_prompt=video_prompt,
        )
        text = self._chat_text(user_content)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids):
        self.token_ids_cache += token_ids
        text = self.tokenizer.decode(self.token_ids_cache)
        if "\ufffd" in text and len(self.token_ids_cache) < 9:
            return ""
        self.token_ids_cache.clear()
        return text.replace("\ufffd", "")

    def _token_id(self, token):
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            return None
        return ids[0]

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def bos_token(self):
        return self.tokenizer.bos_token

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def img_start_token(self):
        return self._token_id("<|vision_start|>")

    @property
    def img_context_token(self):
        return self._token_id("<|image_pad|>")

    @property
    def video_context_token(self):
        return self._token_id("<|video_pad|>")


class Request(BaseHTTPRequestHandler):
    tokenizer = None

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/bos_id":
            value = self.tokenizer.bos_id
            self._send_json({"bos_id": -1 if value is None else value})
            return

        if self.path == "/eos_id":
            value = self.tokenizer.eos_id
            self._send_json({"eos_id": -1 if value is None else value})
            return

        if self.path == "/img_start_token":
            value = self.tokenizer.img_start_token
            self._send_json({"img_start_token": -1 if value is None else value})
            return

        if self.path == "/img_context_token":
            value = self.tokenizer.img_context_token
            self._send_json({"img_context_token": -1 if value is None else value})
            return

        if self.path == "/video_context_token":
            value = self.tokenizer.video_context_token
            self._send_json({"video_context_token": -1 if value is None else value})
            return

        self._send_json({"error": "unknown endpoint"})

    def do_POST(self):
        raw_data = self.rfile.read(int(self.headers["content-length"]))
        req = json.loads(raw_data.decode())

        if self.path == "/encode":
            prompt = req.get("text", "")
            b_img_prompt = bool(req.get("img_prompt", False))

            if b_img_prompt:
                token_ids = self.tokenizer.encode_vpm(
                    content=prompt,
                    num_img=int(req.get("num_img", 1)),
                    img_token_num=int(req.get("img_token_num", 256)),
                    video_prompt=bool(req.get("video_prompt", False)),
                )
            else:
                token_ids = self.tokenizer.encode(prompt)

            self._send_json({"token_ids": -1 if token_ids is None else token_ids})
            return

        if self.path == "/decode":
            token_ids = req.get("token_ids", [])
            text = self.tokenizer.decode(token_ids)
            self._send_json({"text": "" if text is None else text})
            return

        self._send_json({"error": "unknown endpoint"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="10.122.86.184")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()

    tokenizer = TokenizerHttp(model_path=args.model_path, enable_thinking=args.enable_thinking)
    Request.tokenizer = tokenizer

    print(
        f"bos_id={tokenizer.bos_id} bos_token={tokenizer.bos_token} "
        f"eos_id={tokenizer.eos_id} eos_token={tokenizer.eos_token}"
    )
    print(
        f"vision_start={tokenizer.img_start_token} "
        f"image_pad={tokenizer.img_context_token} "
        f"video_pad={tokenizer.video_context_token}"
    )

    host = (args.host, args.port)
    print("http://%s:%s" % host)
    server = HTTPServer(host, Request)
    server.serve_forever()
