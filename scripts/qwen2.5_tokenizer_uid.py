from transformers import AutoTokenizer, PreTrainedTokenizerFast
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import argparse
import uuid

# 全局字典：存储 uid 到 Tokenizer_Http 实例的映射
tokenizers = {}

class Tokenizer_Http():
    def __init__(self):
        model_id = "qwen2.5_tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        ]
        self.token_ids = []

    def encode(self, prompt, last_reply=None):
        if last_reply is not None:
            self.messages.append({"role": "assistant", "content": last_reply})
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # print("生成的文本:\n============\n", text, "============\n")
            self.token_ids = self.tokenizer.encode(text)[:-3]
        self.messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("生成的文本:\n============\n", text, "============\n")
        token_ids = self.tokenizer.encode(text)
        # 找出新增部分
        diff = token_ids[len(self.token_ids):]
        self.token_ids = token_ids
        print(self.decode(diff))
        return token_ids, diff

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

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
    
    def reset(self, system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
        self.messages = [
            {"role": "system", "content": system_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(text)[:-3]
        self.token_ids = token_ids
        print(self.decode(token_ids))
        return token_ids
        

class Request(BaseHTTPRequestHandler):
    timeout = 5
    server_version = 'Apache'

    def do_GET(self):
        print("GET 请求路径:", self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        # 新增接口：获取 uid
        if '/get_uid' in self.path:
            new_uid = str(uuid.uuid4())
            print("新 uid:", new_uid)
            # 为该 uid 创建一个新的 Tokenizer_Http 实例
            tokenizers[new_uid] = Tokenizer_Http()
            msg = json.dumps({'uid': new_uid})
        elif '/bos_id' in self.path:
            # 获取 uid 参数（例如 ?uid=xxx）
            uid = self.get_query_param("uid")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                bos_id = instance.bos_id
                msg = json.dumps({'bos_id': bos_id if bos_id is not None else -1})
        elif '/eos_id' in self.path:
            uid = self.get_query_param("uid")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                eos_id = instance.eos_id
                msg = json.dumps({'eos_id': eos_id if eos_id is not None else -1})
        else:
            msg = json.dumps({'error': 'Invalid GET endpoint'})

        print("响应消息:", msg)
        self.wfile.write(msg.encode())

    def do_POST(self):
        content_length = int(self.headers.get('content-length', 0))
        data = self.rfile.read(content_length).decode()
        print("POST 请求路径:", self.path)
        print("接收到的数据:", data)
        req = json.loads(data)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if '/encode' in self.path:
            # 请求数据中必须包含 uid, text, 和可选的 last_reply
            uid = req.get('uid')
            prompt = req.get('text')
            last_reply = req.get('last_reply')
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                token_ids, diff = instance.encode(prompt, last_reply)
                msg = json.dumps({'token_ids': token_ids, 'diff': diff})
        elif '/decode' in self.path:
            uid = req.get('uid')
            token_ids = req.get('token_ids')
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                text = instance.decode(token_ids)
                msg = json.dumps({'text': text})
        elif '/reset' in self.path:
            uid = req.get("uid")
            system_prompt = req.get("system_prompt")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                if system_prompt is not None:
                    print("system_prompt:", system_prompt)
                    token_ids = instance.reset(system_prompt)
                    msg = json.dumps({'token_ids': token_ids})
                else:
                    token_ids = instance.reset()
                    msg = json.dumps({'token_ids': token_ids})
        else:
            msg = json.dumps({'error': 'Invalid POST endpoint'})

        print("响应消息:", msg)
        self.wfile.write(msg.encode())

    def get_query_param(self, key):
        """
        辅助函数：从 GET 请求的 URL 中获取查询参数的值
        例如：/bos_id?uid=xxx
        """
        from urllib.parse import urlparse, parse_qs
        query = urlparse(self.path).query
        params = parse_qs(query)
        values = params.get(key)
        return values[0] if values else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=12345)
    args = parser.parse_args()

    host = (args.host, args.port)
    print('Server running at http://%s:%s' % host)
    server = HTTPServer(host, Request)
    server.serve_forever()
