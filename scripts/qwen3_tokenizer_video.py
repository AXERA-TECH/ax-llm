from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import AddedToken
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import argparse

def _prompt_split_image(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_img_token,
):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}"
                + f"<row_{n_h + 1}_col_{n_w + 1}>"
                + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(
    image_seq_len, fake_token_around_image, image_token, global_img_token
):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows,
    image_cols,
    image_seq_len,
    fake_token_around_image,
    image_token,
    global_img_token,
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(
        image_seq_len,
        image_rows,
        image_cols,
        fake_token_around_image,
        image_token,
        global_img_token,
    )

class Tokenizer_Http():

    def __init__(self):

        path = 'qwen3-vl-tokenizer'
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True,
                                                       use_fast=False)

    def encode(self, content):
        text = [f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n']
        input_ids = self.tokenizer(text)
        return input_ids["input_ids"][0]

    def encode_vpm(self, content="Describe this image.", num_img=1, img_token_num=256):

        # official implementation 
        imgs_token = '<|vision_start|>' +  '<|video_pad|>'*img_token_num*num_img + '<|vision_end|>'
        
        text = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{imgs_token}{content}<|im_end|>\n<|im_start|>assistant\n'
        
        output_kwargs = {'text_kwargs': {'padding': True, 'return_tensors': 'pt'}, 'images_kwargs': {'return_tensors': 'pt'}, 'audio_kwargs': {'padding': True, 'return_tensors': 'pt'}, 'videos_kwargs': {'fps': 2.0, 'return_tensors': 'pt'}, 'common_kwargs': {'return_tensors': 'pt'}}
        
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return text_inputs["input_ids"].tolist()[0]

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids,
                                     clean_up_tokenization_spaces=False)

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
        return self.tokenizer.encode("<|vision_start|>")[0]

    @property
    def img_context_token(self):
        return self.tokenizer.encode("<|video_pad|>")[0]

tokenizer = Tokenizer_Http()

print(tokenizer.bos_id, tokenizer.bos_token, tokenizer.eos_id,
      tokenizer.eos_token)
token_ids = tokenizer.encode_vpm()
# [151644, 8948, 198, 56568, 104625, 100633, 104455, 104800, 101101, 32022, 102022, 99602, 100013, 9370, 90286, 21287, 42140, 53772, 35243, 26288, 104949, 3837, 105205, 109641, 67916, 30698, 11, 54851, 46944, 115404, 42192, 99441, 100623, 48692, 100168, 110498, 1773, 151645, 151644, 872, 198,
# 151646,
# 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
# 151647,
# 198, 5501, 7512, 279, 2168, 19620, 13, 151645, 151644, 77091, 198]
# 118
print(token_ids)
print(len(token_ids))
token_ids = tokenizer.encode("hello world")
# [151644, 8948, 198, 56568, 104625, 100633, 104455, 104800, 101101, 32022, 102022, 99602, 100013, 9370, 90286, 21287, 42140, 53772, 35243, 26288, 104949, 3837, 105205, 109641, 67916, 30698, 11, 54851, 46944, 115404, 42192, 99441, 100623, 48692, 100168, 110498, 1773, 151645, 151644, 872, 198, 14990, 1879, 151645, 151644, 77091, 198]
# 47
print(token_ids)
print(len(token_ids))


class Request(BaseHTTPRequestHandler):
    #通过类继承，新定义类
    timeout = 5
    server_version = 'Apache'

    def do_GET(self):
        print(self.path)
        #在新类中定义get的内容（当客户端向该服务端使用get请求时，本服务端将如下运行）
        self.send_response(200)
        self.send_header("type", "get")  #设置响应头，可省略或设置多个
        self.end_headers()

        if self.path == '/bos_id':
            bos_id = tokenizer.bos_id
            # print(bos_id)
            # to json
            if bos_id is None:
                msg = json.dumps({'bos_id': -1})
            else:
                msg = json.dumps({'bos_id': bos_id})
        elif self.path == '/eos_id':
            eos_id = tokenizer.eos_id
            if eos_id is None:
                msg = json.dumps({'eos_id': -1})
            else:
                msg = json.dumps({'eos_id': eos_id})
        elif self.path == '/img_start_token':
            img_start_token = tokenizer.img_start_token
            if img_start_token is None:
                msg = json.dumps({'img_start_token': -1})
            else:
                msg = json.dumps({'img_start_token': img_start_token})
        elif self.path == '/img_context_token':
            img_context_token = tokenizer.img_context_token
            if img_context_token is None:
                msg = json.dumps({'img_context_token': -1})
            else:
                msg = json.dumps({'img_context_token': img_context_token})
        else:
            msg = 'error'

        print(msg)
        msg = str(msg).encode()  #转为str再转为byte格式

        self.wfile.write(msg)  #将byte格式的信息返回给客户端

    def do_POST(self):
        #在新类中定义post的内容（当客户端向该服务端使用post请求时，本服务端将如下运行）
        data = self.rfile.read(int(
            self.headers['content-length']))  #获取从客户端传入的参数（byte格式）
        data = data.decode()  #将byte格式转为str格式

        self.send_response(200)
        self.send_header("type", "post")  #设置响应头，可省略或设置多个
        self.end_headers()

        if self.path == '/encode':
            req = json.loads(data)
            print(req)
            prompt = req['text']
            b_img_prompt = False
            if 'img_prompt' in req:
                b_img_prompt = req['img_prompt']
            if b_img_prompt:
                token_ids = tokenizer.encode_vpm(prompt, req["num_img"], req["img_token_num"])
            else:
                token_ids = tokenizer.encode(prompt)
            
            if token_ids is None:
                msg = json.dumps({'token_ids': -1})
            else:
                msg = json.dumps({'token_ids': token_ids})

        elif self.path == '/decode':
            req = json.loads(data)
            token_ids = req['token_ids']
            text = tokenizer.decode(token_ids)
            if text is None:
                msg = json.dumps({'text': ""})
            else:
                msg = json.dumps({'text': text})
        else:
            msg = 'error'
        print(msg)
        msg = str(msg).encode()  #转为str再转为byte格式

        self.wfile.write(msg)  #将byte格式的信息返回给客户端


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--host', type=str, default='localhost')
    args.add_argument('--port', type=int, default=8080)
    args = args.parse_args()

    host = (args.host, args.port)  #设定地址与端口号，'localhost'等价于'127.0.0.1'
    print('http://%s:%s' % host)
    server = HTTPServer(host, Request)  #根据地址端口号和新定义的类，创建服务器实例
    server.serve_forever()  #开启服务
