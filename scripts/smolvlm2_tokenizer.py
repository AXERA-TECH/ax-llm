from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import AddedToken
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import argparse
from datetime import timedelta
from num2words import num2words

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import  Optional

DEFAULT_VIDEO_INTRO = (
    "You are provided the following series of {frame_count} frames from a {video_duration} [H:MM:SS] video.\n"
)
DEFAULT_MEDIA_OUTTRO = "\n\n"
FRAME_TIMESTAMP_MESSAGE = "\nFrame from {timestamp}:"

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


@dataclass
class VideoMetadata(Mapping):
    total_num_frames: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    video_backend: Optional[str] = None
    frames_indices: Optional[list[int]] = None

    def __iter__(self):
        return (f.name for f in fields(self))

    def __len__(self):
        return len(fields(self))

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    @property
    def timestamps(self) -> list[float]:
        "Timestamps of the sampled frames in seconds."
        if self.fps is None or self.frames_indices is None:
            raise ValueError("Cannot infer video `timestamps` when `fps` or `frames_indices` is None.")
        return [frame_idx / self.fps for frame_idx in self.frames_indices]

    def update(self, dictionary):
        for key, value in dictionary.items():
            if hasattr(self, key):
                setattr(self, key, value)

class Tokenizer_Http():

    def __init__(self):
        self.token_ids_cache = []
        path = "smolvlm2-tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True,
                                                       use_fast=False)

        self.fake_image_token = getattr(self.tokenizer, "fake_image_token", "<fake_token_around_image>")
        self.image_token = getattr(self.tokenizer, "image_token", "<image>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.end_of_utterance_token = getattr(self.tokenizer, "end_of_utterance_token", "<end_of_utterance>")
        self.global_image_token = getattr(self.tokenizer, "global_image_token", "<global-img>")
        self.image_seq_len = 64
        self.video_token = getattr(self.tokenizer, "video_token", "<video>")

    def expand_text_with_image_tokens(self, text, image_rows, image_cols):
        prompt_strings = []
        image_rows = image_rows if image_rows is not None else [[0] * len(text)]
        image_cols = image_cols if image_cols is not None else [[0] * len(text)]
        for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
            print("sample",sample)
            print("sample_row",sample_rows)
            # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
            image_prompt_strings = []
            for n_rows, n_cols in zip(sample_rows, sample_cols):
                image_prompt_string = get_image_prompt_string(
                    n_rows,
                    n_cols,
                    self.image_seq_len,
                    image_token=self.image_token,
                    fake_token_around_image=self.fake_image_token,
                    global_img_token=self.global_image_token,
                )
                image_prompt_strings.append(image_prompt_string)

            split_sample = sample.split(self.image_token)
            if len(split_sample) == 0:
                raise ValueError("The image token should be present in the text.")

            # Place in the image prompt strings where the image tokens are
            sample = split_sample[0]
            for i, image_prompt_string in enumerate(image_prompt_strings):
                print("i",i)
                sample += image_prompt_string + split_sample[i + 1]
            prompt_strings.append(sample)

        return prompt_strings

    # just support one video
    def expand_text_with_video_tokens(self, text,  video_metadata):
        # num_frames = video_inputs["pixel_values"].shape[1]
        # video_metadata = iter(video_inputs["video_metadata"])
        video_metadata = VideoMetadata(**video_metadata)
        num_frames = video_metadata.total_num_frames
        prompt_strings = []
        for sample in text:
            while self.video_token in sample:
                # metadata = next(video_metadata)
                metadata = video_metadata
                if metadata.fps is None:
                    print(
                        "SmolVLM requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                        "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                        "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                    )
                    metadata.fps = 24  # Set the default fps to 24 for BC, otherwise `timestamps` can't be inferred
                timestamps = [(int(second // 60), int(second % 60)) for second in metadata.timestamps]
                duration = int(metadata.duration) if metadata.duration is not None else int(metadata.timestamps[-1])
                duration_td = timedelta(seconds=int(duration))
                image_prompt_strings = DEFAULT_VIDEO_INTRO.format(
                    frame_count=num2words(num_frames), video_duration=str(duration_td)
                )
                for timestamp in timestamps:
                    image_prompt_string = _prompt_single_image(
                        self.image_seq_len,
                        image_token=self.image_token,
                        fake_token_around_image=self.fake_image_token,
                        global_img_token=self.global_image_token,
                    )
                    timestamp = f"{timestamp[0]:02d}:{timestamp[1]:02d}"
                    image_prompt_string = FRAME_TIMESTAMP_MESSAGE.format(timestamp=timestamp) + image_prompt_string
                    image_prompt_strings += image_prompt_string

                image_prompt_strings += DEFAULT_MEDIA_OUTTRO
                sample = sample.replace(self.video_token, image_prompt_strings, 1)
            prompt_strings.append(sample)
        return prompt_strings

    def encode(self, content):
        text = [f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n']
        input_ids = self.tokenizer(text)
        return input_ids["input_ids"][0]

    def encode_vpm(self, content="Describe this image.", num_img=1, img_token_num=256, video_prompt=False):

        # official implementation 
        if video_prompt:
            video_metadata = {
                    "total_num_frames": num_img,
                    "fps": None,
                    "duration": None,
                    "frames_indices": list(range(num_img)),
                    "height": 512,
                    "width": 512,
                }
                
            text = [f'<|im_start|>User: <video>{content}<end_of_utterance>\nAssistant:']
            text = self.expand_text_with_video_tokens(text,  video_metadata)
            print("prompt string",text)
        else:
            img_token = "<image>"*(num_img//5)
            text = [f'<|im_start|>User:{img_token}{content}<end_of_utterance>\nAssistant:']
            text = self.expand_text_with_image_tokens(text, [[2]*(num_img//5)], [[2]*(num_img//5)])
            print("prompt string",text)
        
        output_kwargs = {'text_kwargs': {'add_special_tokens': False, 'padding': False, 'is_split_into_words': False}}
        
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return text_inputs["input_ids"][0]

    

    def decode(self, token_ids):
        self.token_ids_cache += token_ids
        text = self.tokenizer.decode(self.token_ids_cache)
        if "\ufffd" in text and len(self.token_ids_cache) < 9:
            print("text 中包含非法字符")
            return ""
        else:
            self.token_ids_cache.clear()
            return text.replace("\ufffd","")
    
    # def decode(self, token_ids):
    #     return self.tokenizer.decode(token_ids,
    #                                  clean_up_tokenization_spaces=False)

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
        return self.tokenizer.encode("<|image_pad|>")[0]
    
    @property
    def video_context_token(self):
        return self.tokenizer.encode("<|video_pad|>")[0]

tokenizer = Tokenizer_Http()

print(tokenizer.bos_id, tokenizer.bos_token, tokenizer.eos_id,
      tokenizer.eos_token)
# token_ids = tokenizer.encode_vpm()
# [151644, 8948, 198, 56568, 104625, 100633, 104455, 104800, 101101, 32022, 102022, 99602, 100013, 9370, 90286, 21287, 42140, 53772, 35243, 26288, 104949, 3837, 105205, 109641, 67916, 30698, 11, 54851, 46944, 115404, 42192, 99441, 100623, 48692, 100168, 110498, 1773, 151645, 151644, 872, 198,
# 151646,
# 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648, 151648,
# 151647,
# 198, 5501, 7512, 279, 2168, 19620, 13, 151645, 151644, 77091, 198]
# 118
# print(token_ids)
# print(len(token_ids))
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
        elif self.path == '/video_context_token':
            video_context_token = tokenizer.video_context_token
            if video_context_token is None:
                msg = json.dumps({'video_context_token': -1})
            else:
                msg = json.dumps({'video_context_token': video_context_token})
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
                token_ids = tokenizer.encode_vpm(prompt, req["num_img"], req["img_token_num"], req["video_prompt"])
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
