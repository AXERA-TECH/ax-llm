import base64
import glob
from openai import OpenAI
import cv2
import tqdm

BASE_URL = "http://localhost:8000/v1"

def img_to_data_url(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


client = OpenAI(api_key="not-needed", base_url=BASE_URL)

def generate_caption_for_image(img_path: str) -> str:
    image_data = img_to_data_url(img_path)

    # 注意：messages 要是 list
    messages = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in one concise English sentence."
                },
                {
                    "type": "image_url",
                    "image_url": image_data,  # 如果你的服务要求 {\"url\": xxx} 再按你那边的格式来
                },
            ],
        }
    

    # 评测用非流式会简单很多
    resp = client.chat.completions.create(
        model="AXERA-TECH/Qwen3-VL-2B-Instruct-GPTQ-Int4",
        messages=messages,
        stream=False,
        max_tokens=64,
    )

    # 不同实现返回结构可能略有差异，这里做一个比较稳的写法
    msg = resp.choices[0].message

    # 如果 content 是字符串
    if isinstance(msg.content, str):
        text = msg.content
    else:
        # 也可能是一个 list（和输入类似），那就把其中的 text 拼一下
        parts = []
        for part in msg.content:
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
        text = "".join(parts)

    return text.strip()


import json
import os
from pycocotools.coco import COCO
import argparse

parser = argparse.ArgumentParser(description="Generate captions for images in COCO dataset.")
parser.add_argument("--cocoRoot", type=str, default="./coco2014", help="Path to the directory containing COCO images.")
args = parser.parse_args()


COCO_ROOT = args.cocoRoot
ANN_FILE = os.path.join(COCO_ROOT, "annotations", "captions_val2017.json")
IMG_DIR = os.path.join(COCO_ROOT, "val2017")

coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()  # 默认是 5000 张 val 图

results = []

for i, img_id in tqdm.tqdm(enumerate(img_ids), total=len(img_ids)):
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]  # 如 "COCO_val2014_000000123456.jpg"
    img_path = os.path.join(IMG_DIR, file_name)

    try:
        caption = generate_caption_for_image(img_path)
    except Exception as e:
        print(f"[ERROR] image_id={img_id}, file={file_name}, err={e}")
        caption = ""  # 或者跳过，这里简单处理

    results.append({
        "image_id": int(img_id),
        "caption": caption
    })

    if (i + 1) % 50 == 0:
        print(f"Processed {i+1}/{len(img_ids)} images")

# 写出 COCO 官方评测所需的 results.json
with open("coco_val2017_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f)
