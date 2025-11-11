import base64
import glob
from openai import OpenAI
import cv2

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


def test(openai_messages):
    client = OpenAI(api_key="not-needed", base_url=BASE_URL)

    stream = client.chat.completions.create(
        model="AXERA-TECH/Qwen3-VL-2B-Instruct-GPTQ-Int4",
        messages=openai_messages,
        stream=True,
    )
    out_chunks = []
    for ev in stream:
        delta = ev.choices[0].delta
        if delta and delta.content:
            out_chunks.append(delta.content)
            print(delta.content, end="", flush=True)
    print()
    assistant_text = "".join(out_chunks).strip()

def test_image():
    image_data = img_to_data_url("../demo_cv308/frame_0075.jpg")

    openai_messages = {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述一下这张图片"},
            {"type": "image_url", "image_url": image_data},
        ],
    }
    

    test(openai_messages)

def test_video():
    image_list = glob.glob("../demo_cv308/*.jpg")
    image_list.sort()
    
    image_data_list = [img_to_data_url(img) for img in image_list]
    
    openai_messages = {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述一下这个视频"},
            {"type": "image_url", "is_video":True, "image_url": image_data_list},
        ],
    }
    
    test(openai_messages)

test_video()
