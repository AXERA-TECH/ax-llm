import base64
from openai import OpenAI
import cv2


def img_to_data_url(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


image_data = img_to_data_url("/home/axera/ax-llm/demo_cv308/frame_0075.jpg")
# print(image_data)

BASE_URL = "http://localhost:8000/v1"
client = OpenAI(api_key="not-needed", base_url=BASE_URL)

openai_messages = {
    "role": "user",
    "content": [
        {"type": "text", "text": "描述一下这张图片"},
        {"type": "image_url", "image_url": image_data},
    ],
}

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
