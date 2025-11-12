# gradio_chat_single_turn.py
import re
import subprocess
import gradio as gr
import base64, cv2, os, tempfile
from openai import OpenAI
import requests

def get_all_local_ips():
    result = subprocess.run(['ip', 'a'], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有IPv4
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)

    # 过滤掉回环地址
    real_ips = [ip for ip in ips if not ip.startswith('127.')]

    return real_ips



# ---------- Helpers ----------
def img_to_data_url_from_cvframe(frame):
    import base64, cv2
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def img_to_data_url_from_path(img_path: str) -> str:
    import cv2, base64
    img = cv2.imread(img_path)
    return img_to_data_url_from_cvframe(img)

def video_to_data_urls(video_path: str, frame_stride: int = 30, max_frames: int = 8):
    import cv2, base64
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total / frame_stride > max_frames:
        frame_stride = int(total/max_frames)
    
    urls = []
    idx = 0
    first_preview = None
    while len(urls) < max_frames and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            break
        b64 = base64.b64encode(buf).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        urls.append(data_url)
        if first_preview is None:
            first_preview = data_url
        idx += frame_stride
    cap.release()
    return urls, first_preview

def save_preview_image_from_data_url(data_url: str) -> str:
    # 仅用于在 Chatbot 里显示缩略图
    comma = data_url.find(",")
    if comma == -1:
        return ""
    b64 = data_url[comma+1:]
    raw = base64.b64decode(b64)
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="preview_")
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(raw)
    return tmp_path

def build_messages(prompt: str, image_path: str | None, video_path: str | None,
                   prefer_video: bool, frame_stride: int, max_frames: int):
    content = []
    if prompt and prompt.strip():
        content.append({"type": "text", "text": prompt.strip()})

    if video_path and os.path.exists(video_path) and prefer_video:
        urls, first_preview = video_to_data_urls(video_path, frame_stride=frame_stride, max_frames=max_frames)
        content.append({"type": "image_url", "is_video":True, "image_url": urls})
        media_desc = f"（视频抽帧：{len(urls)} 帧，步长 {frame_stride}）"
        return {"role": "user", "content": content}, first_preview, media_desc

    if image_path and os.path.exists(image_path):
        u = img_to_data_url_from_path(image_path)
        content.append({"type": "image_url", "image_url": u})
        media_desc = "（已附带图片）"
        return {"role": "user", "content": content}, u, media_desc

    if video_path and os.path.exists(video_path):
        urls, first_preview = video_to_data_urls(video_path, frame_stride=frame_stride, max_frames=max_frames)
        content.append({"type": "image_url", "is_video":True, "image_url": urls})
        media_desc = f"（视频抽帧：{len(urls)} 帧，步长 {frame_stride}）"
        return {"role": "user", "content": content}, first_preview, media_desc

    return {"role": "user", "content": content if content else [{"type": "text", "text": prompt or ""}]}, None, ""

# ---------- Gradio callback (single-turn, stream) ----------
def run_single_turn(prompt, image_file, video_file, prefer_video, frame_stride, max_frames,
                    base_url, model, api_key, chatbot_state):
    """
    单轮：每次发送都会重置聊天历史，只显示本轮的 user/assistant 两个气泡。
    """
    try:
        # 清空历史（单轮），构造用户气泡
        chatbot_state = []

        # 准备文件路径
        image_path = image_file if isinstance(image_file, str) else (image_file.name if image_file else None)
        video_path = video_file if isinstance(video_file, str) else (video_file.name if video_file else None)

        # 构造 messages 和预览
        messages, preview_data_url, media_desc = build_messages(
            prompt=prompt or "",
            image_path=image_path,
            video_path=video_path,
            prefer_video=bool(prefer_video),
            frame_stride=int(frame_stride),
            max_frames=int(max_frames),
        )

        # 组装用户气泡（Markdown）：文本 + 预览图/视频说明
        user_md = (prompt or "").strip()
        if media_desc:
            user_md = (user_md + "\n\n" if user_md else "") + f"> {media_desc}"
        if preview_data_url:
            # user_md = (user_md + "\n\n" if user_md else "") + f"![preview]({preview_path})"
            user_md = (user_md + "\n\n" if user_md else "") + f"![preview]({preview_data_url})"

        chatbot_state.append((user_md or "(空提示)", ""))  # assistant 先空字符串，等待流式填充
        yield chatbot_state  # 先把用户气泡渲染出来

        # 调后端（流式）
        client = OpenAI(api_key=api_key or "not-needed", base_url=base_url.strip())
        stream = client.chat.completions.create(
            model=model.strip(),
            messages=messages,
            stream=True,
        )

        bot_chunks = []
        # 先补一个空 assistant 气泡
        if len(chatbot_state) == 1:
            chatbot_state[0] = (chatbot_state[0][0], "")
            yield chatbot_state

        # 逐 chunk 更新 assistant 气泡（Markdown）
        for ev in stream:
            delta = getattr(ev.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                bot_chunks.append(delta.content)
                chatbot_state[-1] = (chatbot_state[-1][0], "".join(bot_chunks))
                yield chatbot_state

        # 结束再确保收尾
        chatbot_state[-1] = (chatbot_state[-1][0], "".join(bot_chunks) if bot_chunks else "(empty response)")
        yield chatbot_state

    except Exception as e:
        chatbot_state.append((
            chatbot_state[-1][0] if chatbot_state else "(request)",
            f"**Error:** {e}"
        ))
        yield chatbot_state

# ---------- Gradio UI ----------
with gr.Blocks(css="""
    #chat, 
    #chat * {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }

    #chat .message,
    #chat [data-testid="bot"],
    #chat [data-testid="user"] {
        font-size: 18px !important;
    }
""",title="AXERA Qwen3 VL") as demo:
    axera_logo = img_to_data_url_from_path("/home/axera/ax-llm/build/axera_logo.png")
    gr.Markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <img src="{axera_logo}" alt="axera_logo" style="height: 60px;">
        </div>
        """
    )

    chatbot = gr.Chatbot(
        label="对话",
        bubble_full_width=False,
        height=500,
        avatar_images=(None, None),  # 可替换头像
        latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                          {"left": "$", "right": "$", "display": False}],
        show_copy_button=True,
        render_markdown=True,
        elem_id="chat"
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", placeholder="输入你的提示语", lines=2)
            with gr.Row():
                send_btn = gr.Button("发送 ▶️", variant="primary")
                clear_btn = gr.Button("清空")
                stop_btn = gr.Button("停止 ■", variant="stop")
            with gr.Row():
                image = gr.Image(type="filepath", label="上传图片（可选）")
                video = gr.Video(label="上传视频（可选）")
            
        with gr.Column(scale=1):
            base_url = gr.Textbox(value="http://localhost:8000/v1", label="Base URL")
            model = gr.Textbox(value="AXERA-TECH/Qwen3-VL-2B-Instruct-GPTQ-Int4", label="Model")
            api_key = gr.Textbox(value="not-needed", label="API Key", type="password")
            with gr.Row():
                prefer_video = gr.Checkbox(True, label="如果有视频，优先使用视频抽帧")
                frame_stride = gr.Slider(1, 90, value=30, step=1, label="视频抽帧间隔")
                max_frames = gr.Slider(1, 8, value=8, step=1, label="最多抽帧数")
            

    # 单轮对话需要一个 state 来承载当前这轮的气泡
    state = gr.State([])

    send_btn.click(
        fn=run_single_turn,
        inputs=[prompt, image, video, prefer_video, frame_stride, max_frames, base_url, model, api_key, state],
        outputs=chatbot,
        show_progress=True,
        queue=True,
    )
    
    def stop_stream():
        url = "http://localhost:8000/v1/stop"
        response = requests.get(url)
        if response.status_code == 200:
            print("Stream stopped successfully")
        else:
            print(f"Failed to stop stream: {response.status_code} - {response.text}")

    stop_btn.click(
        fn=stop_stream,
        outputs=chatbot,
        show_progress=True,
        queue=True,
    )

    def clear_all():
        return [], "", None, None, True, 30, 8
    clear_btn.click(clear_all, None, [chatbot, prompt, image, video, prefer_video, frame_stride, max_frames])

if __name__ == "__main__":
    ips = get_all_local_ips()
    for ip in ips:
        print(f"* Running on local URL:  http://{ip}:7860")
    ip = "0.0.0.0"
    demo.launch(server_name=ip, server_port=7860)
