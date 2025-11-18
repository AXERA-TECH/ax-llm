import subprocess
import time
import gradio as gr
from openai import OpenAI
import requests
import json
import re

# Base URL of your API server; adjust host and port as needed
API_URL = "http://0.0.0.0:8000/v1"
MODEL = "AXERA-TECH/Qwen3-1.7B"

def get_all_local_ips():
    result = subprocess.run(['ip', 'a'], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有IPv4
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)

    # 过滤掉回环地址
    real_ips = [ip for ip in ips if not ip.startswith('127.')]

    return real_ips


def reset_chat(system_prompt):
    """
    Calls the /api/reset endpoint (POST) to initialize a new conversation.
    If system_prompt is provided, include it in the request body.
    Returns empty history and clears input. On error, shows error in chat.
    """
    payload = {}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    try:
        response = requests.post(f"{API_URL}/reset", json=payload)
        response.raise_for_status()
    except Exception as e:
        # Return error in chat if reset fails
        return [("Error resetting chat:", str(e))], ""
    # On successful reset, clear chat history and input
    return [], ""


def build_messages(prompt: str):
    content = []
    if prompt and prompt.strip():
        content.append({"type": "text", "text": prompt.strip()})

    return {"role": "user", "content": content if content else [{"type": "text", "text": prompt or ""}]}

# ---------- Gradio callback (single-turn, stream) ----------
def run_single_turn(prompt, chatbot_state):
    try:
        # 清空历史（单轮），构造用户气泡
        # chatbot_state = []

        # 构造 messages 和预览
        messages = build_messages(
            prompt=prompt or "",
        )

        user_md = (prompt or "").strip()

        chatbot_state.append((user_md or "(空提示)", ""))  # assistant 先空字符串，等待流式填充
        yield chatbot_state, chatbot_state  # 先把用户气泡渲染出来

        # 调后端（流式）
        client = OpenAI(api_key="not-needed", base_url=API_URL.strip())
        stream = client.chat.completions.create(
            model=MODEL.strip(),
            messages=messages,
            stream=True,
        )

        bot_chunks = []
        # 先补一个空 assistant 气泡
        # if len(chatbot_state) == 1:
        chatbot_state[-1] = (chatbot_state[-1][0], "")
        yield chatbot_state, chatbot_state 

        # 逐 chunk 更新 assistant 气泡（Markdown）
        for ev in stream:
            delta = getattr(ev.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                ctx = delta.content
                if "<think>" in delta.content:
                    ctx = delta.content.replace("<think>", "【思考中】")
                
                if "</think>" in delta.content:
                    ctx = delta.content.replace("</think>", "【思考结束】")
                
                bot_chunks.append(ctx)
                chatbot_state[-1] = (chatbot_state[-1][0], "".join(bot_chunks))
                yield chatbot_state, chatbot_state 

        # 结束再确保收尾
        chatbot_state[-1] = (chatbot_state[-1][0], "".join(bot_chunks) if bot_chunks else "(empty response)")
        yield chatbot_state, chatbot_state 

    except Exception as e:
        chatbot_state.append((
            chatbot_state[-1][0] if chatbot_state else "(request)",
            f"**Error:** {e}"
        ))
        yield chatbot_state, chatbot_state 



def stop_generate():
    try:
        requests.get(f"{API_URL}/stop")
    except Exception as e:
        print(e)
    

# Build the Gradio interface优化布局
with gr.Blocks(theme=gr.themes.Soft(font="Consolas"), fill_width=True) as demo:
    gr.Markdown("<h2 style='text-align:center;'>🚀 Chatbot Demo with Axare API Backend</h2>")
    
    # 使用Row包裹左右两个主要区域
    with gr.Row():
        # 左侧聊天主区域（占3/4宽度）
        with gr.Column(scale=3):
            system_prompt = gr.Textbox(label="System Prompt", placeholder="Optional system prompt", lines=2, value="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
            reset_button = gr.Button("🔄 Reset Chat")
            chatbot = gr.Chatbot(elem_id="chatbox", label="Axera Chat",height=500)
            user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=2)
            with gr.Row():
                send_button = gr.Button("➡️ Send", variant="primary")
                stop_button = gr.Button("🛑 Stop", variant="stop")

        # 右侧参数设置区域（占1/4宽度）
        with gr.Column(scale=1):
            temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.7, label="Temperature")
            repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, step=0.01, value=1.0, label="Repetition Penalty")
            top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Top-p Sampling")
            top_k = gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top-k Sampling")
    
    
    chat_state = gr.State([])
        
    reset_button.click(
        fn=reset_chat,
        inputs=system_prompt,
        outputs=[chatbot, user_input],  
    ).then(
        lambda: [],
        inputs=None,
        outputs=chat_state
    )
    
    send_button.click(
        fn=run_single_turn,
        inputs=[user_input, chat_state],  
        outputs=[chatbot, chat_state],      
        show_progress=True,
        queue=True,
    )
    
    stop_button.click(
        fn=stop_generate
    )
    

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  # adjust as needed
