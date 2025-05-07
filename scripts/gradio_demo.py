import time
import gradio as gr
import requests
import json

# Base URL of your API server; adjust host and port as needed
API_URL = "http://localhost:8000"


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
        response = requests.post(f"{API_URL}/api/reset", json=payload)
        response.raise_for_status()
    except Exception as e:
        # Return error in chat if reset fails
        return [("Error resetting chat:", str(e))], ""
    # On successful reset, clear chat history and input
    return [], ""


def stream_generate(history, message, temperature, repetition_penalty, top_p, top_k):
    """
    Sends the user message and sampling parameters to /api/generate.
    Streams the response chunks and updates the last bot message in history.
    Clears input after sending. On error, shows error in chat.
    """
    history = history + [(message, "")]
    yield history, ""
    payload = {
        "prompt": message,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top-p": top_p,
        "top-k": top_k
    }
    try:
        response = requests.post(f"{API_URL}/api/generate", json=payload, timeout=(3.05, None))
        response.raise_for_status()
    except Exception as e:
        history[-1] = (message, f"Error: {str(e)}")
        yield history, ""
        return
    time.sleep(0.1)
    
    while True:
        time.sleep(0.01)
        response = requests.get(
                f"{API_URL}/api/generate_provider"
            )
        data = response.json()
        chunk:str = data.get("response", "") 
        done = data.get("done", False)
        if done:
            break
        if chunk.strip() == "":
            continue
        history[-1] = (message, history[-1][1] + chunk)
        yield history, ""
       
    print("end")
    

def stop_generate():
    try:
        requests.get(f"{API_URL}/api/stop")
    except Exception as e:
        print(e)
    
# Build the Gradio interface optimized for PC with spacious layout
# custom_css = """
# .gradio-container {
#     max-width: 1400px;
#     margin: auto;
#     padding: 20px;
# }
# .gradio-container > * {
#     margin-bottom: 20px;
# }
# #chatbox .overflow-y-auto {
#     height: 600px !important;
# }
# """

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

    # Wire up events: reset clears chat and input
    reset_button.click(fn=reset_chat, inputs=system_prompt, outputs=[chatbot, user_input])
    # send streams chat and clears input
    send_button.click(
        fn=stream_generate,
        inputs=[chatbot, user_input, temperature, repetition_penalty, top_p, top_k],
        outputs=[chatbot, user_input]
    )
    
    stop_button.click(
        fn=stop_generate
    )
    
    # allow Enter key to send
    user_input.submit(
        fn=stream_generate,
        inputs=[chatbot, user_input, temperature, repetition_penalty, top_p, top_k],
        outputs=[chatbot, user_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  # adjust as needed
