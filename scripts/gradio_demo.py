import mimetypes
import os
import gradio as gr
import requests
import json, time

base_url = "http://0.0.0.0:8000"

def upload_image(file_path):
    if file_path is None:
        return None
    # Gradio File component returns a tempfile-like object
    # file_path = image.name
    filename = os.path.basename(file_path)
    # Guess MIME type
    mime_type, _ = mimetypes.guess_type(filename)
    mime_type = mime_type or 'application/octet-stream'
    # Open file in binary mode for upload
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    # Prepare multipart form data
    files = {
        'image': (filename, file_bytes, mime_type)
    }
    # Send to upload endpoint
    resp = requests.post(
        f'{base_url}/api/upload',
        files=files
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get('file_path')

def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]

def stop_generation():
    try:
        requests.get(f'{base_url}/api/stop')
    except:
        pass

def respond(prompt, image:gr.Image, temp, rep_penalty, tp, tk, history=None):
    if history is None:
        history = []
    if not prompt.strip():
        return history
    # append empty response to history
    if  image is None:
        file_path = None
    else:
        file_path = upload_image(image)
        history.append((f'![]({file_path})', None))
        relative_path = os.path.relpath(file_path)
        # html = f"<img src='{relative_path}' style='max-width:300px;'/>"
        # history.append((html, None))
        # print(relative_path)
        
    
    history.append((prompt, ""))
    yield history
    # stream updates
    
    
    payload = {
        "prompt": prompt,
        "temperature": temp,
        "repetition_penalty": rep_penalty,
        "top-p": tp,
        "top-k": tk
    }
    if file_path:
        payload["file_path"] = file_path

    response = requests.post(
        f'{base_url}/api/generate',
        json=payload
    )
    response.raise_for_status()
    

    while True:
        time.sleep(0.01)
        response = requests.get(
                f'{base_url}/api/generate_provider'
            )
        data = response.json()
        chunk:str = data.get("response", "") 
        done = data.get("done", False)
        if done:
            break
        if chunk.strip() == "":
            continue
        history[-1] = (prompt, history[-1][1] + chunk)
        yield history
       
    print("end")
    
     


def chat_interface():
    with gr.Blocks(theme=gr.themes.Soft(font="Consolas"), fill_width=True) as demo:
        gr.Markdown("## Chat with LLM\nUpload an image and chat with the model!")
        with gr.Row():
            image = gr.Image(label="Upload Image", type="filepath")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600)
                prompt = gr.Textbox(placeholder="Type your message...", label="Prompt", value="描述一下这张图片")
                with gr.Row():
                    btn_chat = gr.Button("Chat", variant="primary")
                    btn_stop = gr.Button("Stop", variant="stop")

            with gr.Column(scale=1):
                temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.7, label="Temperature")
                repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, step=0.01, value=1.0, label="Repetition Penalty")
                top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Top-p Sampling")
                top_k = gr.Slider(minimum=0, maximum=100, step=1, value=40, label="Top-k Sampling")

            btn_stop.click(fn=stop_generation, inputs=None, outputs=None)
            btn_chat.click(
                fn=respond,
                inputs=[prompt, image, temperature, repetition_penalty, top_p, top_k, chatbot],
                outputs=chatbot
            )
        print("http://"+get_local_ip()+":7860")
        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    chat_interface()
