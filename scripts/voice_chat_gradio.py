#!/usr/bin/env python3
"""
AI 语音对话 - Gradio 版本（流式显示，支持/no_think开关）
按住录音 -> ASR识别 -> AI对话 -> TTS播放
"""

import os
import re
import base64
import tempfile
import requests
from openai import OpenAI
import gradio as gr

# ==================== 配置 ====================
ASR_IP = "10.126.33.252"
ASR_PORT = 8080
ASR_MODEL = "sensevoice"
ASR_LANGUAGE = "auto"

CHAT_IP = "10.126.33.252"
CHAT_PORT = 8000
CHAT_MODEL = "AXERA-TECH/Qwen3-1.7B"

TTS_URL = "http://10.126.33.252:8081/tts"
TTS_VOICE = "zf_xiaoxiao"
TTS_LANG = "z"
TTS_SPEED = 1.0

# ==================== 服务调用函数 ====================

def asr_recognize(audio_path: str) -> str:
    """调用 ASR 服务识别语音"""
    if not audio_path:
        return ""
    
    client = OpenAI(
        base_url=f'http://{ASR_IP}:{ASR_PORT}/v1',
        api_key="dummy_key"
    )
    
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model=ASR_MODEL,
            language=ASR_LANGUAGE,
            file=f
        )
    
    return transcription.text.strip()


def chat_stream(user_message: str, history: list, enable_thinking: bool = True):
    """
    流式调用 AI 对话服务
    enable_thinking: 是否启用thinking，如果为False则添加/no_think
    """
    client = OpenAI(
        api_key="not-needed",
        base_url=f"http://{CHAT_IP}:{CHAT_PORT}/v1"
    )
    
    # 构建消息历史
    messages = [{"role": "system", "content": "You are a helpful assistant named Lisa. Please respond in Chinese."}]
    
    # 添加历史对话
    for msg in history:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                messages.append({"role": "user", "content": msg.get("content", "")})
            elif msg.get("role") == "assistant":
                content = msg.get("content", "")
                messages.append({"role": "assistant", "content": content})
    
    # 添加当前消息（如果禁用thinking，则添加/no_think后缀）
    if not enable_thinking:
        user_message = user_message + " /no_think"
    
    messages.append({"role": "user", "content": user_message})
    
    # 流式获取回复
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
    )
    
    full_text = ""
    for ev in stream:
        delta = getattr(ev.choices[0], "delta", None)
        if delta and getattr(delta, "content", None):
            full_text += delta.content
            yield full_text


def parse_thinking(text: str) -> tuple:
    """解析 thinking 和回复内容"""
    thinking_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        reply = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return thinking, reply
    return "", text


def tts_synthesize(text: str) -> str:
    """调用 TTS 服务合成语音，返回音频文件路径"""
    payload = {
        "sentence": text,
        "voice_name": TTS_VOICE,
        "lang_code": TTS_LANG,
        "speed": TTS_SPEED,
        "sample_rate": 24000
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(TTS_URL, json=payload, headers=headers, timeout=30)
    
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(response.content)
            return f.name
    else:
        print(f"TTS 请求失败: {response.status_code}")
        return None


# ==================== 主处理函数 ====================

def voice_chat_stream(audio, history, enable_thinking, thinking_state, reply_state):
    """
    处理语音对话流程（流式）
    enable_thinking: 是否显示AI思考过程
    """
    if audio is None:
        yield history, None, "请先录音", "", "", "请先录音"
        return
    
    if history is None:
        history = []
    
    # 处理录音文件
    if isinstance(audio, tuple):
        import numpy as np
        import soundfile as sf
        sr, data = audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, data, sr)
            audio_path = f.name
    else:
        audio_path = audio
    
    try:
        # 1. ASR 识别
        yield history, None, "🎤 识别中...", "", "", "语音识别中..."
        asr_text = asr_recognize(audio_path)
        
        if not asr_text:
            yield history, None, "语音识别失败", "", "", "语音识别失败"
            return
        
        # 显示是否使用了/no_think
        display_asr = asr_text
        if not enable_thinking:
            display_asr = asr_text + " /no_think"
        
        # 添加用户消息到历史（如果禁用thinking，则保存带/no_think的版本）
        history.append({"role": "user", "content": display_asr})
        yield history.copy(), None, f"🎤 你说: {asr_text}\n\n🤖 AI思考中...", "", "", f"识别: {asr_text}"
        
        # 2. AI 对话（流式）
        full_text = ""
        thinking_content = ""
        reply_content = ""
        
        for chunk in chat_stream(asr_text, history[:-1], enable_thinking):
            full_text = chunk
            thinking, reply = parse_thinking(full_text)
            
            if thinking and not thinking_content:
                thinking_content = thinking
            
            if reply:
                reply_content = reply
            
            # 构建显示文本
            display_text = f"🎤 你说: {asr_text}\n\n"
            
            # 如果启用thinking且存在thinking内容，则显示
            if enable_thinking and thinking_content:
                display_text += f"🤔 AI思考过程:\n{thinking_content}\n\n"
            
            if reply_content:
                display_text += f"🤖 AI回复:\n{reply_content}"
            else:
                display_text += "🤖 AI回复:\n（思考中...）"
            
            # 更新历史中的AI回复
            temp_history = history.copy()
            temp_history.append({"role": "assistant", "content": reply_content or full_text})
            
            yield temp_history, None, display_text, thinking_content, reply_content, f"AI回复中..."
        
        # 最终回复
        final_thinking, final_reply = parse_thinking(full_text)
        if not final_thinking:
            final_thinking = thinking_content
        if not final_reply:
            final_reply = reply_content or full_text
        
        # 更新最终历史
        final_history = history.copy()
        final_history.append({"role": "assistant", "content": final_reply})
        
        # 3. TTS 合成
        final_display = f"🎤 你说: {asr_text}\n\n"
        
        if enable_thinking and final_thinking:
            final_display += f"🤔 AI思考过程:\n{final_thinking}\n\n"
        
        final_display += f"🤖 AI回复:\n{final_reply}\n\n🔊 生成语音中..."
        
        yield final_history, None, final_display, final_thinking, final_reply, "生成语音中..."
        
        audio_path_result = tts_synthesize(final_reply)
        
        final_display = f"🎤 你说: {asr_text}\n\n"
        
        if enable_thinking and final_thinking:
            final_display += f"🤔 AI思考过程:\n{final_thinking}\n\n"
        
        final_display += f"🤖 AI回复:\n{final_reply}\n\n✅ 完成"
        
        yield final_history, audio_path_result, final_display, final_thinking, final_reply, "完成"
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        yield history, None, f"错误: {str(e)}", "", "", f"错误: {str(e)}"
    finally:
        if isinstance(audio, tuple) and 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)


def clear_conversation():
    """清空对话"""
    return [], None, "对话已清空", "", "", "", True  # 返回默认值保持开关开启


# ==================== Gradio 界面 ====================

with gr.Blocks(title="AI 语音对话") as demo:
    gr.Markdown("# 🎙️ AI 语音对话")
    gr.Markdown("点击录音按钮说话，松开后自动识别、对话、播放语音回复。")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 录音组件
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 按住录音，松开提交",
            )
            
            # 控制选项
            with gr.Row():
                enable_thinking = gr.Checkbox(
                    label="🤔 显示AI思考过程",
                    value=True,
                    info="关闭后自动添加 /no_think"
                )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
        
        with gr.Column(scale=2):
            # 对话显示
            chatbot = gr.Chatbot(
                label="对话记录",
                height=300,
            )
            
            # 详细显示区域
            detail_text = gr.Textbox(
                label="详细信息",
                lines=10,
                interactive=False
            )
            
            # 音频输出
            audio_output = gr.Audio(label="🔊 AI 语音回复", autoplay=True)
    
    # 隐藏的状态
    thinking_state = gr.State("")
    reply_state = gr.State("")
    status_state = gr.State("")
    
    # 录音完成自动提交
    audio_input.stop_recording(
        fn=voice_chat_stream,
        inputs=[audio_input, chatbot, enable_thinking, thinking_state, reply_state],
        outputs=[chatbot, audio_output, detail_text, thinking_state, reply_state, status_state]
    )
    
    # 清空对话
    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chatbot, audio_output, detail_text, thinking_state, reply_state, status_state, enable_thinking]
    )
    
    gr.Markdown("---")
    gr.Markdown('💡 **使用说明**: 点击麦克风 🎤 录音，说完停止。勾选"显示AI思考过程"可查看thinking内容，取消勾选则自动添加`/no_think`让AI直接回复。')


if __name__ == "__main__":
    print("=" * 50)
    print("🎙️ AI 语音对话 - Gradio 版本")
    print("=" * 50)
    print(f"ASR 服务: http://{ASR_IP}:{ASR_PORT}")
    print(f"AI 对话: http://{CHAT_IP}:{CHAT_PORT}")
    print(f"TTS 服务: {TTS_URL}")
    print("=" * 50)
    
    SSL_CERT = "cert.pem"
    SSL_KEY = "key.pem"
    
    use_https = os.path.exists(SSL_CERT) and os.path.exists(SSL_KEY)
    
    if use_https:
        print(f"✅ HTTPS 模式（支持麦克风访问）")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True,
            ssl_certfile=SSL_CERT,
            ssl_keyfile=SSL_KEY,
            ssl_verify=False
        )
    else:
        print(f"⚠️  HTTP 模式（仅 localhost 支持麦克风）")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True
        )
