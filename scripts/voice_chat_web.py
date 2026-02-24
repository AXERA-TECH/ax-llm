#!/usr/bin/env python3
"""
语音对话网页应用 - HTTPS 版本
按住录音 -> ASR识别 -> AI对话 -> TTS播放
"""

import os
import sys
import io
import re
import base64
import tempfile
import requests
import soundfile as sf
import numpy as np
import ssl
import socket
import threading
import time
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI
from flask_cors import CORS

# ==================== 配置 ====================
ASR_IP = "10.126.33.252"
ASR_PORT = 3000
ASR_MODEL = "sensevoice"
ASR_LANGUAGE = "auto"

CHAT_IP = "10.126.33.252"
CHAT_PORT = 8000
CHAT_MODEL = "AXERA-TECH/Qwen3-1.7B"

TTS_URL = "http://10.126.33.252:8081/tts"
TTS_VOICE = "zf_xiaoxiao"
TTS_LANG = "z"
TTS_SPEED = 1.0

# 服务器配置
HOST = "0.0.0.0"
PORT = 3000
# 设置为 True 启用 HTTPS（推荐用于麦克风访问）
USE_HTTPS = True

# ==================== 初始化 Flask ====================
app = Flask(__name__)
CORS(app)

# ==================== HTML 模板 ====================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <title>AI 语音对话</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        
        h1 {
            color: white;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .container {
            width: 100%;
            max-width: 600px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .status {
            text-align: center;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .status.idle {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .status.recording {
            background: #ffebee;
            color: #d32f2f;
            animation: pulse 1.5s infinite;
        }
        
        .status.processing {
            background: #fff3e0;
            color: #f57c00;
        }
        
        .status.speaking {
            background: #e8f5e9;
            color: #388e3c;
        }
        
        .status.error {
            background: #ffebee;
            color: #c62828;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .mic-button {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 48px;
            cursor: pointer;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 30px;
            user-select: none;
            -webkit-user-select: none;
            touch-action: none;
        }
        
        .mic-button:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
        }
        
        .mic-button:active, .mic-button.recording {
            transform: scale(0.95);
            background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%);
            box-shadow: 0 5px 20px rgba(211, 47, 47, 0.4);
        }
        
        .mic-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .dialog {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: #667eea;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        
        .message.ai {
            background: #f5f5f5;
            color: #333;
            margin-right: auto;
            border-bottom-left-radius: 4px;
        }
        
        .message-label {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 4px;
            opacity: 0.8;
        }
        
        .hint {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 20px;
        }
        
        .permission-tip {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 12px;
            margin-bottom: 20px;
            border-radius: 4px;
            font-size: 14px;
            color: #e65100;
            display: none;
        }
        
        .audio-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            margin-top: 10px;
        }
        
        .audio-indicator span {
            width: 4px;
            height: 20px;
            background: #388e3c;
            border-radius: 2px;
            animation: sound 0.5s infinite ease-in-out;
        }
        
        .audio-indicator span:nth-child(2) { animation-delay: 0.1s; }
        .audio-indicator span:nth-child(3) { animation-delay: 0.2s; }
        .audio-indicator span:nth-child(4) { animation-delay: 0.3s; }
        .audio-indicator span:nth-child(5) { animation-delay: 0.4s; }
        
        @keyframes sound {
            0%, 100% { height: 10px; }
            50% { height: 25px; }
        }
        
        .volume-bar {
            width: 100%;
            height: 4px;
            background: #ddd;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
            display: none;
        }
        
        .volume-fill {
            height: 100%;
            background: #d32f2f;
            width: 0%;
            transition: width 0.1s ease;
        }
    </style>
</head>
<body>
    <h1>🎙️ AI 语音对话</h1>
    
    <div class="container">
        <div id="permissionTip" class="permission-tip">
            ⚠️ 需要麦克风权限。如果未看到授权提示，请检查浏览器设置。
        </div>
        
        <div id="status" class="status idle">按住麦克风按钮开始录音</div>
        
        <button id="micBtn" class="mic-button">🎤</button>
        
        <div class="volume-bar" id="volumeBar">
            <div class="volume-fill" id="volumeFill"></div>
        </div>
        
        <div id="dialog" class="dialog"></div>
        
        <div class="hint">按住麦克风说话，松开自动识别和回复</div>
    </div>

    <script>
        let mediaRecorder = null;
        let audioChunks = [];
        let audioContext = null;
        let analyser = null;
        let microphone = null;
        let volumeInterval = null;
        
        let isRecording = false;
        let isProcessing = false;
        
        const statusEl = document.getElementById('status');
        const micBtn = document.getElementById('micBtn');
        const dialogEl = document.getElementById('dialog');
        const permissionTip = document.getElementById('permissionTip');
        const volumeBar = document.getElementById('volumeBar');
        const volumeFill = document.getElementById('volumeFill');
        
        // 绑定事件 - 同时支持鼠标和触摸
        micBtn.addEventListener('mousedown', handleStart);
        micBtn.addEventListener('touchstart', handleStart, { passive: false });
        
        document.addEventListener('mouseup', handleEnd);
        document.addEventListener('touchend', handleEnd);
        
        // 防止触摸时的默认行为（如滚动、缩放）
        micBtn.addEventListener('touchmove', (e) => e.preventDefault(), { passive: false });
        
        function handleStart(e) {
            e.preventDefault();
            startRecording();
        }
        
        function handleEnd(e) {
            if (isRecording) {
                e.preventDefault();
                stopRecording();
            }
        }
        
        async function startRecording() {
            if (isProcessing || isRecording) return;
            
            try {
                // 请求麦克风权限
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000
                    } 
                });
                
                permissionTip.style.display = 'none';
                
                // 创建音频上下文用于分析音量
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                microphone = audioContext.createMediaStreamSource(stream);
                microphone.connect(analyser);
                analyser.fftSize = 256;
                
                // 创建 MediaRecorder
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.start(100); // 每100ms收集一次数据
                isRecording = true;
                
                // 更新UI
                statusEl.textContent = '🔴 正在录音... (松开结束)';
                statusEl.className = 'status recording';
                micBtn.classList.add('recording');
                volumeBar.style.display = 'block';
                
                // 开始音量检测
                startVolumeDetection();
                
            } catch (err) {
                console.error('录音失败:', err);
                permissionTip.style.display = 'block';
                
                if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                    statusEl.textContent = '❌ 麦克风权限被拒绝';
                    statusEl.className = 'status error';
                    alert('请允许麦克风访问权限。\\n\\n如果已经拒绝，请按以下步骤操作：\\n1. 点击地址栏左侧的 🔒 图标\\n2. 找到"麦克风"设置并改为"允许"\\n3. 刷新页面');
                } else if (err.name === 'NotFoundError') {
                    statusEl.textContent = '❌ 未找到麦克风设备';
                    statusEl.className = 'status error';
                } else {
                    statusEl.textContent = '❌ 无法访问麦克风: ' + err.message;
                    statusEl.className = 'status error';
                }
            }
        }
        
        function startVolumeDetection() {
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            volumeInterval = setInterval(() => {
                if (!analyser) return;
                
                analyser.getByteFrequencyData(dataArray);
                
                // 计算平均音量
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    sum += dataArray[i];
                }
                const average = sum / dataArray.length;
                
                // 更新音量条
                const percentage = Math.min(100, (average / 128) * 100);
                volumeFill.style.width = percentage + '%';
            }, 100);
        }
        
        async function stopRecording() {
            if (!isRecording || !mediaRecorder) return;
            
            isRecording = false;
            micBtn.classList.remove('recording');
            volumeBar.style.display = 'none';
            clearInterval(volumeInterval);
            
            statusEl.textContent = '⏳ 正在处理...';
            statusEl.className = 'status processing';
            isProcessing = true;
            micBtn.disabled = true;
            
            // 停止音量检测
            if (microphone) {
                microphone.disconnect();
                microphone = null;
            }
            if (audioContext) {
                await audioContext.close();
                audioContext = null;
            }
            
            mediaRecorder.stop();
            
            mediaRecorder.onstop = async () => {
                // 停止所有音轨
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                
                // 创建音频文件
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                
                try {
                    // 发送到服务器处理
                    await processAudio(audioBlob);
                    
                } catch (err) {
                    console.error('处理失败:', err);
                    statusEl.textContent = '❌ 处理失败，请重试';
                    statusEl.className = 'status error';
                } finally {
                    isProcessing = false;
                    micBtn.disabled = false;
                }
            };
        }
        
        async function processAudio(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                // 显示用户消息
                addMessage('user', result.asr_text);
                // 显示 AI 回复
                addMessage('ai', result.ai_text);
                // 播放音频
                playAudio(result.audio_base64);
            } else {
                statusEl.textContent = '❌ ' + (result.error || '处理失败');
                statusEl.className = 'status error';
            }
        }
        
        function addMessage(role, text) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;
            
            const label = document.createElement('div');
            label.className = 'message-label';
            label.textContent = role === 'user' ? '👤 你' : '🤖 AI';
            
            const content = document.createElement('div');
            content.textContent = text;
            
            msgDiv.appendChild(label);
            msgDiv.appendChild(content);
            dialogEl.appendChild(msgDiv);
            dialogEl.scrollTop = dialogEl.scrollHeight;
        }
        
        function playAudio(base64Data) {
            statusEl.textContent = '🔊 正在播放...';
            statusEl.className = 'status speaking';
            
            // 添加音频可视化指示器
            const indicator = document.createElement('div');
            indicator.className = 'audio-indicator';
            indicator.innerHTML = '<span></span><span></span><span></span><span></span><span></span>';
            const lastMsg = dialogEl.lastElementChild;
            if (lastMsg) {
                lastMsg.appendChild(indicator);
            }
            
            const audio = new Audio('data:audio/wav;base64,' + base64Data);
            audio.onended = () => {
                statusEl.textContent = '按住麦克风按钮开始录音';
                statusEl.className = 'status idle';
                if (indicator.parentNode) {
                    indicator.remove();
                }
            };
            audio.onerror = () => {
                statusEl.textContent = '❌ 音频播放失败';
                statusEl.className = 'status idle';
                if (indicator.parentNode) {
                    indicator.remove();
                }
            };
            audio.play().catch(err => {
                console.error('播放失败:', err);
                statusEl.textContent = '❌ 播放失败（可能被浏览器阻止）';
                statusEl.className = 'status error';
                if (indicator.parentNode) {
                    indicator.remove();
                }
            });
        }
        
        // 页面加载时检查麦克风权限
        async function checkPermission() {
            try {
                const result = await navigator.permissions.query({ name: 'microphone' });
                if (result.state === 'denied') {
                    permissionTip.style.display = 'block';
                    statusEl.textContent = '⚠️ 麦克风权限已被拒绝';
                    statusEl.className = 'status error';
                }
            } catch (e) {
                // 某些浏览器不支持 permissions API
            }
        }
        
        checkPermission();
    </script>
</body>
</html>
'''

# ==================== API 路由 ====================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/process', methods=['POST'])
def process():
    """处理音频：ASR -> Chat -> TTS"""
    try:
        # 1. 保存上传的音频文件
        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # 2. ASR 识别
            asr_text = asr_recognize(tmp_path)
            if not asr_text:
                return jsonify({'success': False, 'error': '语音识别失败'})
            
            print(f"[ASR] 识别结果: {asr_text}")
            
            # 3. AI 对话（过滤 thinking）
            ai_text = chat_with_ai(asr_text)
            if not ai_text:
                return jsonify({'success': False, 'error': 'AI对话失败'})
            
            print(f"[AI] 回复内容: {ai_text}")
            
            # 4. TTS 合成
            audio_base64 = tts_synthesize(ai_text)
            if not audio_base64:
                return jsonify({'success': False, 'error': '语音合成失败'})
            
            return jsonify({
                'success': True,
                'asr_text': asr_text,
                'ai_text': ai_text,
                'audio_base64': audio_base64
            })
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"处理异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ==================== 服务调用函数 ====================

def asr_recognize(audio_path: str) -> str:
    """调用 ASR 服务识别语音"""
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


def chat_with_ai(user_message: str) -> str:
    """调用 AI 对话服务，过滤 thinking 内容"""
    client = OpenAI(
        api_key="not-needed",
        base_url=f"http://{CHAT_IP}:{CHAT_PORT}/v1"
    )
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant named Lisa. Please respond in Chinese."
        },
        {
            "role": "user", 
            "content": user_message
        }
    ]
    
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
    
    # 过滤掉 thinking 内容 (Qwen3 模型会在 <think>...</think> 中包含思考过程)
    cleaned_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()
    
    return cleaned_text


def tts_synthesize(text: str) -> str:
    """调用 TTS 服务合成语音，返回 base64 编码的音频"""
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
        # 转换为 base64
        audio_base64 = base64.b64encode(response.content).decode('utf-8')
        return audio_base64
    else:
        print(f"TTS 请求失败: {response.status_code} - {response.text}")
        return None


# ==================== 生成自签名证书 ====================

def generate_self_signed_cert(cert_path="cert.pem", key_path="key.pem"):
    """生成自签名 SSL 证书"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime
    import ipaddress
    
    # 检查是否已有证书
    if os.path.exists(cert_path) and os.path.exists(key_path):
        print(f"使用现有证书: {cert_path}, {key_path}")
        return cert_path, key_path
    
    print("正在生成自签名 SSL 证书...")
    
    # 生成私钥
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # 获取本机所有 IP 地址
    alt_names = [x509.DNSName(u"localhost")]
    
    try:
        # 获取所有网络接口的 IP
        hostname = socket.gethostname()
        local_ip = socket.getaddrinfo(hostname, None, socket.AF_INET)[0][4][0]
        alt_names.append(x509.IPAddress(ipaddress.ip_address(u"127.0.0.1")))
        alt_names.append(x509.IPAddress(ipaddress.ip_address(local_ip)))
        
        # 尝试获取更多 IP
        for addr_info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = addr_info[4][0]
            if ip not in [str(a.value) for a in alt_names if hasattr(a, 'value')]:
                try:
                    alt_names.append(x509.IPAddress(ipaddress.ip_address(ip)))
                except:
                    pass
    except Exception as e:
        print(f"获取 IP 地址警告: {e}")
        alt_names.append(x509.IPAddress(ipaddress.ip_address(u"127.0.0.1")))
    
    # 生成证书
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"CN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Beijing"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Beijing"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Voice Chat"),
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName(alt_names),
        critical=False,
    ).sign(key, hashes.SHA256())
    
    # 保存证书和私钥
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    print(f"证书已生成: {cert_path}, {key_path}")
    return cert_path, key_path


def test_services():
    """测试各服务是否可达"""
    print("\n测试服务连接...")
    
    # 测试 ASR
    try:
        response = requests.get(f"http://{ASR_IP}:{ASR_PORT}/v1/models", timeout=5)
        print(f"  [OK] ASR 服务: http://{ASR_IP}:{ASR_PORT}")
    except Exception as e:
        print(f"  [WARN] ASR 服务连接失败: {e}")
    
    # 测试 Chat
    try:
        response = requests.get(f"http://{CHAT_IP}:{CHAT_PORT}/v1/models", timeout=5)
        print(f"  [OK] Chat 服务: http://{CHAT_IP}:{CHAT_PORT}")
    except Exception as e:
        print(f"  [WARN] Chat 服务连接失败: {e}")
    
    # 测试 TTS
    try:
        response = requests.get(TTS_URL.replace('/tts', '/'), timeout=5)
        print(f"  [OK] TTS 服务: {TTS_URL}")
    except Exception as e:
        print(f"  [WARN] TTS 服务连接失败: {e}")


def check_port_available(port):
    """检查端口是否被占用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('0.0.0.0', port))
    sock.close()
    return result != 0


# ==================== 主函数 ====================

if __name__ == '__main__':
    import ipaddress
    
    print("=" * 60)
    print("🎙️ AI 语音对话网页服务")
    print("=" * 60)
    
    # 检查端口
    if not check_port_available(PORT):
        print(f"\n❌ 错误: 端口 {PORT} 已被占用！")
        print(f"请尝试: lsof -i :{PORT} | grep LISTEN")
        print(f"或修改代码中的 PORT = {PORT} 为其他端口如 8080, 3000 等")
        sys.exit(1)
    
    # 获取本机 IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.getaddrinfo(hostname, None, socket.AF_INET)[0][4][0]
    except:
        local_ip = "127.0.0.1"
    
    print(f"\n本机 IP: {local_ip}")
    print(f"绑定地址: {HOST}:{PORT}")
    
    # 测试后端服务
    test_services()
    
    print("=" * 60)
    
    if USE_HTTPS:
        print("✅ HTTPS 模式（支持麦克风访问）")
        try:
            cert_file, key_file = generate_self_signed_cert()
        except Exception as e:
            print(f"❌ 证书生成失败: {e}")
            print("切换到 HTTP 模式...")
            USE_HTTPS = False
            cert_file = None
        
        if USE_HTTPS:
            print(f"\n请使用以下地址访问（注意是 https）：")
            print(f"  🔗 https://localhost:{PORT}")
            print(f"  🔗 https://127.0.0.1:{PORT}")
            print(f"  🔗 https://{local_ip}:{PORT}")
            print(f"\n⚠️  首次访问浏览器会显示'不安全'警告，请：")
            print(f"   1. 点击'高级'或'详细信息'")
            print(f"   2. 点击'继续前往'或'接受风险并继续'")
            print("=" * 60)
            
            try:
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_context.load_cert_chain(cert_file, key_file)
                print("\n🚀 启动 HTTPS 服务...")
                app.run(host=HOST, port=PORT, ssl_context=ssl_context, debug=False, threaded=True)
            except Exception as e:
                print(f"\n❌ HTTPS 启动失败: {e}")
                print("切换到 HTTP 模式...")
                USE_HTTPS = False
    
    if not USE_HTTPS:
        print("⚠️  HTTP 模式（仅 localhost/127.0.0.1 支持麦克风）")
        print(f"\n请使用以下地址访问：")
        print(f"  🔗 http://localhost:{PORT}")
        print(f"  🔗 http://127.0.0.1:{PORT}")
        print(f"\n❗ 注意：通过 http://{local_ip}:{PORT} 访问时")
        print(f"   浏览器会阻止麦克风访问！")
        print(f"\n💡 如需外网访问麦克风，请使用 HTTPS 模式")
        print("=" * 60)
        print("\n🚀 启动 HTTP 服务...")
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
