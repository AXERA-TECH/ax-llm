import requests
import json
import soundfile as sf
import numpy as np

def tts_request(text, voice="zf_xiaoxiao", lang="z", speed=1.0, format="wav"):
    url = "http://10.126.33.252:8081/tts"
    
    payload = {
        "sentence": text,
        "voice_name": voice,
        "lang_code": lang,
        "speed": speed,
        "sample_rate": 24000
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            if format == "wav":
                # 保存WAV文件
                with open("output.wav", "wb") as f:
                    f.write(response.content)
                print("WAV文件已保存: output.wav")
                
                # 也可以直接播放
                import io
                audio_data, sr = sf.read(io.BytesIO(response.content))
                return audio_data, sr
            else:
                # 处理PCM数据
                import struct
                audio_data = struct.unpack(f'{len(response.content)//4}f', response.content)
                print(f"收到 {len(audio_data)} 个PCM样本")
                return np.array(audio_data), 24000
        else:
            print(f"请求失败: {response.status_code}")
            print(response.text)
            return None, None
            
    except Exception as e:
        print(f"请求异常: {e}")
        return None, None

# 使用示例
if __name__ == "__main__":
    text = "你好，这是一个TTS测试。欢迎使用语音合成服务。"
    
    # 获取WAV格式
    audio, sr = tts_request(text, format="wav")
    
    if audio is not None:
        print(f"音频时长: {len(audio)/sr:.2f} 秒")
        
        # 播放音频（可选）
        # import sounddevice as sd
        # sd.play(audio, sr)
        # sd.wait()