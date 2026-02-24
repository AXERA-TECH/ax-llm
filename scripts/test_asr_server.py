import argparse
from openai import OpenAI

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--audio", "-a", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=True, choices=["sensevoice", "whisper_tiny", "whisper_base", "whisper_small", "whisper_turbo"])
    parser.add_argument("--language", "-l", type=str, required=True)
    return parser.parse_args()

args = get_args()
client = OpenAI(
    base_url=f'http://{args.ip}:{args.port}/v1',
    api_key="dummy_key"
)
audio_file = open(args.audio, "rb")

transcription = client.audio.transcriptions.create(
    model=args.model, 
    language=args.language,
    file=audio_file
)

audio_file.close()

print(transcription.text)