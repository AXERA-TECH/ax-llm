export AXLLM_LOG_LEVEL=warn

./build/install/bin/axllm tts_voice_clone Qwen3-TTS-12Hz-0.6B-Base-AX650 \
--ref_audio zero_shot_prompt.wav \
--x_vector_only_mode \
--text "恭喜发财，红包拿来！" \
--language Chinese \
--max_new_tokens 80 \
--output out.wav