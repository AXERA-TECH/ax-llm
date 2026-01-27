import os
import json
import glob
import base64
import argparse

from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tokenizer_path', type=str, help='tokenizer path(hf path or local dir, for example: "Qwen/Qwen3-VL-2B-Instruct" "Qwen/Qwen3-1.7B')
    parser.add_argument('--text', type=str, help='', default="hello")
    args = parser.parse_args()


    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
    except:
        tokenizer = None
    if None == tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        except:
            tokenizer = None
    if None == tokenizer:
        print("Default load tokenizer failed for ", args.tokenizer_path)
    
    if None != tokenizer:
        ids = tokenizer.encode(args.text)
        print("ids size: ", len(ids))
        print("ids: ", ids)
        
        text = tokenizer.decode(ids)
        print("text: ", text)
        
        # text = tokenizer.decode([0, 994, 1322, 2])
        # print("text: ", text)
        
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},{"role": "user", "content": "你好"},{"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}],
            tokenize=False,
            add_generation_prompt=True
        )
        print("text: \n", text)
        print("ids: \n", tokenizer.encode(text))
        
        
        
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},{"role": "user", "content": "你好"}],
            tokenize=False,
            add_generation_prompt=True
        )
        print("text: \n", text)
        print("ids: \n", tokenizer.encode(text))

if __name__ == '__main__':
    main()