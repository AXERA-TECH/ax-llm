# AX-LLM

![GitHub License](https://img.shields.io/github/license/AXERA-TECH/ax-llm)

| Platform | Build Status |
| -------- | ------------ |
| AX650    | ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-llm/build_650.yml)|

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。

### 已支持芯片

- AX650A/AX650N
  - SDK ≥ v1.45.0_P31
- AX630C
  - SDK ≥ v2.0.0_P7

### 已支持模型

- Qwen2.5-0.5B/1.5B/3B/7B
- Qwen3-0.6B/1.7B/4B/8B

### 获取地址

- [百度网盘](https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e)
- [Google Drive](https://drive.google.com/drive/folders/1i8xdD2PWDlueouds6F1dhMc72n3v_aER?usp=sharing)

## 源码编译

- clone 本项目
    ```shell
    git clone --recursive https://github.com/AXERA-TECH/ax-llm.git
    cd ax-llm
    ```
- 仔细阅读 `build.sh` ，并在 `build.sh` 正确修改 `BSP_MSP_DIR` 变量后，运行编译脚本
    ```shell
    ./build.sh
    ```
- 正确编译后，`build/install/` 目录，应有以下文件（百度网盘中有预编译的可执行程序）
  ```
  $ tree install
    install
    └── bin
        ├── gradio_demo.py
        ├── main
        ├── main_api
        └── qwen2.5_tokenizer_uid.py
  ```
  
## 运行示例

### Qwen2.5-1.5B-ctx

#### 运行支持上下文的 tokenizer 服务器
```shell
python qwen2.5_tokenizer_uid.py 
Server running at http://127.0.0.1:12345
```

#### 运行命令行 llm
```shell
./run_qwen2.5_1.5b_ctx_ax650.sh 
[I][                            Init][ 110]: LLM init start
[I][                            Init][  34]: connect http://127.0.0.1:12345 ok
[I][                            Init][  57]: uid: 4bba0928-fada-4329-903e-3b6e52d68791
bos_id: -1, eos_id: 151645
100% | ████████████████████████████████ |  31 /  31 [18.94s<18.94s, 1.64 count/s] init post axmodel ok,remain_cmm(1464 MB)
[I][                            Init][ 188]: max_token_len : 2559
[I][                            Init][ 193]: kv_cache_size : 256, kv_cache_num: 2559
[I][                            Init][ 201]: prefill_token_num : 128
[I][                            Init][ 205]: grp: 1, prefill_max_token_num : 1
[I][                            Init][ 205]: grp: 2, prefill_max_token_num : 512
[I][                            Init][ 205]: grp: 3, prefill_max_token_num : 1024
[I][                            Init][ 205]: grp: 4, prefill_max_token_num : 1536
[I][                            Init][ 205]: grp: 5, prefill_max_token_num : 2048
[I][                            Init][ 209]: prefill_max_token_num : 2048
[I][                     load_config][ 282]: load config: 
{
    "enable_repetition_penalty": false,
    "enable_temperature": true,
    "enable_top_k_sampling": true,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 218]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
[I][          GenerateKVCachePrefill][ 271]: input token num : 21, prefill_split_num : 1 prefill_grpid : 2
[I][          GenerateKVCachePrefill][ 308]: input_num_token:21
[I][                            main][ 230]: precompute_len: 21
[I][                            main][ 231]: system_prompt: You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
prompt >> hello,my name is allen,who are you
[I][                      SetKVCache][ 531]: prefill_grpid:2 kv_cache_num:512 precompute_len:21 input_num_token:18
[I][                      SetKVCache][ 534]: current prefill_max_token_num:1920
[I][                             Run][ 660]: input token num : 18, prefill_split_num : 1
[I][                             Run][ 686]: input_num_token:18
[I][                             Run][ 829]: ttft: 539.49 ms
Hello Allen! I'm sorry, but I'm an AI language model and I don't have a name. I'm just here to help you with any questions or information you need. How can I assist you today?

[N][                             Run][ 943]: hit eos,avg 10.80 token/s

[I][                      GetKVCache][ 500]: precompute_len:83, remaining:1965
prompt >> 我叫什么名字
[I][                      SetKVCache][ 531]: prefill_grpid:2 kv_cache_num:512 precompute_len:83 input_num_token:12
[I][                      SetKVCache][ 534]: current prefill_max_token_num:1920
[I][                             Run][ 660]: input token num : 12, prefill_split_num : 1
[I][                             Run][ 686]: input_num_token:12
[I][                             Run][ 829]: ttft: 538.67 ms
你的名字是Allen。

[N][                             Run][ 943]: hit eos,avg 10.57 token/s

[I][                      GetKVCache][ 500]: precompute_len:100, remaining:1948

```

#### 运行api以及gradio demo
##### 启动服务器
```shell
./run_qwen2.5_1.5b_ctx_ax650_api.sh 
[I][                            Init][ 110]: LLM init start
[I][                            Init][  34]: connect http://10.126.33.124:12345 ok
[I][                            Init][  57]: uid: 13c64c2a-9b4e-4875-91f4-fa9f426e3726
bos_id: -1, eos_id: 151645
  3% | ██                                |   1 /  31 [0.15s<4.77s, 6.49 count/s] tokenizer init ok[I][                            Init][  26]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  31 /  31 [2.97s<2.97s, 10.44 count/s] init post axmodel ok,remain_cmm(1464 MB)[I][                            Init][ 188]: max_token_len : 2559
[I][                            Init][ 193]: kv_cache_size : 256, kv_cache_num: 2559
[I][                            Init][ 201]: prefill_token_num : 128
[I][                            Init][ 205]: grp: 1, prefill_max_token_num : 1
[I][                            Init][ 205]: grp: 2, prefill_max_token_num : 512
[I][                            Init][ 205]: grp: 3, prefill_max_token_num : 1024
[I][                            Init][ 205]: grp: 4, prefill_max_token_num : 1536
[I][                            Init][ 205]: grp: 5, prefill_max_token_num : 2048
[I][                            Init][ 209]: prefill_max_token_num : 2048
[I][                     load_config][ 282]: load config: 
{
    "enable_repetition_penalty": false,
    "enable_temperature": true,
    "enable_top_k_sampling": true,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 218]: LLM init ok
Server running on port 8000...
```
获取板端 ip，并修改 gradio 代码中的 ip 地址
```
import time
import gradio as gr
import requests
import json

# Base URL of your API server; adjust host and port as needed
API_URL = "http://x.x.x.x:8000"
...

```
运行 gradio_demo.py
```
python gradio_demo.py 
/home/axera/ax-llm/scripts/gradio_demo.py:102: UserWarning: You have not specified a value for the `type` parameter. Defaulting to the 'tuples' format for chatbot messages, but this is deprecated and will be removed in a future version of Gradio. Please set type='messages' instead, which uses openai-style dictionaries with 'role' and 'content' keys.
  chatbot = gr.Chatbot(elem_id="chatbox", label="Axera Chat",height=500)
* Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
```

![](scripts/gradio_demo.png)

## Reference

- [Qwen](https://huggingface.co/Qwen)

## 技术讨论

- Github issues
- QQ 群: 139953715

