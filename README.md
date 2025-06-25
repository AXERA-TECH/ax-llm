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

- Qwen1.5-0.5B/1.8B/4B
- Qwen2-0.5B/1.5B
- ChatGLM3-6B
- MiniCPM-2B
- TinyLLaMa-1.1B
- Llama2-7B
- Llama3-8B
- Phi-2
- Phi-3-mini

### 关联项目

- [Huggingface](https://huggingface.co/AXERA-TECH)

## 源码编译

- 在 Host 上下载 axcl llm 对应分支
- 
    ```shell
    git clone -b axcl-context-kvcache https://github.com/AXERA-TECH/ax-llm.git
    cd ax-llm
    ```
- 本地编译
    ```shell
    mkdir build
    cd build
    cmake ..
    make install -j4
    ```
- 正确编译后，`build/install/bin` 目录，应有以下文件（百度网盘中有预编译的可执行程序）
  ```
  (base) axera@dell:~/samples/ax-llm/build$ tree install
  install
  └── bin
      ├── main
      ├── main_api
  ```
  
## 运行示例

### Qwen2.5-1.5B-Instruct

完整的内容获取请参考 [huggingface.co/AXERA-TECH/Qwen2.5-1.5B-Instruct](https://huggingface.co/AXERA-TECH/Qwen2.5-1.5B-Instruct)

```shell
(base) axera@dell:~/qtang/llm-test/qwen2.5-1.5b-ctx$ ./run_qwen2.5_1.5b_ctx_axcl_x86.sh
[I][                            Init][ 136]: LLM init start
[I][                            Init][  34]: connect http://127.0.0.1:12345 ok
[I][                            Init][  57]: uid: 13c9ef69-1330-46b4-aa67-5fb88fd7a569
bos_id: -1, eos_id: 151645
  6% | ███                               |   2 /  31 [4.26s<65.97s, 0.47 count/s] embed_selector init ok
[I][                             run][  30]: AXCLWorker start with devid 0
100% | ████████████████████████████████ |  31 /  31 [48.95s<48.95s, 0.63 count/s] init post axmodel ok,remain_cmm(4610 MB)4853 MB)
[I][                            Init][ 237]: max_token_len : 2047
[I][                            Init][ 240]: kv_cache_size : 256, kv_cache_num: 2047
[I][                            Init][ 248]: prefill_token_num : 128
[I][                            Init][ 252]: grp: 1, prefill_max_token_num : 1
[I][                            Init][ 252]: grp: 2, prefill_max_token_num : 128
[I][                            Init][ 252]: grp: 3, prefill_max_token_num : 256
[I][                            Init][ 252]: grp: 4, prefill_max_token_num : 384
[I][                            Init][ 252]: grp: 5, prefill_max_token_num : 512
[I][                            Init][ 252]: grp: 6, prefill_max_token_num : 640
[I][                            Init][ 252]: grp: 7, prefill_max_token_num : 768
[I][                            Init][ 252]: grp: 8, prefill_max_token_num : 896
[I][                            Init][ 252]: grp: 9, prefill_max_token_num : 1024
[I][                            Init][ 256]: prefill_max_token_num : 1024
________________________
|    ID| remain cmm(MB)|
========================
|     0|           4610|
¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
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

[I][                            Init][ 279]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
[I][          GenerateKVCachePrefill][ 336]: input token num : 21, prefill_split_num : 1 prefill_grpid : 2
[I][          GenerateKVCachePrefill][ 373]: input_num_token:21
[I][                            main][ 236]: precompute_len: 21
[I][                            main][ 237]: system_prompt: You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
prompt >> who are you
[I][                      SetKVCache][ 629]: prefill_grpid:2 kv_cache_num:128 precompute_len:21 input_num_token:11
[I][                      SetKVCache][ 632]: current prefill_max_token_num:896
[I][                             Run][ 870]: input token num : 11, prefill_split_num : 1
[I][                             Run][ 902]: input_num_token:11
[I][                             Run][1031]: ttft: 418.11 ms
I am Qwen, a large language model created by Alibaba Cloud. I'm here to help you with questions and provide information on a variety of topics.
Please ask your question or let me know if you'd like to discuss something, and I'll do my best to assist you.

[N][                             Run][1183]: hit eos,avg 8.88 token/s

[I][                      GetKVCache][ 598]: precompute_len:89, remaining:935
prompt >> q
[I][                             run][  80]: AXCLWorker exit with devid 0
(base) axera@dell:~/qtang/llm-test/qwen2.5-1.5b-ctx$
```

## Reference

- [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

## 技术讨论

- Github issues
- QQ 群: 139953715

