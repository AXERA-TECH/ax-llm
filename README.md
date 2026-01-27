# AX-LLM

![GitHub License](https://img.shields.io/github/license/AXERA-TECH/ax-llm)

| Platform | Build Status |
| -------- | ------------ |
| AX650    | ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-llm/build_650.yml?internvl2)|

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。

### 已支持芯片

- AX650A/AX650N
  - SDK ≥ v3.6.2

### 已支持模型

- FastVLM-0.5B
- FastVLM-1.5B
- FastVLM-1.5B-GPTQ-Int4


## 源码编译
即可以交叉编译，也可以在开发板上编译。    
-  clone 本项目  
    ```shell
    git clone --recurse-submodules --shallow-submodules --depth 1 -b axcl-fastvlm https://github.com/Nnow2024/ax-llm.git
    cd ax-llm
    ```
- 编译  
  ```
  ./build.sh
  ```
- 正确编译后，`build/install/bin` 目录，应有以下文件（百度网盘中有预编译的可执行程序）
  ```
  $ tree install/bin/
    install/bin/
    ├── main
    ├── main_api
  ```

## 运行示例
### 1. 设置环境变量
设置 LD_LIBRARY_PATH={path to build/install/lib}:$LD_LIBRARY_PATH

### 2. 图像理解

1) 将 `build/install/bin`目录下的文件和编译好的模型都拷贝到爱芯板子上，下载[AXERA-TECH/FastVLM-0.5B](https://huggingface.co/AXERA-TECH/FastVLM-0.5B)仓库。
2) 仓库下载完成后，运行 `run_axcl_x86.sh`
```shell
chmod +x main* run*
./run_axcl_x86.sh
```
output:
```
[I][                            Init][ 162]: LLM init start
tokenizer_type = 3
stop_tokens size: 2
151645
151645
  7% | ███                               |   2 /  27 [0.18s<2.48s, 10.87 count/s] embed_selector init ok
[I][                             run][  30]: AXCLWorker start with devid 0
100% | ████████████████████████████████ |  27 /  27 [20.66s<20.66s, 1.31 count/s] init post axmodel ok,remain_cmm(6394 MB)[I][                            Init][ 313]: image encoder input nhwc@uint8
[I][                            Init][ 338]: image encoder output float32

[I][                            Init][ 350]: max_token_len : 1024
[I][                            Init][ 353]: kv_cache_size : 128, kv_cache_num: 1024
[I][                            Init][ 361]: prefill_token_num : 128
[I][                            Init][ 365]: grp: 1, prefill_max_token_num : 1
[I][                            Init][ 365]: grp: 2, prefill_max_token_num : 128
[I][                            Init][ 365]: grp: 3, prefill_max_token_num : 256
[I][                            Init][ 365]: grp: 4, prefill_max_token_num : 512
[I][                            Init][ 365]: grp: 5, prefill_max_token_num : 640
[I][                            Init][ 369]: prefill_max_token_num : 640
________________________
|    ID| remain cmm(MB)|
========================
|     0|           6228|
¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
[I][                     load_config][ 282]: load config:
{
    "enable_repetition_penalty": false,
    "enable_temperature": true,
    "enable_top_k_sampling": true,
    "enable_top_p_sampling": false,
    "penalty_window": 30,
    "repetition_penalty": 2,
    "temperature": 0.1,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 466]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
prompt >> who are you
image >>
[I][                             Run][ 718]: input token num : 27, prefill_split_num : 1
[I][                             Run][ 733]: prefill grpid 2
[I][                             Run][ 760]: input_num_token:27
[I][                             Run][ 889]: ttft: 153.21 ms
I'm an AI language model and I don't have personal identity or a physical body. I exist solely as a digital entity created by Apple Inc. and I am designed to assist and provide information to users.

[N][                             Run][1041]: hit eos,avg 17.21 token/s

prompt >> describe the image.
image >> ./images/image_1.jpg
[I][                          Encode][ 563]: image encode time : 49.20 ms, size : 57344
[I][                          Encode][ 610]: imgs_embed.size() : 1, media token size : 64
[I][                             Run][ 718]: input token num : 94, prefill_split_num : 1
[I][                             Run][ 733]: prefill grpid 2
[I][                             Run][ 760]: input_num_token:94
[I][                             Run][ 889]: ttft: 149.93 ms
The image depicts a panda bear in a naturalistic setting, surrounded by greenery and a wooden structure. The panda appears to be resting or hiding among the plants, with its head turned towards the camera. The panda's distinctive black and white fur is clearly visible, with its black ears, eyes, and nose contrasting against its white face and body. The background features a mix of green leaves, bamboo shoots, and other vegetation, as well as a wooden structure that resembles a tree trunk or a small fence. The overall scene suggests a peaceful and natural environment, possibly a zoo or wildlife sanctuary.

[N][                             Run][1041]: hit eos,avg 17.11 token/s

prompt >> q
[I][                             run][  80]: AXCLWorker exit with devid 0
```

## Reference

- [AXERA-TECH/FastVLM-0.5B](https://huggingface.co/AXERA-TECH/FastVLM-0.5B)
- [AXERA-TECH/FastVLM-1.5B](https://huggingface.co/AXERA-TECH/FastVLM-1.5B)
- [AXERA-TECH/FastVLM-1.5B-GPTQ-Int4](https://huggingface.co/AXERA-TECH/FastVLM-1.5B-GPTQ-Int4)

## 技术讨

- Github issues
- QQ 群: 139953715
