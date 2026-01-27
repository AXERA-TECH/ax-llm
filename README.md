# AX-LLM

![GitHub License](https://img.shields.io/github/license/AXERA-TECH/ax-llm)

| Platform | Build Status |
| -------- | ------------ |
| AX650    | ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-llm/build_650.yml)|

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。

### 已支持芯片

- AX650A/AX650N
  - SDK ≥ v3.6.2
- AX630C
  - SDK ≥ v3.0.0

### 已支持模型

- FastVLM-0.5B
- FastVLM-1.5B
- FastVLM-1.5B-GPTQ-Int4

### 获取地址

- [Huggingface](https://huggingface.co/AXERA-TECH)
  - [FastVLM-0.5B](https://huggingface.co/AXERA-TECH/FastVLM-0.5B) 
  - [FastVLM-1.5B](https://huggingface.co/AXERA-TECH/FastVLM-1.5B)
  - [FastVLM-1.5B-GPTQ-Int4](https://huggingface.co/AXERA-TECH/FastVLM-1.5B-GPTQ-Int4)

## 源码编译

- 在 Host 上下载 axcl llm 对应分支
    ```shell
    git clone --recurse-submodules --shallow-submodules --depth 1 -b ax-fastvlm https://github.com/Nnow2024/ax-llm.git
    cd ax-llm
    ```
- 本地编译
    ```shell
    sudo apt install libopencv-dev build-essential 
    ./build.sh
    ```
- 正确编译后，`build/install/bin` 目录
  ```
  $ tree install/bin/
    install/bin/
    ├── main
    ├── main_api
  ```
  其中 `main` 就是 Huggingface 仓库中对应的 `main_ax650`
  
## 运行示例

### FastVLM-1.5B-GPTQ-Int4

```shell
root@ax650:~/FastVLM-1.5B-GPTQ-Int4# ./run_ax650_1024.sh
[I][                            Init][ 134]: LLM init start
tokenizer_type = 3
  6% | ███                               |   2 /  31 [1.09s<16.97s, 1.83 count/s] embed_selector init ok
100% | ████████████████████████████████ |  31 /  31 [3.39s<3.39s, 9.14 count/s] init post axmodel ok,remain_cmm(8619 MB)[I][                            Init][ 284]: image encoder input nhwc@uint8
[I][                            Init][ 308]: image encoder output float32

[I][                            Init][ 318]: image_encoder_height : 1024, image_encoder_width: 1024
[I][                            Init][ 320]: max_token_len : 1024
[I][                            Init][ 323]: kv_cache_size : 256, kv_cache_num: 1024
[I][                            Init][ 331]: prefill_token_num : 128
[I][                            Init][ 335]: grp: 1, prefill_max_token_num : 1
[I][                            Init][ 335]: grp: 2, prefill_max_token_num : 128
[I][                            Init][ 335]: grp: 3, prefill_max_token_num : 256
[I][                            Init][ 335]: grp: 4, prefill_max_token_num : 512
[I][                            Init][ 335]: grp: 5, prefill_max_token_num : 640
[I][                            Init][ 339]: prefill_max_token_num : 640
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

[I][                            Init][ 348]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
prompt >> who are you
image >>
[I][                          Encode][ 470]: input_ids size: 22
[I][                             Run][ 604]: input token num : 22, prefill_split_num : 1
[I][                             Run][ 619]: prefill grpid 2
[I][                             Run][ 646]: input_num_token:22
[I][                             Run][ 770]: ttft: 128.29 ms
I am an AI language model, I am here to help answer any questions you may have. How can I assist you today?

[N][                             Run][ 879]: hit eos,avg 19.84 token/s

prompt >> describe the image
image >> ./images/ssd_horse.jpg
[I][                          Encode][ 442]: image encode time : 232.04 ms, size : 393216
[I][                          Encode][ 496]: imgs_embed.size() : 1, media token size : 256
[I][                             Run][ 604]: input token num : 280, prefill_split_num : 3
[I][                             Run][ 619]: prefill grpid 4
[I][                             Run][ 646]: input_num_token:128
[I][                             Run][ 646]: input_num_token:128
[I][                             Run][ 646]: input_num_token:24
[I][                             Run][ 770]: ttft: 417.89 ms
In the image, a young man is riding a brown horse in a fenced area. The horse is standing still, and the rider is holding the reins. The horse has a white blaze on its face and is wearing a bridle. The rider is wearing a blue hoodie and jeans. In front of the horse, there is a brown dog standing on the ground, looking up at the horse. In the background, there is a silver pickup truck parked near a fence. There are also some trees and other people visible in the background. The ground is covered with dirt.

[N][                             Run][ 879]: hit eos,avg 19.85 token/s

prompt >> q
```

## Reference

- [FastVLM-0.5B](https://huggingface.co/apple/FastVLM-0.5B)
- [FastVLM-1.5B](https://huggingface.co/apple/FastVLM-1.5B)

## 技术讨论

- Github issues
- QQ 群: 139953715

