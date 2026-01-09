# AX-LLM

![GitHub License](https://img.shields.io/github/license/AXERA-TECH/ax-llm)

| Platform | Build Status |
| -------- | ------------ |
| AX650    | ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-llm/build_650.yml?internvl2)|

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。

### 已支持芯片

- AX650A/AX650N
  - SDK ≥ v1.45.0_P31

### 已支持模型

- AXERA-TECH/InternVL3_5-1B_GPTQ_INT4

## 源码编译

-  clone 本项目  
    ```shell
    git clone -b ax-internvl-3_5 https://github.com/AXERA-TECH/ax-llm.git
    cd ax-llm
    ```
- clone `ax650n_bsp_sdk` 仓库 (可选)  
    ```shell
    git cloen https://github.com/AXERA-TECH/ax650n_bsp_sdk
    ```
- 仔细阅读 `build.sh` ，并在 `build.sh` 正确修改 `BSP_MSP_DIR` 变量后(该变量表示`ax650n_bsp_sdk`代码位置)，运行编译脚本  
    ```shell
    ./build.sh
    ```
- 正确编译后，`build_650/install/bin` 目录，应有以下文件（百度网盘中有预编译的可执行程序）
  ```
  $ tree install/bin/
    .
    ├── gradio_demo.py
    ├── internvl3-5-1b_tokenizer.txt
    ├── main
    ├── main_api
    ├── openai_cli.py
    ├── post_config.json
    ├── qwen3_tokenizer.py
    └── run_internvl_3-5_1b_448_ax650.sh

    0 directories, 8 files
  ```
  
## 运行示例

### 1. 图像理解

![panda.jpg](assets/image_1.jpg)

1) 将 `build_650/install/bin` 目录下的文件和编译好的模型都拷贝到爱芯板子上, 可以直接从 [AXERA-TECH/InternVL3_5-1B_GPTQ_INT4](https://huggingface.co/AXERA-TECH/InternVL3_5-1B_GPTQ_INT4) 下载.
2) 运行 `run_internvl_3-5_1b_448_ax650.sh`  

```sh
root@ax650 ~/yongqiang/push_hugging_face/InternVL3_5-1B_GPTQ_INT4 # ./run_internvl_3-5_1b_448_ax650.sh
[I][                            Init][ 135]: LLM init start
[I][                            Init][ 137]: Total CMM:7915 MB
tokenizer_type = 3
  3% | ██                                |   1 /  31 [0.71s<21.92s, 1.41 count/s] tokenizer init ok[I][                            Init][  26]: LLaMaEmbedSelector use mmap
  6% | ███                               |   2 /  31 [0.71s<11.05s, 2.81 count/s] embed_selector init ok[I][                            Init][ 182]: attr.axmodel_num:28
100% | ████████████████████████████████ |  31 /  31 [2.06s<2.06s, 15.03 count/s] init post axmodel ok,remain_cmm(6940 MB)[I][                            Init][ 240]: image encoder feature outputs:0
103% | ██████████████████████████████████ |  32 /  31 [2.32s<2.25s, 13.79 count/s] init vpm axmodel ok,remain_cmm(6588 MB)[I][                            Init][ 280]: image encoder input nhwc@uint8
[I][                            Init][ 305]: image encoder output float32

[I][                            Init][ 335]: max_token_len : 2047
[I][                            Init][ 340]: kv_cache_size : 1024, kv_cache_num: 2047
[I][                            Init][ 348]: prefill_token_num : 128
[I][                            Init][ 352]: grp: 1, prefill_max_token_num : 1
[I][                            Init][ 352]: grp: 2, prefill_max_token_num : 128
[I][                            Init][ 352]: grp: 3, prefill_max_token_num : 256
[I][                            Init][ 352]: grp: 4, prefill_max_token_num : 384
[I][                            Init][ 352]: grp: 5, prefill_max_token_num : 512
[I][                            Init][ 352]: grp: 6, prefill_max_token_num : 640
[I][                            Init][ 352]: grp: 7, prefill_max_token_num : 768
[I][                            Init][ 352]: grp: 8, prefill_max_token_num : 896
[I][                            Init][ 352]: grp: 9, prefill_max_token_num : 1024
[I][                            Init][ 356]: prefill_max_token_num : 1024
[I][                     load_config][ 281]: load config:
{
    "enable_repetition_penalty": true,
    "enable_temperature": true,
    "enable_top_k_sampling": true,
    "enable_top_p_sampling": false,
    "penalty_window": 30,
    "repetition_penalty": 1.2,
    "temperature": 0.7,
    "top_k": 10,
    "top_p": 0.9
}

[I][                            Init][ 373]: LLM init ok
[I][                            Init][ 375]: Left CMM:6588 MB
Type "q" to exit, Ctrl+c to stop current running
prompt(输入q退出) >> 介绍一下你自己
image(回车键跳过) >>
[I][                             Run][ 713]: input token num : 21, prefill_split_num : 1
[I][                             Run][ 747]: input_num_token:21
[I][                             Run][ 976]: ttft: 83.79 ms
我被称为"语言模型-1.0"，来自上海人工智能实验室。我的开发团队致力于为用户提供高效、准确和个性化的AI服务。作为一款先进的自然语言处理（NLP）模型，我旨在帮助用户解决各种语言相关问题，并提供有用的信息和建议。我的设计目标是能够以自然流畅的方式与人类进行交互，无论是回答问题、提供建议还是执行任务。

[N][                             Run][1102]: hit eos,avg 19.79 token/s

prompt(输入q退出) >> 请你详细描述下面这幅图
image(回车键跳过) >> assets/image_1.jpg
[I][                     EncodeImage][ 481]: image encode time : 408.467987 ms, size : 1
[I][                          Encode][ 636]: input_ids size:284
[I][                          Encode][ 644]: offset 15
[I][                          Encode][ 673]: img_embed.size:1, 262144
[I][                          Encode][ 689]: out_embed size:290816
[I][                          Encode][ 690]: input_ids size 284
[I][                          Encode][ 692]: position_ids size:284
[I][                             Run][ 713]: input token num : 284, prefill_split_num : 3
[I][                             Run][ 747]: input_num_token:128
[I][                             Run][ 747]: input_num_token:128
[I][                             Run][ 747]: input_num_token:28
[I][                             Run][ 976]: ttft: 270.76 ms
这是一幅生动的图片，展示了一只大熊猫正在自然环境中觅食的情景。画面中，大熊猫正低头在植物丛中寻找食物。它的毛发呈白色，背部和腹部有黑色斑点。周围绿意盎然，各种灌木和植物环绕着它，显得生机勃勃。背景的木质结构可能是一把竹竿或长椅，进一步暗示这可能是动物园或野生动物保护区。整个场景充满了自然的气息，让人感受到大自然的可爱与生机。

[N][                             Run][1102]: hit eos,avg 19.86 token/s

prompt(输入q退出) >>

```

## Reference

- [OpenGVLab/InternVL3_5-1B](https://huggingface.co/OpenGVLab/InternVL3_5-1B)

## 技术讨

- Github issues
- QQ 群: 139953715
