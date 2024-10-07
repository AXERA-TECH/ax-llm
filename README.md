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

### LLM 编译器相关

- [Pulsar2 llm build](https://pulsar2-docs.readthedocs.io/zh-cn/latest/appendix/build_llm.html)
  - 默认 w8a16 量化
  - 支持从 Huggingface 仓库直转
  - 支持自定义 prompt 仿真运行

### 速度评估

- [Benchmark](benchmark/) 常见开源大模型推理耗时统计，基于 *AXera-Pi Pro* 、*AXera-Pi 2* 实测。

### 已验证过的模型

- Qwen1.5-0.5B/1.8B/4B
- Qwen2-0.5B/1.5B/4B
- Qwen2.5-0.5B/1.5B/3B
- ChatGLM3-6B
- MiniCPM-1B/2B
- MiniCPM-V 2.0
- TinyLLaMa-1.1B
- Llama2-7B
- Llama3-8B
- Llama3.2-1B/3B
- Phi-2
- Phi-3-mini
- SmolLM-135M/360M
- OpenBuddy-3B

### 获取地址

- 百度网盘
  - [AX650N](https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e)
  - [AX630C](https://pan.baidu.com/s/1X0aJTQM0bl8wsraspHnDUw?pwd=ifg5)
- [Google Drive](https://drive.google.com/drive/folders/1i8xdD2PWDlueouds6F1dhMc72n3v_aER?usp=sharing)

## 源码编译

- 递归 clone 本项目，确保所有 `submodule` 正确 clone
    ```shell
    git clone --recursive https://github.com/AXERA-TECH/ax-llm.git
    cd ax-llm
    ```
- 仔细阅读 `build.sh` ，并在 `build.sh` 正确修改 `BSP_MSP_DIR` 变量后，运行编译脚本
    ```shell
    ./build.sh
    ```
- 正确编译后，`build/install/bin` 目录，应有以下文件（百度网盘中有预编译的可执行程序）
  ```
  $ tree install/bin/
    install/bin/
    ├── main
    ├── run_bf16.sh
    └── run_qwen_1.8B.sh
  ```
  
## 运行示例

### Qwen2.5-0.5B

```shell
root@ax650:llm-test/qwen2.5-0.5B# ./run_qwen2.5_0.5B_prefill_ax650.sh
[I][                            Init][ 125]: LLM init start
  3% | ██                                |   1 /  27 [0.27s<7.16s, 3.77 count/s] tokenizer init ok[I][                            Init][  26]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  27 /  27 [2.64s<2.64s, 10.21 count/s] init post axmodel ok,remain_cmm(11442 MB)[I][                            Init][ 241]: max_token_len : 1023
[I][                            Init][ 246]: kv_cache_size : 128, kv_cache_num: 1023
[I][                            Init][ 254]: prefill_token_num : 128
[I][                            Init][ 263]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
>> 你是谁
[I][                             Run][ 484]: ttft: 125.02 ms
我是来自阿里云的超大规模语言模型，我叫通义千问。
[N][                             Run][ 623]: hit eos,avg 29.29 token/s

>> Translate to English：天气预报说今天下午会下雨，请带上雨伞
[I][                             Run][ 484]: ttft: 124.31 ms
The weather forecast says it will rain today, so bring an umbrella.
[N][                             Run][ 623]: hit eos,avg 29.97 token/s
```

### MiniCPM-V 2.0
This sample need use the minicpm-v branch.
```shell
root@ax650:/llm-test/minicpm-v-2.0# ./run_minicpmv.sh
[I][                            Init][ 125]: LLM init start
  2% | █                                 |   1 /  44 [0.21s<9.28s, 4.74 count/s] tokenizer init ok[I][                            Init][  26]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  44 /  44 [35.30s<35.30s, 1.25 count/s] init vpm axmodel ok,remain_cmm(8037 MB)[I][                            Init][ 284]: max_token_len : 1023
[I][                            Init][ 289]: kv_cache_size : 2304, kv_cache_num: 1023
[I][                            Init][ 297]: prefill_token_num : 128
[I][                            Init][ 306]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
prompt >> 图片中有什么
image >> pig.jpg
[I][                          Encode][ 365]: image encode time : 784.114014 ms
[I][                             Run][ 589]: ttft: 523.50 ms
这幅图片展示了一只小猪站在一个色彩斑斓的户外场景中。这只小猪有着棕色、白色和黑色的毛皮，看起来像是一只家养的猪。它站在一片郁郁葱葱的绿色草地上，背景中有各种物品，包括椅子、雨伞和玩具等，暗示着一个花园或户外娱乐区。小猪周围摆放着各种物品，包括一个看起来像是雨伞的粉红色物品，以及一些看起来像是玩具的物品，可能是沙滩玩具，因为它们有着类似的设计。整个场景给人一种轻松、休闲的感觉，小猪似乎在享受它的户外时光。

[N][                             Run][ 728]: hit eos,avg 5.41 token/s
```

## Reference

- [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)
- [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)
- [MiniCPM-V 2.0](https://huggingface.co/openbmb/MiniCPM-V-2)

## 技术讨论

- Github issues
- QQ 群: 139953715

