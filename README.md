# AX-LLM

![GitHub License](https://img.shields.io/github/license/AXERA-TECH/ax-llm)

| Platform | Build Status |
| -------- | ------------ |
| AX650    | ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-llm/build_650.yml)|

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。

### 分支说明

- [ax-context(default)](https://github.com/AXERA-TECH/ax-llm/tree/ax-context)
  - AX650A/AX650N/AX8850/AX630C Host 运行 LLM 使用
- [ax-internvl](https://github.com/AXERA-TECH/ax-llm/tree/ax-internvl)
  - AX650A/AX650N/AX8850/AX630C Host 运行 InternVL 系列使用
- [axcl-context](https://github.com/AXERA-TECH/ax-llm/tree/axcl-context)
  - AX650N/AX8850 EP 的主控运行 LLM 系列使用
- [axcl-internvl](https://github.com/AXERA-TECH/ax-llm/tree/axcl-internvl)
  - AX650N/AX8850 EP 的主控运行 InternVL 系列使用

### 已支持芯片

- AX650A/AX650N
  - SDK ≥ v3.6.2
- AX630C
  - SDK ≥ v3.0.0

### 已支持模型

- Qwen2.5
- Qwen3
- MiniCPM
- SmolLM2
- Llama3

### 获取地址

我们的 ModelZoo 已迁移到 [Huggingface](https://huggingface.co/AXERA-TECH), 例如：

- [Qwen2.5-7B-Instruct](https://huggingface.co/AXERA-TECH/Qwen2.5-7B-Instruct)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/AXERA-TECH/Qwen2.5-1.5B-Instruct)

## 当前分支（axllm）

本分支统一输出可执行文件名为 `axllm`，根据运行环境自动选择 AX650 片上后端或 AXCL PCIe 后端。

### 安装方式（推荐）

使用根目录的安装脚本：

```shell
./install.sh
```

或使用一行命令下载并执行（默认分支 `axllm`）：

```shell
curl -fsSL https://raw.githubusercontent.com/AXERA-TECH/ax-llm/axllm/install.sh | bash
```

脚本逻辑：

- **AX650 片上后端**
  - 条件：`/proc/ax_proc/board_id` 包含 `AX650` 且本机有 `gcc`
  - 行为：自动下载 BSP（msp_3.6.2）和交叉工具链，编译并安装到 `/usr/bin/axllm`
- **AXCL PCIe 后端**
  - 条件：可运行 `axcl-smi` 且存在 `/usr/include/axcl/` 与 `/usr/lib/axcl/`
  - 行为：使用系统 AXCL 头文件与库编译后端并安装到 `/usr/bin/axllm`

默认从当前仓库 `origin` 拉取 **axllm 分支**（可通过环境变量覆盖）：

```shell
REPO_URL=git@github.com:AXERA-TECH/ax-llm.git BRANCH=axllm ./install.sh
```

卸载：

```shell
./uninstall.sh
```

### 编译方式（手动）

如果需要手动编译，请按后端选择对应脚本：

```shell
# AX650 片上后端
./build_ax650.sh

# AXCL PCIe 后端（x86）
./build_axcl_x86.sh

# AXCL PCIe 后端（aarch64 交叉编译）
./build_axcl_aarch64.sh
```

编译完成后，`build*/install/bin/` 目录下会生成 `axllm`（本分支已统一命名）。

## 源码编译

- clone 本项目（带子模块）
  ```shell
  git clone --recursive https://github.com/AXERA-TECH/ax-llm.git
  cd ax-llm
  ```
- 使用本分支推荐的安装脚本或对应后端脚本编译

## 使用方式

编译/安装后运行：

```shell
axllm
```

如需 API/Gradio 示例，可继续使用 `scripts/` 下的脚本（与分支功能一致）。

## 运行示例
### 命令行对话
```shell
$ axllm run smollm2-360m-ax650/
[I][                            Init][ 127]: LLM init start
tokenizer_type = 1
 97% | ████████████████████████████████  |  34 /  35 [1.98s<2.03s, 17.21 count/s] init post axmodel ok,remain_cmm(11249 MB)
[I][                            Init][ 188]: max_token_len : 2047
[I][                            Init][ 191]: kv_cache_size : 320, kv_cache_num: 2047
[I][                            Init][ 194]: prefill_token_num : 128
[I][                            Init][ 198]: grp: 1, prefill_max_kv_cache_num : 1
[I][                            Init][ 198]: grp: 2, prefill_max_kv_cache_num : 128
[I][                            Init][ 198]: grp: 3, prefill_max_kv_cache_num : 256
[I][                            Init][ 198]: grp: 4, prefill_max_kv_cache_num : 384
[I][                            Init][ 198]: grp: 5, prefill_max_kv_cache_num : 512
[I][                            Init][ 198]: grp: 6, prefill_max_kv_cache_num : 640
[I][                            Init][ 198]: grp: 7, prefill_max_kv_cache_num : 768
[I][                            Init][ 198]: grp: 8, prefill_max_kv_cache_num : 896
[I][                            Init][ 198]: grp: 9, prefill_max_kv_cache_num : 1024
[I][                            Init][ 203]: prefill_max_token_num : 1024
[I][                            Init][  27]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  35 /  35 [1.98s<1.98s, 17.70 count/s] embed_selector init ok
[I][                     load_config][ 282]: load config: 
{
    "enable_repetition_penalty": false,
    "enable_temperature": false,
    "enable_top_k_sampling": false,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 224]: LLM init ok
Type "q" to exit
Ctrl+c to stop current running
"reset" to reset kvcache
"dd" to remove last conversation.
"pp" to print history.
----------------------------------------
prompt >> hello,my name is Allen
[I][                      SetKVCache][ 357]: prefill_grpid:2 kv_cache_num:128 precompute_len:0 input_num_token:27
[I][                      SetKVCache][ 359]: current prefill_max_token_num:1024
[I][                      SetKVCache][ 360]: first run
[I][                             Run][ 412]: input token num : 27, prefill_split_num : 1
[I][                             Run][ 474]: ttft: 177.20 ms
Hello, Allen. How can I assist you today?

[N][                             Run][ 554]: hit eos,avg 26.19 token/s

[I][                      GetKVCache][ 331]: precompute_len:38, remaining:986
prompt >> 
```

### 服务(兼容 OpenAI API)

```shell
$ axllm serve smollm2-360m-ax650/
[I][                            Init][ 127]: LLM init start
tokenizer_type = 1
 97% | ████████████████████████████████  |  34 /  35 [1.99s<2.05s, 17.09 count/s] init post axmodel ok,remain_cmm(11249 MB)
[I][                            Init][ 188]: max_token_len : 2047
[I][                            Init][ 191]: kv_cache_size : 320, kv_cache_num: 2047
[I][                            Init][ 194]: prefill_token_num : 128
[I][                            Init][ 198]: grp: 1, prefill_max_kv_cache_num : 1
[I][                            Init][ 198]: grp: 2, prefill_max_kv_cache_num : 128
[I][                            Init][ 198]: grp: 3, prefill_max_kv_cache_num : 256
[I][                            Init][ 198]: grp: 4, prefill_max_kv_cache_num : 384
[I][                            Init][ 198]: grp: 5, prefill_max_kv_cache_num : 512
[I][                            Init][ 198]: grp: 6, prefill_max_kv_cache_num : 640
[I][                            Init][ 198]: grp: 7, prefill_max_kv_cache_num : 768
[I][                            Init][ 198]: grp: 8, prefill_max_kv_cache_num : 896
[I][                            Init][ 198]: grp: 9, prefill_max_kv_cache_num : 1024
[I][                            Init][ 203]: prefill_max_token_num : 1024
[I][                            Init][  27]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  35 /  35 [1.99s<1.99s, 17.58 count/s] embed_selector init ok
[I][                     load_config][ 282]: load config: 
{
    "enable_repetition_penalty": false,
    "enable_temperature": false,
    "enable_top_k_sampling": false,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 224]: LLM init ok
Starting server on port 8000 with model 'AXERA-TECH/SmolLM2-360M-Instruct'...
OpenAI API Server starting on http://0.0.0.0:8000
Max concurrency: 1
Models: AXERA-TECH/SmolLM2-360M-Instruct
```
### 测试 OpenAI API
```shell
python scripts/openai_demo.py --model AXERA-TECH/SmolLM2-360M-Instruct --api_url http://127.0.0.1:8000/v1
```

## Reference

- [Qwen](https://huggingface.co/Qwen)

## 技术讨论

- Github issues
- QQ 群: 139953715
