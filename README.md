# AX-LLM

![GitHub License](https://img.shields.io/github/license/AXERA-TECH/ax-llm)

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。


### 已支持芯片

- AX650A/AX650N
  - SDK ≥ v3.6.2
- AX630C
  - SDK ≥ v3.0.0

### 已支持模型

#### LLM
- Qwen2.5
- Qwen3
- MiniCPM
- SmolLM2
- Llama3
- HY-MT1.5-1.8B
- ...

#### VLM（多模态）
- Qwen3-VL-2B-Instruct
- Qwen3.5-2B
- Qwen3-VL-Embedding-2B（Embedding，多模态）
- SmolVLM2-500M-Video-Instruct
- FastVLM-1.5B-GPTQ-Int4
- InternVL3_5-1B-GPTQ-INT4
- PaddleOCR-VL-1.5
- ...

### 获取地址

我们的 ModelZoo 已迁移到 [Huggingface](https://huggingface.co/AXERA-TECH)

## 当前分支（axllm）

本分支统一输出可执行文件名为 `axllm`，根据运行环境自动选择 AX650 片上后端或 AXCL PCIe 后端。

### 安装方式（推荐）

使用根目录的安装脚本：

```shell
./install.sh
```

Windows + AXCL + MinGW64 可使用：

```bat
install.bat
```

或使用一行命令下载并执行（默认分支 `axllm`）：

```shell
curl -fsSL https://raw.githubusercontent.com/AXERA-TECH/ax-llm/axllm/install.sh | bash
```

脚本逻辑：

- **AX650 片上后端**
  - 条件：`/proc/ax_proc/board_id` 包含 `AX650` 且本机有 `gcc`
  - 行为：自动下载 BSP（msp_3.6.2），编译并安装到 `/usr/bin/axllm`
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

Windows 本地安装可使用：

```bat
uninstall.bat
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

Windows 下编译 AXCL 后端时，请显式关闭 `BUILD_AX650`，并传入 `AXCL_DIR`：

```powershell
# MinGW64
cmake -S . -B build_windows_mingw -G "MinGW Makefiles" `
  -DBUILD_AX650=OFF -DBUILD_AXCL=ON `
  -DAXCL_DIR="C:\Program Files\AXCL\axcl\out\axcl_win_x64"
cmake --build build_windows_mingw --parallel

# MSVC
cmake -S . -B build_windows_msvc -G "Visual Studio 17 2022" -A x64 `
  -DBUILD_AX650=OFF -DBUILD_AXCL=ON `
  -DAXCL_DIR="C:\Program Files\AXCL\axcl\out\axcl_win_x64"
cmake --build build_windows_msvc --config Release --parallel
```

## 使用方式

编译/安装后运行：

```shell
axllm
```

如需 API/Gradio 示例，可继续使用 `scripts/` 下的脚本（与分支功能一致）。

### Docker（CI 导出镜像）

本仓库提供 GitHub Actions 工作流 `Docker Images` 用于构建并导出 Docker 镜像（按后端/架构拆分），产物为 `.tar.gz`：

- `docker-axcl-amd64`：AXCL（PCIe）x86_64 Host
- `docker-axcl-arm64`：AXCL（PCIe）aarch64 Host
- `docker-ax650-arm64`：AX650（板端）aarch64

下载 artifact 后加载镜像：

```shell
docker load -i axllm-axcl-amd64.tar.gz
```

如需将镜像推送到 GHCR，可在 Actions 手动触发 `Docker Images` 并将 `publish_ghcr=true`，随后可通过 `ghcr.io/<owner>/` 拉取：

```shell
docker pull ghcr.io/<owner>/axllm-axcl-amd64:<sha>
```

AXCL 运行示例（Host 已安装 AXCL Driver，需透传设备节点；最简单可用 `--privileged`）：

```shell
docker run --rm --network host --privileged \
  -v /path/to/models:/models \
  axllm-axcl-amd64:<sha> serve /models/<model_dir> --port 8000
```

AX650 运行示例（板端，需挂载 `/soc` 提供运行库）：

```shell
docker run --rm --network host --privileged \
  -v /soc:/soc:ro \
  -e LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH \
  -v /path/to/models:/models \
  axllm-ax650-arm64:<sha> serve /models/<model_dir> --port 8000
```

### VLM 使用说明

- 使用 VLM 模型目录运行 `axllm run <vlm_model_path>`
- 每轮输入 `prompt` 后，会提示 `image >>`
  - 直接回车：本轮仅文本对话
  - 输入图片路径：图文对话
  - 输入 `video:<frames_dir>`：视频/多帧对话（按文件名排序读取帧）
  - 输入 `video:<video_file>[:<fps>]`：单个视频文件按采样 FPS 均匀抽帧，默认 `fps=2`，显式 `fps` 支持正整数或正小数

VLM 模型的 `config.json` 需包含（或等价字段）：

- `vlm_type`
- `filename_image_encoder_axmodel`
- `vision_patch_size`

可选字段（建议保留，未配置时会自动从视觉编码模型输入形状推断）：

- `vision_width`
- `vision_height`
- `full_attention_interval`（Qwen3.5 等混合 linear/full attention 模型需要，用于标记每 N 层为 full-attention）

### Embedding（/v1/embeddings）使用说明

`axllm` 支持在 `serve` 模式下加载 Embedding 模型，并提供 OpenAI 兼容的 `/v1/embeddings` 接口（Embedding 模型 **不支持** `run` 交互模式）。

同时提供 llama.cpp 兼容的 Embedding 路径：

- `POST /embedding`：单条 embedding（llama.cpp 风格，返回 `{ "embedding": [...], "model": "..." }`）
- `POST /embeddings`：批量 embedding（llama.cpp 风格，返回 `[{ "index": 0, "embedding": [...] }, ...]`）

1) 在 Embedding 模型目录的 `config.json` 中启用开关：

```json
{
  "is_embedding": true
}
```

2) 启动服务：

```shell
axllm serve <embedding_model_dir> --port 8000
```

启动后会在终端打印本机可访问的完整 API URL（包含 `127.0.0.1` 与本机网卡 IP）。

3) 调用示例：

```shell
# 健康检查
curl -s http://127.0.0.1:8000/health

# 获取已注册模型列表
curl -s http://127.0.0.1:8000/v1/models

# 生成 embedding（input 支持 string 或 string 数组）
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model_name>","input":["hello","world"]}'
```

也可参考 Python 示例：`python3 scripts/openai_embedding_demo.py --model <model_name> --api_url http://127.0.0.1:8000/v1`（需先 `pip install openai`）。

#### 多模态 Embedding（messages 扩展）

对于支持视觉输入的 Embedding 模型（如 `Qwen3-VL-Embedding-2B`），`/v1/embeddings` 额外支持传入 `messages`（格式与 `/v1/chat/completions` 相同），从而实现图文 embedding：

```shell
python3 scripts/openai_vl_embedding_demo.py \
  --model <model_name> \
  --api_url http://127.0.0.1:8000/v1 \
  --image /path/to/image.jpg \
  --text "Describe this image"
```

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

纯文本对话：
```shell
python scripts/openai_demo.py --model AXERA-TECH/SmolLM2-360M-Instruct --api_url http://127.0.0.1:8000/v1
```

自定义 prompt：
```shell
python scripts/openai_demo.py --model AXERA-TECH/SmolLM2-360M-Instruct --prompt "请介绍一下你自己"
```

图文对话（VLM，图片会自动 base64 编码发送）：
```shell
python scripts/openai_demo.py --model AXERA-TECH/Qwen3-VL-2B-Instruct --image /path/to/image.jpg --prompt "描述一下这张图片"
```

音频对话（Gemma4，当前每条消息支持单个音频文件）：
```shell
python scripts/openai_demo.py --model AXERA-TECH/gemma-4-E2B-it --audio /path/to/audio.wav --prompt "Transcribe the speech in its original language. Output only the transcription."
```

音频转写（OpenAI Audio API）：
```shell
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "model=AXERA-TECH/gemma-4-E2B-it" \
  -F "file=@/path/to/audio.wav" \
  -F "response_format=json"
```

音频翻译（OpenAI Audio API）：
```shell
curl http://127.0.0.1:8000/v1/audio/translations \
  -F "model=AXERA-TECH/gemma-4-E2B-it" \
  -F "file=@/path/to/audio.wav" \
  -F "response_format=text"
```

`response_format` 目前支持 `json`、`text`、`srt`、`vtt`。

> **参数说明**
> | 参数 | 说明 | 默认值 |
> |------|------|--------|
> | `--model` | 模型名称（必填） | — |
> | `--api_url` | API 地址 | `http://127.0.0.1:8000/v1` |
> | `--prompt` | 用户 prompt 文本 | `hello` |
> | `--image` | 图片路径（可选，VLM 模式） | — |
> | `--audio` | 音频路径（可选，Gemma4 模式） | — |
>
> 当指定 `--image` 或 `--audio` 时，脚本会将媒体读取并以 `data:...;base64,...` 格式嵌入请求，服务端会自动解码为临时文件供对应编码器使用。

## 技术讨论

- Github issues
- QQ 群: 139953715
