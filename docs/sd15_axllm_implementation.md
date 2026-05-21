# AXLLM 支持 SD1.5 生图的实现说明

本文说明本分支中，`axllm serve` 是如何在 AX650 上以纯 C++ 方式运行
基于 SD1.5 的 `lcm-lora-sdv1-5` 模型，并通过 OpenAI 兼容图片接口提供给
前端 WebUI 使用的。

本次实现主要落在以下文件：

- `src/runner/image/sd15_image_generator.cpp`
- `src/runner/image/sd15_image_generator.hpp`
- `src/main.cpp`
- `src/runner/ax_model_runner/ax_model_runner_ax650.cpp`
- `third_party/openai-api.cpp` 现有的 OpenAI 接口路由逻辑

## 1. 目标

目标是支持如下启动方式：

```bash
axllm serve /path/to/lcm-lora-sdv1-5 --port 18100
```

启动后，对外提供：

- `GET /v1/models`
- `POST /v1/images/generations`
- `POST /v1/images/edits`

并且在图片生成阶段不再调用外部 Python 推理脚本，而是全部在 C++ 中完成。

## 2. AXLLM 假设的模型仓库结构

当前实现假设图片模型仓库根目录下可以包含多套 variant 子目录。推荐在
`config.json` 中显式声明 `image_variants`：

```json
{
  "image_variants": [
    {
      "id": "lcm-lora-sdv1-5-512x512",
      "dir": "models",
      "size": "512x512",
      "chip": "ax650",
      "supports_img2img": true
    },
    {
      "id": "lcm-lora-sdv1-5-768x1024",
      "dir": "models_1024x768",
      "size": "768x1024",
      "chip": "ax650",
      "supports_img2img": false
    }
  ]
}
```

典型仓库结构如下：

```text
lcm-lora-sdv1-5/
├── config.json
├── models/
│   ├── text_encoder/sd15_text_encoder_sim.axmodel
│   ├── tokenizer/{vocab.json, merges.txt, ...}
│   ├── unet.axmodel
│   ├── vae_decoder.axmodel
│   ├── vae_encoder.axmodel
│   ├── time_input_txt2img.npy
│   └── time_input_img2img.npy
└── models_1024x768/
    ├── text_encoder/sd15_text_encoder_sim.axmodel
    ├── tokenizer/{vocab.json, merges.txt, ...}
    ├── unet.axmodel
    ├── vae_decoder.axmodel
    └── time_input_txt2img.npy
```

代码会优先读取 `image_variants`。如果旧仓库没有这个字段，则回退到内置候选目录表。
每个 variant 是否真的可用，不只靠目录名判断，而是会用真实 `axmodel` 和真实 tensor
shape 进一步校验。

## 3. 推理链路总览

### 3.1 txt2img

`txt2img` 的 C++ 推理流程如下：

1. 用 CLIP BPE tokenizer 对 prompt 做编码
2. 运行 `text_encoder`
3. 生成初始 latent noise
4. 运行 4 步 LCM denoise loop，核心模型是 `unet`
5. 对 latent 做 `1 / 0.18215` 缩放
6. 运行 `vae_decoder`
7. 编码为 PNG，并按 OpenAI 接口格式返回

### 3.2 img2img

`img2img` 的 C++ 推理流程如下：

1. 读取前端上传的图片字节流
2. resize，并转成 NCHW、归一化到 `[-1, 1]`
3. 运行 `vae_encoder`
4. 对编码后的 latent 采样，再注入噪声
5. 运行 denoise loop
6. 运行 `vae_decoder`
7. 编码 PNG 并返回

## 4. 为什么要单独实现 `sd15_image_generator`

AXLLM 原本的核心路径是 `LLM.cpp`，主要围绕：

- tokenizer
- token embedding
- KV cache
- decoder group

这一套结构是为 LLM/VLM 设计的，不适合直接塞进 SD1.5 这种 diffusion pipeline。

所以这里单独实现了图片生成路径：

- `sd15::ImageGenerator`：抽象接口
- `Sd15ImageGenerator`：SD1.5 的具体实现
- `src/main.cpp`：在 server 模式下识别图片模型，并把 OpenAI 图片接口回调注册进去

这样图片模型和文本 LLM 路径是分离的，互不污染。

## 5. OpenAI 图片接口是如何接进来的

在 `src/main.cpp` 的 `run_server_mode()` 里：

1. 调用 `create_image_generator()`
2. 执行 `generator->init(config.image_model_dir, err)`，扫描可用图片 variant
3. 把每个真实 variant 注册成一个 OpenAI model id

当前实际暴露的模型 id 例如：

- `lcm-lora-sdv1-5-512x512`
- `lcm-lora-sdv1-5-768x1024`

这里有一个明确的设计决策：

- 不再暴露 `dall-e-2`、`dall-e-3` 这种占位别名
- `/v1/models` 只返回真实可运行的模型

这样可以和前端 “fetch 真实模型列表并选择其中一个模型通信” 的行为保持一致。

## 6. 为什么必须自己实现 CLIP tokenizer

最开始，SD1.5 路径复用了 AXLLM 通用 tokenizer 方案，并导出了一个
`tokenizer.txt`。这样做虽然能跑，但 prompt token id 和 HuggingFace
`CLIPTokenizer` 实际输出不一致。

典型现象是：

- API 返回成功
- 但图像只有马赛克、纹理块，没有语义内容

根因是：

- SD1.5 的 text encoder 必须严格使用 CLIP BPE 分词
- 通用 LLM tokenizer 行为和 CLIP tokenizer 不是一回事

所以最终在 `sd15_image_generator.cpp` 里新增了本地
`ClipBPETokenizer`，直接读取：

- `tokenizer/vocab.json`
- `tokenizer/merges.txt`

在 C++ 中完成与 Python CLIP tokenizer 对齐的编码。

修完后，C++ 和 Python 参考实现的 prompt ids 才完全一致。

## 7. 为了让 SD1.5 在 AX650 上稳定运行，对 runner 做了什么适配

### 7.1 推理前后做 cache sync

SD1.5 的 `axmodel` 路径要求：

- 写入 input buffer 后，推理前必须 flush
- 读取 output buffer 前，推理后必须 invalidate

因此图片路径显式开启了：

- `set_auto_sync_before_inference(true)`
- `set_auto_sync_after_inference(true)`

作用对象包括：

- text encoder
- unet
- vae decoder
- vae encoder

### 7.2 按 tensor 实际字节数推断 dtype

虽然模型目录里仍保留有 dtype 配置，但在 AX650 上，真正应该信的是：

- runtime 暴露出来的 tensor buffer 大小
- 逻辑元素个数

因此代码里新增了按 tensor 大小反推 dtype 的逻辑，支持：

- `fp32`
- `fp16`
- `bf16`

对当前这套 650 上的 SD1.5 模型，最终验证到的大部分关键 tensor 都是 `fp32`。

## 8. 导致“马赛克图”的 scheduler bug

这次最关键的一个问题是 scheduler 常量写错了。

Python 参考实现是：

```python
betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
```

但最初的 C++ 写法把：

- `0.00085`
- `0.012`

直接当成了已经开方后的值。

这会导致整条 `alphas_cumprod` 曲线都错掉。于是即使：

- prompt ids 正确
- prompt embeddings 正确
- `unet` 单步输出也正确

最终 latent 轨迹还是会跑偏，图像就会变成马赛克。

修正方式是：

```cpp
constexpr float kBetaStartSqrt = 0.029154759f;  // sqrt(0.00085)
constexpr float kBetaEndSqrt = 0.109544512f;    // sqrt(0.012)
```

修正后，C++ 的轨迹才能和 Python 参考实现对齐。

## 9. 为什么 `models_1024x768` 最终暴露成 `768x1024`

另一个图像错误来自尺寸解释错误。

目录名是 `models_1024x768`，但真实 `vae_decoder` 输出 shape 是：

```text
[1, 3, 1024, 768]
```

也就是说：

- height = `1024`
- width = `768`

按照 OpenAI 风格的 `WxH` 表示法，正确暴露应该是：

- `768x1024`

如果错误地把它暴露成 `1024x768`，后续在 C++ 解码图片时就会按错误的宽高去解释
tensor，最后出现横向重复、条纹、拼接错乱等问题。

因此当前实现优先从 `vae_decoder` 的真实输出 shape 推断最终尺寸，只把目录名映射
作为 fallback。

## 10. 为什么后来改成“按需加载 variant”

在 AX650 上，如果把两套 SD1.5 variant 同时常驻在内存中，运行时容易出现不稳定：

- 启动能成功
- `/v1/models` 能返回
- 但真正请求时 `text_encoder` 可能报 `ret=0x8006008a`

这个问题本质上是板端运行时资源压力导致的，不是 OpenAI 接口本身的问题。

为了同时满足：

- 一个端口暴露多个真实模型 id
- 650 上尽量稳定

当前实现改成了：

1. 启动时扫描所有候选 variant
2. 每个 variant 启动时先完整校验一遍，得到元数据
3. 常驻保存的只有：
   - `ImageModelVariant`
   - variant 对应目录路径
   - fallback size 信息
4. 真正收到请求时，再按 `model id` 加载当前需要的 runtime
5. 切换到另一个模型时，替换掉上一个 active runtime

这样可以保留：

- `/v1/models` 同时列出多个真实模型

同时避免：

- 多套大模型 runtime 长期并驻带来的 650 不稳定问题

代价是：

- 第一次切换到另一个模型 id 时，会比同模型连续请求更慢

这是当前版本有意接受的折中。

## 11. 当前能力矩阵

当前已经验证的能力如下：

| Model ID | txt2img | img2img | 说明 |
|---|---|---|---|
| `lcm-lora-sdv1-5-512x512` | 支持 | 支持 | 有 `vae_encoder.axmodel` |
| `lcm-lora-sdv1-5-768x1024` | 支持 | 不支持 | 当前仓库未提供对应 `vae_encoder.axmodel` |

对于不支持 `img2img` 的 variant，AXLLM 不会崩，而是返回明确错误：

```json
{
  "error": {
    "code": "image_request_error",
    "message": "img2img is not supported for this model variant",
    "type": "image_request_error"
  }
}
```

## 12. 典型请求示例

### 12.1 拉取模型列表

```bash
curl http://127.0.0.1:18100/v1/models
```

### 12.2 512x512 txt2img

```bash
curl -X POST http://127.0.0.1:18100/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lcm-lora-sdv1-5-512x512",
    "prompt": "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "size": "512x512",
    "n": 1,
    "response_format": "b64_json",
    "seed": 0
  }'
```

### 12.3 512x512 img2img

```bash
curl -X POST http://127.0.0.1:18100/v1/images/edits \
  -F model=lcm-lora-sdv1-5-512x512 \
  -F prompt='Astronauts in a jungle, cold color palette, muted colors, detailed, 8k' \
  -F size=512x512 \
  -F response_format=b64_json \
  -F image=@init.png
```

### 12.4 768x1024 txt2img

```bash
curl -X POST http://127.0.0.1:18100/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lcm-lora-sdv1-5-768x1024",
    "prompt": "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "size": "768x1024",
    "n": 1,
    "response_format": "b64_json",
    "seed": 0
  }'
```

## 13. 当前限制

1. 新仓库建议使用 `config.json/image_variants` 显式声明 variant；旧仓库仍通过固定候选目录表兼容。
2. `img2img` 是否支持，取决于该 variant 是否提供 `vae_encoder.axmodel`。
3. 多模型切换现在稳定性更好，但首次切换有加载开销。
4. 当前这条 SD1.5 路径主要针对 AX650 片上运行做了验证，AXCL 不是这次主验证目标。

## 14. 这次实际上新增了什么

如果用一句话概括，这个分支中 AXLLM 对 SD1.5 的支持，实际是由下面几部分组成的：

- 一条独立的 C++ 图片生成 pipeline
- CLIP 兼容 tokenizer
- AX650 runtime 的 cache sync 和 dtype 处理
- `axllm serve` 中的 OpenAI 图片接口接线
- 只暴露真实模型的 `/v1/models`
- 面向 650 稳定性的按需加载 variant 机制

也就是说，这个分支里“AXLLM 支持 SD1.5 推理”并不是简单包一层 Python，而是已经把
核心推理链路接进了 C++ 和 `axllm serve`。
