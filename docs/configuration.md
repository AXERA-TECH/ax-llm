# 配置文件说明（模型目录 `config.json`）

`axllm` 启动时读取 `<model_dir>/config.json`。下面按用途分组列出所有可用字段;**未列为必填的都是可选**,不填用默认值。

> 路径类字段(`filename_*` / `template_filename_axmodel` / `post_config_path` 等)相对模型目录解析。

## 必填

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_name` | string | 模型名(显示在 `/v1/models`、日志) |
| `tokenizer_type` | string | 分词器类型(如 `Qwen3` / `Qwen3VL` / `Gemma4VL` / `SmolLM2` …) |
| `url_tokenizer_model` | string | 分词器文件路径或 HTTP 地址 |
| `template_filename_axmodel` | string | 每层 axmodel 文件名模板,含 `%d`(如 `qwen3_p128_l%d_together.axmodel`) |
| `axmodel_num` | int | transformer 层数 |
| `filename_post_axmodel` | string | 输出 logits 的 post axmodel |
| `filename_tokens_embed` | string | token embedding 权重(bf16 bin) |
| `tokens_embed_num` | int | 词表大小 |
| `tokens_embed_size` | int | embedding 维度 |

## 通用 / 分词

| 字段 | 默认 | 说明 |
|---|---|---|
| `system_prompt` | 空 | 缺省系统提示词,缺失时自动前置 |
| `post_config_path` | `post_config.json` | 采样配置文件 |
| `bos` / `eos` | `true` / `false` | 是否加 BOS/EOS |
| `pad_token_id` | 0 | pad token id |
| `thinking_mode` / `enable_thinking` | 模型默认 | 推理(思考)模式开关,用于 Qwen3 / MiniCPM5 等带思考链的模型;也可在 `/v1/chat/completions` 请求里按次覆盖 |

## 加载与内存

| 字段 | 默认 | 说明 |
|---|---|---|
| `use_mmap_load_embed`(别名 `b_use_mmap_load_embed`) | `false` | embedding 用 mmap 加载(省内存) |
| `use_mmap_load_layer`(别名 `b_use_mmap_load_layer`) | `true`(仅 AX650) | 层权重用 mmap |
| `dynamic_load_enable` | `false` | 动态加载层(省 CMM,降速),见 README |
| `dynamic_load_pool_size` | `2` | 动态加载常驻层数(仅 enable 时) |
| **`mem_guard_enable`** | `true` | **加载前内存预检总开关**(见下「内存安全预检」) |
| **`mem_guard_floor_mb`** | `128` | 估算占用之上额外保留的安全余量(MB) |
| **`mem_guard_on_unsafe`** | `prompt` | 不安全时:`prompt`(有TTY弹Y/N,无TTY=abort)/ `abort` / `warn` |

### 内存安全预检（防止超 CMM/DDR 加载导致驱动崩溃）

开启后,加载模型前会按**文件大小**估算各部分占用,并对照剩余内存判断是否安全:

- **CMM**(设备显存,各层 / post / 视觉&音频编码器):用现有 `remain` 查询(AX650 读 `/proc/ax_proc/mem_cmm_info`,AXCL 经 axcl-smi),多卡按各卡分别核算。
- **DDR**(主机内存,token embedding 非 mmap 时 / Gemma per-layer 权重):读 `/proc/meminfo` 的 **MemAvailable**(已扣可回收 buffer/cache,是"真正可用",避免误报)。
- 若 `剩余 - 估算 < mem_guard_floor_mb` → 按 `mem_guard_on_unsafe` 处理:`abort` 直接中止并报错;`warn` 仅告警继续;`prompt` 在交互式终端弹 `[y/N]`(默认 N=不加载),无终端(serve/docker)时退化为 `abort`。
- 关闭:`mem_guard_enable=false`。
- 说明:`dynamic_load_enable=true` 时层权重会在加载后释放,故预检不计层权重(只算 post/编码器/主机部分)。

## 注意力(混合注意力 / 长上下文模型)

| 字段 | 默认 | 说明 |
|---|---|---|
| `full_attention_interval` | 0 | 每 N 层(1-indexed)为 full-attention,其余 linear(如 Qwen3.5) |
| `layer_types` | 空 | 显式每层类型数组(`full_attention`/`linear_attention`/`sliding_attention`) |
| `sliding_window` | 0 | 滑动窗口大小 |
| `num_kv_shared_layers` | 0 | 末尾共享 KV 的层数 |

## 多槽前缀 KV 缓存（serve 多用户/多提示词加速）

| 字段 | 默认 | 说明 |
|---|---|---|
| `kv_cache_slots` | 1 | 槽数,1=关闭(等于现状) |
| `kv_cache_slot_location` | `device` | `device`(零拷贝指针切换)/ `host`(省 CMM,切换拷贝) |

详见 [multi_slot_kv_cache.md](multi_slot_kv_cache.md)。

## VLM / 视觉 / 音频

| 字段 | 默认 | 说明 |
|---|---|---|
| `vlm_type`(别名 `VLM_TYPE`) | `None` | `Qwen2_5VL`/`Qwen3VL`/`InternVL3`/`FastVLM`/`SmolVLM2`/`PaddleOCRVL`/`Gemma4VL`/`MiniCPMV46VL` |
| `filename_image_encoder_axmodel` | — | 视觉编码器 axmodel(VLM 必填) |
| `filename_audio_encoder_axmodel_5s` / `_30s` | — | Gemma4 音频编码器(ASR) |
| `vision_cache_dir` | 空 | 视觉 embedding 磁盘缓存目录 |
| `vision_width` / `vision_height` | 448 | 视觉输入尺寸(未配置时按编码器输入形状自动推断) |
| `vision_patch_size` / `vision_temporal_patch_size` / `vision_spatial_merge_size` | 14 / 2 / 2 | patchify 参数 |
| `vision_fps` / `vision_tokens_per_second` | 1 / 1 | 视频时间缩放(Qwen2.5-VL mRoPE) |
| `vision_num_frames` / `vision_do_sample_frames` | 0 / true | 视频抽帧上限 / 是否均匀抽帧 |
| `video_processor` | — | 视频处理器选择(部分模型) |

## Embedding

| 字段 | 默认 | 说明 |
|---|---|---|
| `is_embedding`(别名 `embedding`/`embedding_type`) | `false` | 以 Embedding 模式启动,提供 `/v1/embeddings`(不支持 `run`) |

## Gemma4 per-layer 投影

| 字段 | 说明 |
|---|---|
| `hidden_size_per_layer_input` | per-layer 投影维度(>0 时启用) |
| `rms_norm_eps` | RMSNorm eps |
| `filename_tokens_embed_per_layer` / `filename_per_layer_model_projection` / `filename_per_layer_projection_norm` | per-layer 权重文件 |

## 服务（serve）

| 字段 | 默认 | 说明 |
|---|---|---|
| `port` | 8000 | 监听端口 |
| `server_timeout_ms` | 300000 | 请求超时(并发排队也复用该值) |
| `server_default_max_tokens` | — | 请求默认 max_tokens |
| `server_max_output_tokens` | — | 输出 token 硬上限 |
| `server_forced_prompt_text` | — | 强制提示词(如 OCR 规整) |

## AXCL（PCIe 多卡）

| 字段 | 默认 | 说明 |
|---|---|---|
| `devices` | `[0]` | 使用的设备 id 列表(多卡张量并行) |

## 图像生成（SD1.5）

| 字段 | 说明 |
|---|---|
| `model_type` / `task_type` = `image_generation` 或 `is_image_generation=true` | 以图像生成模式启动,提供 `/v1/images/*` |
| `image_model_dir` | 图像模型根目录 |
