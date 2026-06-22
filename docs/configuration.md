# 配置文件说明（模型目录 `config.json`）

`axllm` 启动时读取 `<model_dir>/config.json`。下面按用途分组列出所有可用字段;**未列为必填的都是可选**,不填用默认值。

> 路径类字段(`filename_*` / `template_filename_axmodel` / `post_config_path` 等)相对模型目录解析。

## 必填

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_name` | string | 模型名(显示在 `/v1/models`、日志) |
| `tokenizer_type` | string | 分词器类型(如 `Qwen3` / `Qwen3VL` / `Gemma4VL` / `SmolLM2` …) |
| `url_tokenizer_model` | string | **本地**分词器文件路径(如 `qwen3_tokenizer.txt`)。⚠ 字段名带 `url`、默认值带 `http` 都是历史遗留;**当前只读本地文件,不支持 HTTP** |
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
| `enable_thinking`(bool) 或 `thinking_mode`(string) | 模型默认 | 思考链开关(Qwen3 / MiniCPM5 等)。`enable_thinking`: `true`/`false`;`thinking_mode`: `think`/`no_think`/`default`(`default`/`auto`=按模型默认)。也可在 `/v1/chat/completions` 请求里按次覆盖 |

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

开启后分**两道**核对剩余内存,任一不安全都按 `mem_guard_on_unsafe` 处理。

**① 加载前（按文件大小估算）** —— 拦住"明显放不下"的模型:

- **CMM**(设备显存,各层 / post / 视觉&音频编码器):AX650 读 `/proc/ax_proc/mem_cmm_info`,AXCL 用 `axcl_GetCMMRemain`,多卡按各卡分别核算。
- **DDR**(主机内存,token embedding 非 mmap 时 / Gemma per-layer 权重):读 `/proc/meminfo` 的 **MemAvailable**(已扣可回收 buffer/cache,是"真正可用",避免误报)。

**② 加载中（实测外推）** —— 文件大小估不到引擎加载每层时额外分配的 KV/IO 缓冲(约比纯权重多 ~30%,且随 context 长度增长)。故加载层时按**实测的每层 CMM 增量**外推"剩余层 + post 尾部",一旦预计突破 floor 就**在分配前中止**(此时只加载了头几层,可干净回收,驱动不崩)。多卡并行加载时任一卡触发会停下其它卡。注:此阶段不再交互弹窗(①已问过),`prompt` 在此等同 `abort`。

判定:若 `剩余 - 估算 < mem_guard_floor_mb` → `abort` 直接中止并报错;`warn` 仅告警继续;`prompt` 在交互式终端弹 `[y/N]`(默认 N=不加载),无终端(serve/docker)时退化为 `abort`。

- 关闭:`mem_guard_enable=false`。
- `dynamic_load_enable=true`:层权重加载后即释放,①不计层权重,②仍按实测拦截层 IO。
- 多槽(`kv_cache_slots>1`)的 N×KV 申请本就按已知真实尺寸精确预算,并把 `mem_guard_floor_mb` 作为保留余量(取与内置 256MB/512MB 的较大者)。

## 注意力(混合注意力 / 长上下文模型)

| 字段 | 默认 | 说明 |
|---|---|---|
| `full_attention_interval` | 0 | 每 N 层(1-indexed)为 full-attention,其余 linear(如 Qwen3.5) |
| `layer_types` | 空 | 显式每层类型数组(`full_attention`/`linear_attention`/`sliding_attention`) |
| `sliding_window` | 0 | 滑动窗口大小 |
| `num_kv_shared_layers` | 0 | 末尾共享 KV 的层数 |

> 这几项也可不在 `config.json` 顶层写:`full_attention_interval` / `num_kv_shared_layers` 会回退读 `text_config.*`;`sliding_window` / `layer_types` 未配置时会自动从模型目录下分词器的 sidecar config 读取。

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

## Embedding

| 字段 | 默认 | 说明 |
|---|---|---|
| `is_embedding`(别名 `embedding`;旧 `embedding_type`/`EMBEDDING_TYPE` 已废弃) | `false` | 以 Embedding 模式启动,提供 `/v1/embeddings`(不支持 `run`) |

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
| `server_default_max_tokens` | 0 | 请求未带 max_tokens 时的默认值(0=用内置默认) |
| `server_max_output_tokens` | 0 | 输出 token 硬上限(0=不额外限制) |
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
