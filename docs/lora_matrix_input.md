# LoRA 适配与运行时切换指南

本文介绍 `feat/chat-lora-task-runtime` 分支实现的 AX650 LoRA 矩阵输入方案，覆盖模型适配、adapter 打包、运行时加载、请求切换和验证方法。以下说明以本分支源码为准，模型参数以实际发布包为准。

## 1. 交付范围与依赖

本仓库提供推理 runtime、adapter 打包脚本和请求接口。将 Hugging Face 模型转换为带 LoRA 输入的 AXModel，需要另行准备支持 LoRA matrix-input 的 Pulsar2 编译器及对应模型配置；克隆本分支不会同时获得编译器源码、训练权重或编译产物。

本分支基于 Jina 多任务支持继续开发，已经包含 `feat/jina-v5-nano-lora-input-runtime` 的 runtime 提交，以及 chat 请求切换和模型包相对路径处理。学习这些 runtime 功能不需要再合并旧分支。

获取本分支源码：

```bash
git clone --branch feat/chat-lora-task-runtime --single-branch https://github.com/AXERA-TECH/ax-llm.git
cd ax-llm
git submodule update --init --recursive
git submodule status
```

`third_party/tokenizer.axera` 使用 `.gitmodules` 中指定的依赖仓库，其配套分支为 `feat/chat-lora-task-runtime`。访问私有依赖仓库需要预先获得授权，并为 Git 配置相应的 HTTPS 凭据；主仓库的访问权限不代表同时拥有私有子模块权限。没有依赖仓库权限时，需要先由交付方提供访问权限或匹配版本的源码包。

本分支固定的子模块版本如下。普通 `git submodule update --init --recursive` 按主仓库记录的精确提交检出，`.gitmodules` 的分支名称不替代该固定版本；复现时不要使用 `--remote` 升级依赖，也不要直接替换为最新 tokenizer 并假设 API、模板和多模态 token 行为一致。

| 子模块 | 固定提交 |
| --- | --- |
| `third_party/tokenizer.axera` | `404c00951dfa938a4e74c7b1452d0928e35b392a` |
| `third_party/openai-api.cpp` | `14db1ad0cdd6d3af4d031d7376ff182336c28fbb` |
| `third_party/SimpleCV` | `7729953c7c75ffaea441f70e0a9eb62790f3eb2b` |

已有克隆在更新本分支后，应同步子模块地址，再初始化：

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status
```

依赖齐备后，在支持的 Linux 交叉编译环境中使用仓库脚本：

```bash
./build_ax650.sh
```

该脚本默认使用 `build` 目录，安装产物位于 `build/install/bin/axllm`。BSP 和工具链可通过脚本支持的 `AXLLM_BSP_MSP_DIR`、`AXLLM_TOOLCHAIN_BIN_DIR` 等变量指定。通用 `install.sh` 默认选择 `axllm` 分支，使用它时须显式设置 `BRANCH=feat/chat-lora-task-runtime`。

`axllm` 和 AXModel 的实际运行需要目标板环境。在板端执行 `axllm version`，可以检查编译提交、分支、tokenizer 提交及后端，避免混用其他分支的二进制。

## 2. 为什么把 LoRA 权重作为输入

对输入特征按行排列的线性层，普通计算与 LoRA 计算可以写成：

```text
普通线性层：Y = X Wᵀ
LoRA 线性层：Y = X Wᵀ + (X Aᵀ) B'ᵀ
B' = (alpha / rank) B
A.shape = [rank, in_features]
B.shape = [out_features, rank]
```

偏置项按原模型处理。基座权重 W 编译进模型，A 和 B' 暴露为运行时输入。因此只需更换小矩阵，就能复用同一套模型句柄和基座权重。

| 方案 | 编译产物 | 切换方式 | 主要代价 |
| --- | --- | --- | --- |
| 普通模型 | 固定权重 AXModel | 此配置没有 adapter 切换 | 无额外 LoRA 分支计算 |
| 先合并 LoRA 再编译 | 每个 adapter 对应完整模型 | 更换整套模型或服务 | 多套产物和模型加载成本 |
| 本分支 matrix-input | 带 LoRA 运算和输入的 AXModel，加 adapter BF16 文件 | 覆盖 LoRA 输入缓冲区 | LoRA 运算、持久输入缓冲区、切换时文件读取与拷贝 |
| shared-weight patch | 共享基座与任务权重 patch | 更新模型 WBT segment | 专用引擎接口和 patch 格式 |

后两条路径共用部分配置名称和 `SetSharedWeightTask` 入口，但机制不同。纯 matrix-input 配置通过 `load_lora_adapter` 写输入，不走 `AX_ENGINE_PatchHandleWBT`。README 中共享基座部分对 `AX_ENGINE_CreateHandleV4` 的要求属于 shared-weight 路径，不能直接套用到纯 matrix-input 配置。

## 3. 模型适配流程

### 3.1 确认基座与 adapter 契约

先固定基座模型、tokenizer、adapter、编译器和 runtime 的版本。检查 adapter 是否来自同一基座，层数、投影维度、rank 和目标模块是否与待编译图一致。相同文件长度只能证明布局长度匹配，不能证明模型语义匹配。

本分支独立打包脚本的契约是：

- PEFT `LORA`，`bias` 为 `none`，A/B 均为二维 BF16 源张量。
- 层编号从 0 开始连续，每层包含全部七类投影的 A/B。
- 目标模块恰好为 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。
- A 的第一维与 B 的第二维等于 rank，同类投影的形状跨层一致。
- 不支持 DoRA、RSLoRA、QALoRA、`modules_to_save`、分层 rank/alpha 等扩展；具体拒绝项见脚本的 `validate_config`。

其他模型拓扑、部分层 LoRA、部分投影 LoRA 或其他 rank，不能只修改 JSON 就假设兼容，需要同步调整编译图、打包器和 runtime 契约并验证。

### 3.2 在编译图中保留 LoRA 分支

编译器需要在各目标线性层的原始基座输出上增加低秩分支，并让 A/B 成为图输入。普通的无 LoRA 输入 AXModel 无法通过增加 `lora_mode` 配置获得此能力。

每层必须暴露以下 14 个输入，名称与 runtime 精确匹配：

```text
lora_q_proj_a       lora_q_proj_b
lora_k_proj_a       lora_k_proj_b
lora_v_proj_a       lora_v_proj_b
lora_o_proj_a       lora_o_proj_b
lora_gate_proj_a    lora_gate_proj_b
lora_up_proj_a      lora_up_proj_b
lora_down_proj_a    lora_down_proj_b
```

约定输入为 BF16，形状对应 `[rank, in_features]` 和 `[out_features, rank]`。编译阶段应检查所有 prefill/decode group 的名称、字节数和布局兼容，并确认 LoRA 分支没有被常量折叠或优化删除。runtime 按图中输入的字节数切分文件，不会自动转置、转换 dtype 或推断另一个 rank。

转换基座时应保留未合并的基座权重，避免已合并的增量在运行时再计算一次。编译器校验、导出和编译命令依赖所使用的 Pulsar2 版本及模型类型，应使用该版本的配套配置；本仓库的 `build_ax650.sh` 编译的是 runtime，不编译 Hugging Face 模型。

### 3.3 分别验证编译图和完整模型

先对单个投影、单层比较 `base + LoRA` 的参考输出，再验证完整 prefill/decode、多个 group 及切换后的结果。编译 rank、目标模块或模型维度改变时，应重新编译匹配的 AXModel。量化基座与额外 BF16 分支的误差、算力和内存开销也需要实测。

## 4. Adapter 打包与文件布局

独立打包器位于 `scripts/pack_lora_adapter.py`，输入目录包含：

```text
adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

在准备好的 Python 环境安装依赖后执行：

```bash
python -m pip install torch numpy ml_dtypes safetensors
python scripts/pack_lora_adapter.py ./adapter ./packed_adapter
```

该脚本使用 PyTorch 读取 safetensors；实际版本应与适配环境匹配。它将 A 保持原始排列，将 B 乘以 `alpha / rank` 后转换为 BF16，按以下顺序连续写入各层文件：

```text
q_proj_A, q_proj_B, k_proj_A, k_proj_B, v_proj_A, v_proj_B,
o_proj_A, o_proj_B, gate_proj_A, gate_proj_B,
up_proj_A, up_proj_B, down_proj_A, down_proj_B
```

每层 `.bf16.bin` 为无头的连续数据，无张量名称和分隔符。文件长度必须等于该层 14 个图输入的 `nSize` 之和。BF16 的逻辑数据量为所有 A/B 元素数之和乘 2 字节；若编译布局存在额外 padding，打包输出也必须匹配实际 IO 契约。

输出示例：

```text
packed_adapter/
├── layer_00.bf16.bin
├── ...
├── layer_35.bf16.bin
└── manifest.json
```

`manifest.json` 记录 rank、alpha、缩放约定、模块顺序、形状及文件 SHA256 等信息，用于发布审计和离线验证。**当前 runtime 不读取该 manifest，也不会在启动时校验 SHA256**；这些检查需在发布或部署前单独执行。

本分支独立打包器的版本为 `0.2`，只接受 BF16 源张量。部分发布包由编译器侧的其他版本打包器生成，可能记录 F32 源张量或不同 packer 版本；只要输出满足 BF16、缩放、顺序和图 IO 契约，runtime 加载格式相同。不要将其他打包器的源 dtype 支持范围视为本脚本的能力，也不要再次给已缩放的 B 乘系数。

## 5. 模型包配置

以下为 36 层模型的 LoRA 配置片段，需合入已有的完整模型 `config.json`。其中 `task-a`、`task-b` 为示例名称：

```json
{
  "axmodel_num": 36,
  "dynamic_load_enable": false,
  "kv_cache_slots": 1,
  "lora_mode": "matrix_input",
  "shared_weight_task_id": "task-a",
  "shared_weight_adapters": [
    {
      "task_id": "task-a",
      "lora_adapter_template_filename": "lora/task-a/layer_%02d.bf16.bin"
    },
    {
      "task_id": "task-b",
      "lora_adapter_template_filename": "lora/task-b/layer_%02d.bf16.bin"
    }
  ]
}
```

| 字段 | 含义 |
| --- | --- |
| `lora_mode` | 设为 `matrix_input` 才启用这条 LoRA 加载路径 |
| `shared_weight_task_id` | 启动默认 adapter，应显式指向已配置且含 LoRA 文件模板的 task |
| `shared_weight_adapters` | 启动时声明可选 task，task 名称应唯一 |
| `lora_adapter_template_filename` | 按层编号格式化的文件名，相对模型包目录解析 |
| `axmodel_num` | 文本层数，必须与图和 adapter 文件数量一致 |
| `dynamic_load_enable` | 本指南使用 `false`，保持文本层句柄常驻 |
| `kv_cache_slots` | 本指南使用 `1`，避免跨 adapter 复用其他前缀槽 |

这些字段复用了早期 shared-weight 接口的命名；纯 matrix-input 配置不需要填 shared-base 模型和 WBT patch 字段。默认 task 不匹配时，当前初始化代码可能跳过默认权重加载，应在发布前主动检查，不能依赖它自动选择首个 adapter。

模型包仍需保留普通推理要求的 tokenizer、embedding、post、文本层和视觉模型等配置。LoRA 目录只增加 adapter 文件，不替代这些基础文件。

## 6. 启动时如何加载

```text
解析 config.json，按模型包目录解析 adapter 相对路径
  → 初始化基座模型句柄和所有层 IO
  → 校验每个已配置 adapter 的每层文件
  → 加载 shared_weight_task_id 指定的默认 adapter
  → 开始接收请求
```

`PreflightConfiguredLoraAdapters` 在 IO 初始化之后校验全部已配置且含 LoRA 模板的 task：打开每层文件，检查长度为正，确认 group 0 中 14 个规范输入各出现一次，并比较文件总长度。未激活 task 的文件损坏也会阻止启动。

`load_lora_adapter` 对每层执行：

1. 将该层文件读入临时 CPU buffer。
2. 按规范输入名称定位 group 0 的目的缓冲区，核对精确长度。
3. 依固定顺序将 A/B 数据 `memcpy` 到对应的 `pVirAddr`。
4. 对每个输入调用 `AX_SYS_MflushCache`，让 NPU 可见更新的数据。

group 0 持有规范 LoRA 输入缓冲区，其他 group 复用这些地址，因此 adapter 不需要为 prefill 和 decode 各加载一份。CPU 临时文件 buffer 在函数返回时释放，当前 adapter 的矩阵保留在设备可访问的 IO 内存中。启动校验所有 adapter 不等于把所有 adapter 都常驻设备内存。

## 7. 请求如何触发切换

在板端启动服务，模型目录以实际部署位置为准：

```bash
axllm serve ./model_package --port 8000
```

使用已配置的 task 发起 chat 请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"example-model","task_id":"task-b","messages":[{"role":"user","content":"请给出三个配色建议。"}],"max_tokens":64}'
```

示例中的 `example-model` 应替换为包内实际配置的模型名称。Embedding 模式下，`/v1/embeddings` 也通过相同的 task 字段选择 adapter，但需遵循对应模型的 embedding 请求格式。

请求字段优先级为非空字符串 `task_id` → `adapter_task` → `task`。这些字段选择预先配置的 task，不是客户端上传路径，也不实现在线注册一个新 adapter。

```text
解析请求 task
  → SetSharedWeightTask
  → 查找 task 配置
  → 若与当前 task 相同则直接复用
  → 若不同则先使 active task 标记失效
  → 逐层 load_lora_adapter，覆盖同一套 LoRA 输入
  → 更新 active task，按需处理 media encoder
  → ResetKVCache
  → 处理本次请求
```

纯 matrix-input 切换复用已有基座句柄，不重新加载整套基座 AXModel。当前实现每次跨 task 切换都从文件读取目标权重，没有把所有 adapter 缓存到 CPU 或设备内存。

| 请求情况 | 实际行为 |
| --- | --- |
| 与当前 task 相同 | 跳过 LoRA 重载，可能按需检查 media encoder |
| 选择另一个已配置 task | 逐层覆盖 LoRA 输入，成功后重置 KV 缓存 |
| 省略 task 或全部为空 | 继续使用当前设备状态；不会自动恢复默认 task |
| 指定未知 task | 返回错误，不使用其他 adapter 代替 |
| task 没有可用 LoRA 或 shared-weight payload | 切换失败 |

`model` 字段用于 API 模型名称，不替代这里的 task 选择。当前没有通过空 task 卸载 LoRA 或切到纯基座的专用接口。若需要无增量参考，应使用单独的基座包，或验证过的零增量 adapter 方案。

## 8. 缓存、并发与失败处理

LoRA 改变各层隐状态，旧 adapter 下产生的 K/V 无法作为新 adapter 的上下文继续使用。切换成功后，`ResetKVCache` 清理已缓存的 token、历史、相关状态，并清零设备 KV。新请求应提供完整对话历史，重新执行 prefill。

本分支的多槽前缀缓存没有按 adapter 标识隔离，`ResetKVCache` 也没有遍历并作废全部 `kv_slots_` 快照。LoRA task 切换应保持 `kv_cache_slots: 1`。若要组合多槽缓存，需要先补齐 task 参与缓存键、切换失效或独立缓存生命周期等机制，再验证跨 task 的前缀命中行为。

当前 active task 为进程全局状态，没有按请求同时保留多套 adapter。不同 task 的请求应串行执行；自定义服务接入 `LLM::SetSharedWeightTask` 时也需要串行化“切换 + 本次推理”整体，不能只保护切换函数。不要把请求带有 `task_id` 理解为并发隔离。

切换逐层写入，**不是原子事务，也没有自动回滚**。代码在修改第一层之前清空 active task 标记，因此中途出错后，下次显式指定任意有效 task 都会重新完整加载，不会因同名 task 快速路径误信任混合状态。

但省略 task 的请求不会触发重载：切换失败后应停止发送无 task 请求，显式重载一个有效 task；若无法确认设备状态，应重启服务。运行期间也不要替换 adapter 文件，启动预校验不能覆盖之后发生的文件变动。

切换可附带 adapter 专用视觉/音频 encoder profile；只有请求需要 media 且 profile 不同时才尝试重新初始化相应 encoder。仅含文本 LoRA 路径的两个 task 可以共用原有图像编码器。media encoder 更新和 KV 重置的错误路径也应纳入集成验证。

## 9. 支持边界与验证清单

本指南对应的实际 LoRA 文件加载、输入覆盖与缓存刷新实现在 AX650 后端。AXCL 当前继承了基类的 LoRA 默认空实现，返回成功不代表实际加载了权重，不能据此宣称支持 AXCL LoRA。若要移植，应补充设备内存传输、group 共享和验证逻辑后再测。

模型层动态换入换出、同一进程并发混跑 task、多 adapter 同时叠加、运行时改变 rank、任意模块拓扑，以及 adapter 的在线注册，都不属于本文已经验证的使用方式。

建议按以下顺序验收：

1. 离线核对基座、tokenizer、adapter 来源与版本、打包 manifest、所有权重的 SHA256。
2. 检查每层全部 group 的 LoRA 输入名称、dtype、形状、字节数和共享 buffer。
3. 对单层输出、完整 prefill 和 decode 与 Hugging Face 参考比较；误差阈值应考虑基座量化。
4. 分别验证 task-a、task-b，再验证 a → b → a 与同 task 连续请求，确认切回后结果一致。
5. 验证未知 task、缺失或截断的文件、切换失败后的恢复；不要只检查 HTTP 成功状态。
6. 覆盖单 token、多 chunk、长上下文和图文输入，核对切换后 KV 无污染。
7. 分别记录启动耗时、切换耗时、首 token 延迟、token/s、CMM 和 RSS；跨 task 首请求包含加载及上下文重算成本。

以公开 Qwen3-VL-4B LoRA 包的 rank 16、36 层布局为例，每层 adapter 文件为 1,835,008 字节，一套共约 63 MiB。这是原始 payload 大小，不能直接作为 CMM 增量；实际内存还受缓冲区对齐、模型图、KV 和运行库影响。

常见问题：

| 现象 | 优先检查 |
| --- | --- |
| 私有子模块无法访问，或出现 `not our ref` | 仓库访问权限、Git 凭据、是否同步了 `.gitmodules` 地址，以及固定提交是否可获取 |
| 缺失 `lora_*` 输入 | 是否误用了普通 AXModel，或编译图删掉了 LoRA 分支 |
| adapter 文件长度不匹配 | rank、投影维度、层号、dtype、padding 和打包顺序 |
| 加载成功但答案异常 | 基座版本、B 是否重复缩放、文件内容与参考输出 |
| 首次切换延迟明显增加 | 文件读取、36 层拷贝、缓存刷新、KV 清空与重新 prefill |
| 省略 task 后没有回到默认行为 | 当前实现保留上一次 active adapter |
| 重启前无法恢复一致结果 | 检查部分切换失败、文件变动、并发和 media profile 错误 |

## 10. 源码阅读顺序

以下链接定位到文件，函数名称便于在当前提交中检索：

| 文件 | 关注内容 |
| --- | --- |
| [`scripts/pack_lora_adapter.py`](../scripts/pack_lora_adapter.py) | `validate_config`、`load_and_validate_tensors`、`pack`：源格式约束、缩放及输出布局 |
| [`src/main.cpp`](../src/main.cpp) | 配置解析、`resolve_config_paths`、`adapter_task_from_request`、chat/embedding 请求处理 |
| [`src/runner/LLM.hpp`](../src/runner/LLM.hpp) | `LLMSharedWeightAdapter`、`LLMAttrType`、公开切换接口 |
| [`src/runner/LLM.cpp`](../src/runner/LLM.cpp) | `PreflightConfiguredLoraAdapters`、`Init`、`SetSharedWeightTask`、`ResetKVCache` |
| [`src/runner/ax_model_runner/ax_model_runner.hpp`](../src/runner/ax_model_runner/ax_model_runner.hpp) | 后端抽象和默认实现，移植时要检查空实现 |
| [`src/runner/ax_model_runner/ax_model_runner_ax650.cpp`](../src/runner/ax_model_runner/ax_model_runner_ax650.cpp) | group IO 共享、`validate_lora_adapter`、`load_lora_adapter` |

学习时可沿“配置与请求 → LLM task 状态 → 各层 runner → 输入缓冲区”追踪调用，再结合所使用编译器的图生成与 packer 实现，核对两端是否遵守同一契约。
