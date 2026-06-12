# 多槽前缀 KV 缓存(Multi-Slot Prefix KV Cache)

分支:`ax-prefix-cache-multi-slot`

## 背景 / 动机

`serve` 模式是**单实例、串行处理**(`setMaxConcurrency(1)`),但会有多个用户 / 多套系统提示词轮流请求。
当前只有**一份**上下文(`last_tokens_ids` + `precompute_len` + 设备上的一份 K/V),所以一旦请求的前缀和上一次不同,就会触发 `ResetKVCache` 全量重算 prefill,非常慢。

OpenAI chat 协议是无状态的(每次发全量 history),因此**不需要 session id**:token 前缀本身就是 key。
不同用户/系统提示词 → 不同前缀 → 落到不同槽;同一会话多轮 → 新前缀以旧前缀开头 → 命中自己的槽。

## 目标

- 缓存**多份** prefix token 及其 KV(槽 slot)。
- 份数由 config 配置;一次仍只处理一个请求。
- 每个请求按 token **最长公共前缀**匹配槽;命中则复用其 KV,只对增量做 prefill;不命中则覆盖**最久未使用**(LRU)的槽,整段 prefill。
- KV 内存位置可配:
  - `device`:每槽一份设备 K/V buffer,激活时**零拷贝重绑定**引擎输入张量,切换最快;代价是 N× 设备 CMM。
  - `host`(DDR):单份设备 K/V + 每槽一份 host K/V,激活时把旧槽 D2H、新槽 H2D;**省 CMM**(无论多少槽只占 1 份设备 buffer),切换有拷贝开销。
- 文本 LLM 与 **VLM 均支持**。VLM 下视觉编码结果(vision embedding)随 KV 一起常驻在槽里,追问/复用时**跳过 vision encoder 重跑**。

## 配置(模型目录 `config.json`)

| 键 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `kv_cache_slots` | int | `1` | 前缀缓存槽数。`1` = 完全等于现状(关闭多槽)。 |
| `kv_cache_slot_location` | string | `device` | `device`(零拷贝指针切换) / `host`(后续支持)。 |

默认 `kv_cache_slots=1`,对既有模型零影响。仅当显式设 `>1` 才启用。

### 显存/内存自适应(放不下就降级 + 报警)

开启多槽时**先评估再分配**,避免配置过大导致 OOM:

- device 模式:先为每层建立槽元数据并算出**一份槽的设备 CMM 占用**(各层 K+V,按设备汇总),查询每张卡的剩余 CMM(`get_remaining_cmm_size` / `axcl_GetCMMRemain`),保留 256MB 余量后算出最多能开几份;按各设备取最小值并以 config 为上限。实际分配若因碎片不足,会把所有层裁剪到真正分配成功的份数。
- host 模式:按空闲主机内存(`sysinfo`,保留 512MB 余量)估算。
- 若实际能开的份数 **< config 请求**,会 `⚠` 告警并按能开的份数启用;一份都开不下则关闭多槽。例如配置 4 份但卡上只够 3 份,就用 3 份并告警。

示例日志:
```
kv slot budget: dev=0 per_slot=139MB free=6474MB margin=256MB -> max_slots=44
⚠ kv_cache_slots=9999 requested but device CMM only fits 44; reducing to 44
```

## 设计

### 一个"槽"包含什么

复制现有单上下文的全部状态:

- 设备侧:每层一份 `K_cache` / `V_cache` 设备 buffer(device 模式;slot0 复用引擎原生 buffer,slot1..N-1 额外分配)。
- 主机侧(每槽各一份,放在 `LLM::Impl`):
  - `last_tokens_ids`、`precompute_len`
  - `linear_state_snapshots_`(linear 注意力回滚快照)
  - `cached_mrope_next_pos`、`full_cache_valid_slots_` / `full_cache_has_sparse_slots_`
  - VLM:`vision_state` 相关记录(v1 主要保证文本 LLM;VLM 谨慎纳入)
  - LRU 时间戳

### 关键思路:激活即重绑定,KV 永不在槽间拷贝

两个后端的 KV 布局相同:所有 shape group **共享同一份** `K_cache`/`V_cache` 设备 buffer(group0 分配,其余 alias)。
- 引擎从 `io_data[grp].pInputs[]`(AX650)/ `axcl_EngineSetInputBufferByIndex`(AXCL)读 buffer 地址。
- `LLM.cpp` 从 `get_input(grpid,"K_cache").{phyAddr,pVirAddr}` 读地址(AXCL 用 `phyAddr`,AX650 用 `pVirAddr`)。

因此"激活槽 S" = 把每个 group 的 `K_cache`/`V_cache` 输入**重绑定到槽 S 的 buffer**(同时更新 `io_data` / SetInputBuffer 与 `mgroup_input_tensors` 描述符,并重建 map)。
之后**所有现有的 prefill/decode/SetKVCache 代码原样操作当前绑定的 buffer**,天然写进槽 S,槽间零拷贝。

### 后端新增接口(`ax_runner_base`,默认 `-1`=不支持)

```cpp
virtual int kv_cache_slots_init(int num_slots);   // 为 K_cache/V_cache 各多分配 num_slots-1 份等大 buffer
virtual int kv_cache_slots_activate(int slot);    // 把所有 group 的 K_cache/V_cache 重绑定到该槽
virtual int kv_cache_slots_count() const;         // 返回已分配槽数(默认 1)
```

- AX650:`AX_SYS_MemAllocCached` 分配槽 buffer;activate 改写 `io_data[grp].pInputs[*]` 与 `mgroup_input_tensors[grp][*]` 的 `phyAddr`+`pVirAddr`。
- AXCL:`axcl_Malloc` 分配设备 buffer;activate 改写 `mgroup_input_tensors[grp][*].phyAddr` 并对每 group 调 `axcl_EngineSetInputBufferByIndex`。
- 每层一个 runner,各自管理自己的 N 份 K/V。

### 引擎侧(`LLM::Impl`)

- 新增 `struct KvSlot { ... }` 与 `std::vector<KvSlot> slots_`、`int active_slot_`、单调递增 `lru_tick_`。
- 请求入口(`Run(history,...)`)在 tokenize 后、进入现有前缀复用逻辑前:
  1. 对每个槽用 `diff_token_ids` 求与 `new_tokens` 的公共前缀长度,选最长且 `>0` 的为命中槽;都为 0 则选 LRU 槽并 `ResetKVCache`(只重置该槽)。
  2. 对每层 `runner.kv_cache_slots_activate(slot)`;把该槽的 host 状态载入工作变量(`last_tokens_ids/precompute_len/...`)。
  3. 走现有 append/rollback/prefix-reuse/recompute 逻辑。
  4. 跑完把更新后的 host 状态写回该槽,刷新 LRU。
- 删除一直未启用的 `b_os_kvcache` 旁路;`Get/SetKVCache` 简化为设备路径。

### 约束 / 边界(v1)

- 与 `dynamic_load_enable` 暂不同时启用(动态换 handle 会重置 IO 绑定);若同时配置则告警并退回单槽。
- 多卡 TP:槽 buffer 按层所在卡分配。
- VLM:先保证文本与"文本前缀"复用正确;含视觉注入的请求按现状逻辑工作但不跨槽复用视觉。

## 测试

- AXCL:`10.126.126.1`(8× AX650N PCIe),`build_axcl_x86.sh`。
- AX650 板端:`10.126.35.191` / `10.126.35.234`,`build_ax650.sh` 交叉编译后推送运行。
- 用例:多套系统提示词 A/B/C 轮流 serve 请求 → 命中槽 TTFT 显著下降;超过份数后 LRU 覆盖最久未用;两后端、正确性与数值一致。
