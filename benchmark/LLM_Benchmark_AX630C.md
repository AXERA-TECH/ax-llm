# LLM Benchmark(AX630C)

Benchmark 是了解硬件平台网络模型运行速度的最佳途径。以下数据基于 AXera-Pi 2 测试获取，**仅供社区参考，不代表商业交付最终性能**。

### 工具链版本
- Pulsar2 3.2

### 数据记录

Decode 性能

| 模型名称      | 参数量 | Generate（token/s） |
| ------------- | ------ | ------------------- |
| TinyLlama-1.1 | 1.1B   | 5.4                 |
| Qwen2.0       | 0.5B   | 10.7                |
| Qwen2.0       | 1.5B   | 3.7                 |
| MiniCPM       | 1.2B   | 3.9                 |
| Llama3.2      | 1.2B   | 4.5                 |

Prefill 性能

| 模型名称      | 参数量 | Prompt length | TTFT（ms） | Prefill（token/s） |
| ------------- | ------ | ------------- | ---------- | ------------------ |
| TinyLlama-1.1 | 1.1B   | 128           | 880        | 145                |
| MiniCPM       | 1.2B   | 128           | 1100       | 115                |
| Qwen2.0       | 0.5B   | 128           | 360        | 355                |
|               | 1.5B   | 128           | 1080       | 118                |
| Llama3.2      | 1.2B   | 128           | 880        | 145                |

