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

### 已支持模型

- MiniCPM-V 2.0

### 获取地址

- [百度网盘](https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e)

## 源码编译

- 递归 clone 本项目，确保所有 `submodule` 正确 clone
    ```shell
    git clone -b minicpm-v --recursive https://github.com/AXERA-TECH/ax-llm.git
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

### minicpm v 2.0

```shell
root@ax650:/llm-test/minicpm-v-2.0# ./run_minicpmv-2.sh
[I][                            Init][ 125]: LLM init start
2% | █                                 |   1 /  44 [0.21s<9.11s, 4.83 count/s] tokenizer init ok
[I][                            Init][  26]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  44 /  44 [33.54s<33.54s, 1.31 count/s] init vpm axmodel ok,remain_cmm(8086 MB)
[I][                            Init][ 284]: max_token_len : 1023
[I][                            Init][ 289]: kv_cache_size : 2304, kv_cache_num: 1023
[I][                            Init][ 297]: prefill_token_num : 128
[I][                            Init][ 306]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
prompt >> 描述下图片
image >> ssd_dog.jpg
[I][                          Encode][ 365]: image encode time : 728.507019 ms
[I][                             Run][ 589]: ttft: 520.94 ms
这幅图片展示了一只大而毛茸茸的狗，可能是拉布拉多或类似品种，坐在黄色和红色相间的门廊上。这只狗看起来在休息，它的目光朝向相机，表情平静。在狗的后面，有一辆红色自行车，车架上有黑色的装饰，停放在门廊上。自行车上挂着几个行李袋，表明它可能用于旅行或运输。背景中，可以看到一辆白色车辆，可能是汽车，停在门廊的后面。整个场景暗示了一个家庭环境，可能是在住宅区。

[N][                             Run][ 728]: hit eos,avg 5.55 token/s
```

## Reference

- [MiniCPM-V-2](openbmb/MiniCPM-V-2)

## 技术讨论

- Github issues
- QQ 群: 139953715
