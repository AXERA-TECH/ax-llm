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

- InternVL2-1B

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

### InternVL2 1B

![dog](https://github.com/user-attachments/assets/fa58faaa-48ed-4550-a37c-8d6b39eef9b8)

```shell
root@ax650:/llm-test/internvl2-1b# ./run_internvl2.sh
[I][                            Init][ 127]: LLM init start
bos_id: -1, eos_id: 151645
  3% | ██                                |   1 /  28 [0.01s<0.14s, 200.00 count/s] tokenizer init ok[I][                            Init][  26]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  28 /  28 [1.43s<1.43s, 19.58 count/s] init vpm axmodel ok,remain_cmm(11116 MB)B)
[I][                            Init][ 275]: max_token_len : 1023
[I][                            Init][ 280]: kv_cache_size : 128, kv_cache_num: 1023
[I][                            Init][ 288]: prefill_token_num : 128
[I][                            Init][ 290]: vpm_height : 224,vpm_width : 224
[I][                            Init][ 299]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
prompt >> 描述下图片
image >> images/ssd_dog.jpg
[I][                          Encode][ 351]: image encode time : 397.944000 ms, size : 57344
[I][                             Run][ 561]: ttft: 124.73 ms

这张图片展示了一只狗坐在阳台上，正在用嘴咬着一辆自行车。阳台的背景是黄色的墙壁，可以看到外面的街道和一辆停着的汽车。阳台的窗户上挂着白色的窗帘，阳光透过窗户照进来，给整个场景增添了一种温暖的氛围。

[N][                             Run][ 700]: hit eos,avg 31.16 token/s
```

## Reference

- [InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B)

## 技术讨论

- Github issues
- QQ 群: 139953715
