# AX-LLM

![GitHub License](https://img.shields.io/github/license/AXERA-TECH/ax-llm)

| Platform | Build Status |
| -------- | ------------ |
| AX650    | ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-llm/build_650.yml?internvl2)|

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。

### 已支持芯片

- AX650A/AX650N
  - SDK ≥ v1.45.0_P31

### 已支持模型

- Qwen2.5-VL-3B-Instruct

### 获取地址

comming soon

## 源码编译

-  clone 本项目  
    ```shell
    git clone  https://github.com/AXERA-TECH/ax-llm.git
    cd ax-llm
    ```
- 编译  
  ```
  mkdir build 
  cd build && cmake ..
  make install -j4
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

### 1. 图像理解

![demo.jpg](assets/demo.jpg)

#### 1. 首先启动 HTTP Tokenizer Server  
```
cd scripts
python qwen2_tokenizer_image_448.py --host {your host} --port {your port}   # 和 run_qwen2_5vl_image.sh 中一致
```

#### 2. 在板子上运行模型  
1) 先修改 `run_qwen2_5vl_image.sh` 中的http host.  
2) 将 `scripts/run_qwen2_5vl_image.sh`, `src/post_config.json` ,`build/install/bin/main`, `assets/demo.jpg` 拷贝到爱芯板子上  
3) 运行 `run_qwen2_5vl_image.sh`  
```shell
root@ax650 Qwen2.5-VL-3B-Instruct-Infer # bash run_qwen2_5vl_image.sh 
[I][                            Init][ 129]: LLM init start
bos_id: -1, eos_id: 151645
  2% | █                                 |   1 /  40 [0.05s<2.00s, 20.00 count/s] tokenizer init ok[I][                            Init][  26]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  40 /  40 [19.48s<19.48s, 2.05 count/s] init vpm axmodel ok,remain_cmm(2559 MB)650-prefill_320/qwen2_5_vl_p320_l35_together.axmodel ok
[I][                            Init][ 277]: max_token_len : 1023
[I][                            Init][ 282]: kv_cache_size : 256, kv_cache_num: 1023
[I][                            Init][ 290]: prefill_token_num : 320
[I][                            Init][ 292]: vpm_height : 1024,vpm_width : 392
[I][                            Init][ 301]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
prompt >> prompt >> Describe this image.
image >> demo.jpg
[I][                          Encode][ 416]: image encode time : 794.763000 ms, size : 524288
[I][                             Run][ 633]: ttft: 43535.27 ms
The image shows a person and a dog sitting on a sandy beach. The person is wearing a plaid shirt and shorts, and the dog is wearing a harness. They appear to be looking at something on a device, possibly a phone or tablet, which the person is holding. The beach is sandy and there are footprints in the sand. The background shows the ocean with waves crashing onto the shore, and the sun is low in the sky, suggesting it might be early morning or late afternoon.

[N][                             Run][ 774]: hit eos,avg 0.03 token/s
```

### 2. 视频理解

<div style="
    display: grid;
    grid-template-columns: repeat(4, 1fr);  /* 4列等宽 */
    grid-template-rows: repeat(2, 1fr);     /* 2行等高 */
    gap: 10px;                              /* 图片间距 */
    width: 80%;                             /* 容器宽度 */
    margin: 0 auto;                         /* 居中显示 */
">
    <img src="demo_cv308/frame_0075.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0077.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0079.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0081.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0083.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0085.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0087.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0089.jpg" style="width: 100%; height: 100%; object-fit: cover;">
</div>

#### 1. 首先启动 HTTP Tokenizer Server  
```
cd scripts
python qwen2_tokenizer_video_308.py --host {your host} --port {your port}   # 和 run_qwen2_5vl_video.sh 中一致
```

#### 2. 在板子上运行模型  
1) 先修改 `run_qwen2_5vl_video.sh` 中的http host.  
2) 将 `scripts/run_qwen2_5vl_video.sh`, `src/post_config.json` ,`build/install/bin/main`, `demo_cv308` 拷贝到爱芯板子上  
3) 运行 `run_qwen2_5vl_video.sh`  
```shell

```

## 图像理解推理速度  
| Stage | Time |
|------|------|
| Image Encoder (448x448) | 790 ms  | 
| Prefill (320) |  43535.27 ms    |
| Decode  |   token/s |

## 视频理解推理速度  
| Stage | Time |
|------|------|
| Image Encoder (8x308x308) |  ms  | 
| Prefill (512) |   ms    |
| Decode  |   token/s |

## Reference

- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## 技术讨论

- Github issues
- QQ 群: 139953715
