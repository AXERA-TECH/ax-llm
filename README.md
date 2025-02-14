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
- AX630C
  - SDK ≥ v2.0.0_P7

### 已支持模型

- Qwen1.5-0.5B/1.8B/4B
- Qwen2-0.5B/1.5B
- ChatGLM3-6B
- MiniCPM-2B
- TinyLLaMa-1.1B
- Llama2-7B
- Llama3-8B
- Phi-2
- Phi-3-mini

### 获取地址

- [百度网盘](https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e)
- [Google Drive](https://drive.google.com/drive/folders/1i8xdD2PWDlueouds6F1dhMc72n3v_aER?usp=sharing)

## 源码编译

- 在 Host 上下载 axcl llm 对应分支
    ```shell
    git clone -b axcl-llm https://github.com/AXERA-TECH/ax-llm.git
    cd ax-llm
    ```
- 本地编译
    ```shell
    mkdir build
    cd build
    cmake ..
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

### Phi-3-mini-int8

```shell
./run_phi3_mini.sh
[I][                            Init][  71]: LLM init start
100% | ████████████████████████████████ |  35 /  35 [28.39s<28.39s, 1.23 count/s] init post axmodel okremain_cmm(9045 MB))
[I][                            Init][ 180]: max_token_len : 1023
[I][                            Init][ 185]: kv_cache_size : 3072, kv_cache_num: 1023
[I][                            Init][ 199]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
>>
>> who are you?
 I am Phi, an AI developed by Microsoft, designed to assist and provide information to users across a wide range of topics. How can I assist you today?

[N][                             Run][ 388]: hit eos,avg 4.40 token/s

>>
>> use c program language implement calculate sum 1-9
 Certainly! To calculate the sum of numbers from 1 to 9 in C, you can use a simple loop. Here's a small program that does exactly that:

`c
#include <stdio.h>

int main() {
    int sum = 0; // Initialize sum to 0

    // Loop from 1 to 9 and add each number to sum
    for(int i = 1; i <= 9; i++) {
        sum += i;
    }

    printf("The sum of numbers from 1 to 9 is: %d\n", sum);

    return 0;
}
`

This program initializes a variable `sum` to 0. It then iterates from 1 to 9, adding each number to `sum`. Finally, it prints the result.

The sum of numbers from 1 to 9 is 45, as calculated by the loop in the program.

[N][                             Run][ 388]: hit eos,avg 4.37 token/s

```

### TinyLLaMa-1.1B-BF16


https://github.com/AXERA-TECH/ax-llm/assets/46700201/e592f78a-03ca-4824-b2d3-46d48740dbef


```shell
# ./run_bf16.sh
[I][                            Init][  71]: LLM init start
100% | ████████████████████████████████ |  25 /  25 [21.82s<21.82s, 1.15 count/s] init post axmodel okremain_cmm(3760 MB)
[I][                            Init][ 162]: max_token_len : 1023
[I][                            Init][ 167]: kv_cache_size : 256, kv_cache_num: 1023
[I][                            Init][ 176]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
>> write a c++ program to calculate 1-9 sum

`c++
#include <iostream>

using namespace std;

int main() {
    int sum = 0;
    int num;

    cout << "Enter a number: ";
    cin >> num;

    for (int I = 1; I <= num; i++) {
        sum += i;
    }

    cout << "The sum of 1 to " << num << " is: " << sum << endl;

    return 0;
}
`


this program prompts the user to enter a number, then calculates the sum of 1 to `num` using a loop. The loop iterates from 1 to `num`, adding each number to the sum variable `sum`. Finally, the program prints the sum of 1 to `num`.

[N][                             Run][ 366]: hit eos,avg 10.14 token/s

>> where is shenzhen
Shenzhen is a city located in southern China, on the southern coast of the Pearl River Delta. It is the capital of Guangdong province and one of the most important economic and cultural centers in China. Shenzhen is known for its innovation, technology, and entrepreneurship, and it is home to many of China's largest companies, including Huawei, Lenovo, and Xiaomi. The city is also a hub for international trade and investment, with many multinational corporations and financial institutions setting up offices and operations in Shenzhen

[N][                             Run][ 366]: hit eos,avg 10.16 token/s

```

### Qwen1.5-1.8B-int8


https://github.com/AXERA-TECH/ax-llm/assets/46700201/6788565f-19a5-45cb-9d17-41e2260886a2


```shell
# ./run_qwen_1.8B.sh
[I][                            Init][  71]: LLM init start
100% | ████████████████████████████████ |  27 /  27 [23.68s<23.68s, 1.14 count/s] init post axmodel okremain_cmm(4140 MB)
[I][                            Init][ 162]: max_token_len : 1023
[I][                            Init][ 167]: kv_cache_size : 2048, kv_cache_num: 1023
[I][                            Init][ 176]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
>> 给我写个C++代码计算1-9的和
以下是一个简单的C++代码，用于计算1-9的和：

`cpp
#include <iostream>

int main() {
    int sum = 1;
    for (int i = 1; i <= 9; i++) {
        sum += i;
    }
    std::cout << "The sum of 1-9 is: " << sum << std::endl;
    return 0;
}
`

在这个代码中，我们首先定义了一个变量`sum`，并将其初始化为1。然后，我们使用一个`for`循环，从1开始，每次增加1，直到10。在每次循环中，我们都会将当前的数`i`加到`sum`中。最后，我们在主函数中打印出`sum`的值，即1-9的和。


[N][                             Run][ 366]: hit eos,avg 8.00 token/s

>> 深圳在哪
深圳位于中国广东省南部，珠江口东侧，东临大亚湾，西濒珠江口，南界深圳湾，北界香港特别行政区，是中国四大一线城市之一，也是中国经济最发达的城市之一。


[N][                             Run][ 366]: hit eos,avg 8.44 token/s
```



## Reference

- [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)
- [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)

## 技术讨论

- Github issues
- QQ 群: 139953715

