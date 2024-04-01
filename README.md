# AX-LLM

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。

### 已支持芯片

- AX650A/AX650N

### 已支持模型

- TinyLLaMa-1.1B
- Qwen1.5-1.8B

### 获取地址

- [百度网盘](https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e)

## 源码编译

### *TODO*

## 运行示例

### TinyLLaMa-1.1B-BF16
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

- [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat)
