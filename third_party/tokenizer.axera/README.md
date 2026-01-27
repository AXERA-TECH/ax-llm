# tokenizer
方便好用的纯纯 C++ tokenizer 封装

## 支持的模型
[ModelType](include/BaseTokenizer.hpp#L6)

## 导出
导出模型的 `hf` 的路径的 `tokenizer` 到 `txt` 文件，方便在 C++ 中使用
```shell
cd tests
python convert_tokenizer.py --tokenizer_path Qwen/Qwen3-1.7B --dst_path assets/qwen3_tokenizer.txt 
```

## 使用
在 C++ 中使用导出的 `tokenizer` 文件，参考 `tests/main.cpp`
```C++
// 加载导出的 tokenizer 文件
std::shared_ptr<BaseTokenizer> tokenizer = create_tokenizer((ModelType)model_type.value());
if (!tokenizer->load("assets/qwen3_tokenizer.txt"))
{
    fprintf(stderr, "load tokenizer failed");
    return -1;
}

// 构建输入的 contents
std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "<think>\n\n</think>\n\n你好！有什么我可以帮助你的吗？"}};
std::vector<int> ids = tokenizer->encode(contents);
std::string text = tokenizer->decode(ids);
printf("text: \n%s\n", text.c_str());
```
输出
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么我可以帮助你的吗？<|im_end|>
```
加入新的对话内容
```C++
// 添加新的 USER 输入
contents.push_back({USER, TEXT, "你能做什么"});
auto ids2 = tokenizer->encode(contents);
text = tokenizer->decode(ids2);
printf("text: \n%s\n", text.c_str());
```
输出
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么我可以帮助你的吗？<|im_end|>
<|im_start|>user
你能做什么<|im_end|>
<|im_start|>assistant
```
VL模型如下
```C++
// 添加新的 IMAGE 输入，图片的数量为1，图片的 token 数量为 256
contents.push_back({USER, IMAGE, "帮我看看这张图片", 1, 256});

// 添加新的 VIDEO 输入，视频的图像数量为8，视频的 token 数量为 256
contents.push_back({USER, VIDEO, "帮我看看这个视频", 8, 256});
```
输出
```
过于先进，不便展示。
```


## reference
- [mnn-llm](https://github.com/wangzhaode/mnn-llm)
- [llm-export](https://github.com/wangzhaode/llm-export)
