---
license: mit
pipeline_tag: image-text-to-text
library_name: transformers
base_model:
  - OpenGVLab/InternVL2_5-8B
base_model_relation: finetune
datasets:
  - OpenGVLab/MMPR-v1.1
language:
  - multilingual
tags:
  - internvl
  - custom_code
---

# InternVL2_5-8B-MPO

[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVL)  [\[📜 InternVL 1.0\]](https://huggingface.co/papers/2312.14238)  [\[📜 InternVL 1.5\]](https://huggingface.co/papers/2404.16821)  [\[📜 InternVL 2.5\]](https://huggingface.co/papers/2412.05271)  [\[📜 InternVL2.5-MPO\]](https://huggingface.co/papers/2411.10442)

[\[🆕 Blog\]](https://internvl.github.io/blog/)  [\[🗨️ Chat Demo\]](https://internvl.opengvlab.com/)  [\[🤗 HF Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL)  [\[🚀 Quick Start\]](#quick-start)  [\[📖 Documents\]](https://internvl.readthedocs.io/en/latest/)

<div align="center">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/64006c09330a45b03605bba3/zJsd2hqd3EevgXo6fNgC-.png">
</div>

## Introduction

We introduce InternVL2.5-MPO, an advanced multimodal large language model (MLLM) series that demonstrates superior overall performance. This series builds upon InternVL2.5 and Mixed Preference Optimization.

![image/png](https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/images/overview_performance.png)

## InternVL 2.5 Family

In the following table, we provide an overview of the InternVL2.5-MPO series.

|     Model Name      |                                       Vision Part                                       |                                 Language Part                                  |                            HF Link                             |
| :-----------------: | :-------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :------------------------------------------------------------: |
| InternVL2_5-1B-MPO  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |   [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-1B-MPO)  |
| InternVL2_5-2B-MPO  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) | [internlm2_5-1_8b-chat](https://huggingface.co/internlm/internlm2_5-1_8b-chat) | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-2B-MPO)  |
| InternVL2_5-4B-MPO  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |     [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)     | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-4B-MPO)  |
| InternVL2_5-8B-MPO  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |   [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat)   | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO)  |
| InternVL2_5-26B-MPO |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |  [internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)  | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-26B-MPO) |
| InternVL2_5-38B-MPO |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |    [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-38B-MPO) |
| InternVL2_5-78B-MPO |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |    [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)    | [🤗 link](https://huggingface.co/OpenGVLab/InternVL2_5-78B-MPO) |

## Model Architecture

As shown in the following figure, [InternVL2.5-MPO](https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/) retains the same model architecture as [InternVL 2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/) and its predecessors, InternVL 1.5 and 2.0, following the "ViT-MLP-LLM" paradigm. In this new version, we integrate a newly incrementally pre-trained InternViT with various pre-trained LLMs, including InternLM 2.5 and Qwen 2.5, using a randomly initialized MLP projector.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64119264f0f81eb569e0d569/BiiyXN6NOk0p-3rl3ueyL.png)

As in the previous version, we applied a pixel unshuffle operation, reducing the number of visual tokens to one-quarter of the original. Besides, we adopted a similar dynamic resolution strategy as InternVL 1.5, dividing images into tiles of 448×448 pixels. The key difference, starting from InternVL 2.0, is that we additionally introduced support for multi-image and video data.

## Key Designs

### Multi-Modal Preference Dataset

MMPR is a large-scale and high-quality multimodal reasoning preference dataset. This dataset includes about 3 million samples.

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/619507e7b74b6c591f794340/mmXL47UPDFwYOWdn9Z6j5.jpeg)
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/619507e7b74b6c591f794340/6fnvI_wCd9JXAs6vYthaG.jpeg)

To construct this dataset, we propose an efficient data construction pipeline. Specifically, we categorize the multimodal data into **samples with clear ground truths** and **samples without clear ground truths**.

- **For samples with clear ground truths:**
  the model is prompted to first provide the reasoning process and then give the final answer in the format like `Final Answer: ***`.
  Responses matching the ground truth answer constitute the positive set \\(\mathcal{Y}_p\\), while those that do not match make up the negative set \\(\mathcal{Y}_n\\). Additionally, responses that fail to provide a clear final answer are also merged into \\(\mathcal{Y}_n\\).
  Given these responses labeled as positive or negative, we build the preference pairs by selecting a chosen response \\(y_c\\) from \\(\mathcal{Y}_p\\) and a negative response \\(y_r\\) from \\(\mathcal{Y}_n\\).

- **For samples without clear ground truths:**
  we propose a simple yet effective method: Dropout Next-Token Prediction (Dropout NTP).
  Specifically, we use the responses generated by InternVL2-8B as chosen answers.
  Given the chosen answer, we truncate it by half and then prompt InternVL2-8B to complete the remaining
  portion of the truncated answer without access to the image input.
  This generated completion serves as the rejected answer for the paired sample.
  It is worth noting that while the responses generated by InternVL2-8B may not be perfect,
  the completions generated without the image input will introduce more hallucinations than those
  generated with the image input.
  Therefore, the partial order relationship between the chosen and rejected responses holds true.

The data construction pipeline is open-sourced, see more details in our [document](https://internvl.readthedocs.io/en/latest/internvl2.0/preference_optimization.html#generate-additional-preference-data).


### Mixed Preference Optimization

The key insight behind MPO is that *an effective PO process should enable the model to learn the relative preference between pairs of responses, the absolute quality of individual responses, and the process for generating preferred responses.* We define the training objective as a combination of
preference loss \\(\mathcal{L}_{\text{p}}\\),
quality loss \\(\mathcal{L}_{\text{q}}\\),
and generation loss \\(\mathcal{L}_{\text{g}}\\),
referred to as Mixed Preference Optimization:

$$
\mathcal{L}=w_{p}\cdot\mathcal{L}_{\text{p}} + w_{q}\cdot\mathcal{L}_{\text{q}} + w_{g}\cdot\mathcal{L}_{\text{g}},
$$

where \\(w_{*}\\) represents the weight assigned to each loss component.
In this work, we empirically compare different variants of preference loss.
Based on the experimental results, we use DPO as our preference loss and BCO as our quality loss.

Specifically, the DPO serves as the preference loss to enable the model to learn the
relative preference between chosen and rejected responses.
This algorithm optimizes the following loss function:

$$
\mathcal{L}_{\text{p}}=-\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_c \mid x\right)}{\pi_0\left(y_c \mid x\right)}-\beta \log \frac{\pi_\theta\left(y_r \mid x\right)}{\pi_0\left(y_r \mid x\right)}\right),
$$

where \\(\beta\\) is the KL penalty coefficient, and \\(x\\), \\(y_c\\), and \\(y_r\\) are user query, chosen response, and rejected response, respectively.
The policy model \\(\pi_\theta\\) is initialized from model \\(\pi_0\\).

Additionally, the BCO loss is employed as the quality loss, which helps the model to understand the absolute quality of individual responses.
The loss function is defined as:

$$
\mathcal{L}_{\text{q}}=\mathcal{L}_{\text{q}}^+ + \mathcal{L}_{\text{q}}^-,
$$

where \\(\mathcal{L}_{\text{q}}^{+}\\) and \\(\mathcal{L}_{\text{q}}^{+}\\) represent the loss for chosen and rejected responses, respectively.
Each response type's loss is calculated independently, requiring the model to differentiate the absolute quality of individual responses. The loss terms are given by:

$$
\mathcal{L}_{\text{q}}^+=-\log \sigma\left(\beta \log \frac{\pi_\theta\left(y_c \mid x\right)}{\pi_0\left(y_c \mid x\right)} - \delta\right),
$$

$$
\mathcal{L}_{\text{q}}^-=-\log \sigma\left(-\left(\beta \log \frac{\pi_\theta\left(y_r \mid x\right)}{\pi_0\left(y_r \mid x\right)} - \delta\right) \right),
$$

where \\(\delta\\) represents the reward shift, calculated as the moving average of previous rewards to stabilize training.

Finally, the SFT loss is used as the generation loss to help the model learn the generation process of preferred responses.
The loss function is defined as:

$$
\mathcal{L}_{\text{gen}}=-\frac{\log\pi_\theta\left(y_c \mid x\right)}{\left| y_c \right|}.
$$

## Evaluation on Multimodal Capability

To comprehensively compare InternVL's performance before and after MPO, we employ the benchmarks from OpenCompass Learderboard, including both well-established classic datasets and newly introduced ones. These benchmarks span a wide range of categories, aiming to provide a thorough and balanced assessment of InternVL’s capabilities across various multimodal tasks. We provide the evaluation results in the tables behind.

| Model               | Avg. | MMBench v1.1 | MMStar | MMMU | MathVista | HallusionBench | AI2D | OCRBench | MMVet |
| ------------------- | ---- | ------------ | ------ | ---- | --------- | -------------- | ---- | -------- | ----- |
| InternVL2-5-1B      | 54.9 | 66.5         | 51.3   | 41.2 | 47.1      | 39.4           | 69.0 | 77.4     | 47.2  |
| InternVL2-5-1B-MPO  | 56.4 | 67.2         | 49.7   | 40.8 | 53.0      | 40.0           | 69.4 | 83.6     | 47.2  |
| InternVL2-5-2B      | 59.9 | 70.9         | 54.3   | 43.2 | 51.1      | 42.3           | 74.9 | 80.2     | 62.6  |
| InternVL2-5-2B-MPO  | 62.0 | 71.6         | 55.0   | 45.0 | 56.4      | 43.0           | 75.3 | 84.2     | 65.4  |
| InternVL2-5-4B      | 65.1 | 78.2         | 58.7   | 51.8 | 60.8      | 46.6           | 81.4 | 82.0     | 61.5  |
| InternVL2-5-4B-MPO  | 67.6 | 78.6         | 60.2   | 51.6 | 65.3      | 47.8           | 82.0 | 88.0     | 67.1  |
| InternVL2-5-8B      | 68.9 | 82.5         | 63.2   | 56.2 | 64.5      | 49.0           | 84.6 | 82.1     | 62.8  |
| InternVL2-5-8B-MPO  | 70.4 | 82.4         | 65.7   | 54.9 | 68.9      | 51.4           | 84.5 | 88.3     | 66.9  |
| InternVL2-5-26B     | 71.6 | 84.6         | 66.5   | 60.7 | 68.0      | 55.8           | 86.2 | 85.4     | 65.4  |
| InternVL2-5-26B-MPO | 72.7 | 84.2         | 67.2   | 57.7 | 72.8      | 55.3           | 86.2 | 91.2     | 67.1  |
| InternVL2-5-38B     | 73.5 | 85.4         | 68.5   | 64.6 | 72.4      | 57.9           | 87.6 | 84.1     | 67.2  |
| InternVL2-5-38B-MPO | 75.5 | 85.6         | 69.8   | 64.1 | 73.8      | 61.5           | 88.1 | 88.5     | 72.5  |
| InternVL2-5-78B     | 75.2 | 87.5         | 69.5   | 70.0 | 70.6      | 57.4           | 89.1 | 85.3     | 71.8  |
| InternVL2-5-78B-MPO | 76.6 | 87.3         | 73.1   | 68.3 | 73.8      | 58.7           | 89.3 | 91.2     | 71.4  |


## Quick Start

We provide an example code to run `InternVL2_5-8B-MPO` using `transformers`.

> Please use transformers>=4.37.2 to ensure the model works normally.

### Model Loading

#### 16-bit (bf16 / fp16)

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-8B-MPO"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
```

#### BNB 8-bit Quantization

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2_5-8B-MPO"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval()
```

#### Multiple GPUs

The reason for writing the code this way is to avoid errors that occur during multi-GPU inference due to tensors not being on the same device. By ensuring that the first and last layers of the large language model (LLM) are on the same device, we prevent such errors.

```python
import math
import torch
from transformers import AutoTokenizer, AutoModel

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

path = "OpenGVLab/InternVL2_5-8B-MPO"
device_map = split_model('InternVL2_5-8B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
```

### Inference with Transformers

```python
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'OpenGVLab/InternVL2_5-8B-MPO'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)

# pure-text conversation (纯文本对话)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Can you tell me a story?'
response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# single-image single-round conversation (单图单轮对话)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# single-image multi-round conversation (单图多轮对话)
question = '<image>\nPlease describe the image in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Please write a poem according to the image.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

question = '<image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'What are the similarities and differences between these two images.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list,
                               history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')

# batch inference, single image per sample (单图批处理)
pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

video_path = './examples/red-panda.mp4'
pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
question = video_prefix + 'What is the red panda doing?'
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

question = 'Describe this video in detail.'
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')
```

#### Streaming Output

Besides this method, you can also use the following code to get streamed output.

```python
from transformers import TextIteratorStreamer
from threading import Thread

# Initialize the streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
# Define the generation configuration
generation_config = dict(max_new_tokens=1024, do_sample=False, streamer=streamer)
# Start the model chat in a separate thread
thread = Thread(target=model.chat, kwargs=dict(
    tokenizer=tokenizer, pixel_values=pixel_values, question=question,
    history=None, return_history=False, generation_config=generation_config,
))
thread.start()

# Initialize an empty string to store the generated text
generated_text = ''
# Loop through the streamer to get the new text as it is generated
for new_text in streamer:
    if new_text == model.conv_template.sep:
        break
    generated_text += new_text
    print(new_text, end='', flush=True)  # Print each new chunk of generated text on the same line
```

## Finetune

Many repositories now support fine-tuning of the InternVL series models, including [InternVL](https://github.com/OpenGVLab/InternVL), [SWIFT](https://github.com/modelscope/ms-swift), [XTurner](https://github.com/InternLM/xtuner), and others. Please refer to their documentation for more details on fine-tuning.

## Deployment

### LMDeploy

LMDeploy is a toolkit for compressing, deploying, and serving LLMs & VLMs.

```sh
pip install lmdeploy>=0.6.4
```

LMDeploy abstracts the complex inference process of multi-modal Vision-Language Models (VLM) into an easy-to-use pipeline, similar to the Large Language Model (LLM) inference pipeline.

#### A 'Hello, world' Example

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2_5-8B-MPO'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
response = pipe(('describe this image', image))
print(response.text)
```

If `ImportError` occurs while executing this case, please install the required dependency packages as prompted.

#### Multi-images Inference

When dealing with multiple images, you can put them all in one list. Keep in mind that multiple images will lead to a higher number of input tokens, and as a result, the size of the context window typically needs to be increased.

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL2_5-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg',
    'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)
```

#### Batch Prompts Inference

Conducting inference with batch prompts is quite straightforward; just place them within a list structure:

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2_5-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image_urls=[
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg",
    "https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/det.jpg"
]
prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print(response)
```

#### Multi-turn Conversation

There are two ways to do the multi-turn conversations with the pipeline. One is to construct messages according to the format of OpenAI and use above introduced method, the other is to use the `pipeline.chat` interface.

```python
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2_5-8B-MPO'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8)
sess = pipe.chat(('describe this image', image), gen_config=gen_config)
print(sess.response.text)
sess = pipe.chat('What is the woman doing?', session=sess, gen_config=gen_config)
print(sess.response.text)
```

#### Service

LMDeploy's `api_server` enables models to be easily packed into services with a single command. The provided RESTful APIs are compatible with OpenAI's interfaces. Below are an example of service startup:

```shell
lmdeploy serve api_server OpenGVLab/InternVL2_5-8B-MPO --server-port 23333
```

To use the OpenAI-style interface, you need to install OpenAI:

```shell
pip install openai
```

Then, use the code below to make the API call:

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'describe this image',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
            },
        }],
    }],
    temperature=0.8,
    top_p=0.8)
print(response)
```

## License

This project is released under the MIT License. This project uses the pre-trained internlm2_5-7b-chat as a component, which is licensed under the Apache License 2.0.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{wang2024mpo,
  title={Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization},
  author={Wang, Weiyun and Chen, Zhe and Wang, Wenhai and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Zhu, Jinguo and Zhu, Xizhou and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2411.10442},
  year={2024}
}
@article{chen2024expanding,
  title={Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  author={Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2412.05271},
  year={2024}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
@inproceedings{chen2024internvl,
  title={Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24185--24198},
  year={2024}
}
```
