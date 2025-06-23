---
license: other
license_name: qwen
license_link: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE
pipeline_tag: image-text-to-text
library_name: transformers
base_model:
  - OpenGVLab/InternViT-300M-448px-V2_5
  - Qwen/Qwen2.5-1.5B
base_model_relation: merge
datasets:
  - OpenGVLab/MMPR-v1.2
language:
  - multilingual
tags:
  - internvl
  - custom_code
---

# InternVL3-2B

[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVL)  [\[📜 InternVL 1.0\]](https://huggingface.co/papers/2312.14238)  [\[📜 InternVL 1.5\]](https://huggingface.co/papers/2404.16821)  [\[📜 InternVL 2.5\]](https://huggingface.co/papers/2412.05271)  [\[📜 InternVL2.5-MPO\]](https://huggingface.co/papers/2411.10442)  [\[📜 InternVL3\]](TBD)

[\[🆕 Blog\]](https://internvl.github.io/blog/)  [\[🗨️ Chat Demo\]](https://internvl.opengvlab.com/)  [\[🤗 HF Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL)  [\[🚀 Quick Start\]](#quick-start)  [\[📖 Documents\]](https://internvl.readthedocs.io/en/latest/)

<div align="center">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/64006c09330a45b03605bba3/zJsd2hqd3EevgXo6fNgC-.png">
</div>

## Introduction

We introduce InternVL3, an advanced multimodal large language model (MLLM) series that demonstrates superior overall performance.
Compared to InternVL 2.5, InternVL3 exhibits superior multimodal perception and reasoning capabilities, while further extending its multimodal capabilities to encompass tool usage, GUI agents, industrial image analysis, 3D vision perception, and more.
Additionally, we compare InternVL3 with  Qwen2.5 Chat models, whose corresponding pre-trained base models are employed as the initialization of the langauge component in InternVL3. Benefitting from Native Multimodal Pre-Training, the InternVL3 series achieves even better overall text performance than the Qwen2.5 series.

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/overall.png)

## InternVL3 Family

In the following table, we provide an overview of the InternVL3 series.

|  Model Name   |                                       Vision Part                                       |                                 Language Part                                  |                         HF Link                          |
| :-----------: | :-------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :------------------------------------------------------: |
| InternVL3-1B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |            [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)            | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-1B)  |
| InternVL3-2B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |            [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)            | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-2B)  |
| InternVL3-8B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |              [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)              | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-8B)  |
| InternVL3-9B  | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) | [internlm3-8b-instruct](https://huggingface.co/internlm/internlm3-8b-instruct) | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-9B)  |
| InternVL3-14B | [InternViT-300M-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-300M-448px-V2_5) |             [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B)             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-14B) |
| InternVL3-38B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |             [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-38B) |
| InternVL3-78B |   [InternViT-6B-448px-V2_5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)   |             [Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B)             | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-78B) |

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/overall-table.png)

## Model Architecture

As shown in the following figure, [InternVL3](https://internvl.github.io/blog/2025-04-11-InternVL-3/) retains the same model architecture as [InternVL 2.5](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/) and its predecessors, InternVL 1.5 and 2.0, following the "ViT-MLP-LLM" paradigm. In this new version, we integrate a newly incrementally pre-trained InternViT with various pre-trained LLMs, including InternLM 3 and Qwen 2.5, using a randomly initialized MLP projector.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/64119264f0f81eb569e0d569/BiiyXN6NOk0p-3rl3ueyL.png)

As in the previous version, we applied a pixel unshuffle operation, reducing the number of visual tokens to one-quarter of the original. Besides, we adopted a similar dynamic resolution strategy as InternVL 1.5, dividing images into tiles of 448×448 pixels. The key difference, starting from InternVL 2.0, is that we additionally introduced support for multi-image and video data.

Notably, in InternVL3, we integrate the [Variable Visual Position Encoding (V2PE)](https://arxiv.org/abs/2412.09616), which utilizes smaller, more flexible position increments for visual tokens. Benefiting from V2PE, InternVL3 exhibits better long context understanding capabilities compared to its predecessors.

## Training Strategy

### Native Multimodal Pre-Training

We propose a [Native Multimodal Pre-Training](TBD) approach that consolidates language and vision learning into a single pre-training stage.
In contrast to standard paradigms that first train a language-only model and subsequently adapt it to handle additional modalities, our method interleaves multimodal data (e.g., image-text, video-text, or image-text interleaved sequences) with large-scale textual corpora. This unified training scheme allows the model to learn both linguistic and multimodal representations simultaneously, ultimately enhancing its capability to handle vision-language tasks without the need for separate alignment or bridging modules.
Please see [our paper](TBD) for more details.

### Supervised Fine-Tuning

In this phase, the techniques of random JPEG compression, square loss re-weighting, and multimodal data packing proposed in [InternVL2.5](https://arxiv.org/abs/2412.05271) are also employed in the InternVL3 series.
The main advancement of the SFT phase in InternVL3 compared to InternVL2.5 lies in the use of higher-quality and more diverse training data.
Specifically, we further extend  training samples for tool use, 3D scene understanding, GUI operations, long context tasks, video understanding, scientific diagrams, creative writing, and multimodal reasoning.

### Mixed Preference Optimization

During Pre-training and SFT, the model is trained to predict the next token conditioned on previous ground-truth tokens.
However, during inference, the model predicts each token based on its own prior outputs. 
This discrepancy between ground-truth tokens and model-predicted tokens introduces a distribution shift, which can impair the model’s Chain-of-Thought (CoT) reasoning capabilities.
To mitigate this issue, we employ [MPO](https://arxiv.org/abs/2411.10442), which introduces additional supervision from both positive and negative samples to align the model response distribution with the ground-truth distribution, thereby improving reasoning performance.
Specifically, the training objective of MPO is a combination of
preference loss \\(\mathcal{L}_{\text{p}}\\),
quality loss \\(\mathcal{L}_{\text{q}}\\),
and generation loss \\(\mathcal{L}_{\text{g}}\\),
which can be formulated as follows:


$$
\mathcal{L}=w_{p}\cdot\mathcal{L}_{\text{p}} + w_{q}\cdot\mathcal{L}_{\text{q}} + w_{g}\cdot\mathcal{L}_{\text{g}},
$$


where \\(w_{*}\\) represents the weight assigned to each loss component. Please see [our paper](https://arxiv.org/abs/2411.10442) for more details about MPO.


### Test-Time Scaling

Test-Time Scaling has been shown to be an effective method to enhance the reasoning abilities of LLMs and MLLMs.
In this work, we use the Best-of-N evaluation strategy and employ [VisualPRM-8B](https://huggingface.co/OpenGVLab/VisualPRM-8B) as the critic model to select the best response for reasoning and mathematics evaluation.

## Evaluation on Multimodal Capability

### Multimodal Reasoning and Mathematics

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/reasoning.png)

### OCR, Chart, and Document Understanding

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/ocr.png)

### Multi-Image & Real-World Comprehension

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/multi-images.png)

### Comprehensive Multimodal & Hallucination Evaluation

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/comprehensive.png)

### Visual Grounding

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/grounding.png)

### Multimodal Multilingual Understanding

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/multilingual.png)

### Video Understanding

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/video.png)

### GUI Grounding

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/gui.png)

### Spatial Reasoning

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/vsi.png)

## Evaluation on Language Capability

We compare InternVL3 with  Qwen2.5 Chat models, whose corresponding pre-trained base models are employed as the initialization of the langauge component in InternVL3.
Benefitting from Native Multimodal Pre-Training, the InternVL3 series achieves even better overall text performance than the Qwen2.5 series.
Please note that the evaluation scores of Qwen2.5 series  may differ from those officially reported, as we have adopted the prompt versions provided in the table across all datasets for OpenCompass evaluation.

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/text.png)

## Ablation Study

### Native Multimodal Pre-Training

We conduct experiments on the InternVL2-8B model while keeping its architecture, initialization parameters, and training data entirely unchanged. Traditionally, InternVL2-8B employs a training pipeline that begins with an MLP warmup phase for feature alignment followed by an Instruction Tuning stage. In our experiments, we substitute the conventional MLP warmup phase with a native multimodal pre-training process. This modification isolates the contribution of native multimodal pre-training to the overall multimodal capability of the model.

The evaluation results in the Figure below shows that the model with native multimodal pre-training exhibits performance on most benchmarks that is comparable to the fully multi-stage-trained InternVL2-8B baseline. Furthermore, when followed by instruction tuning on higher-quality data, the model demonstrates further performance gains across evaluated multimodal tasks. These findings underscore the efficiency of native multimodal pre-training in imparting powerful multimodal capabilities to MLLMs.

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/ablation-native.png)

### Mixed Preference Optimization

As shown in the table below, models fine-tuned with MPO demonstrate superior reasoning performance across seven multimodal reasoning benchmarks compared to their counterparts without MPO. Specifically, InternVL3-78B and InternVL3-38B outperform their counterparts by 4.1 and 4.5 points, respectively. Notably, the training data used for MPO is a subset of that used for SFT, indicating that the performance improvements primarily stem from the training algorithm rather than the training data.

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/ablation-mpo.png)

### Variable Visual Position Encoding

As reported in the table below, the introduction of V2PE leads to significant performance gains across most evaluation metrics. In addition, our ablation studies—by varying the positional increment \\( \delta \\)—reveal that even for tasks primarily involving conventional contexts, relatively small \\( \delta \\) values can achieve optimal performance. These findings provide important insights for future efforts aimed at refining position encoding strategies for visual tokens in MLLMs.

![image/png](https://huggingface.co/datasets/Weiyun1025/InternVL-Performance/resolve/main/internvl3/ablation-v2pe.png)

## Quick Start

We provide an example code to run `InternVL3-2B` using `transformers`.

> Please use transformers>=4.37.2 to ensure the model works normally.

### Model Loading

#### 16-bit (bf16 / fp16)

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL3-2B"
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
path = "OpenGVLab/InternVL3-2B"
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
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
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

path = "OpenGVLab/InternVL3-2B"
device_map = split_model('InternVL3-2B')
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
import math
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

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
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

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
path = 'OpenGVLab/InternVL3-2B'
device_map = split_model('InternVL3-2B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
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
# if lmdeploy<0.7.3, you need to explicitly set chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
pip install lmdeploy>=0.7.3
```

LMDeploy abstracts the complex inference process of multi-modal Vision-Language Models (VLM) into an easy-to-use pipeline, similar to the Large Language Model (LLM) inference pipeline.

#### A 'Hello, world' Example

```python
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL3-2B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
response = pipe(('describe this image', image))
print(response.text)
```

If `ImportError` occurs while executing this case, please install the required dependency packages as prompted.

#### Multi-images Inference

When dealing with multiple images, you can put them all in one list. Keep in mind that multiple images will lead to a higher number of input tokens, and as a result, the size of the context window typically needs to be increased.

```python
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = 'OpenGVLab/InternVL3-2B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))

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
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL3-2B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))

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
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL3-2B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))

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
lmdeploy serve api_server OpenGVLab/InternVL3-2B --chat-template internvl2_5 --server-port 23333 --tp 1
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

This project is released under the MIT License. This project uses the pre-trained Qwen2.5 as a component, which is licensed under the Qwen License.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2024expanding,
  title={Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  author={Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2412.05271},
  year={2024}
}
@article{wang2024mpo,
  title={Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization},
  author={Wang, Weiyun and Chen, Zhe and Wang, Wenhai and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Zhu, Jinguo and Zhu, Xizhou and Lu, Lewei and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2411.10442},
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