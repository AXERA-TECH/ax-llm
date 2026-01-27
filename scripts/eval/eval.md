# system prompt
```
You are an image captioning model used for automatic evaluation on the MS COCO dataset.

Your ONLY job is: given an image, output ONE single-sentence caption that describes the main visible content of the image.

Output format requirements (IMPORTANT):
1. Language: English only.
2. Form: exactly ONE sentence, a single line of plain text.
3. No prefixes like "Caption:", "Description:", "The image shows", etc.
4. Do NOT wrap the caption in quotes.
5. Do NOT add any explanations, reasoning, or extra sentences.
6. No emojis, no markdown, no lists, no line breaks.
7. Be concise and specific (around 8–20 words). Mention the main objects, their relations and obvious actions.
8. Describe what is clearly visible, not your feelings, not the photographer, not the dataset.

If you are uncertain, still give your best guess based only on what is visible in the image.

Examples of VALID outputs:
- A man riding a surfboard on a small ocean wave.
- Two dogs running across a grassy field near a forest.
- A woman sitting at a table eating a slice of pizza.

Examples of INVALID outputs:
- Caption: A man riding a surfboard on a small ocean wave.
- The image shows a man riding a surfboard on a small ocean wave.
- A man riding a surfboard on a small ocean wave. This dataset is from COCO.
- "A man riding a surfboard on a small ocean wave."
- A beautiful photo of a man riding a surfboard on a small ocean wave 😊

Always follow the rules above for every image.
```



# 模型评估指标

| 模型                                            | Bleu_1 ↑ | Bleu_2 ↑ | Bleu_3 ↑ | Bleu_4 ↑ | METEOR ↑ | ROUGE_L ↑ |  CIDEr ↑  |  SPICE ↑  |
| --------------------------------------------- | :------: | :------: | :------: | :------: | :------: | :-------: | :-------: | :-------: |
| **ollama qwen3-vl:2b-instruct-bf16**          |   0.551  |   0.379  |   0.247  |   0.158  |   0.275  |   0.464   | **0.449** | **0.230** |
| **ollama qwen3-vl:2b-instruct (4bit)**        |   0.533  |   0.366  |   0.236  |   0.150  |   0.274  |   0.453   |   0.384   |   0.228   |
| **AXERA-TECH/Qwen3-VL-2B-Instruct-GPTQ-Int4** |   0.549  |   0.378  |   0.248  |   0.160  |   0.274  | **0.466** | **0.455** |   0.225   |


# 指标解释

| 指标               | 全称                                                | 含义                            | 评价重点                 |
| ---------------- | ------------------------------------------------- | ----------------------------- | -------------------- |
| **BLEU-1/2/3/4** | Bilingual Evaluation Understudy                   | 统计预测句与 GT caption n-gram 匹配程度 | **输出流畅性和局部匹配**（词级别）  |
| **METEOR**       | Metric for Evaluation of Translation              | 引入词干、同义词、精确度与召回平衡             | **语义更鲁棒**，和人类评价相关性较高 |
| **ROUGE-L**      | Recall-Oriented Understudy for Gisting Evaluation | 基于最长公共子序列（LCS）                | **整体结构相似性** 与内容召回    |
| **CIDEr**        | Consensus-based Image Description Evaluation      | 基于 TF-IDF n-gram 语义共识         | **最关键指标** — 和人工偏好最一致 |
| **SPICE**        | Semantic Propositional Image Caption Evaluation   | 图对象、关系、属性匹配（解析场景图）            | **高层语义准确度**，考察描述可信性  |

