# LocateAnything-3B (grounding / detection VLM)

`serve`/`run` support for [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
(AXERA build: `AXERA-TECH/LocateAnything-3B`), a Qwen2.5-3B visual-grounding model that does
zero-shot **object detection, phrase grounding, OCR / scene-text detection, document layout,
GUI grounding and pointing**. It is a direct port of the model's own
`infer_locateanything_axengine.py` reference.

## How it works

- **Vision**: image resized to 560×560 (Pillow bicubic), normalized `pixel/127.5 - 1`,
  patchified to `[1600, 3, 14, 14]`, run through `image_encoder_mlp.axmodel` → `400×2048`
  visual tokens (`VLMType::LocateAnythingVL`, reusing the PaddleOCR-VL `encode_block_normalized_float`
  path + a fixed-560 `LocateAnythingImageProcessor`).
- **Prompt** (`tokenizer_type = LocateAnything`): `<image N><img><IMG_CONTEXT>×400</img>` +
  your instruction; the 400 image embeddings are injected at the `<IMG_CONTEXT>` (id 151665) positions.
- **Output**: the model emits geometry as special tokens that render as plain text:
  - box: `<box><x1><y1><x2><y2></box>`
  - point: `<box><x><y></box>`
  - optional label: `<ref>...</ref>` before a box group
  - each `<N>` is a coordinate **normalized to 0–1000**; pixel = `N / 1000 * image_dim`.

## config.json

The shipped `config.json` omits the vision fields — add them:

```json
{
    "model_name": "AXERA-TECH/LocateAnything-3B",
    "url_tokenizer_model": "qwen2_5_tokenizer.txt",
    "tokenizer_type": "LocateAnything",
    "template_filename_axmodel": "qwen2_p128_l%d_together.axmodel",
    "axmodel_num": 36,
    "filename_post_axmodel": "qwen2_post.axmodel",
    "filename_tokens_embed": "model.embed_tokens.weight.bfloat16.bin",
    "tokens_embed_num": 152681,
    "tokens_embed_size": 2048,
    "vlm_type": "LocateAnythingVL",
    "filename_image_encoder_axmodel": "image_encoder_mlp.axmodel",
    "vision_width": 560,
    "vision_height": 560,
    "vision_patch_size": 14,
    "use_mmap_load_embed": true,
    "use_mmap_load_layer": true,
    "devices": [0]
}
```

## Usage

```shell
axllm serve /path/to/LocateAnything-3B --port 8010
```

Send an image + a task instruction via the OpenAI chat API (one image per request):

```json
{
  "model": "AXERA-TECH/LocateAnything-3B",
  "temperature": 0,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": "Locate all the instances that matches the following description:person"}
  ]}]
}
```

### Task prompts (from the reference)

| task | instruction template | output |
|--|--|--|
| object detection | `Locate all the instances that matches the following description:{categories}` | boxes |
| phrase grounding (single) | `Locate a single instance that matches the following description: {phrase}.` | box |
| phrase grounding (multi) | `Locate all the instances that match the following description: {phrase}.` | boxes |
| text grounding | `Please locate the text referred as {phrase}.` | boxes |
| scene-text detection / OCR | `Detect all the text in box format.` | `<ref>text</ref>` + boxes |
| document layout | `Detect all the objects in the image that belong to the category set: {categories}.` | boxes |
| GUI grounding (box) | `Locate the region that matches the following description: {phrase}.` | box |
| GUI grounding / pointing | `Point to: {phrase}.` | point |

## Notes

- **Use greedy decoding** (temperature 0 / sampling off) for the most reliable, well-formed
  geometry — the same guidance as other ≤4B models.
- Geometry is returned as text (`<box>…</box>`); parse it client-side and scale by the image
  size (`N/1000*dim`). Server-side structured-box output is a possible follow-up.
- Only supports Qwen2.5-based LocateAnything; one image per request.
