# Vision/VLM Branch Patterns (Notes)

This repo's `axllm` branch currently runs **text-only** models by:

- `tokenizer->encode(history)` to token ids
- compute `tokens_diff` vs `last_tokens_ids` (append-only fast path)
- `SetKVCache(k/v, precompute_len, input_num_token)`
- build `out_embed` for `tokens_diff` via `embed_selector.getByIndex(id)`
- `Run(out_embed)` to do prefill/decode
- `GetKVCache(...)`, append assistant reply, update `last_tokens_ids`

So the "context" (KV cache) support on `axllm` is based on **token-id diff** + caching.

Below are distilled patterns from the VLM branches (Qwen/InternVL/FastVLM/SmolVLM2, incl. multi-frame/video variants).

## 1) How Images/Videos Get Into The Prompt

All VLM branches follow the same high-level idea:

1. The tokenizer's chat template emits **placeholder tokens** for each media item.
2. The vision encoder produces **per-placeholder embedding vectors**.
3. The code finds the placeholder positions in `input_ids` and **replaces** the corresponding token embeddings in `out_embed`.

The differences are in (a) which placeholder tokens are used and (b) how to locate them.

### A. Qwen3/Qwen2.5-VL style

- Prompt uses `<|vision_start|> ... <|vision_end|>` wrapping.
- Inside the vision block, placeholders are repeated:
  - image: `<|image_pad|>` repeated `num_media * num_media_tokens`
  - video: `<|video_pad|>` repeated `num_media * num_media_tokens`
- Injection offsets are typically found by scanning for `vision_start_token_id` and taking `offset=i+1` (first placeholder after start).

### B. InternVL style

- Prompt uses `<img> ... </img>` wrapping.
- Placeholder token is usually `<IMG_CONTEXT>` repeated `num_media_tokens`.
- Some templates place `content.data` before placeholders; others after.
- Injection offsets are found by scanning `input_ids` for `IMAGE_CONTEXT_TOKEN` (or via `<|vision_start|>` then next token).

### C. FastVLM style

- Similar to InternVL: uses a simple placeholder token (e.g. `"<image>"`) repeated.
- Code scans for a fixed `IMAGE_CONTEXT_TOKEN` id in `input_ids` and overwrites those slots.

### D. SmolVLM2 style

- Placeholder token is `"<image>"` (often a single token id).
- The template may include additional "header tokens" around the image tokens.
- Injection offsets are found by detecting **runs** of consecutive `IMAGE_CONTEXT_TOKEN` ids, pushing the first id of each run.

## 2) Vision Encoder Output Shapes And Preprocess

There are two broad encoder IO styles:

### A. "Classic image encoder" (single image -> embedding sequence)

Common in `ax-fastvlm`, `ax-internvl`:

- Determine input layout:
  - NCHW float input: normalize `(x/255 - mean) / std`, write as `float` into input tensor.
  - NHWC u8 input: resize + RGB, memcpy into input tensor.
- Determine output dtype:
  - if output size matches `elem_count * 2` => bf16 output
  - if output size matches `elem_count * 4` => fp32 output (then convert to bf16)
- Result is a flat bf16 array whose length is `(num_media_tokens * tokens_embed_size)` (conceptually).

### B. Qwen-VL "video processor" (frames -> patches -> embedding sequence)

Common in `ax-qwen2_5-vl`, `ax-qwen3-vl`, `axcl-qwen3-vl`:

- Preprocess frames (even for image) via a "video-like" patching pipeline:
  - resize to `(vision_config.height, vision_config.width)`
  - RGB
  - temporal patching (`temporal_patch_size`)
  - spatial merge (`spatial_merge_size`)
  - patch size (`patch_size`)
- Produces `pixel_values` per grid segment (for videos: multiple segments).
- Each segment is fed to `image_encoder` to produce an embedding block.
- Tracks `cfg.image_grid_thw` and/or `cfg.video_grid_thw` for mRoPE.

## 3) Extra Side-Inputs Used By Some VLMs

### A. mRoPE / position ids (Qwen-VL)

Qwen-VL branches compute `position_ids` (3 x seq_len) based on:

- `input_ids`
- `cfg.image_grid_thw` / `cfg.video_grid_thw`
- vision settings: `spatial_merge_size`, sometimes video time scaling (`second_per_grid_ts`)

These `position_ids` are then used in prefill by writing them into the model's `indices` input
instead of a simple monotonically increasing index.

### B. deepstack features (some AXCL Qwen-VL variants)

Some image encoders output additional tensors (e.g. 3 "deepstack features").
During prefill, for tokens where `visual_pos_mask[j] == 1`, the code adds those features
into the intermediate embedding stream (bf16->fp32 add -> bf16).

### C. visual_pos_mask

Computed from `input_ids` by marking positions that equal `image_token_id` or `video_token_id`.
Used to align deepstack features with only the visual placeholder positions.

## 4) How Multi-Image / Multi-Frame Is Represented

All branches encode multi-image as:

- `Content{ role=USER, type=IMAGE, data=prompt, num_media=N, num_media_tokens=T }`
- Tokenizer repeats placeholder tokens `N*T`.
- Vision encoder returns `N` blocks, each block length `T * tokens_embed_size`.

For video:

- Some branches treat each temporal grid segment as one "media block".
- Some compute `cfg.video_grid_thw = {{grid_t, grid_h, grid_w}}` and then internally expand.

## 5) Key Gap vs `axllm` (Context Support)

The VLM branches above usually build `input_ids` for the full prompt and then build the full
`out_embed` (text token embeddings + injected vision embeddings) in one go.

`axllm` is different:

- It only materializes embeddings for `tokens_diff` (incremental tail) to support KV-cache context.
- It currently has **no hook** to replace placeholder token embeddings with vision embeddings.

So a pluggable image encoder for `axllm` must solve:

- When `tokens_diff` contains placeholder ids, produce embeddings that include:
  - normal token embeddings for text tokens
  - vision embeddings for the placeholder slots
- While keeping the existing "token-id diff + KV cache" logic intact.

Practical implication:

- The "media -> placeholder slots" mapping must be reproducible from `(history, token_ids)` and/or
  persisted state, so incremental encoding can inject the correct vision embeddings only for the
  newly appended part of the conversation.

## 6) Abstraction Axes For A Pluggable Vision Module (Proposed)

To unify the branches without changing `axllm`'s main control flow, the pluggable module needs
to cover these responsibilities:

- `Tokenizer side`:
  - define placeholder tokens + how many tokens per media item (`num_media_tokens`)
  - define how to locate placeholder offsets in `input_ids`
  - (optional) provide `position_ids` generation rules (mRoPE)
- `Vision side`:
  - load/init image encoder axmodel(s)
  - preprocess image/video frames
  - produce per-media embedding blocks (bf16, `tokens_embed_size` aligned)
  - (optional) deepstack features
- `Injection side`:
  - given `input_ids` and media blocks, produce an "embedding stream" where placeholder slots
    are overwritten by vision embeddings
  - for `axllm` context mode: support doing the above for **only the tail** (`tokens_diff`),
    while keeping placeholder alignment correct.

These notes are intentionally implementation-agnostic so we can use them as a checklist when
refactoring `axllm` to support LLM + VLM with context.

## 7) Current `axllm` Implementation Notes (This Branch)

This branch adds a pluggable vision module behind a runtime config switch (no compile-time toggle):

- `config.json`: `vlm_type` (or `VLM_TYPE`) selects the vision module.
- If `vlm_type != "None"(0)`, `filename_image_encoder_axmodel` must be set.
 - Vision preprocessing backend is selected at **CMake configure time**:
   - Prefer OpenCV if found.
   - Otherwise fall back to `third_party/SimpleCV` and print a CMake warning (slight differences vs OpenCV are possible).

VLM runtime data flow (keeps the existing token-diff + KV-cache logic):

- Tokenizer still generates placeholder tokens based on `Content.num_media` and `Content.num_media_tokens`.
- Vision module prepares a copy of `history` where those two fields are filled, then:
  - encodes images/videos with `image_encoder.axmodel`
  - builds a `pos2vision` mapping for placeholder positions in `input_ids`
  - (Qwen-VL) computes `position_ids` (mRoPE) and a decode start override
- The LLM loop only builds embeddings for `tokens_diff`, and replaces only the vision placeholder slots in that tail.
