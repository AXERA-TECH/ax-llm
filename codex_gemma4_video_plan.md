# Gemma4 Video Support Plan

Last updated: 2026-05-07

## Goal

Add Gemma4 video understanding support to `axllm` on AX650, while keeping the current text/image/audio paths stable.

This document is meant to be the running handoff note for the video work, so later memory compression or model handoff can recover the exact status and plan quickly.

## Current Status

Gemma4 video is not implemented in runtime yet, but most of the plumbing is already present.

Update for 2026-05-07:

- Phase 1 is now implemented in `ax-llm`.
- Gemma4 video input is no longer a no-op.
- Runtime now loads `video_processor.num_frames` from the model package when available, and falls back to `32`.
- Runtime now dynamically down-samples frames to fit the current prefill budget.
- Runtime now compares "keep current chat history" vs "treat this video request as a fresh chat" for Gemma4 video.
- If prior text history would reduce the selected video frame count, runtime now automatically handles that video request as a fresh chat to preserve video quality.
- Board validation on `ssh 650` succeeded with a frame directory input built from repeated sample images.
- Phase 2 raw video container support is now implemented for the runtime path used by both `axllm run` and `axllm serve`.
- Single-file video inputs are decoded to a temporary frame directory with `ffmpeg`, then fed back into the existing frame-based Gemma4 video path.
- Board validation on `ssh 650` succeeded for:
  - local `.mp4` file path through `/v1/chat/completions`
  - `data:video/mp4;base64,...` through `/v1/chat/completions`

Implemented already:

- `src/main.cpp`
  - OpenAI chat parsing already accepts `video_url` / `video`.
  - Interactive mode already accepts `video:<frames_dir>`.
- `third_party/tokenizer.axera/src/Gemma4Tokenizer.hpp`
  - Gemma4 tokenizer already has a `VIDEO` branch and emits video placeholders.
- `src/runner/vision/vision_module.cpp`
  - `Gemma4VL` already resolves `image_pad_id`, `video_pad_id`, `audio_pad_id`.
  - `BuildInjectionState()` already treats `video_pad_id_` as a media placeholder.

Current remaining gaps:

- Raw video decode currently extracts all frames first, then applies runtime frame sampling.
- This is correct, but not yet optimal for long videos.
- If later needed, we can replace it with sparse extraction so long container inputs do less disk IO.
- When video quality protection triggers, backend state is reset for that video request only; frontend-visible message history is not rewritten.
- If later needed, a true session id / server-side conversation state can make this quality-driven reset fully explicit to the frontend.

## Facts Confirmed From Current Model Package

From `gemma-4-E2B-it/config.json`:

- `vlm_type = "Gemma4VL"`
- `filename_image_encoder_axmodel = "gemma4_vision_h336_w480_t70.axmodel"`
- `vision_width = 480`
- `vision_height = 336`

From `gemma_4_e2b_it_tokenizer/processor_config.json`:

- `video_processor.max_soft_tokens = 70`
- `video_processor.num_frames = 32`
- `video_processor.patch_size = 16`
- `video_processor.pooling_kernel_size = 3`
- `video_processor.do_sample_frames = true`

Important conclusion:

- The packaged Gemma4 image encoder already outputs `70` visual tokens per frame.
- Gemma4 video therefore does not need a new video encoder axmodel.
- The correct direction is to reuse the existing `gemma4_vision_h336_w480_t70.axmodel` and run it once per sampled frame.

## Critical Capacity Constraint

The current compiled Gemma4 text model on AX650 has:

- max prefill total capacity around `1152`
- current VIT output per frame: `70` visual tokens

So full official processor behavior is not directly usable as-is:

- official processor config says `num_frames = 32`
- `32 * 70 = 2240` visual tokens
- `2240` already exceeds the current compiled prefill capacity before adding chat template tokens or text prompt

This means:

- we cannot simply "follow HF processor config and always keep 32 frames"
- frame count must be capped by runtime context budget

This is the most important implementation constraint for Gemma4 video on the current package.

## Chosen Technical Direction

### 1. Reuse the existing Gemma4 image encoder per frame

For each sampled video frame:

- resize to the current Gemma4 vision profile (`336 x 480`)
- run the same `Gemma4ImageProcessor(...)`
- run the same `encode_block_normalized_float(...)`
- produce one embedding block of `70 x hidden_size`

Runtime representation:

- one sampled frame == one media block
- `out_num_media_for_tokenizer = sampled_frame_count`
- `out_num_media_tokens = tokens_per_block_` (`70` for current package)

This matches the current `VisionModule` design and avoids introducing a second encoder path.

### 2. Add dynamic frame downsampling based on prompt budget

Gemma4 video frame count must be limited by two caps:

- processor-side cap: `video_processor.num_frames` (currently `32`)
- runtime-side cap: current request must still fit the compiled prefill budget

The runtime-side cap should be computed dynamically from the real chat template, not from a rough formula.

Recommended implementation:

1. Build a temporary copy of `history`.
2. For the Gemma4 video content, set candidate `num_media = N`, `num_media_tokens = tokens_per_block_`.
3. Call `tokenizer_->encode(temp_history)`.
4. Check whether total prompt length stays within current prefill budget.
5. Use binary search or decrement search to find the largest valid `N`.

Why this is the right way:

- it automatically includes:
  - `<bos>`
  - `<|turn>` / `<turn|>`
  - Gemma4 image/video wrapper tokens
  - user text prompt
  - previous chat history
- it avoids hand-counting special tokens
- it remains correct when tokenizer behavior changes

If the computed valid frame count is `0`, runtime should return a clear error instead of silently skipping:

- example: history already too long for any video frame injection

### 3. Prefer frame-directory input as the primary board-side path

For AX650 runtime, the lowest-risk v1 path is:

- interactive:
  - `video:<frames_dir>`
- OpenAI chat:
  - local directory path
  - or ordered image-frame list if API side is extended later

Reason:

- current image loading path is already stable
- board build often falls back to `SimpleCV`, not full OpenCV video decoding
- adding container decoding (`mp4`, `mov`, `webm`) is a separate dependency problem

### 4. Raw video container decoding now uses `ffmpeg`

Current implementation:

1. If the VIDEO input is already a frame directory, keep the old path.
2. If the VIDEO input is an ordered image list, keep the old path.
3. If the VIDEO input is a single non-image file, call `ffmpeg` to extract frames into a temp directory.
4. Filter the extracted files to supported image frames.
5. Apply Gemma4 runtime frame-cap logic on top of that frame list.

Why this direction was chosen:

- It fixes the actual OpenAI API regression without depending on OpenCV video support at build time.
- It reuses the already validated frame-directory path instead of introducing a second Gemma4 video preprocessing branch.
- It works for both local file paths and base64 uploads because both eventually resolve to a temporary video file path.

## Proposed Implementation Scope

### Phase 1: Get Gemma4 video working on current AX650 package

Target:

- `axllm run <model_dir>` supports `video:<frames_dir>`
- `axllm serve <model_dir>` supports video messages when the resolved media path is a frame directory
- frame count is automatically downsampled to fit current prefill budget
- no model recompilation required

Code areas:

- `src/runner/LLM.hpp`
  - add Gemma4 video config fields if needed
- `src/main.cpp`
  - load Gemma4 video config from `processor_config.json`
- `src/runner/vision/vision_module.hpp`
  - add Gemma4 video config members if needed
- `src/runner/vision/vision_module.cpp`
  - implement `Gemma4VL + VIDEO` branch
  - add prompt-budget-aware frame cap logic
- `src/runner/utils/`
  - add small frame sampling helper if needed

### Phase 2: Support uploaded video container files

Status:

- Done for the current AX650 path, using `ffmpeg`.

Validated board-side behavior:

- `axllm serve` accepts a local `.mp4` path in `video_url.url`
- `axllm serve` accepts a `data:video/mp4;base64,...` payload
- runtime logs now show:
  - raw video extraction
  - dynamic Gemma4 frame selection
  - normal multimodal prefill/decode execution

## Expected Runtime Behavior

For Gemma4 video, the runtime should behave like this:

1. User supplies one video input in a user turn.
2. Runtime loads frames or decodes video.
3. Runtime caps frame count to:
   - `min(actual_frame_count, configured_video_num_frames, prompt_budget_cap)`
4. Runtime uniformly samples frames in temporal order.
5. Each sampled frame is processed with the existing Gemma4 image VIT path.
6. Tokenizer emits one video placeholder block per sampled frame.
7. `BuildInjectionState()` injects visual embeddings into every video placeholder slot.
8. Text generation continues as a normal multimodal prompt.

## Validation Checklist

### Unit / local logic

- frame sampler keeps order stable
- frame sampler reduces long videos uniformly
- prompt-budget search returns a frame cap that really fits tokenizer output length
- zero-budget case returns explicit error

### AX650 board validation

- text-only still works
- single-image still works
- single-audio still works
- `video:<frames_dir>` works on first turn
- multi-turn after video works while context remains under compiled prefill budget
- long frame directories are automatically downsampled instead of failing
- no silent `warning + skip` for Gemma4 video anymore

### OpenAI API validation

Phase 1:

- `chat/completions` with video frame directory path works

Phase 2:

- `chat/completions` with base64 video file works
- `chat/completions` with a local raw video file works

## Risks

### 1. Prompt budget is the real limiter, not only frame count

Even if `video_processor.num_frames = 32`, current AX650 package cannot hold that many frames in prefill.

If this is ignored, runtime will either:

- exceed prefill capacity
- fail group selection
- or force a later silent degradation

### 2. Board-side video container decoding may not be available

The current build can run with `SimpleCV`, and that does not imply container video decode support.

So raw `.mp4` support should be treated as an optional backend capability, not a guaranteed assumption.

### 3. Multi-turn video conversations will shrink available future budget

After a video turn, the conversation history already contains:

- all video placeholder tokens
- all generated answer tokens

So the next turn may allow fewer or even zero video frames unless history is reset or the model is recompiled with a larger prefill budget.

## Recommendation

Start with Phase 1 first.

Reason:

- it is fully aligned with the current `axllm` architecture
- it reuses the already validated Gemma4 image VIT path
- it avoids introducing a new model compile step
- it keeps the main risk focused on token-budget control, which is the real blocker

After Phase 1 is stable on AX650, decide whether Phase 2 raw video-container decoding is actually needed by the frontend path.

## Immediate Next Step

Implement the `Gemma4VL + VIDEO` branch in `vision_module.cpp` with:

- frame-directory input
- dynamic prompt-budget frame cap
- per-frame reuse of the existing Gemma4 image encoder path

Then board-validate on `ssh 650` before touching README or public-facing usage docs.
