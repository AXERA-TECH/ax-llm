# LocateAnything WebUI

A small, dependency-free (Python stdlib only) web front-end for an ax-llm `serve` instance
running [LocateAnything-3B](../../docs/locate_anything.md) — real-time object detection and
phrase grounding with incremental box drawing.

## Run

```bash
AXLLM_SERVE_URL=http://127.0.0.1:8010 \
AXLLM_IMAGE_DIR=/path/to/sample_images \
python3 locateanything_webui.py --port 7861
```

Then open `http://<host>:7861`. Flags: `--host`, `--port`, `--serve-url`, `--image-dir`, `--model`.

The backend serves the UI, lists thumbnails from `AXLLM_IMAGE_DIR`, and proxies detection to
the model's `/v1/chat/completions` (`stream=true`), turning the streamed `<box>`/`<ref>`
tokens into clean SSE events for real-time drawing.

## Using the UI

1. **Pick an image** — the top banner auto-scrolls (hover to pause, mouse-wheel to scroll,
   click a thumbnail to load). Or use the **Upload** button.
2. **Choose a task**:
   - **Object detection** — edit the category chips (each gets its own bright color, reused
     for its boxes). Detection runs one query per category so labels/colors stay separate.
   - **Phrase grounding** — type a natural-language description in the *Phrase* box
     (e.g. `the dog on the left`, `the car in front`) to locate a specific instance.
     Position/attribute phrases work best; use English.
3. **Max targets** slider caps how many objects to look for (16 / 64 / 256).
4. Press **Detect**. A scanning animation plays during prefill + image encoding, then boxes
   are drawn one by one as they stream in. The status light shows idle / encoding / detecting
   / done. **Stop** aborts.

## Per-image presets (optional): `tags.json`

Drop a `tags.json` next to the images in `AXLLM_IMAGE_DIR` to give each image a preset
category (for detection) and a preset phrase (for grounding):

```json
{
  "dogs.jpg":   { "tags": ["dog"],               "phrase": "the dog in the center" },
  "safari.jpg": { "tags": ["zebra", "elephant"], "phrase": "the elephant" }
}
```

- `tags`   — categories used in **Object detection** mode.
- `phrase` — the sentence used in **Phrase grounding** mode.

Clicking a thumbnail loads **both** (category chips + phrase box); the task selector just
decides which one is used. In detection mode a click also asks — via a small modal — before
replacing your current categories. A bare list (`"dogs.jpg": ["dog"]`) is still accepted
(category only, no phrase).

## Notes

- Serve-only model (like embedding models) — no interactive `run` mode needed.
- Greedy decoding (the default here) gives the most reliable geometry.
- Images with very many / densely packed / macro-scale targets can collapse to a single
  full-frame box — pick images where objects sit at a normal scale, and keep the target
  count modest (≤ ~10), for the cleanest, snappiest demo.
- Box coordinates are normalized 0–1000; the UI scales them by the image size.
