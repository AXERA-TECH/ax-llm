# LocateAnything WebUI

A small, dependency-free (Python stdlib only) web front-end for an ax-llm `serve`
instance running [LocateAnything-3B](../../docs/locate_anything.md) — zero-shot object
detection / phrase grounding / OCR with real-time incremental box drawing.

## Run

```bash
AXLLM_SERVE_URL=http://127.0.0.1:8010 \
AXLLM_IMAGE_DIR=/path/to/sample_images \
python3 locateanything_webui.py --port 7861
```

Then open `http://<host>:7861`. Flags: `--host`, `--port`, `--serve-url`, `--image-dir`, `--model`.

The backend serves the UI, lists sample thumbnails from `AXLLM_IMAGE_DIR`, and proxies
detection to the model's `/v1/chat/completions` with `stream:true`, parsing complete
`<box>`/`<ref>` tokens into clean SSE events (`status` / `box` / `done`) for the browser.

## Features

- Auto-scrolling thumbnail banner (hover pauses; mouse-wheel scrolls; click to load).
- Editable category chips, each with its own bright color (reused for that category's boxes).
- Tasks: object detection (per-category, one query each so labels/colors stay separate),
  scene-text / OCR, phrase grounding. Upload button, "max targets" slider (16 / 64 / 256).
- Canvas with a scanning animation during prefill+image-encoding, then boxes drawn
  one-by-one in real time as they stream; status light (idle / encoding / detecting / done).
- Custom (non-`window.confirm`) modal: clicking a thumbnail can switch the categories to
  that image's tags.

## Per-image tags (optional)

Put a `tags.json` next to the images in `AXLLM_IMAGE_DIR`:

```json
{ "street.jpg": ["car", "person"], "sushi.jpg": ["sushi"] }
```

Each image then carries suggested categories; selecting it offers to switch to them.

## Notes

- Serve-only model (like embedding models) — no interactive `run` mode needed.
- Use greedy decoding (the default here) for the most reliable geometry.
- Geometry coords are normalized 0–1000; the UI scales them by the image size.
