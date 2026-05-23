# Runpod Background Removal (rembg)

[![Runpod](https://api.runpod.io/badge/your-org/runpod-rembg)](https://console.runpod.io/hub/your-org/runpod-rembg)

Serverless background removal worker powered by [rembg](https://github.com/danielgatis/rembg) on Runpod. Accepts public image URLs or base64-encoded images and returns the cut-out RGBA PNG (or WebP/JPG), an optional alpha mask, an optional composited image on a custom background color, and the foreground bounding box. Supports the U2Net, IS-Net, SAM, and BiRefNet model families with GPU acceleration via `onnxruntime-gpu`.

## Features

- Multiple input forms — `image_url`, `image_b64`, `image_urls[]`, `images[]` (mixed url/b64 batch)
- 10 supported models — fast (`u2netp`), human (`u2net_human_seg`), cloth (`u2net_cloth_seg`), anime (`isnet-anime`), best general (`isnet-general-use`), high-res portrait (`birefnet-portrait`), and more
- Alpha matting for fine edges (hair, fur)
- Optional mask-only output (single-channel PNG)
- Composite onto custom solid background color (`#RRGGBB`, named, `[R,G,B,A]`)
- Output as PNG, WebP, or JPG with quality control
- Longest-edge resize before processing (`max_size`)
- Foreground bounding-box detection
- Per-request session caching — one model load per worker lifetime

## Input schema

### Image sources (provide at least one)

| Field | Type | Description |
| --- | --- | --- |
| `image_url` | string | Single image URL |
| `image_b64` | string | Raw base64 or `data:image/png;base64,...` data URI |
| `image_urls` | string[] | List of image URLs (batch) |
| `images` | object[] | List of `{"type": "url"\|"b64", "data": "..."}` (batch, can mix URL + base64; each entry may also carry `"options": {...}` for per-item overrides) |

### Knobs (all optional; per-image overrides via `images[].options` or `items_options[i]`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | string | env `REMBG_MODEL` or `"u2net"` | One of the 10 allowed models below |
| `alpha_matting` | bool | `false` | Enable alpha matting for finer edges |
| `alpha_matting_foreground_threshold` | int | `240` | Foreground threshold (0-255) |
| `alpha_matting_background_threshold` | int | `10` | Background threshold (0-255) |
| `alpha_matting_erode_size` | int | `10` | Erode size for trimap |
| `post_process_mask` | bool | `false` | Apply post-processing to the predicted mask |
| `only_mask` | bool | `false` | Return single-channel mask instead of cut-out |
| `bgcolor` | string\|int[] | `null` (transparent) | Composite onto this background. Accepts `"transparent"`, `"#FFFFFF"`, `"white"`, `[255,255,255]`, or `[255,255,255,255]` |
| `output_format` | string | `"png"` | One of `png`, `webp`, `jpg` |
| `quality` | int | `92` | Used for WebP/JPG encoding |
| `return_mask` | bool | `false` | Return the alpha mask as a separate PNG |
| `return_bbox` | bool | `false` | Return `[x0, y0, x1, y1]` of non-zero alpha |
| `max_size` | int | `null` | Resize so the longest edge is at most this many pixels before processing |

### Allowed models

| Model | Approx. size | Best for |
| --- | --- | --- |
| `u2net` | ~176 MB | General-purpose default |
| `u2netp` | ~4.7 MB | Fast / mobile / batch |
| `u2net_human_seg` | ~176 MB | Humans specifically |
| `u2net_cloth_seg` | ~176 MB | Clothing segmentation (3-class) |
| `silueta` | ~43 MB | Smaller alternative to u2net |
| `isnet-general-use` | ~178 MB | Best general quality |
| `isnet-anime` | ~178 MB | Anime / illustration |
| `sam` | ~340+ MB | Segment Anything (prompt-driven) |
| `birefnet-general` | ~440 MB | Best high-resolution general |
| `birefnet-portrait` | ~440 MB | Best portrait quality |

## Example requests

OCR a single URL with the default model:
```json
{
  "input": {
    "image_url": "https://example.com/photo.jpg"
  }
}
```

Use alpha matting for finer edges:
```json
{
  "input": {
    "image_url": "https://example.com/portrait.jpg",
    "alpha_matting": true,
    "alpha_matting_foreground_threshold": 240,
    "alpha_matting_background_threshold": 10,
    "alpha_matting_erode_size": 10
  }
}
```

Return only the alpha mask:
```json
{
  "input": {
    "image_url": "https://example.com/photo.jpg",
    "only_mask": true,
    "output_format": "png"
  }
}
```

Composite onto a custom background (white) as JPEG:
```json
{
  "input": {
    "image_url": "https://example.com/photo.jpg",
    "bgcolor": "#FFFFFF",
    "output_format": "jpg",
    "quality": 90
  }
}
```

WebP output, resized so the longest edge is at most 1024 px, with bbox:
```json
{
  "input": {
    "image_url": "https://example.com/photo.jpg",
    "output_format": "webp",
    "max_size": 1024,
    "return_bbox": true,
    "return_mask": true
  }
}
```

Multi-image batch with mixed url + base64 and a per-item model override:
```json
{
  "input": {
    "images": [
      {"type": "url", "data": "https://example.com/portrait.jpg",
       "options": {"model": "birefnet-portrait"}},
      {"type": "b64", "data": "iVBORw0KG..."},
      {"type": "url", "data": "https://example.com/anime.png",
       "options": {"model": "isnet-anime"}}
    ],
    "max_size": 1280
  }
}
```

## Output shape

```json
{
  "results": [
    {
      "input": {"kind": "image_url", "url": "https://..."},
      "image_b64": "iVBORw0KG...",
      "mime_type": "image/png",
      "format": "png",
      "model": "u2net",
      "alpha_matting": false,
      "post_process_mask": false,
      "bgcolor": null,
      "original_size": [1920, 1080],
      "processed_size": [1024, 576],
      "mask_b64": "iVBORw0KG...",
      "mask_mime_type": "image/png",
      "bbox": [120, 60, 940, 540]
    }
  ],
  "model": "u2net",
  "count": 1,
  "defaults": {
    "alpha_matting": false,
    "post_process_mask": false,
    "only_mask": false,
    "output_format": "png",
    "max_size": null
  }
}
```

Notes:
- `image_b64` is the primary output — the cut-out (or mask, if `only_mask`).
- `mask_b64` is only present if `return_mask: true`.
- `bbox` is only present if `return_bbox: true`. It is `null` if no foreground pixels were found.
- Per-input failures attach an `error` field and skip that input; the rest of the batch still returns.
- Top-level `error` is only set when no inputs were provided or the top-level `model` is unknown.

## Local testing

```bash
pip install Pillow numpy requests
python3 test_handler.py
```

The test suite stubs `rembg` so it can run without the heavy ONNX runtime installed. It exercises:
- `parse_color` ("transparent" / hex / named / RGB-list)
- Input dispatch (`image_url`, `image_b64`, `images[]`, mixed batch)
- Error paths (missing input, unknown model)
- End-to-end pipeline with a mocked `rembg.remove`
- Mask-only mode, composite onto background, JPEG/WebP output, batch processing, bbox computation, and `max_size` resize

Expected output ends with `ALL TESTS PASSED`.

## Deployment

1. Build the image: `docker build -t runpod-rembg .`
2. Push to your registry, or connect this repo to Runpod Hub via `.runpod/hub.json`.
3. Create a GitHub release to trigger Hub ingestion.

## Notes on GPU acceleration

- The Docker image is based on `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` and installs `rembg[gpu]`, which pulls `onnxruntime-gpu`.
- The first call for a given `model` triggers a model download and ONNX session creation. The `_SESSION_CACHE` keeps the session in memory for the worker lifetime, so subsequent requests reuse the same loaded model.
- BiRefNet and SAM are the heaviest models (~440 MB) and benefit the most from GPU acceleration. `u2netp` is lightweight enough to run efficiently on CPU.
- If a GPU is not available, `onnxruntime` will fall back to CPU automatically; latency for large models (BiRefNet, IS-Net) will be significantly higher.
- The `containerDiskInGb` in `.runpod/hub.json` is set to `25` GB to comfortably hold the largest combinations of models you may load during a worker's lifetime.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `REMBG_MODEL` | `u2net` | Default model used when a request does not specify `model`. |
| `PYTHONUNBUFFERED` | `1` | Stream logs immediately. |
