"""Runpod serverless handler for background removal using rembg.

Accepts one or more images via URL, raw base64, or data URI. For each input
runs rembg.remove(...) with a configurable model and post-processing options,
and returns the cut-out foreground (RGBA PNG/WebP/JPG, or alpha mask), plus
optional composited image on a chosen background color/image and a bounding
box of the foreground content.
"""
from __future__ import annotations

import base64
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

try:
    import runpod
except Exception:
    runpod = None

try:
    import rembg
    from rembg import new_session, remove as rembg_remove
except Exception:
    rembg = None
    new_session = None
    rembg_remove = None

try:
    import numpy as np
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Pillow and numpy are required for image processing."
    ) from e




ALLOWED_MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
    "birefnet-general",
    "birefnet-portrait",
]

ALLOWED_FORMATS = {"png", "webp", "jpg", "jpeg"}

DEFAULT_MODEL = os.getenv("REMBG_MODEL", "u2net")




_SESSION_CACHE: Dict[str, Any] = {}


def get_session(model_name: str) -> Any:
    """Return a cached rembg session for the given model, creating it lazily."""
    if rembg is None or new_session is None:
        raise RuntimeError(
            "rembg is not installed. Install with `pip install rembg[gpu]`."
        )
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"unknown model '{model_name}'. allowed: {ALLOWED_MODELS}"
        )
    if model_name not in _SESSION_CACHE:
        _SESSION_CACHE[model_name] = new_session(model_name)
    return _SESSION_CACHE[model_name]




_DATA_URI_RE = re.compile(r"^data:([^;,]+);base64,(.+)$", re.IGNORECASE | re.DOTALL)


def download_bytes(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def decode_image_b64(s: str) -> bytes:
    """Decode raw base64 or data URI into image bytes."""
    if not isinstance(s, str) or not s:
        raise ValueError("image base64 data must be a non-empty string")
    m = _DATA_URI_RE.match(s.strip())
    payload = m.group(2) if m else s
    payload = "".join(payload.split())
    try:
        return base64.b64decode(payload, validate=False)
    except Exception as e:
        raise ValueError(f"invalid base64 image data: {e}") from e


def collect_inputs(inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize image_url/image_b64/image_urls/images into an ordered list.

    Each returned item is one of:
      {"kind": "image_url", "url": str}
      {"kind": "image_b64", "bytes": bytes}
      {"kind": "image_b64", "error": str}  # invalid input recorded inline
    """
    items: List[Dict[str, Any]] = []

    single_url = inp.get("image_url")
    if isinstance(single_url, str) and single_url:
        items.append({"kind": "image_url", "url": single_url})

    single_b64 = inp.get("image_b64")
    if isinstance(single_b64, str) and single_b64:
        try:
            items.append(
                {"kind": "image_b64", "bytes": decode_image_b64(single_b64)}
            )
        except Exception as e:
            items.append({"kind": "image_b64", "error": str(e)})

    urls = inp.get("image_urls") or []
    if isinstance(urls, str):
        urls = [urls]
    for u in urls:
        if isinstance(u, str) and u:
            items.append({"kind": "image_url", "url": u})

    images = inp.get("images") or []
    if isinstance(images, dict):
        images = [images]
    for i, img in enumerate(images):
        if not isinstance(img, dict):
            items.append(
                {"kind": "image_b64", "error": f"images[{i}] is not an object"}
            )
            continue
        t = img.get("type")
        data = img.get("data")
        if t == "url":
            if not isinstance(data, str) or not data:
                items.append(
                    {"kind": "image_url", "error": f"images[{i}].data missing"}
                )
            else:
                items.append({"kind": "image_url", "url": data})
        elif t == "b64":
            try:
                b = decode_image_b64(data)
                items.append({"kind": "image_b64", "bytes": b})
            except Exception as e:
                items.append({"kind": "image_b64", "error": str(e)})
        else:
            items.append(
                {
                    "kind": "image_b64",
                    "error": f"images[{i}].type must be 'url' or 'b64'",
                }
            )

    return items




_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def parse_color(c: Any) -> Optional[Tuple[int, int, int, int]]:
    """Parse a color spec into an RGBA tuple or None for transparent.

    Accepts:
      - "transparent" / None / "" / "none"  -> None
      - "#RGB", "#RGBA", "#RRGGBB", "#RRGGBBAA" (with or without leading #)
      - "red", "white", "black", named color (basic palette)
      - [r,g,b] or [r,g,b,a] list/tuple of ints 0..255
    """
    if c is None:
        return None
    if isinstance(c, str):
        s = c.strip().lower()
        if s in ("", "transparent", "none"):
            return None
        named = {
            "white": (255, 255, 255, 255),
            "black": (0, 0, 0, 255),
            "red": (255, 0, 0, 255),
            "green": (0, 128, 0, 255),
            "blue": (0, 0, 255, 255),
            "gray": (128, 128, 128, 255),
            "grey": (128, 128, 128, 255),
            "yellow": (255, 255, 0, 255),
        }
        if s in named:
            return named[s]
        m = _HEX_RE.match(s)
        if m:
            h = m.group(1)
            if len(h) == 3:
                r, g, b = (int(ch * 2, 16) for ch in h)
                return (r, g, b, 255)
            if len(h) == 4:
                r, g, b, a = (int(ch * 2, 16) for ch in h)
                return (r, g, b, a)
            if len(h) == 6:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
            if len(h) == 8:
                return (
                    int(h[0:2], 16),
                    int(h[2:4], 16),
                    int(h[4:6], 16),
                    int(h[6:8], 16),
                )
        raise ValueError(f"unrecognized color string: {c!r}")
    if isinstance(c, (list, tuple)):
        vals = [int(v) for v in c]
        if len(vals) == 3:
            return (vals[0], vals[1], vals[2], 255)
        if len(vals) == 4:
            return (vals[0], vals[1], vals[2], vals[3])
        raise ValueError(f"color list must have 3 or 4 ints, got {len(vals)}")
    raise ValueError(f"unsupported color type: {type(c).__name__}")




def open_image_rgba(image_bytes: bytes) -> Image.Image:
    """Open arbitrary image bytes and return an RGBA PIL Image."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        return im.convert("RGBA")


def resize_longest_edge(img: Image.Image, max_size: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img
    scale = float(max_size) / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.LANCZOS)


def alpha_bbox(img: Image.Image) -> Optional[List[int]]:
    """Return [x0, y0, x1, y1] bbox of non-zero-alpha pixels, or None if empty."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        return None
    return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]


def composite_on_background(
    fg: Image.Image, bg_color: Tuple[int, int, int, int]
) -> Image.Image:
    """Composite the RGBA foreground onto a solid RGBA background color."""
    if fg.mode != "RGBA":
        fg = fg.convert("RGBA")
    bg = Image.new("RGBA", fg.size, bg_color)
    bg.alpha_composite(fg)
    return bg


def encode_output(
    img: Image.Image,
    output_format: str,
    quality: int = 92,
) -> Tuple[bytes, str]:
    """Encode an image into the given format. Returns (bytes, mime_type)."""
    fmt = output_format.lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(
            f"unsupported output_format '{output_format}'. allowed: {sorted(ALLOWED_FORMATS)}"
        )
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue(), "image/png"
    if fmt == "webp":
        img.save(buf, format="WEBP", quality=int(quality), lossless=False)
        return buf.getvalue(), "image/webp"
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    rgb.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue(), "image/jpeg"




_BOOL_KEYS = {
    "alpha_matting",
    "post_process_mask",
    "only_mask",
    "return_mask",
    "return_bbox",
}


def _opt(item_opts: Dict[str, Any], top: Dict[str, Any], key: str, default: Any) -> Any:
    """Per-item options override top-level options, which override defaults."""
    if key in item_opts:
        return item_opts[key]
    if key in top:
        return top[key]
    return default


def process_one(
    image_bytes: bytes,
    top: Dict[str, Any],
    item_opts: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the rembg pipeline on a single image and return a result dict."""
    model = _opt(item_opts, top, "model", DEFAULT_MODEL)
    if model not in ALLOWED_MODELS:
        raise ValueError(
            f"unknown model '{model}'. allowed: {ALLOWED_MODELS}"
        )

    alpha_matting = bool(_opt(item_opts, top, "alpha_matting", False))
    fg_thr = int(_opt(item_opts, top, "alpha_matting_foreground_threshold", 240))
    bg_thr = int(_opt(item_opts, top, "alpha_matting_background_threshold", 10))
    erode = int(_opt(item_opts, top, "alpha_matting_erode_size", 10))
    post_process = bool(_opt(item_opts, top, "post_process_mask", False))
    only_mask = bool(_opt(item_opts, top, "only_mask", False))
    bgcolor_raw = _opt(item_opts, top, "bgcolor", None)
    output_format = str(_opt(item_opts, top, "output_format", "png")).lower()
    return_mask = bool(_opt(item_opts, top, "return_mask", False))
    return_bbox = bool(_opt(item_opts, top, "return_bbox", False))
    max_size = _opt(item_opts, top, "max_size", None)
    quality = int(_opt(item_opts, top, "quality", 92))

    bgcolor = parse_color(bgcolor_raw)

    src = open_image_rgba(image_bytes)
    original_size = list(src.size)
    if isinstance(max_size, int) and max_size > 0:
        src = resize_longest_edge(src, max_size)
    processed_size = list(src.size)

    session = get_session(model)

    cut = rembg_remove(
        src,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=fg_thr,
        alpha_matting_background_threshold=bg_thr,
        alpha_matting_erode_size=erode,
        only_mask=only_mask,
        post_process_mask=post_process,
    )

    if isinstance(cut, bytes):
        cut = Image.open(io.BytesIO(cut))
    elif isinstance(cut, np.ndarray):
        cut = Image.fromarray(cut)

    bbox: Optional[List[int]] = None
    if return_bbox and not only_mask:
        try:
            bbox = alpha_bbox(cut.convert("RGBA") if cut.mode != "RGBA" else cut)
        except Exception:
            bbox = None

    if only_mask:
        main_img = cut.convert("L") if cut.mode != "L" else cut
        main_bytes, main_mime = encode_output(main_img, output_format, quality)
        main_b64 = base64.b64encode(main_bytes).decode("ascii")
        return {
            "image_b64": main_b64,
            "mime_type": main_mime,
            "format": output_format,
            "model": model,
            "only_mask": True,
            "original_size": original_size,
            "processed_size": processed_size,
        }

    rgba = cut.convert("RGBA") if cut.mode != "RGBA" else cut

    if bgcolor is not None:
        main_img = composite_on_background(rgba, bgcolor)
    else:
        main_img = rgba

    if output_format in ("jpg", "jpeg") and bgcolor is None:
        main_img = composite_on_background(rgba, (255, 255, 255, 255))

    main_bytes, main_mime = encode_output(main_img, output_format, quality)
    result: Dict[str, Any] = {
        "image_b64": base64.b64encode(main_bytes).decode("ascii"),
        "mime_type": main_mime,
        "format": output_format,
        "model": model,
        "alpha_matting": alpha_matting,
        "post_process_mask": post_process,
        "bgcolor": list(bgcolor) if bgcolor is not None else None,
        "original_size": original_size,
        "processed_size": processed_size,
    }

    if return_mask:
        try:
            alpha = rgba.split()[-1]
            mb, mmime = encode_output(alpha, "png", quality)
            result["mask_b64"] = base64.b64encode(mb).decode("ascii")
            result["mask_mime_type"] = mmime
        except Exception as e:
            result["mask_error"] = str(e)

    if return_bbox:
        result["bbox"] = bbox

    return result




def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input") or {}

    items = collect_inputs(inp)
    if not items:
        return {
            "error": "Missing 'image_url' / 'image_b64' / 'image_urls' / 'images' in input."
        }

    top_model = inp.get("model", DEFAULT_MODEL)
    if top_model not in ALLOWED_MODELS:
        return {
            "error": f"unknown model '{top_model}'. allowed: {ALLOWED_MODELS}"
        }

    top: Dict[str, Any] = {
        "model": top_model,
        "alpha_matting": inp.get("alpha_matting", False),
        "alpha_matting_foreground_threshold": inp.get(
            "alpha_matting_foreground_threshold", 240
        ),
        "alpha_matting_background_threshold": inp.get(
            "alpha_matting_background_threshold", 10
        ),
        "alpha_matting_erode_size": inp.get("alpha_matting_erode_size", 10),
        "post_process_mask": inp.get("post_process_mask", False),
        "only_mask": inp.get("only_mask", False),
        "bgcolor": inp.get("bgcolor", None),
        "output_format": inp.get("output_format", "png"),
        "return_mask": inp.get("return_mask", False),
        "return_bbox": inp.get("return_bbox", False),
        "max_size": inp.get("max_size", None),
        "quality": inp.get("quality", 92),
    }

    items_options = inp.get("items_options") or []
    if isinstance(items_options, dict):
        items_options = [items_options]

    raw_images = inp.get("images") or []
    if isinstance(raw_images, dict):
        raw_images = [raw_images]

    outputs: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        out: Dict[str, Any] = {"input": {"kind": item["kind"]}}
        if item["kind"] == "image_url":
            out["input"]["url"] = item.get("url")

        per_item: Dict[str, Any] = {}
        if i < len(items_options) and isinstance(items_options[i], dict):
            per_item.update(items_options[i])
        url_count = 0
        b64_count = 0
        single_url = inp.get("image_url")
        if isinstance(single_url, str) and single_url:
            url_count += 1
        single_b64 = inp.get("image_b64")
        if isinstance(single_b64, str) and single_b64:
            b64_count += 1
        urls_list = inp.get("image_urls") or []
        if isinstance(urls_list, str):
            urls_list = [urls_list]
        offset_before_images = url_count + b64_count + len(urls_list)
        if i >= offset_before_images:
            img_idx = i - offset_before_images
            if 0 <= img_idx < len(raw_images) and isinstance(
                raw_images[img_idx], dict
            ):
                opts_field = raw_images[img_idx].get("options")
                if isinstance(opts_field, dict):
                    per_item.update(opts_field)

        if "error" in item:
            out["error"] = item["error"]
            outputs.append(out)
            continue

        try:
            if item["kind"] == "image_url":
                image_bytes = download_bytes(item["url"])
            else:
                image_bytes = item["bytes"]
        except Exception as e:
            out["error"] = f"fetch failed: {e}"
            outputs.append(out)
            continue

        try:
            result = process_one(image_bytes, top, per_item)
            out.update(result)
        except Exception as e:
            out["error"] = f"processing failed: {e}"

        outputs.append(out)

    return {
        "results": outputs,
        "model": top_model,
        "count": len(outputs),
        "defaults": {
            "alpha_matting": bool(top["alpha_matting"]),
            "post_process_mask": bool(top["post_process_mask"]),
            "only_mask": bool(top["only_mask"]),
            "output_format": top["output_format"],
            "max_size": top["max_size"],
        },
    }


if runpod is not None:
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError(
            "runpod is not installed; cannot start serverless worker"
        )
