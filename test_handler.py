"""Smoke tests for handler.py — runnable WITHOUT rembg installed.

Strategy:
  - Inject a fake `rembg` module into sys.modules BEFORE importing handler.
  - The fake `rembg.remove` returns the input image unchanged (or zeroed for
    only_mask), so the handler exercises the wrapping logic but doesn't need
    an actual ONNX model.

Run with:
    python3 test_handler.py
"""
from __future__ import annotations

import base64
import io
import sys
import time
import traceback
import types
from typing import Any, Dict, List



def _install_fake_rembg() -> Dict[str, Any]:
    """Install a fake rembg module. Returns a dict of call-tracking counters."""
    calls: Dict[str, Any] = {"new_session": [], "remove": []}

    def new_session(model_name: str):
        calls["new_session"].append(model_name)
        return {"__fake_session__": True, "model": model_name}

    def remove(
        img,
        session=None,
        alpha_matting=False,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        only_mask=False,
        post_process_mask=False,
        **kwargs,
    ):
        calls["remove"].append(
            {
                "session": session,
                "alpha_matting": alpha_matting,
                "only_mask": only_mask,
                "post_process_mask": post_process_mask,
            }
        )
        try:
            from PIL import Image
        except Exception:
            return img

        if isinstance(img, Image.Image):
            if only_mask:
                return img.convert("L")
            return img.convert("RGBA")
        return img

    mod = types.ModuleType("rembg")
    mod.new_session = new_session
    mod.remove = remove
    mod.__version__ = "0.0.0-fake"
    sys.modules["rembg"] = mod
    return calls


_FAKE_CALLS = _install_fake_rembg()

if "runpod" not in sys.modules:
    rp = types.ModuleType("runpod")
    serverless = types.ModuleType("runpod.serverless")

    def _start(_cfg):
        return None

    serverless.start = _start
    rp.serverless = serverless
    sys.modules["runpod"] = rp
    sys.modules["runpod.serverless"] = serverless


import handler as H




_FAILURES: List[str] = []
_SKIPPED: List[str] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS {name}")
    else:
        msg = f"  FAIL {name}: {detail}" if detail else f"  FAIL {name}"
        print(msg)
        _FAILURES.append(name)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _have_pillow() -> bool:
    try:
        import PIL

        return True
    except Exception:
        return False




def test_parse_color():
    _section("parse_color helper")
    _check("'transparent' -> None", H.parse_color("transparent") is None)
    _check("'' -> None", H.parse_color("") is None)
    _check("None -> None", H.parse_color(None) is None)
    _check("'none' -> None", H.parse_color("none") is None)

    c = H.parse_color("#fff")
    _check("'#fff' -> (255,255,255,255)", c == (255, 255, 255, 255), str(c))

    c = H.parse_color("#ffffff")
    _check("'#ffffff' -> (255,255,255,255)", c == (255, 255, 255, 255), str(c))

    c = H.parse_color("FFFFFF")
    _check("'FFFFFF' (no #) -> white", c == (255, 255, 255, 255), str(c))

    c = H.parse_color([255, 0, 0])
    _check("[255,0,0] -> (255,0,0,255)", c == (255, 0, 0, 255), str(c))

    c = H.parse_color([10, 20, 30, 128])
    _check("[10,20,30,128] -> RGBA", c == (10, 20, 30, 128), str(c))

    c = H.parse_color("#80800080")
    _check("'#80800080' -> RGBA with alpha 0x80", c == (128, 128, 0, 128), str(c))

    c = H.parse_color("white")
    _check("'white' -> white", c == (255, 255, 255, 255), str(c))

    raised = False
    try:
        H.parse_color("notacolor")
    except ValueError:
        raised = True
    _check("invalid color string raises", raised)

    raised = False
    try:
        H.parse_color([1, 2])
    except ValueError:
        raised = True
    _check("invalid color list raises", raised)


def test_decode_image_b64():
    _section("decode_image_b64 forms")
    raw = base64.b64encode(b"hello").decode("ascii")
    _check("raw base64 decodes", H.decode_image_b64(raw) == b"hello")
    duri = "data:image/png;base64," + raw
    _check("data URI decodes", H.decode_image_b64(duri) == b"hello")
    raised = False
    try:
        H.decode_image_b64("")
    except ValueError:
        raised = True
    _check("empty string raises", raised)


def test_collect_inputs_dispatch():
    _section("collect_inputs — input dispatch")
    items = H.collect_inputs({"image_url": "https://example.com/a.png"})
    _check("image_url -> 1 image_url item", len(items) == 1 and items[0]["kind"] == "image_url")

    items = H.collect_inputs(
        {
            "images": [
                {"type": "url", "data": "https://example.com/x.png"},
                {"type": "b64", "data": base64.b64encode(b"\x89PNG\r\n").decode("ascii")},
                {"type": "b64", "data": "data:image/png;base64,iVBORw0KGgo="},
                {"type": "b64", "data": "!!!not-base64!!!"},
                {"type": "wat", "data": "x"},
            ]
        }
    )
    _check("five items from images[]", len(items) == 5, f"got {len(items)}")
    _check("images[0] -> image_url", items[0]["kind"] == "image_url")
    _check("images[1] -> image_b64 with bytes", items[1]["kind"] == "image_b64" and "bytes" in items[1])
    _check("images[2] (data URI) decoded", items[2]["kind"] == "image_b64" and "bytes" in items[2])
    _check("images[4] unknown type recorded as error", "error" in items[4])

    items = H.collect_inputs(
        {
            "image_url": "https://example.com/one.png",
            "image_urls": ["https://example.com/two.png", "https://example.com/three.png"],
        }
    )
    _check("image_url + image_urls combined", len(items) == 3, f"got {len(items)}")


def test_handler_missing_input():
    _section("handler — missing input returns error")
    out = H.handler({"input": {}})
    _check("error key present", "error" in out, str(out))
    _check(
        "error mentions missing fields",
        "image_url" in out.get("error", "") or "Missing" in out.get("error", ""),
        out.get("error", ""),
    )


def test_handler_unknown_model():
    _section("handler — unknown model returns error")
    out = H.handler(
        {"input": {"image_url": "https://example.com/x.png", "model": "bogus"}}
    )
    _check("error key present", "error" in out, str(out))
    _check("error mentions model", "model" in out.get("error", ""), out.get("error", ""))


def test_end_to_end_with_mocked_rembg():
    _section("end-to-end — mocked rembg + tiny PIL image")
    if not _have_pillow():
        print("  SKIP (Pillow not installed)")
        _SKIPPED.append("test_end_to_end_with_mocked_rembg")
        return

    from PIL import Image

    img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    for y in range(2, 6):
        for x in range(2, 6):
            img.putpixel((x, y), (200, 100, 50, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    b64 = base64.b64encode(png_bytes).decode("ascii")

    pre = len(_FAKE_CALLS["remove"])

    out = H.handler(
        {
            "input": {
                "image_b64": b64,
                "return_bbox": True,
                "return_mask": True,
                "model": "u2netp",
            }
        }
    )

    _check("results length 1", isinstance(out.get("results"), list) and len(out["results"]) == 1, str(out)[:300])
    r = out["results"][0]
    _check("no top-level error", "error" not in out, str(out)[:300])
    _check("no per-item error", "error" not in r, str(r)[:300])
    _check("image_b64 in result", isinstance(r.get("image_b64"), str) and len(r["image_b64"]) > 0)
    _check("mime_type is png", r.get("mime_type") == "image/png", str(r.get("mime_type")))
    _check("format echoed", r.get("format") == "png")
    _check("model echoed", r.get("model") == "u2netp", str(r.get("model")))
    _check("rembg.remove called", len(_FAKE_CALLS["remove"]) == pre + 1)
    _check("session created for u2netp", "u2netp" in _FAKE_CALLS["new_session"])

    try:
        decoded = base64.b64decode(r["image_b64"])
        with Image.open(io.BytesIO(decoded)) as im:
            _check("decoded image opens", im.size == (8, 8), str(im.size))
    except Exception as e:
        _check("decoded image opens", False, str(e))

    bbox = r.get("bbox")
    _check(
        "bbox identifies colored region",
        bbox == [2, 2, 6, 6],
        str(bbox),
    )

    _check("mask_b64 present", isinstance(r.get("mask_b64"), str) and len(r["mask_b64"]) > 0)


def test_end_to_end_only_mask():
    _section("end-to-end — only_mask returns single channel")
    if not _have_pillow():
        print("  SKIP (Pillow not installed)")
        _SKIPPED.append("test_end_to_end_only_mask")
        return

    from PIL import Image

    img = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    out = H.handler(
        {
            "input": {
                "image_b64": b64,
                "only_mask": True,
            }
        }
    )
    r = out["results"][0]
    _check("only_mask flag echoed", r.get("only_mask") is True, str(r))
    _check("image_b64 present", isinstance(r.get("image_b64"), str) and len(r["image_b64"]) > 0)

    try:
        decoded = base64.b64decode(r["image_b64"])
        with Image.open(io.BytesIO(decoded)) as im:
            _check("only_mask result is L or LA mode", im.mode in ("L", "LA"), im.mode)
    except Exception as e:
        _check("only_mask decodes", False, str(e))


def test_end_to_end_composite_bg():
    _section("end-to-end — composite onto white background, output jpg")
    if not _have_pillow():
        print("  SKIP (Pillow not installed)")
        _SKIPPED.append("test_end_to_end_composite_bg")
        return

    from PIL import Image

    img = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
    img.putpixel((3, 3), (200, 100, 50, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    out = H.handler(
        {
            "input": {
                "image_b64": b64,
                "bgcolor": "#FFFFFF",
                "output_format": "jpg",
            }
        }
    )
    r = out["results"][0]
    _check("output format jpg", r.get("format") == "jpg", str(r.get("format")))
    _check("mime_type image/jpeg", r.get("mime_type") == "image/jpeg", str(r.get("mime_type")))
    _check("bgcolor echoed", r.get("bgcolor") == [255, 255, 255, 255], str(r.get("bgcolor")))

    try:
        decoded = base64.b64decode(r["image_b64"])
        with Image.open(io.BytesIO(decoded)) as im:
            _check("jpg decodes", im.mode == "RGB" and im.size == (6, 6), str((im.mode, im.size)))
    except Exception as e:
        _check("jpg decodes", False, str(e))


def test_max_size_resize():
    _section("end-to-end — max_size resizes longest edge")
    if not _have_pillow():
        print("  SKIP (Pillow not installed)")
        _SKIPPED.append("test_max_size_resize")
        return

    from PIL import Image

    img = Image.new("RGBA", (200, 100), (50, 50, 50, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    out = H.handler(
        {
            "input": {
                "image_b64": b64,
                "max_size": 50,
            }
        }
    )
    r = out["results"][0]
    _check("original_size echoed", r.get("original_size") == [200, 100], str(r.get("original_size")))
    _check(
        "processed_size scaled to longest edge 50",
        r.get("processed_size") == [50, 25],
        str(r.get("processed_size")),
    )


def test_batch_input():
    _section("end-to-end — multiple images batch")
    if not _have_pillow():
        print("  SKIP (Pillow not installed)")
        _SKIPPED.append("test_batch_input")
        return

    from PIL import Image

    def _b64_img() -> str:
        i = Image.new("RGBA", (4, 4), (10, 20, 30, 255))
        b = io.BytesIO()
        i.save(b, format="PNG")
        return base64.b64encode(b.getvalue()).decode("ascii")

    b1 = _b64_img()
    b2 = _b64_img()
    out = H.handler(
        {
            "input": {
                "images": [
                    {"type": "b64", "data": b1},
                    {"type": "b64", "data": b2},
                ],
            }
        }
    )
    _check("count == 2", out.get("count") == 2, str(out.get("count")))
    _check("two results", len(out["results"]) == 2, str(len(out["results"])))
    _check("both succeeded", all("image_b64" in r for r in out["results"]))




def main():
    started = time.time()
    suites = [
        test_parse_color,
        test_decode_image_b64,
        test_collect_inputs_dispatch,
        test_handler_missing_input,
        test_handler_unknown_model,
        test_end_to_end_with_mocked_rembg,
        test_end_to_end_only_mask,
        test_end_to_end_composite_bg,
        test_max_size_resize,
        test_batch_input,
    ]
    for fn in suites:
        try:
            fn()
        except Exception:
            print(f"  FAIL {fn.__name__} raised:")
            traceback.print_exc()
            _FAILURES.append(fn.__name__)

    dur = time.time() - started
    print(f"\n--- Done in {dur:.1f}s — {len(_FAILURES)} failure(s), {len(_SKIPPED)} skipped ---")
    if _FAILURES:
        for f in _FAILURES:
            print(f"  FAIL {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
