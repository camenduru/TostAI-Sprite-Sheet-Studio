"""Run spritefusion-pixel-snapper over a batch of PNGs.

The snapper ships as a Rust/WASM module (`process_image`). The editor's pixels
live server-side in numpy, so the module is driven through a small Node bridge
(`snapper_runner.mjs`) rather than loaded in the browser: one process per batch,
so a multi-frame selection is one Node start and not one per frame.

The package location defaults to the checkout this was integrated with and can
be pointed elsewhere with ``SPRITE_SNAPPER_DIR``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import struct
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "snapper_runner.mjs")

DEFAULT_SNAPPER_DIR = os.environ.get(
    "SPRITE_SNAPPER_DIR",
    r"C:\Users\PC\Desktop\monsters\sprite\spritefusion-pixel-snapper")

# The WASM build can take a few frames at once; a 124-cell selection is ~2-6 MB
# of base64 PNG, which is one argv-free stdin write and well under any pipe
# limit. Still, batching keeps a pathological sheet from building one huge
# string in memory at once.
BATCH = 64


class SnapperError(RuntimeError):
    pass


def pkg_path(snapper_dir: str | None = None) -> str:
    return os.path.join(snapper_dir or DEFAULT_SNAPPER_DIR, "pkg",
                        "spritefusion_pixel_snapper.js")


def wasm_path(snapper_dir: str | None = None) -> str:
    return os.path.join(snapper_dir or DEFAULT_SNAPPER_DIR, "pkg",
                        "spritefusion_pixel_snapper_bg.wasm")


def available(snapper_dir: str | None = None) -> bool:
    return bool(shutil.which("node")) and os.path.isfile(pkg_path(snapper_dir)) \
        and os.path.isfile(wasm_path(snapper_dir))


def _png_size(png: bytes):
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n":
        raise SnapperError("snapper returned something that is not a PNG")
    w, h = struct.unpack(">II", png[16:24])
    return int(w), int(h)


# Console-mode children (node.exe) would each get their own flashing console
# window when this server itself has none (pythonw from Startup). Hide it:
# the bridge only talks over pipes, so no window is ever needed.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(batch, colors, pixel_size, palette, snapper_dir):
    node = shutil.which("node")
    if node is None:
        raise SnapperError(
            "Node.js is required to run spritefusion-pixel-snapper "
            "(the WASM module is driven through snapper_runner.mjs)")
    if not os.path.isfile(pkg_path(snapper_dir)):
        raise SnapperError(
            "spritefusion-pixel-snapper was not found at %s -- set "
            "SPRITE_SNAPPER_DIR to the checkout that holds its pkg/ folder"
            % os.path.dirname(pkg_path(snapper_dir)))
    payload = {
        "pkg": pkg_path(snapper_dir),
        "wasm": wasm_path(snapper_dir),
        "colors": None if colors is None else int(colors),
        # 0 means "let the snapper auto-detect"; the CLI/README treat an absent
        # override the same way, and None is what the WASM bridge expects.
        "pixelSize": int(pixel_size) if pixel_size else None,
        "palette": palette or None,
        "images": [base64.b64encode(b).decode("ascii") for b in batch],
    }
    try:
        proc = subprocess.run(
            [node, RUNNER], input=json.dumps(payload).encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
            creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        raise SnapperError("spritefusion-pixel-snapper timed out")
    out = proc.stdout.decode("utf-8", "replace").strip()
    try:
        res = json.loads(out)
    except ValueError:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise SnapperError("snapper bridge failed: %s"
                           % (err[-500:] or "no output"))
    if not res.get("ok"):
        raise SnapperError(res.get("error") or "snapper failed")
    return [base64.b64decode(s) for s in res["images"]]


def snap_pngs(pngs, colors: int | None = 16, pixel_size: int | None = None,
              palette: str | None = None, snapper_dir: str | None = None):
    """Snap each PNG. Returns ``[{"png", "w", "h"}, ...]`` in input order.

    ``pixel_size`` is the override the README calls ``--pixel-size``; pass 0 or
    None to let the snapper auto-detect. ``palette`` is a comma-separated hex
    list, or None for auto.
    """
    pngs = list(pngs or [])
    if not pngs:
        return []
    out = []
    for start in range(0, len(pngs), BATCH):
        for png in _run(pngs[start:start + BATCH], colors, pixel_size,
                        palette, snapper_dir):
            w, h = _png_size(png)
            out.append({"png": png, "w": w, "h": h})
    return out
