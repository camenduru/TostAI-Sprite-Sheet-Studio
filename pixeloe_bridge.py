"""Bridge to the PixelOE pixelization algorithm.

PixelOE ("Detail-Oriented Pixelization based on Contrast-Aware Outline
Expansion") is a different animal from proper-pixel-art. It does not try to
*recover* a grid the source already had; it *makes* one, in two stages:

  1. Contrast-aware outline expansion. The LAB luminance is median/min/max
     filtered locally, a per-pixel weight says how much the pixel is an edge, and
     the image is rebuilt as ``erode * w + dilate * (1 - w)`` so lines thicken
     where contrast is high and thin where it is not. A closing/opening pass
     cleans it up. This is what stops the downscale from eating a one-pixel
     outline before the grid exists.
  2. Contrast-based downsampling. Each ``pixel_size`` block picks its
     representative luminance -- the pixel nearest the block's center, median,
     mean, min or max -- and the A/B channels by median, so a block becomes one
     colour chosen for contrast rather than averaged into mud. Optional colour
     matching, sharpening, k-means quantisation and dithering follow.

Two consequences shape this bridge:

* **PixelOE is RGB-only.** It has no alpha channel and never will -- every stage
  works on luminance and chromaticity. So alpha is stripped before the call and
  put back afterwards, and the op says so. It is not a transparency algorithm.
* **It is not a per-pixel-local operator.** When the image does not divide evenly
  by ``pixel_size`` the package replicate-pads it, and that padding is fed *into*
  the outline expansion and the colour matching -- a padded run genuinely differs
  from an unpadded one, so cropping the padding back off does not recover the
  unpadded result. The bridge therefore requires ``pixel_size`` to divide the
  cell exactly, which makes padding zero and keeps the cell size exact. See
  ``pixelize_cells``.

The package lives in its own checkout; point ``PIXELOE_DIR`` at it. It is a src
layout (``PixelOE/src/pixeloe/...``) and is imported from the checkout rather
than pip, so ``src`` is what goes on ``sys.path``. Only the torch backend is
used: the Slang compute-shader backends need ``slangpy``, which is not a studio
dependency, and ``pixelize`` raises a clear RuntimeError naming that extra if a
Slang backend is ever requested without it.
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PIXELOE_DIR = os.environ.get(
    "PIXELOE_DIR",
    r"C:\Users\PC\Desktop\monsters\sprite\PixelOE")

ENTRY = "src/pixeloe/torch/pixelize.py"

_CACHE = None
_LOCK = threading.Lock()


def _root(pixeloe_dir=None):
    return os.path.abspath(pixeloe_dir or DEFAULT_PIXELOE_DIR)


def _import(pixeloe_dir=None):
    """Import and cache the checkout's ``pixelize``/``to_numpy`` callables."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    root = _root(pixeloe_dir)
    if not os.path.isfile(os.path.join(root, *ENTRY.split("/"))):
        raise RuntimeError(
            "PixelOE was not found at %s -- set PIXELOE_DIR to the checkout "
            "that holds its src/pixeloe/ folder" % root)
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import torch                                   # noqa: F401
        from pixeloe.torch.pixelize import pixelize
        from pixeloe.torch.utils import to_numpy
    except Exception as ex:                            # noqa: BLE001
        raise RuntimeError(
            "PixelOE could not be imported (it needs torch, torchvision, "
            "kornia, numpy, pillow and opencv-python): %s" % ex)
    _CACHE = (pixelize, to_numpy)
    return _CACHE


def available(pixeloe_dir=None) -> bool:
    try:
        _import(pixeloe_dir)
        return True
    except Exception:                                  # noqa: BLE001
        return False


def _divides(cell_w, cell_h, pixel_size):
    return cell_w % pixel_size == 0 and cell_h % pixel_size == 0


def divisors(cell_w, cell_h, lo=2, hi=64):
    """The pixel sizes that divide both cell dimensions, ascending.

    What the op offers as the reason when a size does not fit, so the user is
    told what *would* work rather than only that they were wrong.
    """
    return [k for k in range(lo, hi + 1)
            if _divides(cell_w, cell_h, k)]


def pixelize_cells(cells, pixel_size=8, thickness=3, mode="contrast",
                   colors=0, dither="ordered", sharpen=None,
                   sharpen_factor=0.5, color_match=True, pixeloe_dir=None):
    """Pixelize same-sized straight-alpha RGBA cells, one call for the batch.

    Each cell is stripped to RGB, pixelized, and has its original alpha put back.
    The result is the same ``(H, W, 4)`` uint8 shape as the input, and the shape
    is guaranteed exact because ``pixel_size`` must divide the cell: PixelOE
    replicate-pads an image that does not divide, and that padding changes the
    result, so this refuses instead of padding. A ``ValueError`` names the sizes
    that do divide.

    ``pixel_size`` is the block size, in source pixels, that becomes one output
    pixel -- bigger means coarser art. ``thickness`` is the outline-expansion
    radius; 0 turns stage 1 off. ``mode`` is the downsampler: ``contrast`` (the
    paper's), ``k_centroid``, ``lanczos``, or any ``F.interpolate`` mode.
    ``colors`` of 0 keeps every colour; above 1 enables k-means quantisation.
    ``dither`` is used only when quantising (``ordered``/``none``/``error_diffusion``).
    ``sharpen`` is None, ``unsharp`` or ``laplacian``.

    Returns a list of ``(H, W, 4)`` uint8 arrays in input order.
    """
    pixelize, to_numpy = _import(pixeloe_dir)
    arrs = [np.asarray(c, np.uint8) for c in cells]
    if not arrs:
        return []
    h, w = arrs[0].shape[:2]
    if any(a.shape[:2] != (h, w) for a in arrs):
        raise ValueError("pixelize needs every cell to be the same size")
    ps = int(pixel_size)
    if ps < 1:
        raise ValueError("pixel size is 1 or more, not %d" % ps)
    if not _divides(w, h, ps):
        good = divisors(w, h)
        raise ValueError(
            "cell is %dx%d and pixel size %d does not divide it -- PixelOE "
            "replicate-pads a cell that does not divide, and that padding "
            "changes the result, so it is refused rather than padded. A pixel "
            "size that fits this cell: %s"
            % (w, h, ps, ", ".join(str(d) for d in good) or "none (2..64)"))

    n = len(arrs)
    rgb = np.empty((n, 3, h, w), np.float32)
    for i, a in enumerate(arrs):
        rgb[i] = np.transpose(a[..., :3], (2, 0, 1)).astype(np.float32) / 255.0

    import torch
    with _LOCK:
        t = torch.from_numpy(rgb)
        out = pixelize(
            t,
            pixel_size=ps,
            thickness=max(0, int(thickness)),
            mode=str(mode),
            sharpen_mode=sharpen or None,
            sharpen_factor=float(sharpen_factor),
            do_color_match=bool(color_match),
            do_quant=int(colors) > 1,
            num_colors=max(2, min(256, int(colors))) if int(colors) > 1 else 32,
            dither_mode=str(dither or "ordered"),
            backend="torch",
        )
        outs = to_numpy(out)

    result = []
    for i, o in enumerate(outs):
        o = np.asarray(o, np.uint8)
        if o.shape[:2] != (h, w):
            # Cannot happen once pixel_size divides the cell, and if it ever
            # does the cell size is the one thing that must not drift, so it is
            # a hard error rather than a silent resize.
            raise RuntimeError(
                "PixelOE returned %dx%d for a %dx%d cell at pixel size %d"
                % (o.shape[1], o.shape[0], w, h, ps))
        rgba = np.empty((h, w, 4), np.uint8)
        rgba[..., :3] = o
        rgba[..., 3] = arrs[i][..., 3]
        result.append(rgba)
    return result
