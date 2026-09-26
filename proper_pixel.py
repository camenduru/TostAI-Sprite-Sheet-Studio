"""Bridge to the proper-pixel-art mesh pixelation algorithm.

proper-pixel-art recovers the *true* pixel grid of noisy, high-resolution
"pixel art" -- the kind a generative model or a low-quality web upload produces
-- instead of trusting the source pixels to sit on a grid: Canny edges ->
morphological closing -> probabilistic Hough lines -> outlier-trimmed median
pixel width -> a homogenised mesh, then one representative colour per mesh cell.

This editor's pixels live in numpy, and a selection is a batch of frames that
should share one canonical grid, so the package's animation path is the right
entry point: the mesh (steps 1-6) and the palette (step 7) are fitted once over
a sample of the frames and then applied to every frame (step 8). That is what
keeps an animation from jittering frame to frame, where solving each frame on
its own would move the mesh and shift the palette every frame.

The package lives in its own checkout; point ``SPRITE_PPA_DIR`` at it. It
targets Pillow >= 12 while this runtime may ship an older one, so the single
API it uses that Pillow 11 lacks (``get_flattened_data``) is shimmed to the
``getdata`` it already provides.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PPA_DIR = os.environ.get(
    "SPRITE_PPA_DIR",
    r"C:\Users\PC\Desktop\monsters\sprite\proper-pixel-art")

_VIDEO = None


def _root(ppa_dir=None):
    return os.path.abspath(ppa_dir or DEFAULT_PPA_DIR)


def _patch_pillow():
    """Make the checkout importable on Pillow < 12.

    ``Image.get_flattened_data`` is the flat pixel iterator ``getdata`` already
    is; adding it only when absent leaves a Pillow 12+ runtime untouched.
    """
    from PIL import Image
    if not hasattr(Image.Image, "get_flattened_data"):
        Image.Image.get_flattened_data = Image.Image.getdata


def _video(ppa_dir=None):
    """Import and cache the checkout's ``video`` module (its animation path)."""
    global _VIDEO
    if _VIDEO is not None:
        return _VIDEO
    root = _root(ppa_dir)
    if not os.path.isfile(os.path.join(root, "proper_pixel_art", "__init__.py")):
        raise RuntimeError(
            "proper-pixel-art was not found at %s -- set SPRITE_PPA_DIR to the "
            "checkout that holds its proper_pixel_art/ folder" % root)
    _patch_pillow()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from proper_pixel_art import video
    except Exception as ex:                       # noqa: BLE001
        raise RuntimeError(
            "proper-pixel-art could not be imported (it needs opencv-python, "
            "pillow, numpy, pyyaml and tqdm): %s" % ex)
    _VIDEO = video
    return _VIDEO


def available(ppa_dir=None) -> bool:
    try:
        _video(ppa_dir)
        return True
    except Exception:                             # noqa: BLE001
        return False


def pixelate_frames(arrays, colors=16, pixel_width=0, sample=8, upscale=2,
                    transparent=False, ppa_dir=None):
    """Pixelate same-sized RGBA frames with one shared mesh and palette.

    The mesh and the palette are fitted once over up to ``sample`` of the frames
    and then applied to all of them. Every frame is collapsed to true pixel
    resolution and nearest-upscaled back to its input size, so the frames and
    the sheet keep their size.

    ``colors`` of 0 keeps every colour (skips quantization). ``pixel_width`` of
    0 auto-detects the pixel width.

    Returns ``(frames, (cols, rows))`` where ``frames`` is a list of ``(H, W, 4)``
    uint8 arrays in input order and ``(cols, rows)`` is the detected grid.
    """
    from PIL import Image

    video = _video(ppa_dir)
    arrs = [np.asarray(a, np.uint8) for a in arrays]
    if not arrs:
        return [], (0, 0)
    h, w = arrs[0].shape[:2]
    if any(a.shape[:2] != (h, w) for a in arrs):
        raise ValueError("pixelate needs every frame to be the same size")
    frames = [Image.fromarray(a) for a in arrs]

    n = len(frames)
    picks = np.unique(np.linspace(0, n - 1, max(1, min(int(sample), n)),
                                  dtype=int))
    sampled = [frames[i] for i in picks]

    mesh_lines, factor = video.compute_video_mesh(
        sampled, upscale_factor=max(1, int(upscale)),
        pixel_width=int(pixel_width) or None)
    pipe = video.FramePipeline(
        mesh_lines, factor, (w, h), int(colors) or None, sampled,
        transparent_background=bool(transparent))

    out = [np.asarray(pipe.process(f).resize((w, h), Image.Resampling.NEAREST),
                      np.uint8) for f in frames]
    return out, (len(mesh_lines[0]) - 1, len(mesh_lines[1]) - 1)
