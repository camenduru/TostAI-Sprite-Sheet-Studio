"""Sprite Studio pipeline: raw video -> matted frames -> sprite sheet.

Every stage is a plain function so it can be tested and reused without the web
layer. `run_pipeline(cfg, emit)` is the orchestrator the server calls.

The traps this module encodes -- all of them cost real time to find:

1. **Never paste a frame with itself as the mask.** `Image.paste(im, box, im)`
   reads the third argument as a *blend mask*, and for an RGBA mask it uses the
   pasted image's own alpha band, giving `rgb*a/255` and `a*a/255`. Paste with no
   mask for a straight-alpha sheet.

2. **`ImageDraw.floodfill` silently does nothing when seeded at (0,0) on an image
   built by `Image.fromarray`.** Build the mask with `Image.new("L")` + `putdata`.

3. **`frame_count` must equal `columns * rows`.** Unity's Flipbook node and the
   Particle System's Texture Sheet Animation both assume a packed grid and play
   empty cells as frames. So `columns` must be a divisor of `frame_count`.

4. **32768 px 2D texture limit** (D3D12 / Unity / Godot cap at 16384, so half
   of this range is only safe on hardware that reports the larger figure). Check
   both sheet dimensions, and when it is violated say what to change rather than
   just failing.

5. **VRMBG-3.0 returns a list of raw logits**, needs `.sigmoid()`, takes a
   6-channel input at a fixed square size, and carries its temporal state in
   *normalised* space. Without `torch.cuda.synchronize()` you time kernel launch,
   not compute.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import time

import numpy as np

# cv2 / PIL / torch are imported lazily inside the functions that need them so
# that the pure-geometry helpers (solve_layout, auto_columns) stay importable and
# testable without the heavy stack.

MAX_TEXTURE = 327680
BLACK_MAX = 4
KEY_COLOR = "#00ff00"     # the backdrop colour the key starts on
KEY_TOL = 96              # straight RGB distance from KEY_COLOR
# The band is how deep the colour key may bite from the transparent edge, and it
# is a measured number, not a taste. On the walk clip (backdrop #12ff4d, tol 96)
# the visible key-coloured edge -- alpha >= 32, the part you can actually see --
# is 14417 px over 6 frames, and it sits 1-4 px deep: a 3 px band leaves 1105 of
# it (12%), a 6 px band leaves 142 (1%), a 12 px band 16 (0.1%) and no band at all
# 0. 6 is where the curve flattens and where the cost is still bounded: it
# removes 42795 px against the 3 px band's 41619, so 2.8% more pixels for 96% less
# fringe. It bounds the damage on a key-coloured subject, it does not prevent it --
# 6 px off the contour is still 6 px off the contour, and no band makes a colour
# key safe to run on a subject painted in the backdrop colour.
KEY_EDGE_BAND = 6         # px from transparency the colour key is allowed to act in
LEAK_ALPHA = 200
BRIGHT = 64


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

DEFAULT_CFG = {
    # input
    "video": "",
    "prefix": "",
    # matte
    "do_matte": True,
    # `SPRITE_MODEL_DIR` first, so a container can point at its own copy of the
    # weights; the Desktop path stays the default on this machine. Same
    # convention as the three editor checkouts (SPRITE_PPA_DIR, PIXELOE_DIR,
    # SPRITE_SNAPPER_DIR) -- this was the one external dependency with no
    # override, which made it the only one the container could not relocate.
    "model_dir": os.environ.get("SPRITE_MODEL_DIR",
                                r"C:\Users\PC\Desktop\VRMBG-3.0"),
    "infer_size": 1024,
    "start": 1,
    "end": 0,                 # 0 = to the last frame
    "frame_step": 1,
    # A mask video: an optional second clip that IS the background removal.
    # White is the subject, black is the background -- that is the convention
    # every roto tool and every "alpha matte" export already uses, and it is the
    # one the user named. Its frames are read in lockstep with the source (same
    # start/end/step) and its grey becomes the alpha, so a hand-drawn soft edge
    # stays soft.
    #
    # Non-empty means VRMBG-3.0 is NOT run. The model is not refined by the mask,
    # it is replaced: the usual reason to hand one over is that the model got a
    # clip wrong, and running both would spend the GPU time the mask exists to
    # save. That trade is the whole point of the path, so it is opt-in -- and
    # `ui.html` gates the path on its own checkbox, so a path left in the box
    # from an earlier experiment cannot silently keep the model from running.
    "mask_video": "",
    "mask_invert": False,     # off = white is the subject, black the background
    "mask_binary": False,     # off = the mask's own grey becomes the alpha
    "mask_threshold": 128,    # >= this is opaque, when mask_binary is on
    # repair. Off by default, because it is destructive when the matte did not
    # actually leak: the flood fill walks near-black pixels reachable from the
    # frame border, and that includes the subject's own dark edge where it
    # touches the border -- a boot on the last row, a hand at the side. Asked
    # for, not assumed; I2 in the verifier is what says whether it was needed.
    "do_repair": False,
    "black_max": BLACK_MAX,
    # Whether the flood fill keeps its connectivity test. On it clears only
    # near-black pixels *reachable from the frame border*, which keeps it off the
    # subject's own dark edge; off turns it into a tonal black key, which reaches
    # the backdrop the subject enclosed and eats the subject's dark parts with it
    # -- an outline, a boot, a shadow. Off by default: the tonal key is the mode
    # that actually reaches what a bad matte leaves behind a closed silhouette,
    # and the risk it carries is the user's to accept by unticking -- the editor's
    # "erase a colour" op has carried the same switch all along, as `connected`;
    # this is the generator catching up, not a new idea.
    "repair_border_only": False,
    # Chroma-key repair, the other half of the same job. The flood fill is
    # topological -- it can only reach backdrop connected to the frame border --
    # so it cannot touch the green edge that hugs the subject, which is where a
    # matte usually leaves its leak. This one is tonal instead: pixels within
    # `key_tol` of `key_color` (straight RGB distance, 0 = exact) lose their
    # alpha. Off by default: it is destructive when the matte did not leak, and
    # I5 in the verifier is what says whether it was needed. Reading the clip
    # re-seeds the colour from the clip's own border, so the colour is rarely
    # wrong when it is asked for.
    "do_key": False,
    "key_color": KEY_COLOR,
    "key_tol": KEY_TOL,
    # What the checkbox adds, on top of the band the key always keeps. Ticking
    # it ALSO clears every key-coloured region reachable from the frame border,
    # which reaches leak the band cannot see (a slab of backdrop left along a
    # frame edge, opaque and deeper than six pixels). The box only ever EXTENDS
    # the reach -- ticked removes everything unticked removes and more. It used
    # to swap the band for the connectivity test instead, and that read as the
    # box working backwards: on the walk clip the swap cleaned LESS (5.5% of the
    # visible key edge left, against the band's 0.6%), so ticking it re-opened
    # leak the band had closed. Off by default, like the key itself: both are
    # asked for, not assumed.
    "key_also_connected": False,
    # layout
    "columns": 0,             # 0 = auto (pick the divisor that keeps the sheet smallest)
    "cell_mode": "subject",   # none | subject | fixed
    "cell_w": 0,
    "cell_h": 0,
    "pad": 8,
    "square_cell": False,
    "max_texture": MAX_TEXTURE,
    # playback
    "fps": 24,
    "play_mode": "loop",      # loop | pingpong | once
    "trim_to": 0,             # 0 = last frame; else play only up to this frame (1-based)
    "anchor": "none",         # none | bottom-center
    "blend": "straight",
    # outputs
    "want_sheet": True,
    "want_sidecar": True,
    "want_preview": True,
    "want_gif": False,
    "want_frames": False,
    "gif_scale": 0.5,
    "gif_bg": "checker",
    # verify
    # Off by default (user request 2026-09-25): the gate is opt-in. Kept here as a
    # real key rather than removed so `merge_cfg` still coerces/accepts an explicit
    # do_verify=True from the API, and so the knob is one edit away from coming
    # back. `ui.html` seeds its checkbox from this via /api/defaults.
    "do_verify": False,
    "leak_alpha": LEAK_ALPHA,
    "bright": BRIGHT,
    "erode_max": 24,          # largest connected hole that still passes, px
    "erode_frac": 0.005,      # or this fraction of the subject, whichever bites
}


# Field types, used to coerce whatever the client sends. The HTTP API is a
# boundary: an <input type="range"> in the browser yields a *string*, and
# `max(RGB) < "4"` fails deep inside numpy with a UFuncTypeError that says
# nothing about the real cause. Coerce here rather than trusting the caller.
_INT_FIELDS = ("infer_size", "start", "end", "frame_step", "black_max", "columns",
               "cell_w", "cell_h", "pad", "max_texture", "fps", "trim_to",
               "leak_alpha", "bright", "erode_max", "key_tol", "mask_threshold")
_FLOAT_FIELDS = ("gif_scale",)
_BOOL_FIELDS = ("do_matte", "do_repair", "do_key", "repair_border_only",
                "key_also_connected", "want_sheet", "want_sidecar",
                "want_preview", "want_gif", "want_frames", "do_verify", "square_cell",
                "mask_invert", "mask_binary")


def merge_cfg(user):
    cfg = dict(DEFAULT_CFG)
    for k, v in (user or {}).items():
        if k in cfg:
            cfg[k] = v
    for k in _INT_FIELDS:
        try:
            cfg[k] = int(float(cfg[k]))
        except (TypeError, ValueError):
            cfg[k] = DEFAULT_CFG[k]
    for k in _FLOAT_FIELDS:
        try:
            cfg[k] = float(cfg[k])
        except (TypeError, ValueError):
            cfg[k] = DEFAULT_CFG[k]
    for k in _BOOL_FIELDS:
        v = cfg[k]
        cfg[k] = v if isinstance(v, bool) else str(v).strip().lower() in (
            "1", "true", "yes", "on")
    # Canonical on the way in, so the sidecar, the verifier and the picker all
    # name the same colour however the client spelled it.
    cfg["key_color"] = norm_key_color(cfg["key_color"])
    # A path field, so it is coerced to a string: the browser sends "" for an
    # unticked mask, but a hand-rolled client can send null, and `None` reaching
    # `os.path.exists` is a TypeError from three frames away. `.strip()` because
    # a path copied out of Explorer arrives with a trailing space often enough.
    cfg["mask_video"] = str(cfg["mask_video"] or "").strip()
    return cfg


# --------------------------------------------------------------------------- #
# geometry / layout -- pure, no heavy imports
# --------------------------------------------------------------------------- #

def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


# The three inference sizes the model is offered, largest first. 1024 is the
# size VRMBG-3.0 was trained at; 768 and 640 trade matte quality for speed.
INFER_LADDER = (1024, 768, 640)


def suggest_infer_size(w, h):
    """The largest ladder step that does not exceed the source's longer edge.

    Never ask the model for more resolution than the source actually has -- a
    640x480 clip upscaled into a 1024 square buys nothing and costs ~2x the time
    -- and never go below 640, which is the floor the ladder stops at.

    The *longer* edge decides, because the model takes a square input: 1920x1080
    is asked for 1024, 640x480 for 640, and anything under 640 is still 640.

    A suggestion, not a policy: the page seeds its picker with this when a clip
    is read, and every field stays overridable -- same contract as the backdrop
    colour and the column divisors.
    """
    try:
        size = max(int(w), int(h))
    except (TypeError, ValueError):
        return INFER_LADDER[0]
    # A container that will not report its own size leaves the trained size in
    # place rather than silently dropping to the fastest one.
    if size <= 0:
        return INFER_LADDER[0]
    for step in INFER_LADDER:
        if size >= step:
            return step
    return INFER_LADDER[-1]


def auto_columns(frame_count, cell_w, cell_h, max_texture=MAX_TEXTURE):
    """Pick the column count that makes the sheet as small as possible.

    `columns` must divide `frame_count` (full-grid rule), so the choice is
    discrete. Objective: minimise the largest sheet dimension, which is what
    actually hits the texture cap.

    Tie-break: prefer MORE columns / FEWER rows. Sheets are conventionally wider
    than tall, and for frame counts where c x r and r x c tie on max dimension
    (124 = 4x31 = 31x4) the arbitrary pick was the wrong shape.

    There is deliberately no separate "does it fit the cap" test in the score. A
    grid that fits always beats one that does not, because fitting is a property
    of max(w, h) and max(w, h) is what is being minimised: if any divisor fits,
    the minimum is a fitting one. So set_layout's "no column count that fits" is
    only ever reported when it is true -- every divisor really is over the cap,
    as with 58 frames of 566x640 under 16384 (the closest grid, 29 x 2, is
    16414 x 1280: 30 px over, and 1 x 58, 2 x 29 and 58 x 1 are all worse).
    Raising the cap is the only thing that can rescue a save like that one; the
    column count cannot. `max_texture` stays in the signature because every
    caller states the cap it cares about, but the score does not need it.
    """
    best = None
    for c in divisors(frame_count):
        r = frame_count // c
        w, h = c * cell_w, r * cell_h
        score = (max(w, h), r, abs(w - h))
        if best is None or score < best[0]:
            best = (score, c, r, w, h)
    return best[1], best[2], best[3], best[4]


def round_up(v, step=2):
    return int(math.ceil(v / float(step)) * step)


def solve_layout(frame_count, src_w, src_h, bbox, cfg):
    """Decide the grid and the per-cell crop window.

    Returns a dict with columns, rows, cell_w, cell_h, the crop window
    (x0, y0, w, h) into the source frame, and a list of warnings.
    """
    warns = []
    mode = cfg["cell_mode"]

    if mode == "none":
        cell_w, cell_h = src_w, src_h
    else:
        if mode == "fixed" and cfg["cell_w"] and cfg["cell_h"]:
            cell_w, cell_h = int(cfg["cell_w"]), int(cfg["cell_h"])
        else:
            bx0, by0, bx1, by1 = bbox
            pad = int(cfg["pad"])
            cell_w = round_up((bx1 - bx0 + 1) + 2 * pad)
            cell_h = round_up((by1 - by0 + 1) + 2 * pad)
            if cfg["square_cell"]:
                cell_w = cell_h = max(cell_w, cell_h)
            # A crop window cannot reach past the frame. Padding beyond the
            # source only invents transparent rows/columns -- which for a subject
            # that already touches the edge (a character whose feet sit on the
            # last row) silently shifts the anchor. Clamp, and say so.
            if cell_w > src_w:
                warns.append(
                    "cell_w clamped %d -> %d (pad would reach past the frame's "
                    "left/right edge)" % (cell_w, src_w))
                cell_w = src_w
            if cell_h > src_h:
                warns.append(
                    "cell_h clamped %d -> %d (pad would reach past the frame's "
                    "top/bottom edge)" % (cell_h, src_h))
                cell_h = src_h

    cols = int(cfg["columns"]) if cfg["columns"] else 0
    if cols:
        if frame_count % cols:
            raise ValueError(
                "columns=%d does not divide frame_count=%d, so the grid would "
                "have empty cells (engines play those as blank frames). Valid "
                "values: %s" % (cols, frame_count, divisors(frame_count)))
        rows = frame_count // cols
        sheet_w, sheet_h = cols * cell_w, rows * cell_h
    else:
        cols, rows, sheet_w, sheet_h = auto_columns(
            frame_count, cell_w, cell_h, cfg["max_texture"])

    cap = int(cfg["max_texture"])
    if sheet_w > cap or sheet_h > cap:
        raise ValueError(
            "sheet would be %dx%d, over the %d px 2D texture limit. "
            "Smallest possible cell for this frame count is %dx%d. "
            "Fix by reducing the cell (cell_mode='subject' with a smaller pad, "
            "or set cell_w/cell_h), or by using fewer frames."
            % (sheet_w, sheet_h, cap, sheet_w, sheet_h))

    # crop window: centre the subject bbox in the cell, then clamp inside the frame
    bx0, by0, bx1, by1 = bbox
    cx = (bx0 + bx1 + 1) / 2.0 - cell_w / 2.0
    cy = (by0 + by1 + 1) / 2.0 - cell_h / 2.0
    x0 = int(round(max(0, min(src_w - cell_w, cx)))) if cell_w <= src_w else 0
    y0 = int(round(max(0, min(src_h - cell_h, cy)))) if cell_h <= src_h else 0

    if cell_w <= src_w and cell_h <= src_h:
        # the crop must contain the whole subject or we silently lose pixels
        if not (x0 <= bx0 and bx1 < x0 + cell_w and y0 <= by0 and by1 < y0 + cell_h):
            raise ValueError(
                "crop %dx%d at (%d,%d) would clip the subject (bbox %s). "
                "Increase the cell or reduce pad."
                % (cell_w, cell_h, x0, y0, (bx0, by0, bx1, by1)))

    if cell_w > src_w or cell_h > src_h:
        warns.append(
            "cell %dx%d is larger than the source %dx%d -- frames are centred "
            "and padded with transparency" % (cell_w, cell_h, src_w, src_h))

    return {
        "columns": cols, "rows": rows, "frame_count": frame_count,
        "cell_w": cell_w, "cell_h": cell_h,
        "sheet_w": sheet_w, "sheet_h": sheet_h,
        "crop": [x0, y0, cell_w, cell_h],
        "pad_left": max(0, (cell_w - src_w) // 2),
        "pad_top": max(0, (cell_h - src_h) // 2),
        "warnings": warns,
    }


def frame_to_cell(im, layout):
    """Crop a frame to the layout's window and place it in a cell-sized canvas."""
    from PIL import Image
    x0, y0, cw, ch = layout["crop"]
    sw, sh = im.size
    if cw <= sw and ch <= sh:
        return im.crop((x0, y0, x0 + cw, y0 + ch))
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    canvas.paste(im, (layout["pad_left"], layout["pad_top"]))
    return canvas


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #

def probe(video):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError("cannot open video: %s" % video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("video has no readable frames: %s" % video)
    return {
        "frames": n, "fps": round(fps, 3), "width": w, "height": h,
        "duration": round(n / fps, 2) if fps else None,
        "first_frame": frame,          # BGR
    }


# --------------------------------------------------------------------------- #
# stage: matte
# --------------------------------------------------------------------------- #

class MatteModel:
    """VRMBG-3.0, loaded once and kept warm across jobs.

    Warm matters: cudnn autotuning makes the first ~25 frames 5-10x slower
    (~580 ms/frame vs 261 ms), and loading costs 2-3.5 s. Caching the model in
    the server process means only the very first job pays either cost.
    """

    def __init__(self, model_dir, infer_size=640, half=True):
        import torch
        from transformers import AutoModelForImageSegmentation
        self.torch = torch
        self.infer_size = int(infer_size)
        self.model_dir = model_dir
        model = AutoModelForImageSegmentation.from_pretrained(
            model_dir, trust_remote_code=True)
        model = model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = bool(half and self.device.type == "cuda")
        if self.half:
            model = model.half()
        model = model.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = model

    def __call__(self, rgb, prev_rgb, prev_alpha):
        """One autoregressive step. rgb is uint8 HxWx3, priors are tensors."""
        import cv2
        from torchvision import transforms
        torch = self.torch
        n = self.infer_size
        dtype = next(self.model.parameters()).dtype
        rgb_r = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_LINEAR)
        cur = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )(transforms.ToTensor()(rgb_r)).to(device=self.device, dtype=dtype)
        paired = torch.cat([cur, prev_rgb * prev_alpha], dim=0).unsqueeze(0)
        paired = paired.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            out = self.model(paired)          # list of raw logits
            pred = out[-1].sigmoid().squeeze(0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()      # else you time kernel launch
        return pred, cur

    def zero_state(self):
        torch = self.torch
        dtype = next(self.model.parameters()).dtype
        n = self.infer_size
        return (torch.zeros(3, n, n, device=self.device, dtype=dtype),
                torch.zeros(1, n, n, device=self.device, dtype=dtype))


def read_video(video, start=1, end=0, step=1):
    """Yield (index, BGR frame) for the requested 1-based inclusive range."""
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise ValueError("cannot open video: %s" % video)
    i = 0
    emitted = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if i < start:
            continue
        if end and i > end:
            break
        if (i - start) % step:
            continue
        yield emitted, bgr
        emitted += 1
    cap.release()


def matte_video(video, outdir, cfg, model, emit=None, cancel=None):
    """Write per-frame RGBA PNGs with VRMBG-3.0. Returns the list of paths."""
    import cv2
    os.makedirs(outdir, exist_ok=True)
    prefix = cfg["prefix"] or os.path.splitext(os.path.basename(video))[0]
    total = max(1, count_frames(video, cfg["start"], cfg["end"], cfg["frame_step"]))
    prev_rgb, prev_alpha = model.zero_state()
    paths = []
    times = []
    for idx, bgr in read_video(video, cfg["start"], cfg["end"], cfg["frame_step"]):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = time.time()
        pred, cur = model(rgb, prev_rgb, prev_alpha)
        times.append(time.time() - t)
        alpha = cv2.resize(pred[0].float().cpu().numpy(), (w, h),
                           interpolation=cv2.INTER_LINEAR)
        rgba = np.dstack([rgb, np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)])
        p = os.path.join(outdir, "%s_%03d.png" % (prefix, idx + 1))
        cv2.imwrite(p, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        paths.append(p)
        prev_rgb, prev_alpha = cur, pred        # state in NORMALISED space
        if emit and (idx % 5 == 0 or idx + 1 == total):
            avg = sum(times[-25:]) / len(times[-25:])
            emit("matte", (idx + 1) / float(total),
                 "frame %d/%d  %.0f ms/frame" % (idx + 1, total, avg * 1000))
    return paths


def count_frames(video, start=1, end=0, step=1):
    import cv2
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if end:
        n = min(n, end)
    n = max(0, n - (start - 1))
    return (n + step - 1) // step


def mask_to_alpha(mask_frame, invert=False, binary=False, threshold=128):
    """One mask frame -> an alpha plane. White is the subject, black is the
    background.

    Luminance rather than a single channel: the mask may be greyscale, or
    colour-coded by whatever drew it (a white-on-green export, a blue-screen
    pass), and `COLOR_BGR2GRAY` reads all three the way it reads a neutral one
    -- a grey mask is a fixed point of it, so nothing is lost on the common
    case. A frame that arrives already single-channel is passed straight
    through, and one that carries a real alpha band uses that band, because a
    video with an alpha channel IS a mask and its alpha is the honest answer.

    `binary` is off by default and that is deliberate. A mask's whole advantage
    over a matte is that a person drew its edge; a threshold throws that edge
    away for a staircase. It is here for the two cases where the edge is
    already gone: a mask an encoder has muddied (a background that was black
    decodes to 3,5,2 and leaves the sheet at alpha 3 instead of 0), and one
    that is genuinely black-and-white but arrived soft.

    Returns a uint8 plane, the same size as the mask frame.
    """
    import cv2
    if mask_frame.ndim == 2:
        g = mask_frame
    elif mask_frame.shape[2] == 4:
        g = mask_frame[..., 3]
    else:
        g = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    if invert:
        g = 255 - g
    if binary:
        g = np.where(g >= int(threshold), 255, 0).astype(np.uint8)
    return g


def matte_with_mask(video, mask_video, outdir, cfg, emit=None, cancel=None):
    """Write per-frame RGBA PNGs using a mask video as the alpha.

    The mask is a second clip read in lockstep with the source -- same
    start/end/step -- whose white pixels are the subject and whose black pixels
    are the background. It REPLACES the matte model rather than refining it:
    nothing loads VRMBG-3.0 and nothing runs on the GPU, so the result is
    exactly as good as the mask. That is the trade, and it is why the path is
    opt-in.

    Two things it refuses to do quietly:

    * **A mask that runs out before the source.** `zip` would stop at the
      shorter clip and hand back a sheet with fewer frames than were asked for
      -- and the frame count is what the grid is built from, so the failure
      would surface as a wrong layout, not as a short mask. It is an error
      instead, and it says which frame it died on.
    * **A mask at another resolution.** Resized to the source, and said out loud
      once: a mask at half size is a normal thing to hand over, a silently
      mis-scaled alpha is not.

    The source's own frame count is what the run is built on, so a mask with
    MORE frames in range is fine -- the extras are never read.

    Returns the list of frame paths, the same shape `matte_video` returns.
    """
    import cv2
    os.makedirs(outdir, exist_ok=True)
    prefix = cfg["prefix"] or os.path.splitext(os.path.basename(video))[0]
    total = max(1, count_frames(video, cfg["start"], cfg["end"], cfg["frame_step"]))
    src = read_video(video, cfg["start"], cfg["end"], cfg["frame_step"])
    msk = read_video(mask_video, cfg["start"], cfg["end"], cfg["frame_step"])
    binary = bool(cfg["mask_binary"])
    paths = []
    said_size = False
    for idx, bgr in src:
        if cancel and cancel():
            raise RuntimeError("cancelled")
        try:
            _, mframe = next(msk)
        except StopIteration:
            # The source frame this died on, 1-based, and the last one the mask
            # actually covers -- so the message can name the End value that lines
            # the two up instead of only saying they disagree. Measured on the
            # user's own pair (video 124 frames, mask 121) that is the whole
            # difference between a usable error and a riddle.
            died_on = cfg["start"] + idx * cfg["frame_step"]
            if idx == 0:
                raise ValueError(
                    "the mask video produced no frames at all over the selected "
                    "range (start=%d, end=%s, step=%d) -- is it readable, and does "
                    "the range overlap it?"
                    % (cfg["start"], cfg["end"] or "last", cfg["frame_step"]))
            last = died_on - cfg["frame_step"]
            raise ValueError(
                "the mask video ran out at source frame %d of %d -- it has fewer "
                "frames than the source over the selected range (start=%d, end=%s, "
                "step=%d). It covers up to frame %d, so End=%d lines the two up; "
                "or give a mask that covers the whole range."
                % (died_on, total, cfg["start"], cfg["end"] or "last",
                   cfg["frame_step"], last, last))
        h, w = bgr.shape[:2]
        a = mask_to_alpha(mframe, cfg["mask_invert"], binary, cfg["mask_threshold"])
        if a.shape[:2] != (h, w):
            if not said_size:
                said_size = True
                if emit:
                    emit("matte", 0.0, "mask is %dx%d, resized to the source's %dx%d"
                         % (a.shape[1], a.shape[0], w, h))
            # NEAREST for a binarised mask, or the resize would reinvent the soft
            # edge the threshold was asked to remove.
            a = cv2.resize(a, (w, h), interpolation=(
                cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        p = os.path.join(outdir, "%s_%03d.png" % (prefix, idx + 1))
        cv2.imwrite(p, cv2.cvtColor(np.dstack([rgb, a]), cv2.COLOR_RGBA2BGRA))
        paths.append(p)
        if emit and (idx % 5 == 0 or idx + 1 == total):
            emit("matte", (idx + 1) / float(total), "frame %d/%d" % (idx + 1, total))
    return paths


# --------------------------------------------------------------------------- #
# stage: backdrop repair
# --------------------------------------------------------------------------- #

def border_reachable(passable):
    """Pixels connected to the image border through `passable`, 4-connected.

    Uses cv2.connectedComponents rather than ImageDraw.floodfill. The PIL route
    needs a Python loop over every border pixel and measured 0.51 s/frame on a
    640x640 frame -- 64 s of a 124-frame run, the single biggest cost in the
    pipeline. connectedComponents is one C pass.

    Two PIL traps this also sidesteps: ImageDraw.floodfill silently fills nothing
    when seeded at (0,0) on an Image.fromarray-built image, and it needs a
    hand-built mask via Image.new + putdata to work at all.
    """
    import cv2
    n, labels = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
    if n <= 1:
        return np.zeros(passable.shape, bool)
    touching = set(labels[0].tolist()) | set(labels[-1].tolist())
    touching |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    touching.discard(0)                       # 0 is the background label
    if not touching:
        return np.zeros(passable.shape, bool)
    return np.isin(labels, np.fromiter(touching, dtype=np.int32))


_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def key_rgb(color, default=KEY_COLOR):
    """'#00ff00' -> (0, 255, 0). Anything unparseable -> the default colour.

    Checked rather than trusted: the value arrives from a form, and a typo that
    fell through as black would quietly delete every dark pixel on the rim --
    the one failure of this op that looks like a successful run.
    """
    t = str(color or "").strip()
    if not _HEX_COLOR_RE.match(t):
        t = default
    h = _HEX_COLOR_RE.match(t).group(1)
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    v = int(h, 16)
    return (v >> 16) & 255, (v >> 8) & 255, v & 255


def norm_key_color(color, default=KEY_COLOR):
    """The key colour as lowercase '#rrggbb'."""
    return "#%02x%02x%02x" % key_rgb(color, default)


def backdrop_color(frame_bgr, band=4):
    """The clip's own backdrop colour, read off the first frame's border.

    A suggestion for the colour key, as '#rrggbb': the border is almost always
    backdrop, and nobody can name their green screen by eye (the walk clip's is
    #13ff38, not #00ff00). The *median* of the ring rather than the mean, so a
    subject that touches the border only has to be a minority of it.
    """
    a = np.asarray(frame_bgr)[..., :3][..., ::-1]          # cv2 gives BGR
    ring = np.concatenate([a[:band].reshape(-1, 3), a[-band:].reshape(-1, 3),
                           a[:, :band].reshape(-1, 3), a[:, -band:].reshape(-1, 3)])
    r, g, b = (int(round(float(v))) for v in np.median(ring, axis=0)[:3])
    return "#%02x%02x%02x" % (r, g, b)


def despill_key(img, key_color=KEY_COLOR, band=KEY_EDGE_BAND, floor=BLACK_MAX):
    """Take the key colour OUT of the rim pixels the key itself could not match.

    `repair_key` removes what it can identify by colour. What is left over is
    spill: the backdrop's own light bled into the subject's edge by the encoder
    and the upscale, which leaves *dark* green -- measured on the 00183 clip,
    (0,26,0) to (0,55,0): hue 120 deg, saturation 255, value 23-62, alpha 195-253,
    1-2 px deep along the silhouette. They are 194-233 away from any green
    backdrop key, so no tolerance reaches them without also swallowing the
    subject: at tol 255 a mid-grey is 205 away and goes with them. That is why
    raising the tolerance never cleaned this leak -- the metric was wrong, not the
    dial.

    So do not delete them: take the key's colour out of them. The picked colour's
    dominant channel is clamped to the larger of the other two, which turns
    (0,50,0) into (0,0,0) -- the black outline it was always meant to be -- and
    leaves any pixel whose key channel does not dominate untouched, at any
    brightness. Nothing is removed and no alpha changes, so the silhouette keeps
    its shape; the spill stops being visible because it stops being green.

    Bounded by the same `band` as the key (not by its connectivity test: the spill
    is at the rim by definition, and the box is about which *removals* are allowed),
    and only where the key's channel actually dominates. Nothing is removed, so it
    cannot thin the silhouette. A subject painted in the key colour loses the
    saturation of its outline -- the one cost, and the same one the key carries.

    `floor` is the one non-obvious rule. Clamping a *dark* spill lands on
    near-black, and near-black is exactly what this project's own leak test (I2)
    walks through: measured on the walk clip, 23 despilled pixels were enough to
    bridge the transparent margin to dark *subject* pixels that were previously
    enclosed, and I2 then flagged 144 px of the sprite as leaked backdrop -- a
    false positive manufactured by the repair itself. So the two channels that are
    not the key's are lifted to `floor` (the same `black_max` I2 uses) wherever the
    clamped channel lands below it: `(0,50,0)` becomes `(4,0,4)` rather than
    `(0,0,0)`. At 4/255 the difference is invisible, the outline is still black,
    and a pixel this despill produces can never be mistaken for a leak -- by the
    verifier, or by the flood fill if it runs on the result.

    Returns (image, pixels_changed).
    """
    import cv2
    from PIL import Image
    a = np.array(img)
    alpha = a[..., 3]
    if not (alpha == 0).any():
        return img, 0
    ch = int(np.argmax(key_rgb(key_color)))          # the backdrop's channel
    oth = [i for i in range(3) if i != ch]
    rgb = a[..., :3]
    other_max = np.maximum(rgb[..., oth[0]].astype(np.int16),
                           rgb[..., oth[1]].astype(np.int16))
    spill = rgb[..., ch].astype(np.int16) > other_max
    # Where the pass is allowed to act: the `band` ring around full transparency
    # PLUS every soft-alpha pixel. The band alone undershoots a matte with a wide
    # soft edge -- measured on the 00183 frames the leftover spill sits at
    # alpha 195-253 up to 13 px deep, and on the user's 2x-upscaled sheet up to
    # 21 px, half of it beyond the 6 px band -- and every one of those pixels is
    # semi-transparent, i.e. exactly the blend zone the matte itself created and
    # the one place spill is guaranteed to be harmless to guess at: a pixel that
    # is not fully opaque cannot be load-bearing detail, and despill only removes
    # the key channel's dominance, never alpha. Opaque pixels are reachable only
    # through the band, so a subject's interior saturation is untouched in either
    # checkbox mode.
    dist = cv2.distanceTransform((alpha > 0).astype(np.uint8), cv2.DIST_L2, 3)
    rim = (dist <= band) | (alpha < 255)
    fix = spill & rim & (alpha > 0)
    if not fix.any():
        return img, 0
    a = a.copy()
    plane = a[..., :3].astype(np.int16)
    clamped = np.where(fix, other_max, plane[..., ch])
    plane[..., ch] = clamped
    if floor > 0:
        lift = fix & (clamped < floor)
        for i in oth:
            plane[..., i] = np.where(lift, np.maximum(plane[..., i], floor), plane[..., i])
    a[..., :3] = np.clip(plane, 0, 255).astype(a.dtype)
    return Image.fromarray(a), int(fix.sum())


def repair_backdrop(img, black_max=BLACK_MAX, border_only=True):
    """Clear backdrop that leaked into the matte.

    Topological by default, not tonal: the backdrop is connected to the frame
    border, the subject's own near-black pixels are enclosed by its body.

    That connectivity test is what `border_only` turns off. With it off the pass
    becomes a tonal black key -- every near-black pixel goes, enclosed ones
    included -- which reaches backdrop the subject cut off from the frame border
    (inside a handle, between an arm and the body, behind a closed silhouette)
    and pays for it by eating the subject's own dark parts: an outline, a boot, a
    shadow. Which of those two failure modes is worse is the user's call, so it
    is a separate flag rather than a new default.

    Returns (image, pixels_cleared).
    """
    from PIL import Image
    a = np.array(img)
    ai = a.astype(np.int16)
    near_black = ai[..., :3].max(axis=2) < black_max
    backdrop = border_reachable(near_black) if border_only else near_black
    kill = backdrop & (ai[..., 3] > 0)
    if kill.any():
        a = a.copy()
        a[..., 3][kill] = 0
        a[..., :3][kill] = 0
    return Image.fromarray(a), int(kill.sum())


def repair_key(img, key_color=KEY_COLOR, key_tol=KEY_TOL, band=KEY_EDGE_BAND,
               also_connected=False):
    """Clear a chroma-key leak by colour, wherever around the subject it sits.

    The complement of repair_backdrop. That one is topological: it walks near-black
    pixels reachable from the frame border, so backdrop the matte left *inside* the
    silhouette -- the green edge hugging the subject, which is where a matte
    usually leaks -- is enclosed by the subject and the flood fill never sees it.
    This one is tonal: a pixel whose RGB is within `key_tol` (straight RGB
    distance, 0 = an exact match) of `key_color` loses its alpha.

    What keeps it off the subject is the `band` -- always. Only pixels within
    `band` px of transparency are eligible. The leak is a rim phenomenon; a
    subject painted in the key colour is interior, and erasing that is the
    failure mode of plain select-by-colour. Measured on the walk clip at tol 96,
    91% of the visible key-coloured edge (alpha >= 32) sits within 3 px of
    transparency and the rest within 4, which is what fixes the default band at 6
    rather than 3: see KEY_EDGE_BAND for the whole curve. Transparent pixels
    count as passable, so the backdrop margin is always part of the passable mask
    whatever its RGB happens to be -- which matters because a flood-fill pass
    running first zeroes the RGB of everything it clears.

    `also_connected` then EXTENDS the reach; it can never shrink it. On top of
    the band, every key-coloured region reachable from the frame border is
    cleared too -- the same topological test the flood fill uses, on a colour
    instead of on near-black. A matte leaves the backdrop as one region attached
    to the transparent margin, and that margin reaches the frame border, so the
    ring hugging the subject is reachable while a patch the subject encloses is
    not. This reaches leak the band cannot see at all: a slab of backdrop left
    along a frame edge, opaque and deeper than six pixels. The guarantee it adds
    is the other half: an enclosed patch of the key colour cannot be touched at
    any tolerance.

    It used to be a swap -- the flag replaced the band with the connectivity
    test -- and that read as the box working backwards: the swap *restricted*
    where the band cleaned better (on the walk clip, 0.6% of the visible key
    edge left, against the connectivity test's 5.5%), so ticking it cleaned LESS
    on some clips and re-opened leak the band had closed. A checkbox next to a
    leak has to mean "clean more"; now it does, and monotonicity is checked in
    test_matte_repair (ticked clears a superset of unticked). The editor's
    "erase a colour" keeps the pure swap as `connected`, for a user who wants
    connectivity only.

    `key_tol` is the user's dial either way. Wide is destructive; that is a stated
    tradeoff, not a hidden one.

    A frame with no transparency at all is returned untouched on the band path:
    without a rim there is no leak this op is defined for, and a distance
    transform with no zero pixel is a whole-frame erase. The connectivity test has
    no such problem -- it is defined on any frame, and on a frame the matte failed
    on completely it clears the border-connected key region, which is that leak.

    Returns (image, pixels_cleared).
    """
    import cv2
    from PIL import Image
    a = np.array(img)
    alpha = a[..., 3]
    if not (alpha == 0).any() and not also_connected:
        return img, 0
    rgb = a[..., :3].astype(np.int32)
    key = np.array(key_rgb(key_color), dtype=np.int32)
    near = ((rgb - key) ** 2).sum(axis=2) <= int(key_tol) ** 2
    # distanceTransform measures to the nearest zero pixel, so the map is "how
    # far is this pixel from the transparent background" -- one C pass.
    rim = cv2.distanceTransform((alpha > 0).astype(np.uint8), cv2.DIST_L2, 3) <= band
    kill = near & rim & (alpha > 0)
    if also_connected:
        kill |= border_reachable(near | (alpha == 0)) & near & (alpha > 0)
    if kill.any():
        a = a.copy()
        a[..., 3][kill] = 0
        a[..., :3][kill] = 0
    return Image.fromarray(a), int(kill.sum())


# --------------------------------------------------------------------------- #
# stage: bbox
# --------------------------------------------------------------------------- #

def subject_bbox(paths, alpha_thresh=0):
    """Union bbox of the subject across frames, using alpha > thresh.

    A coverage threshold would under-measure a soft edge and let the crop clip
    it; alpha > 0 measures the full extent.
    """
    from PIL import Image
    x0 = y0 = 10 ** 9
    x1 = y1 = -1
    for p in paths:
        a = np.array(Image.open(p).convert("RGBA"))[..., 3]
        ys, xs = np.where(a > alpha_thresh)
        if len(ys) == 0:
            continue
        y0 = min(y0, int(ys.min())); y1 = max(y1, int(ys.max()))
        x0 = min(x0, int(xs.min())); x1 = max(x1, int(xs.max()))
    if x1 < 0:
        raise ValueError("no subject pixels found in any frame (all transparent)")
    return [x0, y0, x1, y1]


# --------------------------------------------------------------------------- #
# stage: sheet
# --------------------------------------------------------------------------- #

def build_sheet(paths, layout, out_png, emit=None):
    """Assemble the grid. Straight alpha: paste with NO mask."""
    from PIL import Image
    cols, rows = layout["columns"], layout["rows"]
    cw, ch = layout["cell_w"], layout["cell_h"]
    sheet = Image.new("RGBA", (cols * cw, rows * ch), (0, 0, 0, 0))
    for i, p in enumerate(paths):
        im = frame_to_cell(Image.open(p).convert("RGBA"), layout)
        if im.size != (cw, ch):
            raise ValueError("frame %d is %s, expected %dx%d" % (i, im.size, cw, ch))
        sheet.paste(im, ((i % cols) * cw, (i // cols) * ch))   # no mask
        if emit and (i % 16 == 0 or i + 1 == len(paths)):
            emit("sheet", (i + 1) / float(len(paths)), "cell %d/%d" % (i + 1, len(paths)))
    sheet.save(out_png)
    return out_png


def sidecar_dict(cfg, layout, video, n_frames, matte_note):
    last = n_frames - 1
    trim = int(cfg["trim_to"]) - 1 if cfg["trim_to"] else last
    trim = max(0, min(last, trim))
    mode = cfg["play_mode"]
    anims = {
        "play": {"from": 0, "to": trim, "loop": mode == "loop"},
    }
    if mode == "pingpong":
        anims["reverse"] = {"from": 0, "to": trim, "loop": False}
    return {
        "name": cfg["prefix"] or os.path.splitext(os.path.basename(video))[0],
        "sheet": os.path.basename(cfg["_sheet"]),
        "frame_width": layout["cell_w"],
        "frame_height": layout["cell_h"],
        "columns": layout["columns"],
        "rows": layout["rows"],
        "frame_count": layout["frame_count"],
        "fps": cfg["fps"],
        "crop": layout["crop"],
        "matte": ("mask video" if cfg.get("mask_video")
                  else ("vrmbg-3.0" if cfg["do_matte"] else "none")),
        "matte_repair": ", ".join(
            ([("border flood-fill (black_max=%d)" % cfg["black_max"])
              if cfg["repair_border_only"]
              else ("near-black anywhere (black_max=%d)" % cfg["black_max"])]
             if cfg["do_repair"] else [])
            + (["colour key %s (tol=%d, band=%d%s) + despill"
                % (cfg["key_color"], cfg["key_tol"], KEY_EDGE_BAND,
                   " + border-connected" if cfg["key_also_connected"] else "")]
               if cfg["do_key"] else [])) or "none",
        "matte_note": matte_note,
        "anchor": cfg["anchor"],
        "blend": cfg["blend"],
        "play_mode": mode,
        "animations": anims,
        "note": ("%d frames from %s at %.2f fps."
                 % (n_frames, os.path.basename(video), cfg["fps"])),
    }


# --------------------------------------------------------------------------- #
# stage: preview html
# --------------------------------------------------------------------------- #

PREVIEW_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>%(name)s - sprite preview</title>
<style>
  :root { --bd:#d8dde3; --mut:#667; }
  * { box-sizing:border-box; }
  html, body { height:100%%; }
  body { margin:0; background:#f5f7f9; color:#1b1f24;
         font:14px/1.5 "Segoe UI",system-ui,sans-serif; }
  /* The player fills the window. It used to be a 1100px column at the top of a
     plain page, so a 566x640 cell sat in the corner of a mostly empty viewport
     -- and inside the generator's preview panel it was mostly cut off. The sheet
     is the whole point of this page: it gets the space, the controls sit beside
     it, and the canvas is drawn at the size that fits (see fitZoom) so "fills
     the page" is the default rather than something to be scrolled to. */
  .wrap { height:100%%; padding:12px 16px; display:flex; flex-direction:column;
          gap:10px; }
  h1 { font-size:17px; margin:0; }
  .sub { color:var(--mut); font-size:12.5px; }
  .row { display:flex; gap:14px; flex:1; min-height:0; align-items:stretch; }
  .stage { background:#fff; border:1px solid var(--bd); border-radius:10px;
           padding:10px; flex:1; min-width:0; min-height:0; overflow:auto;
           scrollbar-gutter:stable;
           display:flex; align-items:center; justify-content:center; }
  canvas { display:block; margin:auto; border-radius:6px;
           image-rendering:pixelated;
           background:
             linear-gradient(45deg,#e6e6e6 25%%,transparent 25%%,transparent 75%%,#e6e6e6 75%%),
             linear-gradient(45deg,#e6e6e6 25%%,#fff 25%%,#fff 75%%,#e6e6e6 75%%);
           background-size:16px 16px; background-position:0 0,8px 8px; }
  .ctrls { background:#fff; border:1px solid var(--bd); border-radius:10px;
           padding:12px 14px; width:250px; flex:0 0 250px; overflow:auto; }
  /* Narrow window (or the generator's preview panel): the controls go under the
     sheet rather than squeezing it into a strip. */
  @media (max-width:760px){
    .row { flex-direction:column; }
    .ctrls { width:auto; flex:0 0 auto; }
  }
  label { display:block; font-size:12px; color:var(--mut); margin:10px 0 3px; }
  input[type=range] { width:100%%; }
  button { font:inherit; padding:6px 12px; border:1px solid var(--bd);
           background:#fff; border-radius:6px; cursor:pointer; }
  button:hover { background:#eef1f4; }
  .btns { display:flex; gap:8px; margin-top:12px; flex-wrap:wrap; }
  .meta { font-size:12px; color:var(--mut); margin-top:10px; }
  .err { color:#b4232a; }
</style></head><body><div class="wrap">
<h1>%(name)s</h1>
<div class="sub">%(fw)d&times;%(fh)d cells &middot; %(cols)d&times;%(rows)d grid &middot;
%(count)d frames &middot; %(fps)s fps &middot; %(mode)s</div>
<div class="row">
  <div class="stage"><canvas id="cv"></canvas></div>
  <div class="ctrls">
    <label>Frame <span id="fl">0</span> / %(last)d</label>
    <input type="range" id="frame" min="0" max="%(last)d" value="0" step="1">
    <label>Speed <span id="sl">1.0</span>&times;</label>
    <input type="range" id="speed" min="0.1" max="3" value="1" step="0.1">
    <label>Zoom <span id="zl">1.0</span>&times; (fits the window at 1.0)</label>
    <input type="range" id="zoom" min="0.25" max="3" value="1" step="0.05">
    <div class="btns">
      <button id="play">Play</button>
      <button id="stepb">&minus;1</button>
      <button id="stepf">+1</button>
    </div>
    <div class="meta" id="meta"></div>
  </div>
</div>
<script>
const CFG = {
  name:%(name_js)s, sheet:%(sheet_js)s, cellw:%(fw)d, cellh:%(fh)d,
  cols:%(cols)d, rows:%(rows)d, count:%(count)d, fps:%(fps)s, mode:%(mode_js)s
};
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const sheetImg = new Image();
const state = { f:0, t:0, playing:false, dir:1, prev:0, speed:1, zoom:1, loaded:false };
sheetImg.onload = () => { state.loaded = true; draw(); };
sheetImg.onerror = () => {
  document.getElementById('meta').innerHTML =
    '<span class="err">could not load '+CFG.sheet+' (open this page over http, not file://)</span>';
};
sheetImg.src = CFG.sheet;

function cellPos(i) {
  return [ (i %% CFG.cols) * CFG.cellw, Math.floor(i / CFG.cols) * CFG.cellh ];
}
// The size at which the cell exactly fills the stage, which is what the zoom
// slider multiplies: 1.0 is "as big as the window allows", not "100%% of the
// source pixels". A 566x640 cell at 1:1 does not fit a 430px panel at all, and
// that is the size a 1.0 slider used to hand the user. The stage's own padding
// is subtracted (box-sizing is border-box, so clientWidth includes it) and the
// stage keeps a stable scrollbar gutter, so the number cannot oscillate between
// "fits" and "does not fit" as the scrollbar appears.
function fitZoom() {
  const st = document.querySelector('.stage');
  const w = st.clientWidth - 24, h = st.clientHeight - 24;
  if (w < 2 || h < 2) return 1;
  return Math.min(w / CFG.cellw, h / CFG.cellh);
}
function draw() {
  if (!state.loaded || !sheetImg.naturalWidth) return;
  const z = state.zoom * fitZoom();
  cv.width  = Math.max(1, Math.round(CFG.cellw * z));
  cv.height = Math.max(1, Math.round(CFG.cellh * z));
  const [sx, sy] = cellPos(state.f);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(sheetImg, sx, sy, CFG.cellw, CFG.cellh, 0, 0, cv.width, cv.height);
  document.getElementById('frame').value = state.f;
  document.getElementById('fl').textContent = state.f;
  document.getElementById('meta').textContent =
    'cell ' + state.f + ' at sheet (' + sx + ',' + sy + ')  \u00b7  drawn at '
    + Math.round(z * 100) + '%% of ' + CFG.cellw + 'x' + CFG.cellh;
}
// Playback. Two fields, two jobs: `prev` is the previous animation timestamp,
// `t` is the position in frames. They used to share one field (`last`), so the
// end-of-frames test compared a frame position against a millisecond timestamp
// -- never true -- and playback walked off the end of the sheet instead of
// stopping there, leaving a blank canvas and a frame counter past the last
// cell. The bound is CFG.count and what happens at it is CFG.mode: loop wraps,
// pingpong reverses, once stops.
function stop() {
  state.playing = false;
  document.getElementById('play').textContent = 'Play';
}
function advance(dt) {
  const end = CFG.count - 1;
  state.t += dt * CFG.fps * state.speed * state.dir;
  if (CFG.mode === 'pingpong') {
    if (state.t >= end) { state.t = end; state.dir = -1; }
    if (state.t <= 0)   { state.t = 0;   state.dir =  1; }
  } else if (CFG.mode === 'once') {
    if (state.t >= end) { state.t = end; stop(); }
  } else {                                   // loop
    // Wrapped half a frame early, so the rounded value never reaches the frame
    // after the last one: wrapping at CFG.count would hold the last cell for
    // 1.5 frames while the rest of the loop gets 1, which reads as a hitch.
    if (state.t >= CFG.count - 0.5) state.t -= CFG.count;
    if (state.t < 0) state.t += CFG.count;
  }
  // Rounding can land one frame past the last cell, which the sheet does not
  // have: drawImage would paint nothing and the canvas would flicker blank.
  state.f = Math.max(0, Math.min(end, Math.round(state.t)));
}
function loop(now) {
  if (state.playing) {
    const dt = state.prev ? (now - state.prev) / 1000 : 0;
    state.prev = now;
    advance(dt);
  } else {
    state.prev = 0;
  }
  draw();
  requestAnimationFrame(loop);
}
const $ = id => document.getElementById(id);
$('frame').oninput = e => {
  state.f = +e.target.value; state.t = state.f; stop(); draw();
};
$('speed').oninput = e => { state.speed = +e.target.value; $('sl').textContent = state.speed.toFixed(1); };
$('zoom').oninput  = e => { state.zoom  = +e.target.value; $('zl').textContent = state.zoom.toFixed(2); draw(); };
$('play').onclick = () => {
  if (state.playing) { stop(); return; }
  // "once" that already ran to the end starts over, so the button is never
  // dead; pingpong resumes in the direction it was heading.
  if (CFG.mode === 'once' && state.t >= CFG.count - 1) state.t = 0;
  if (CFG.mode !== 'pingpong') state.dir = 1;
  state.playing = true;
  state.prev = 0;
  $('play').textContent = 'Pause';
  draw();
};
$('stepb').onclick = () => { stop();
  state.f = (state.f - 1 + CFG.count) %% CFG.count; state.t = state.f; draw(); };
$('stepf').onclick = () => { stop();
  state.f = (state.f + 1) %% CFG.count; state.t = state.f; draw(); };
document.addEventListener('keydown', e => {
  if (e.key === ' ') { e.preventDefault(); $('play').click(); }
  if (e.key === 'ArrowLeft')  $('stepb').click();
  if (e.key === 'ArrowRight') $('stepf').click();
});
// The cell is sized from the window, so a resize has to redraw it.
window.addEventListener('resize', draw);
requestAnimationFrame(loop);
</script></div></body></html>
"""


def write_preview(out_html, cfg, layout, sidecar):
    js = lambda s: json.dumps(s)
    html = PREVIEW_HTML % {
        "name": sidecar["name"],
        "name_js": js(sidecar["name"]),
        "sheet_js": js(sidecar["sheet"]),
        "sheet": sidecar["sheet"],
        "fw": layout["cell_w"], "fh": layout["cell_h"],
        "cols": layout["columns"], "rows": layout["rows"],
        "count": layout["frame_count"], "last": layout["frame_count"] - 1,
        "fps": cfg["fps"], "mode": cfg["play_mode"], "mode_js": js(cfg["play_mode"]),
    }
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_html


# --------------------------------------------------------------------------- #
# stage: gif
# --------------------------------------------------------------------------- #

def write_gif(paths, out_gif, cfg, layout, emit=None):
    from PIL import Image
    scale = float(cfg["gif_scale"])
    w = max(1, int(layout["cell_w"] * scale))
    h = max(1, int(layout["cell_h"] * scale))
    sq = 8

    # Build the backdrop ONCE. It is identical for every frame, and doing it with
    # a per-frame Python double loop over w*h was pure waste.
    if cfg["gif_bg"] == "checker":
        yy, xx = np.mgrid[0:h, 0:w]
        board = np.where((((xx // sq) + (yy // sq)) % 2)[..., None],
                         np.array([222, 222, 222], np.uint8),
                         np.array([255, 255, 255], np.uint8)).astype(np.uint8)
    else:
        board = np.full((h, w, 3), 245, np.uint8)

    rgb_frames = []
    for i, p in enumerate(paths):
        im = frame_to_cell(Image.open(p).convert("RGBA"), layout).resize(
            (w, h), Image.LANCZOS)
        comp = board.copy()
        a = np.array(im)[..., 3:4].astype(np.uint16)
        src = np.array(im)[..., :3].astype(np.uint16)
        # straight-alpha composite over the board
        comp = ((src * a + comp.astype(np.uint16) * (255 - a)) // 255).astype(np.uint8)
        rgb_frames.append(Image.fromarray(comp, "RGB"))
        if emit and (i % 16 == 0 or i + 1 == len(paths)):
            emit("gif", (i + 1) / float(len(paths)), "frame %d/%d" % (i + 1, len(paths)))

    # Quantise once and reuse the palette: ADAPTIVE per frame is both slower and
    # gives the frames inconsistent colour tables, which flickers in the GIF.
    first = rgb_frames[0].convert("P", palette=Image.ADAPTIVE, colors=255)
    pal = first.getpalette()
    rest = [f.quantize(palette=first, dither=Image.NONE) for f in rgb_frames[1:]]
    dur = int(1000 / max(1, cfg["fps"]))
    first.save(out_gif, save_all=True, append_images=rest,
               duration=dur, loop=0, optimize=False)
    return out_gif


# --------------------------------------------------------------------------- #
# stage: verify
# --------------------------------------------------------------------------- #

def verify_sheet(sheet_path, paths, layout, cfg):
    """Invariants. Returns (ok, list_of_lines)."""
    from PIL import Image
    out = []
    fails = []
    cols, rows = layout["columns"], layout["rows"]
    cw, ch = layout["cell_w"], layout["cell_h"]
    count = layout["frame_count"]

    if count != cols * rows:
        fails.append("NOTFULL  frame_count %d != %d x %d" % (count, cols, rows))
    sheet = Image.open(sheet_path).convert("RGBA")
    if sheet.size != (cols * cw, rows * ch):
        fails.append("GRID  sheet %dx%d != %dx%d" % (sheet.width, sheet.height,
                                                     cols * cw, rows * ch))
    cap = int(cfg["max_texture"])
    if sheet.width > cap or sheet.height > cap:
        fails.append("TEXTURE  %dx%d over the %d cap" % (sheet.width, sheet.height, cap))

    s = np.array(sheet)
    i1 = i2 = i4 = i5 = 0
    # The picked colour's dominant channel: what "spill" means for this key.
    # NOT called `ch` -- that is the cell height two lines up, and shadowing it
    # made every cell one pixel tall and every frame look empty. Named for what it
    # is instead.
    kch = int(np.argmax(key_rgb(cfg["key_color"])))
    koth = [i for i in range(3) if i != kch]
    faint = 0
    leak_px = 0
    lost_px = 0
    subj_px = 0
    largest = 0
    key_px = 0
    key_big = 0
    for i, p in enumerate(paths):
        src_im = Image.open(p).convert("RGBA")
        exp = frame_to_cell(src_im, layout)
        src = np.array(exp).astype(np.int16)
        cy, cx = (i // cols) * ch, (i % cols) * cw
        cell = s[cy:cy + ch, cx:cx + cw].astype(np.int16)
        alpha = cell[..., 3]
        if not (alpha > 0).any():
            fails.append("EMPTY  cell %d fully transparent" % i)
            continue
        m = alpha > 0
        if (m & (np.abs(cell[..., :3] - src[..., :3]).max(axis=2) > 0)).any():
            i1 += 1
            fails.append("PREMULT  cell %d: RGB differs from the frame where alpha>0" % i)
        reach_dark = border_reachable(cell[..., :3].max(axis=2) < cfg["black_max"])
        leak = reach_dark & (alpha >= cfg["leak_alpha"])
        leak_px += int(leak.sum())
        if leak.any():
            i2 += 1
            if i2 <= 5:
                ys, xs = np.where(leak)
                fails.append("BACKDROP  cell %d: %d px opaque reachable-black "
                             "(e.g. (%d,%d))" % (i, int(leak.sum()), int(ys[0]), int(xs[0])))
        faint += int((reach_dark & (alpha > 0) & (alpha < cfg["leak_alpha"])).sum())
        # I5 -- spill on the rim: a pixel near the silhouette whose key channel
        # still dominates. This is the one the eye actually catches, and it is not
        # a near-miss of the key: on the 00183 clip the leftover outline is
        # (0,26,0)-(0,55,0) at alpha 195-253, 194-233 away from any green backdrop
        # key, so no tolerance can reach it without swallowing the subject's
        # mid-tones (a mid-grey is 205 away at tol 255). It is counted here
        # independently of the key's distance for exactly that reason, and after
        # despill_key -- which the key pass always runs -- it is an INVARIANT: the
        # key's channel is clamped to max(others) everywhere on the rim. So when
        # the key is on, this fails; with it off the count is informational, since
        # failing every run over a backdrop nobody asked to key is the same
        # mistake as defaulting do_repair to on. Geometry matches the frame's:
        # crop and pad preserve distances.
        cell_rgb = cell[..., :3].astype(np.int16)
        spill = cell_rgb[..., kch] > np.maximum(cell_rgb[..., koth[0]],
                                                cell_rgb[..., koth[1]])
        if spill.any():
            import cv2
            # The same reach despill_key now claims: the band ring plus the soft
            # edge. Measuring the band alone would pass sheets whose spill sits
            # deeper -- the user's 2x-upscaled sheet left green at 21 px out, all
            # soft alpha the band never saw.
            rim = ((cv2.distanceTransform((alpha > 0).astype(np.uint8),
                                          cv2.DIST_L2, 3) <= KEY_EDGE_BAND)
                   | (alpha < 255))
            n_spill = int((spill & rim & (alpha > 0)).sum())
            key_px += n_spill
            key_big += int((spill & rim & (alpha >= cfg["leak_alpha"])).sum())
            if n_spill and cfg["do_key"]:
                i5 += 1
                if i5 <= 5:
                    ys, xs = np.where(spill & rim & (alpha > 0))
                    fails.append("SPILL  cell %d: %d px of the key's own channel left "
                                 "on the rim within %d px of the edge, e.g. (%d,%d)"
                                 % (i, n_spill, KEY_EDGE_BAND, int(ys[0]), int(xs[0])))
        # I4 -- subject not eroded.
        # NB: this measures the MATTE, not the repair. The repair can only kill
        # pixels with max(RGB) < black_max, so a bright pixel (max > bright) can
        # never be removed by it -- verified empirically: raw model output and
        # repaired output lose exactly the same bright pixels. What this catches
        # is real erosion: a contiguous hole in the subject.
        bright = src[..., :3].max(axis=2) > cfg["bright"]
        if bright.any():
            lost = bright & (alpha == 0)
            n_lost = int(lost.sum())
            subj_px += int(bright.sum())
            if n_lost:
                import cv2
                nlab, _, stats, _ = cv2.connectedComponentsWithStats(
                    lost.astype(np.uint8), 8)
                big = int(stats[1:, 4].max()) if nlab > 1 else 0
                largest = max(largest, big)
                lost_px += n_lost
                frac = n_lost / float(max(1, int(bright.sum())))
                if big >= cfg["erode_max"] or frac > cfg["erode_frac"]:
                    i4 += 1
                    if i4 <= 5:
                        fails.append(
                            "ERODED  cell %d: %d bright px lost alpha "
                            "(largest hole %d px, %.3f%% of the subject)"
                            % (i, n_lost, big, 100 * frac))

    out.append("grid      : %d cells (%dx%d of %dx%d) sheet %dx%d"
               % (count, cols, rows, cw, ch, sheet.width, sheet.height))
    out.append("I1 straight alpha      failed cells: %d" % i1)
    out.append("I2 no backdrop leak    failed cells: %d  (%d px opaque reachable-black)"
               % (i2, leak_px))
    out.append("I4 subject intact      failed cells: %d  (%d of %d bright px lost, "
               "largest hole %d px; tolerance %d px / %.2f%%)"
               % (i4, lost_px, subj_px, largest, cfg["erode_max"],
                  100 * cfg["erode_frac"]))
    out.append("I5 key spill on the rim   failed cells: %d  (%d px whose %s channel "
               "dominates within %d px of the edge or at partial alpha, "
               "%d of them opaque%s)"
               % (i5, key_px, "RGB"[kch], KEY_EDGE_BAND, key_big,
                  "" if cfg["do_key"] else "; key off, informational"))
    out.append("faint fringe (informational, alpha<%d): %d px" % (cfg["leak_alpha"], faint))
    if fails:
        out.append("")
        out.extend(fails[:25])
        if len(fails) > 25:
            out.append("... and %d more" % (len(fails) - 25))
    return (len(fails) == 0), out


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #

def run_pipeline(cfg, workdir, emit=None, cancel=None, model_cache=None):
    """Full run. `emit(stage, frac, msg)` reports progress; returns a result dict."""
    t_all = time.time()
    _st = {"name": None, "t0": time.time()}

    def e(stage, frac, msg=""):
        # fold the previous stage's wall time into the first message of the new
        # one, so the cost of each stage is visible instead of guessed at
        if _st["name"] != stage:
            if _st["name"] is not None:
                msg = "%s  [%s took %.1fs]" % (msg, _st["name"],
                                               time.time() - _st["t0"])
            _st["name"] = stage
            _st["t0"] = time.time()
        if emit:
            emit(stage, frac, msg)

    cfg = merge_cfg(cfg)
    os.makedirs(workdir, exist_ok=True)
    video = cfg["video"]
    if not video or not os.path.exists(video):
        raise ValueError("video not found: %r" % video)
    # Checked here rather than in the matte stage so a typo'd mask fails before
    # the model is loaded, which is 2-3.5 s and ~845 MB of weights to pay for a
    # missing file. The mask is not loaded at all when it is not used, and a
    # blank path is not an error -- it is the default.
    mask_video = cfg.get("mask_video") or ""
    if mask_video and not os.path.exists(mask_video):
        raise ValueError("mask video not found: %r" % mask_video)

    prefix = cfg["prefix"] or os.path.splitext(os.path.basename(video))[0]
    cfg["prefix"] = prefix

    e("probe", 0.0, "reading %s" % os.path.basename(video))
    info = probe(video)
    e("probe", 1.0, "%d frames @ %.2f fps, %dx%d"
      % (info["frames"], info["fps"], info["width"], info["height"]))

    frames_dir = os.path.join(workdir, "frames")
    if mask_video:
        e("matte", 0.0, "mask video %s (VRMBG-3.0 not run)"
          % os.path.basename(mask_video))
        paths = matte_with_mask(video, mask_video, frames_dir, cfg,
                                emit=e, cancel=cancel)
    elif cfg["do_matte"]:
        model = None
        if model_cache is not None:
            key = (cfg["model_dir"], int(cfg["infer_size"]))
            if model_cache.get("key") != key:
                e("model", 0.0, "loading VRMBG-3.0 (first run only)")
                model_cache["model"] = MatteModel(cfg["model_dir"], cfg["infer_size"])
                model_cache["key"] = key
                e("model", 1.0, "model ready")
            model = model_cache["model"]
        else:
            model = MatteModel(cfg["model_dir"], cfg["infer_size"])
        paths = matte_video(video, frames_dir, cfg, model, emit=e, cancel=cancel)
    else:
        e("matte", 0.0, "skipped (using the video's own alpha if any)")
        paths = extract_frames_plain(video, frames_dir, cfg)

    n = len(paths)
    if n == 0:
        raise ValueError("no frames produced")
    e("matte", 1.0, "%d frames" % n)

    if cfg["do_repair"] or cfg["do_key"]:
        n_border = n_key = n_spill = 0
        for i, p in enumerate(paths):
            if cancel and cancel():
                raise RuntimeError("cancelled")
            from PIL import Image
            im = Image.open(p).convert("RGBA")
            b = k = 0
            if cfg["do_repair"]:
                im, b = repair_backdrop(im, cfg["black_max"], cfg["repair_border_only"])
            s = 0
            if cfg["do_key"]:
                im, k = repair_key(im, cfg["key_color"], cfg["key_tol"],
                                   also_connected=cfg["key_also_connected"])
                # and take the key's colour out of what survived it: the spill on
                # the outline is far too dark to match any tolerance (see
                # despill_key), so removing it is not on the table.
                im, s = despill_key(im, cfg["key_color"], floor=cfg["black_max"])
            if b or k:
                im.save(p)
            n_border += b
            n_key += k
            n_spill += s
            if i % 16 == 0 or i + 1 == n:
                e("repair", (i + 1) / float(n),
                  "%d px cleared, %d despilled" % (n_border + n_key, n_spill))
        did = []
        if cfg["do_repair"]:
            did.append("%d px of leaked backdrop cleared%s"
                       % (n_border, " from the border" if cfg["repair_border_only"]
                          else " (near-black anywhere)"))
        if cfg["do_key"]:
            did.append("%d px of %s leak cleared by colour (tol %d, band %d px%s), "
                       "%d px of spill neutralised on the rim"
                       % (n_key, cfg["key_color"], cfg["key_tol"], KEY_EDGE_BAND,
                          " + border-connected" if cfg["key_also_connected"] else "",
                          n_spill))
            if not n_key:
                # The failure that looks like a success: a key colour that is not
                # the backdrop deletes nothing and reports nothing. Say so here,
                # where the setting was made, rather than leaving it to be noticed
                # in the sprite days later. (A clip with no transparency at all
                # also lands here, and the same sentence is still the right one.)
                did.append("nothing matched %s within %d -- that is not the "
                           "backdrop colour, or the tolerance is too tight"
                           % (cfg["key_color"], cfg["key_tol"]))
        e("repair", 1.0, "; ".join(did))
    else:
        e("repair", 1.0, "skipped")

    e("layout", 0.0, "measuring subject")
    bbox = subject_bbox(paths)
    layout = solve_layout(n, info["width"], info["height"], bbox, cfg)
    for w in layout["warnings"]:
        e("layout", 0.5, w)
    e("layout", 1.0, "%dx%d grid of %dx%d, sheet %dx%d"
      % (layout["columns"], layout["rows"], layout["cell_w"], layout["cell_h"],
         layout["sheet_w"], layout["sheet_h"]))

    result = {"workdir": workdir, "prefix": prefix, "frames": paths,
              "layout": layout, "info": info, "bbox": bbox, "artifacts": {}}

    # The sidecar's one-line account of where the alpha came from. Named in full
    # -- which clip, and which of the two switches was on -- because a sheet
    # whose alpha is a hand-made mask and a sheet whose alpha is the model look
    # identical once they are packed, and the sidecar is the only place that
    # difference can survive.
    matte_note = "vrmbg-3.0 autoregressive"
    if mask_video:
        matte_note = ("mask video %s%s%s"
                      % (os.path.basename(mask_video),
                         ", inverted" if cfg["mask_invert"] else "",
                         ", binarised at %d" % cfg["mask_threshold"]
                         if cfg["mask_binary"] else ""))

    if cfg["want_sheet"]:
        sp = os.path.join(workdir, "%s_sheet.png" % prefix)
        cfg["_sheet"] = sp
        build_sheet(paths, layout, sp, emit=e)
        result["artifacts"]["sheet"] = sp
    else:
        cfg["_sheet"] = os.path.join(workdir, "%s_sheet.png" % prefix)

    if cfg["want_sidecar"]:
        side = sidecar_dict(cfg, layout, video, n, matte_note)
        jp = os.path.join(workdir, "%s.json" % prefix)
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(side, fh, indent=2)
        result["artifacts"]["sidecar"] = jp
        result["sidecar"] = side
    else:
        side = sidecar_dict(cfg, layout, video, n, matte_note)
        result["sidecar"] = side

    if cfg["want_preview"]:
        hp = os.path.join(workdir, "%s_preview.html" % prefix)
        write_preview(hp, cfg, layout, side)
        result["artifacts"]["preview"] = hp

    if cfg["want_gif"]:
        gp = os.path.join(workdir, "%s_preview.gif" % prefix)
        write_gif(paths, gp, cfg, layout, emit=e)
        result["artifacts"]["gif"] = gp

    if cfg["want_frames"]:
        result["artifacts"]["frames"] = frames_dir

    if cfg["do_verify"]:
        e("verify", 0.2, "checking invariants")
        if not cfg["want_sheet"]:
            e("verify", 1.0, "skipped (no sheet built)")
            result["verify"] = {"ok": None, "lines": ["skipped: sheet not built"]}
        else:
            ok, lines = verify_sheet(result["artifacts"]["sheet"], paths, layout, cfg)
            result["verify"] = {"ok": ok, "lines": lines}
            e("verify", 1.0, "PASS" if ok else "FAIL")
    else:
        result["verify"] = {"ok": None, "lines": ["skipped"]}

    e("done", 1.0, "finished in %.1fs" % (time.time() - t_all))
    return result


def extract_frames_plain(video, outdir, cfg):
    """No matte: copy frames straight out, keeping any alpha the video carries."""
    import cv2
    os.makedirs(outdir, exist_ok=True)
    prefix = cfg["prefix"]
    paths = []
    for idx, bgr in read_video(video, cfg["start"], cfg["end"], cfg["frame_step"]):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        p = os.path.join(outdir, "%s_%03d.png" % (prefix, idx + 1))
        cv2.imwrite(p, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        paths.append(p)
    return paths
