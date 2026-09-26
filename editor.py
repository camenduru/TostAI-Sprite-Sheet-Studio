"""Sprite sheet editor -- document model and operations.

Why this is server-side
-----------------------
Every operation below is 5-20 lines of numpy. The same work in the browser would
be a canvas re-implementation of flood fill, morphology, alpha compositing and
PNG slicing -- and it would then have to *agree* with the Python verifier that
gates the export. Instead the browser ships **op commands** and receives
**changed cell indices**, then re-fetches only those cells as small PNGs. One
source of truth for pixels, one for invariants.

Conventions
-----------
* A cell is a uint8 RGBA array shaped ``(cell_h, cell_w, 4)``, **straight**
  alpha (RGB is not divided by A). Transparent pixels are kept at RGB (0,0,0).
* Cell index ``i`` sits at column ``i % columns``, row ``i // columns``.
* A ``rect`` argument is ``[x, y, w, h]`` in cell-local pixel coordinates.
* Cell operations are **pure**: they take an array and return a new one. That is
  what makes "which cells actually changed" computable by comparison, and what
  makes undo a matter of storing the pre-state of exactly those cells.

Undo
----
A snapshot stores the pre-state of the touched cells as **PNG bytes**, not raw
arrays. A 512x640 RGBA cell is 1.3 MB raw and typically 5-30 KB as PNG, because
sprite cells are mostly transparent. That turns a 124-cell ``align`` from a
163 MB undo step into a ~2 MB one, which is the difference between 2 undo levels
and 120.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import time

import numpy as np

import pipeline as P

RGBA = "RGBA"
UNDO_BUDGET = 192 * 1024 * 1024      # bytes of encoded snapshot to retain
UNDO_MAX = 160                       # ...and a hard entry cap


# --------------------------------------------------------------------------- #
# codecs
# --------------------------------------------------------------------------- #

def _png(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr, RGBA).save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def _unpng(b):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(b)).convert(RGBA))


def encode_png(arr):
    return _png(arr)


def fit_cell(arr, cw, ch):
    """Nearest-scale an array to the cell, leaving it alone when it already fits.

    The pixel-art detectors return one pixel per grid cell -- the art at true
    resolution. A cell size belongs to the document, not to a detector, so the
    result is scaled back to it rather than letting the document collapse to the
    grid: the frame and the sheet keep their size, and each grid cell reads as
    one crisp block.
    """
    if arr.shape[1] == cw and arr.shape[0] == ch:
        return arr
    from PIL import Image
    return np.array(Image.fromarray(arr, RGBA).resize((cw, ch), Image.NEAREST))


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #

def subject_box(arr, thr=0):
    """Tight bbox of ``alpha > thr`` as (x0, y0, x1, y1) inclusive, or None."""
    a = arr[..., 3] > thr
    if not a.any():
        return None
    ys = np.flatnonzero(a.any(axis=1))
    xs = np.flatnonzero(a.any(axis=0))
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])


ANCHORS = ("top-left", "top-center", "top-right",
           "center-left", "center", "center-right",
           "bottom-left", "bottom-center", "bottom-right",
           "centroid")


def anchor_point(box, anchor, mask=None):
    """Map a bbox to the anchor point named by ``anchor``.

    Every returned point is an **integer pixel**, deliberately. A geometric
    centre of an even-width box sits on a half-pixel, and a cell whose centre is
    at x.5 cannot be translated onto the same pixel as one at y.5 -- the pivot
    would land one pixel apart and the "aligned" sheet would still jitter. A
    pivot is a pixel, not a coordinate, so the centre is floored to one.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0 + 1, y1 - y0 + 1
    lx, cx, rx = x0, x0 + w // 2, x1 + 1
    ty, cy, by = y0, y0 + h // 2, y1 + 1
    if anchor == "centroid":
        if mask is None:
            return (cx, cy)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return (cx, cy)
        return (int(round(float(xs.mean()))), int(round(float(ys.mean()))))
    return {
        "top-left":      (lx, ty),
        "top-center":    (cx, ty),
        "top-right":     (rx, ty),
        "center-left":   (lx, cy),
        "center":        (cx, cy),
        "center-right":  (rx, cy),
        "bottom-left":   (lx, by),
        "bottom-center": (cx, by),
        "bottom-right":  (rx, by),
    }[anchor]


def _shift(arr, dx, dy, wrap=False):
    """Translate a cell *within its own canvas*, filling the vacated area.

    This keeps the shape. To move content onto a **bigger** canvas use
    ``_place`` -- `_shift` cannot grow an array, so using it to "pad" would
    silently clip whatever fell off the edge instead of adding transparency.
    """
    h, w = arr.shape[:2]
    if wrap:
        return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
    out = np.zeros_like(arr)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    dy0, dy1 = max(0, dy), min(h, h + dy)
    dx0, dx1 = max(0, dx), min(w, w + dx)
    if sy0 < sy1 and sx0 < sx1:
        out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


def _place(arr, dx, dy, W, H):
    """Place ``arr`` at (dx, dy) on a fresh transparent W x H canvas."""
    out = np.zeros((H, W, 4), np.uint8)
    h, w = arr.shape[:2]
    sx0, sy0 = max(0, -dx), max(0, -dy)
    sx1, sy1 = min(w, W - dx), min(h, H - dy)
    if sx0 < sx1 and sy0 < sy1:
        out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = arr[sy0:sy1, sx0:sx1]
    return out


def _premul(arr):
    a = arr[..., 3:4].astype(np.float32) / 255.0
    out = arr.astype(np.float32)
    out[..., :3] *= a
    return out


def _unpremul(f):
    a = f[..., 3:4] / 255.0
    out = f.copy()
    np.divide(out[..., :3], np.maximum(a, 1e-6), out=out[..., :3])
    return np.clip(out, 0, 255).astype(np.uint8)


def _ellipse(px):
    import cv2
    k = max(1, int(px) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _mask_grow(mask, px):
    if px <= 0:
        return mask
    import cv2
    return cv2.dilate(mask.astype(np.uint8), _ellipse(px)).astype(bool)


# --------------------------------------------------------------------------- #
# content scaling -- the pixels inside a cell, with the cell left alone
#
# `_shift` translates a cell's content and `resample` rescales the whole sheet,
# but nothing here could change the size of the art *inside* one frame and keep
# the frame. That is what `match_size` needs, and the pivot is the whole point of
# it: a frame scaled about its subject's feet grows where it stands, so
# equalising the sizes does not also shove the frames around.
# --------------------------------------------------------------------------- #

# Filter names the op exposes. Kept as names rather than as cv2's constants
# because cv2 is imported lazily throughout this project (see pipeline.py), and
# a module-level dict of its values would drag it in at import time.
FILTERS = ("nearest", "bilinear", "bicubic", "lanczos")
_CV2_INTERP = {"nearest": "INTER_NEAREST", "bilinear": "INTER_LINEAR",
               "bicubic": "INTER_CUBIC", "lanczos": "INTER_LANCZOS4"}


def _interp(name):
    import cv2
    return getattr(cv2, _CV2_INTERP.get(str(name), "INTER_LINEAR"))


def subject_size(box, how):
    """The subject's size by the dimension the caller asked to match.

    `longest` is there for a pose that changes aspect -- arms out makes a frame
    wider without making the character any taller -- where matching height alone
    would leave the pair looking different sizes.
    """
    w, h = box[2] - box[0] + 1, box[3] - box[1] + 1
    if how == "width":
        return w
    if how == "longest":
        return max(w, h)
    return h


def reference_size(sizes, how):
    """One size to bring every frame to, from the sizes the selection has.

    The **median**, by default, and the choice is the point: a matte that leaked
    one stray opaque pixel inflates that frame's box, so a mean drags the whole
    set toward the bad frame and a max makes every good frame grow to match it.
    The median ignores the outlier, and because the op reports the factor range
    it needed, the outlier is visible rather than contagious.

    `max` is offered for the case where nothing may be shrunk -- upscaling is
    free of detail loss and downscaling is not -- and `first` for matching a
    frame the user is looking at.
    """
    if not sizes:
        return 0
    vals = sorted(sizes.values())
    n = len(vals)
    if how == "max":
        return vals[-1]
    if how == "mean":
        return int(round(sum(vals) / float(n)))
    if how == "min":
        return vals[0]
    return vals[n // 2] if n % 2 else int(round((vals[n // 2 - 1] + vals[n // 2]) / 2.0))


def content_pivot(cell, anchor, thr=0):
    """The pixel on the subject that a content scale pivots about, or None.

    Measured on the subject's own box, not the cell's, because that is the pivot
    that means something here: scaling about the feet grows a character where it
    stands instead of sliding it, so equalising the frames' sizes does not undo
    the alignment `align` just established. Returns a pixel index rather than a
    coordinate, for the reason `anchor_point` gives.

    None means "this cell has no subject", and the caller leaves it alone rather
    than inventing a pivot for it.
    """
    box = subject_box(cell, thr)
    if box is None:
        return None
    return anchor_point(box, anchor, cell[..., 3] > thr)


def _resample(cell, nw, nh, interp):
    """Scale a straight-alpha cell, premultiplied so its edge cannot go dark.

    Interpolating straight RGBA mixes the subject's colour with the RGB *under*
    its transparent neighbours, which in this project is (0,0,0) -- the same
    mistake `op_unpremultiply` exists to undo, and here it would put a black rim
    on every resized frame. Premultiplying around the filter is the standard fix.

    NEAREST deliberately skips that: it copies pixels instead of mixing them, so
    there is nothing to bleed and the result stays byte-exact.
    """
    if (nw, nh) == (cell.shape[1], cell.shape[0]):
        return cell.copy()
    import cv2
    if interp == cv2.INTER_NEAREST:
        return cv2.resize(cell, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return _unpremul(cv2.resize(_premul(cell), (nw, nh), interpolation=interp))


def scale_about(cell, f, px, py, interp):
    """Scale a cell by `f` about (px, py) and put it back on the same canvas.

    The placement offset is an integer pixel, so a *fractional* factor can land
    the pivot one pixel out: the scaled pivot sits on a half-pixel and has to be
    put on one. Content that grows past the cell edge is clipped -- the caller
    counts those frames and says so, because an op that silently eats the top of
    a sprite is the kind of thing this tool is supposed to refuse to hide.
    """
    h, w = cell.shape[:2]
    nw = max(1, int(round(w * f)))
    nh = max(1, int(round(h * f)))
    scaled = _resample(cell, nw, nh, interp)
    return _place(scaled, int(round(px - px * f)), int(round(py - py * f)), w, h)


# --------------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------------- #

class Doc:
    """An open sheet: cells + layout + metadata, with undo/redo."""

    def __init__(self, doc_id, cells, layout, meta, src_sheet=None, src_sidecar=None):
        self.id = doc_id
        self.cells = cells
        self.orig = [c.copy() for c in cells]
        self.layout = dict(layout)
        self.meta = dict(meta)
        self.src_sheet = src_sheet
        self.src_sidecar = src_sidecar
        self.rev = 0
        self.cell_rev = [0] * len(cells)
        self.png_cache = {}
        self.undo = []
        self.redo = []
        self.journal = []
        self.dirty = set()
        self.op_note = None
        self.opened = time.time()
        # Per-frame sound effects. `path` is where the clip currently lives on
        # disk (the upload staging area); save_doc copies it into the output's
        # audio/ folder and the sidecar names that copy, so the JSON an engine
        # reads points at a file shipped beside the sheet.
        self.audio = []
        self._audio_seq = 0

    # -- audio ------------------------------------------------------------- #

    def add_audio(self, frame, path, name=None, volume=1.0):
        self._audio_seq += 1
        entry = {"id": self._audio_seq, "frame": int(frame),
                 "name": name or os.path.basename(path),
                 "path": path, "volume": float(volume)}
        self.audio.append(entry)
        return entry

    def drop_audio(self, aid):
        aid = int(aid)
        before = len(self.audio)
        self.audio = [e for e in self.audio if e["id"] != aid]
        return len(self.audio) != before

    def set_audio_volume(self, aid, volume):
        """Set one attached clip's volume, clamped to 0..1.

        The number is the one save_doc writes into the sidecar, so this is what
        the engine plays at -- not a preview-only setting. Returns the entry, or
        None if there is no clip with that id.
        """
        aid = int(aid)
        for e in self.audio:
            if e["id"] == aid:
                e["volume"] = max(0.0, min(1.0, float(volume)))
                return e
        return None

    def audio_payload(self):
        """What the client needs to list and play the clips (no server paths)."""
        out = []
        for e in sorted(self.audio, key=lambda x: (x["frame"], x["id"])):
            out.append({"id": e["id"], "frame": e["frame"], "name": e["name"],
                        "volume": e["volume"],
                        "url": "/api/editor/%s/audio/%d" % (self.id, e["id"])})
        return out

    # -- geometry ---------------------------------------------------------- #

    @property
    def n(self):
        return len(self.cells)

    @property
    def cell_w(self):
        return int(self.layout["cell_w"])

    @property
    def cell_h(self):
        return int(self.layout["cell_h"])

    @property
    def columns(self):
        return int(self.layout["columns"])

    @property
    def rows(self):
        return int(self.layout["rows"])

    def set_layout(self, columns=None, cell_w=None, cell_h=None, max_texture=None):
        """Change grid geometry, validating the two rules that matter."""
        cols = int(columns if columns is not None else self.columns)
        cw = int(cell_w if cell_w is not None else self.cell_w)
        ch = int(cell_h if cell_h is not None else self.cell_h)
        cap = int(max_texture or self.meta.get("max_texture") or P.MAX_TEXTURE)
        if cols < 1 or cw < 1 or ch < 1:
            raise ValueError("columns/cell size must be positive")
        if self.n % cols:
            raise ValueError(
                "columns=%d does not divide frame_count=%d -- the grid would "
                "have empty cells, which engines play as blank frames. Valid: %s"
                % (cols, self.n, P.divisors(self.n)))
        rows = self.n // cols
        if cols * cw > cap or rows * ch > cap:
            raise ValueError(
                "sheet would be %dx%d, over the %d px 2D texture limit"
                % (cols * cw, rows * ch, cap))
        dim_changed = (cw != self.cell_w) or (ch != self.cell_h)
        self.layout.update({"columns": cols, "rows": rows,
                            "cell_w": cw, "cell_h": ch,
                            "sheet_w": cols * cw, "sheet_h": rows * ch})
        if dim_changed:
            self.cell_rev = [r + 1 for r in self.cell_rev]
            self.png_cache.clear()
        self.rev += 1

    def replace_cells(self, cells):
        self.cells = list(cells)
        self.cell_rev = [0] * len(cells)
        self.png_cache.clear()
        self.orig = [c.copy() for c in self.cells] if not self.dirty else self.orig
        self.rev += 1

    def bump(self, i):
        self.cell_rev[i] += 1
        self.png_cache.pop(i, None)
        self.dirty.add(i)
        self.rev += 1

    # -- pixel access ------------------------------------------------------ #

    def cell_png(self, i):
        if not (0 <= i < self.n):
            raise IndexError("cell %d out of range" % i)
        hit = self.png_cache.get(i)
        if hit is not None and hit[0] == self.cell_rev[i]:
            return hit[1]
        b = _png(self.cells[i])
        self.png_cache[i] = (self.cell_rev[i], b)
        if len(self.png_cache) > 400:
            for k in list(self.png_cache)[:80]:
                self.png_cache.pop(k, None)
        return b

    def compose(self):
        from PIL import Image
        cols, cw, ch = self.columns, self.cell_w, self.cell_h
        sheet = Image.new(RGBA, (cols * cw, self.rows * ch), (0, 0, 0, 0))
        for i, c in enumerate(self.cells):
            sheet.paste(Image.fromarray(c, RGBA),
                        ((i % cols) * cw, (i // cols) * ch))   # no mask: straight alpha
        return sheet

    # -- undo -------------------------------------------------------------- #

    def _trim_undo(self):
        total = 0
        for e in self.undo:
            total += e["bytes"]
        while self.undo and (total > UNDO_BUDGET or len(self.undo) > UNDO_MAX):
            total -= self.undo[0]["bytes"]
            self.undo.pop(0)

    def push_undo(self, entry):
        self.undo.append(entry)
        self._trim_undo()
        self.redo = []

    def snap_now(self, keys, full):
        """Capture the *current* state of ``keys`` as an undo entry."""
        idx = list(range(self.n)) if full else [i for i in keys if 0 <= i < self.n]
        store = {i: _png(self.cells[i]) for i in idx}
        # `orig` is the loaded reference that I1/I4 verify against, and the frame
        # ops (reverse, permute, _set_cells) rewrite it. A full entry has to carry
        # it or undo restores cells without their reference.
        og = ({i: _png(self.orig[i]) for i in idx if i < len(self.orig)}
              if full else None)
        e = {"cells": store, "orig": og, "layout": dict(self.layout),
             "meta": dict(self.meta), "full": full, "n": self.n}
        e["bytes"] = (sum(len(b) for b in store.values())
                      + sum(len(b) for b in (og or {}).values()) + 256)
        return e

    def _restore(self, entry):
        # Bump the revision monotonically rather than nudging counters, so a
        # restored cell can never share a revision with a stale client-side copy.
        self.rev += 1
        if entry["full"]:
            self.cells = [_unpng(entry["cells"][i]) for i in range(entry["n"])]
            # Restore the reference too, and fall back to the restored cell when
            # an entry predates the field -- len(orig) must always equal n, or
            # every op that indexes orig crashes on the tail frames.
            og = entry.get("orig") or {}
            self.orig = [_unpng(og[i]) if i in og else self.cells[i].copy()
                         for i in range(entry["n"])]
            self.cell_rev = [self.rev] * entry["n"]
            # Recompute dirty instead of assuming. A full restore puts the cells
            # back to a state that may or may not equal the loaded reference, and
            # blanket-marking them made I1/I4 report "no reference, not
            # applicable" for the whole document after any frame op + undo.
            self.dirty = {i for i in range(entry["n"])
                          if not np.array_equal(self.cells[i], self.orig[i])}
            self.png_cache.clear()
        else:
            for i, b in entry["cells"].items():
                self.cells[i] = _unpng(b)
                self.cell_rev[i] = self.rev
                # Recomputed against the reference, exactly as the full branch
                # above does -- not merely added. `dirty` is what every "is this
                # cell an unsaved edit" question reads: the Grid panel's
                # "modified N cells" count, and the idle-document eviction in
                # the server, which keeps anything dirty because it cannot be
                # reopened from disk. Adding unconditionally made an undo of a
                # frame-sized op leave those cells marked modified forever, so
                # the count could only ever grow in a session.
                if i < len(self.orig) and np.array_equal(self.cells[i],
                                                         self.orig[i]):
                    self.dirty.discard(i)
                else:
                    self.dirty.add(i)
                self.png_cache.pop(i, None)
        self.layout = dict(entry["layout"])
        self.meta = dict(entry["meta"])

    def undo_once(self):
        """Pop an undo entry, capturing the current state onto the redo stack.

        The redo entry has to be the state *as it is now*. Storing the same
        pre-op entry on both stacks (the obvious implementation) makes redo a
        no-op that silently rewinds a second time.
        """
        if not self.undo:
            return None
        e = self.undo.pop()
        keys = list(range(e["n"])) if e["full"] else sorted(e["cells"])
        post = self.snap_now(keys, e["full"])
        self._restore(e)
        self.redo.append(post)
        return e

    def redo_once(self):
        if not self.redo:
            return None
        e = self.redo.pop()
        keys = list(range(e["n"])) if e["full"] else sorted(e["cells"])
        pre = self.snap_now(keys, e["full"])
        self._restore(e)
        self.undo.append(pre)
        self._trim_undo()
        return e

    # -- export ------------------------------------------------------------ #

    def sidecar(self, sheet_name, audio=None):
        L, M = self.layout, self.meta
        last = self.n - 1
        trim = int(M.get("trim_to") or 0) - 1 if M.get("trim_to") else last
        trim = max(0, min(last, trim))
        mode = M.get("play_mode", "loop")
        anims = {"play": {"from": 0, "to": trim, "loop": mode == "loop"}}
        if mode == "pingpong":
            anims["reverse"] = {"from": 0, "to": trim, "loop": False}
        sc = {
            "name": M.get("name") or "sprite",
            "sheet": sheet_name,
            "frame_width": self.cell_w,
            "frame_height": self.cell_h,
            "columns": self.columns,
            "rows": self.rows,
            "frame_count": self.n,
            "fps": M.get("fps", 24),
            "crop": M.get("crop") or [0, 0, self.cell_w, self.cell_h],
            "matte": M.get("matte", "edited"),
            "matte_repair": M.get("matte_repair", "none"),
            "anchor": M.get("anchor", "none"),
            "blend": M.get("blend", "straight"),
            "play_mode": mode,
            "animations": anims,
            # The video a sheet was cut from used to be a `source` key of its own.
            # Nothing ever read it -- not this tool, not the engine -- and in a
            # pipeline sidecar it duplicated the `note`, so it was dropped rather
            # than carried: a key with no reader is noise in a file the user has to
            # look at. What it recorded is not lost, because the note names it:
            # that is the one place the video was ever wanted, and the editor's
            # note used to drop the name the pipeline had put there. Older
            # sidecars still carry `source` and are read back (see load_doc), so a
            # sheet that has one keeps its provenance through a re-save.
            "note": ("%d frames%s at %.2f fps%s."
                     % (self.n,
                        (" from " + M["source"]) if M.get("source") else "",
                        M.get("fps", 24),
                        ", edited in Sprite Studio" if self.journal else "")),
            # Kept, and read by this project's own suites: it is how an edited
            # sheet is told from a fresh pipeline output, and unlike the video
            # name it is not recoverable from anything else in the file.
            "edited": {
                "ops": self.journal,
                "revision": self.rev,
                "cells_modified": len(self.dirty),
                "editor": "sprite_studio/editor.py",
            },
        }
        # A per-frame sound list, in frame order. Omitted entirely when there is
        # no audio, so a sheet with none has the same sidecar it always had.
        if audio:
            sc["audio"] = audio
        return sc


# --------------------------------------------------------------------------- #
# argument schema -- the UI builds its controls from this
# --------------------------------------------------------------------------- #

def _A(k, t, d, **kw):
    return dict(k=k, t=t, d=d, **kw)


# The `color` control type renders as <input type="color">, whose value is always
# "#rrggbb" -- so a colour arg is one string on the wire, not three numbers, and
# the swatch opens the system picker. This is the one place that string is read.
_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def hex_rgb(value):
    """'#13ff38' -> (19, 255, 56). Raises rather than guessing at a typo.

    The editor's ops refuse bad input with a reason -- set_columns on a
    non-divisor, match_size on a selection with no subject -- and do not fall
    back to a default. That rule matters more here than anywhere else in this
    file, because the fallback for a colour is *black*: a keyer that quietly
    erased black instead of the colour the user named would take out every dark
    pixel on the subject, and the run would still report success. That is the
    same failure `pipeline.key_rgb` guards against from the other side -- it
    falls back to the backdrop green rather than to black, because there a typo
    must not stop a batch. Here the user is looking at the frame and can fix it,
    so the op refuses and says which value it could not read.

    '#abc' shorthand is accepted, because the picker can produce either form
    depending on the browser and rejecting one of them would be arbitrary.
    """
    m = _HEX_COLOR_RE.match(str(value or "").strip())
    if not m:
        raise ValueError("colour %r is not a hex colour like #13ff38" % (value,))
    h = m.group(1)
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    v = int(h, 16)
    return (v >> 16) & 255, (v >> 8) & 255, v & 255


def hex_palette(value):
    """'#ff0000,00ff00' -> (2, 3) uint8. Raises on a bad entry, never drops one.

    The same colours `hex_rgb` reads, comma-separated, in the form the palette
    picker writes into its hidden input. The refusal rule is the same and for the
    same reason: skipping an entry that could not be read would leave the user
    with a palette that is not the one they named, and the run would still report
    success. Blank entries are skipped, because a trailing comma is punctuation
    rather than a colour.
    """
    out = []
    for part in str(value or "").split(","):
        part = part.strip()
        if part:
            out.append(hex_rgb(part))
    if not out:
        raise ValueError("palette %r has no colours in it" % (value,))
    return np.array(out, np.uint8)


_GEO = [_A("anchor", "choice", "bottom-center", label="anchor point",
           opts=list(ANCHORS))]

OPS = {}


def op(name, group, label, args=(), help="", full=False):
    def deco(fn):
        OPS[name] = {"name": name, "group": group, "label": label,
                     "args": list(args), "help": help, "fn": fn, "full": full}
        return fn
    return deco


# --------------------------------------------------------------------------- #
# group: Transform
# --------------------------------------------------------------------------- #

@op("offset", "Transform", "Nudge",
    [_A("dx", "int", 0, min=-4096, max=4096, label="dx (px)"),
     _A("dy", "int", 0, min=-4096, max=4096, label="dy (px)"),
     _A("wrap", "bool", False, label="wrap around")],
    "Translate the pixels inside each selected cell. The canonical fix for a "
    "subject that drifts off its pivot.")
def op_offset(doc, idxs, a):
    if not a["dx"] and not a["dy"]:
        return []
    return _apply(doc, idxs, lambda c: _shift(c, a["dx"], a["dy"], a["wrap"]))


@op("flip_h", "Transform", "Flip horizontal")
def op_flip_h(doc, idxs, a):
    return _apply(doc, idxs, lambda c: c[:, ::-1].copy())


@op("flip_v", "Transform", "Flip vertical")
def op_flip_v(doc, idxs, a):
    return _apply(doc, idxs, lambda c: c[::-1, :].copy())


@op("rotate", "Transform", "Rotate 90/180/270",
    [_A("turns", "choice", "90", label="degrees", opts=["90", "180", "270"])],
    "Rotating by 90 or 270 on non-square cells changes the cell's aspect, so it "
    "is only allowed when the operation covers every cell -- the grid then "
    "swaps its cell dimensions rather than producing a ragged sheet.")
def op_rotate(doc, idxs, a):
    k = {"90": 1, "180": 2, "270": 3}[a["turns"]]
    if k == 2:
        return _apply(doc, idxs, lambda c: np.rot90(c, 2).copy())
    sel = set(idxs)
    if len(sel) != doc.n and doc.cell_w != doc.cell_h:
        raise ValueError(
            "90/270 rotation on a %dx%d cell would change its aspect. "
            "Select every cell (or make the cells square) so the grid can "
            "swap its dimensions." % (doc.cell_w, doc.cell_h))
    out = [np.rot90(c, k).copy() if i in sel else c
           for i, c in enumerate(doc.cells)]
    nh, nw = out[0].shape[:2]
    if (nw, nh) != (doc.cell_w, doc.cell_h):
        doc.set_layout(columns=doc.columns, cell_w=nw, cell_h=nh)
    changed = []
    for i in idxs:
        if not np.array_equal(out[i], doc.cells[i]):
            doc.cells[i] = out[i]
            doc.bump(i)
            changed.append(i)
    return changed


@op("match_size", "Transform", "Match content size",
    [_A("measure", "choice", "height", label="size to match",
        opts=["height", "width", "longest"]),
     _A("reference", "choice", "median", label="reference size",
        opts=["median", "mean", "max", "min", "first"]),
     _A("target", "int", 0, min=0, max=4096, label="target px (0 = derive)"),
     _A("anchor", "choice", "bottom-center", label="pivot that stays put",
        opts=list(ANCHORS)),
     _A("threshold", "int", 0, min=0, max=254, label="alpha threshold"),
     _A("resample", "choice", "bilinear", label="filter", opts=list(FILTERS))],
    "The one-click fix for an animation that pulses: the subject is a little "
    "small in some frames and a little big in others, so the loop looks like it "
    "jumps even though every frame is right on its own. Select every frame and "
    "run this -- it measures the subject in each one and scales each to the same "
    "size. The cell size, the frame count and the grid are untouched; only the "
    "art inside the cells changes. The pivot is what stops it trading a size "
    "jump for a position jump: scaling about the feet leaves the feet on the "
    "pixel they were on -- the anchor is measured again on the result and put "
    "back if the resample moved it -- so an alignment you already ran survives. "
    "Size is the subject's height unless you say otherwise. The reference is the "
    "median of the selection, which one outlying frame moves less than a mean or "
    "a max would; if a frame's number looks wrong in the status line, raise the "
    "alpha threshold, because a stray opaque pixel inflates that frame's box. "
    "The status line reports the factor range, the size actually achieved and "
    "how many frames were clipped at the cell edge -- a subject already taller "
    "than its cell cannot be matched without losing the top of it.")
def op_match_size(doc, idxs, a):
    thr = int(a["threshold"])
    how = a["measure"]
    interp = _interp(a["resample"])

    boxes = {}
    for i in idxs:
        b = subject_box(doc.cells[i], thr)
        if b:
            boxes[i] = b
    if not boxes:
        raise ValueError("no subject found in the selection (alpha threshold %d)"
                         % thr)

    sizes = {i: subject_size(b, how) for i, b in boxes.items()}
    target = int(a["target"])
    if target <= 0:
        # `first` has to fall back when the first selected frame is one of the
        # empty ones -- otherwise the reference is undefined and the op would
        # have to refuse a selection it can perfectly well handle.
        if a["reference"] == "first" and idxs[0] in sizes:
            target = sizes[idxs[0]]
        else:
            target = reference_size(sizes, a["reference"])
    if target < 1:
        raise ValueError("the reference size came out as %d px, so there is "
                         "nothing to match" % target)

    changed, factors, clipped, achieved = [], {}, [], []
    for i in idxs:
        if i not in boxes or sizes[i] <= 0:
            continue
        f = target / float(sizes[i])
        if abs(f - 1.0) < 1e-9:
            continue
        cell = doc.cells[i]
        p = content_pivot(cell, a["anchor"], thr)
        if p is None:
            continue
        out = scale_about(cell, f, p[0], p[1], interp)
        # The scale places its output on an integer offset, and the pivot of a
        # *resampled* subject does not land on the pivot of the original to the
        # pixel -- a fractional factor puts the scaled pivot on a half-pixel and
        # the soft edge an interpolating filter leaves can round the other way.
        # Left there, this op would trade a size jump for a one-pixel position
        # jump, and would quietly undo an alignment the user had already run. So
        # the anchor is measured on the result and put back where it was, which
        # is the same integer translation `align` uses. A shift cannot change
        # any size, so this is free of the thing the op is here to fix.
        nb = subject_box(out, thr)
        if nb is not None:
            na = anchor_point(nb, a["anchor"], out[..., 3] > thr)
            if na != p:
                out = _shift(out, p[0] - na[0], p[1] - na[1])
        if np.array_equal(out, cell):
            continue
        # Whether it fitted is asked of the *box*, not of the result: the
        # result's own box is already clipped, so it cannot answer.
        px, py = p
        x0, y0, x1, y1 = boxes[i]
        if (px + f * (x0 - px) < -0.5 or py + f * (y0 - py) < -0.5
                or px + f * (x1 - px) > cell.shape[1] - 0.5
                or py + f * (y1 - py) > cell.shape[0] - 0.5):
            clipped.append(i)
        doc.cells[i] = out
        doc.bump(i)
        changed.append(i)
        factors[i] = f
        b_out = subject_box(out, thr)
        if b_out is not None:
            achieved.append(subject_size(b_out, how))

    # A frame the op could not act on is part of the report. An animation with a
    # blank frame in it is a real case, and "matched 23 of 24" is a different
    # statement from "matched 23".
    empty = len(idxs) - len(boxes)
    tail = ""
    if empty:
        tail += "; %d frame%s had no subject and %s left alone" % (
            empty, "" if empty == 1 else "s", "was" if empty == 1 else "were")
    if clipped:
        tail += ("; %d clipped at the cell edge -- raise the cell size or lower "
                 "the target" % len(clipped))

    if not changed:
        doc.op_note = ("Match content size: every subject in the selection is "
                       "already %d px %s%s" % (target, how, tail))
        return []
    # The *achieved* range, not just the target: the cell is an integer canvas,
    # so a fractional factor lands the subject within a pixel of what was asked
    # for and a resampled edge is soft enough to measure a pixel either way. The
    # user's question is "are they the same size now", and this answers it.
    got = ("%d" % achieved[0] if len(set(achieved)) == 1
           else "%d-%d" % (min(achieved), max(achieved)))
    doc.op_note = (
        "Match content size: %d frame%s, %s %d-%d px -> %d px, now %s "
        "(factor %.2f-%.2f)%s"
        % (len(changed), "" if len(changed) == 1 else "s", how,
           min(sizes[i] for i in changed), max(sizes[i] for i in changed),
           target, got, min(factors.values()), max(factors.values()), tail))
    return changed


def transform_about(cell, f, ang, px, py, interp):
    """Rotate by ``ang`` degrees and scale by ``f`` about (px, py), same canvas.

    One matrix rather than two passes, because `getRotationMatrix2D` composes
    exactly these two about exactly one pivot -- so there is no rotate-then-scale
    / scale-then-rotate order for a caller to get wrong, and the two cannot
    disagree by a pixel. `expand=False` is the whole point: the output canvas is
    the cell, and content that leaves it is clipped rather than the cell growing.

    Interpolated filters run on premultiplied alpha for the reason `_resample`
    gives -- otherwise the transparent black around a sprite bleeds into its
    edge -- while `nearest` runs straight on the uint8 array, so the pixel-art
    path cannot be altered by a colour-space round trip.
    """
    import cv2
    h, w = cell.shape[:2]
    M = cv2.getRotationMatrix2D((float(px), float(py)), float(ang), float(f))
    if interp == cv2.INTER_NEAREST:
        return cv2.warpAffine(cell, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    return _unpremul(cv2.warpAffine(_premul(cell), M, (w, h), flags=interp,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0)))


@op("transform_content", "Transform", "Transform (box)",
    [_A("dx", "int", 0, min=-4096, max=4096, label="move x"),
     _A("dy", "int", 0, min=-4096, max=4096, label="move y"),
     _A("scale", "float", 1.0, min=0.05, max=4.0, label="scale"),
     _A("angle", "float", 0.0, min=-180.0, max=180.0, label="rotate (deg)"),
     _A("anchor", "choice", "center", label="pivot that stays put",
        opts=list(ANCHORS)),
     _A("threshold", "int", 0, min=0, max=254, label="alpha threshold"),
     _A("resample", "choice", "nearest", label="filter", opts=list(FILTERS))],
    "Move, scale and rotate the art inside each selected frame, leaving the cell "
    "size, the frame count and the grid alone. This is the op behind the box the "
    "canvas draws around a frame's subject: drag inside the box to move it, a "
    "corner or edge handle to scale it, the round handle above it to rotate. The "
    "pivot is a point on the subject's own box, so scaling about `center` grows "
    "the art where it stands instead of sliding it, and the anchor is measured "
    "again on the result and put back if the transform moved it -- without that, "
    "every edit would add the one-pixel drift `align` exists to remove. The move "
    "is applied after the transform, so the box's pivot ends up exactly `dx`, "
    "`dy` from where it was. `nearest` is the default because this is a pixel-art "
    "tool: it keeps the art hard-edged and makes a 90-degree rotation exact, "
    "where an interpolating filter would soften it. Content that leaves the cell "
    "is clipped and the status line says how many frames that happened to -- the "
    "cell size is the pipeline's contract, so growing the canvas is Resize cells "
    "and not something this op does behind your back.")
def op_transform_content(doc, idxs, a):
    thr = int(a["threshold"])
    dx, dy = int(a["dx"]), int(a["dy"])
    f = float(a["scale"])
    ang = float(a["angle"])
    interp = _interp(a["resample"])
    if dx == 0 and dy == 0 and abs(f - 1.0) < 1e-9 and abs(ang) < 1e-9:
        # The form's defaults, and the state a click on Apply with nothing
        # dragged leaves behind. Reporting every frame changed here would put an
        # undo entry on the stack that restores what is already there.
        raise ValueError("nothing to do: move 0, scale 1.0 and rotate 0 would "
                         "report every frame changed and leave it as it was")
    if f <= 0:
        raise ValueError("a scale of %g would erase the frame" % f)

    boxes = {}
    for i in idxs:
        b = subject_box(doc.cells[i], thr)
        if b is not None:
            boxes[i] = b
    if not boxes:
        raise ValueError("no subject found in the selection (alpha threshold %d)"
                         % thr)

    changed, clipped, emptied = [], [], []
    for i in idxs:
        if i not in boxes:
            continue
        cell = doc.cells[i]
        p = anchor_point(boxes[i], a["anchor"], cell[..., 3] > thr)
        out = transform_about(cell, f, ang, p[0], p[1], interp)
        # The anchor is measured again on the result, for the reason match_size
        # gives: warpAffine samples on the integer grid, so a fractional factor
        # or an off-axis angle can leave the pivot a pixel out, and a transform
        # that is supposed to be about a point would quietly walk the sprite
        # across the cell over a few edits.
        nb = subject_box(out, thr)
        if nb is not None:
            na = anchor_point(nb, a["anchor"], out[..., 3] > thr)
            if na != p:
                out = _shift(out, p[0] - na[0], p[1] - na[1])
        if dx or dy:
            out = _shift(out, dx, dy)
        if np.array_equal(out, cell):
            continue
        if not out[..., 3].any():
            # Everything landed off the cell. Skipped rather than committed, so
            # one bad frame in a selection cannot blank itself.
            emptied.append(i)
            continue
        # Whether it fitted is asked of the *box*, not of the result: the
        # result's box is already clipped and so cannot answer. Same reasoning as
        # match_size, and the same corners.
        box = boxes[i]
        sx, sy = p[0] + dx, p[1] + dy
        if (sx + f * (box[0] - p[0]) < -0.5 or sy + f * (box[1] - p[1]) < -0.5
                or sx + f * (box[2] - p[0]) > cell.shape[1] - 0.5
                or sy + f * (box[3] - p[1]) > cell.shape[0] - 0.5):
            clipped.append(i)
        doc.cells[i] = out
        doc.bump(i)
        changed.append(i)

    if not changed:
        raise ValueError(
            "that transform leaves every selected frame as it was (%d frame%s "
            "would have been emptied by it -- the art would land outside the "
            "cell)" % (len(emptied), "" if len(emptied) == 1 else "s")
            if emptied else
            "that transform leaves every selected frame as it was")

    parts = []
    if dx or dy:
        parts.append("move %+d,%+d" % (dx, dy))
    if abs(f - 1.0) >= 1e-9:
        parts.append("scale %.3g" % f)
    if abs(ang) >= 1e-9:
        parts.append("rotate %g deg" % ang)
    tail = ""
    empty = len(idxs) - len(boxes)
    if empty:
        tail += ("; %d frame%s had no subject and %s left alone"
                 % (empty, "" if empty == 1 else "s",
                    "was" if empty == 1 else "were"))
    if clipped:
        tail += ("; %d clipped at the cell edge -- the cell size is the sheet's, "
                 "so use Resize cells if the art needs more room" % len(clipped))
    if emptied:
        tail += ("; %d left alone rather than emptied" % len(emptied))
    doc.op_note = ("Transform (box): %d frame%s, %s about %s, %s%s"
                   % (len(changed), "" if len(changed) == 1 else "s",
                      ", ".join(parts), a["anchor"],
                      "nearest-neighbour" if a["resample"] == "nearest"
                      else "filtered (%s)" % a["resample"], tail))
    return changed


@op("align", "Transform", "Align pivot (de-jitter)",
    [_A("anchor", "choice", "bottom-center", label="align this point",
        opts=list(ANCHORS)),
     _A("reference", "choice", "union", label="reference",
        opts=["union", "first", "last"]),
     _A("threshold", "int", 0, min=0, max=254, label="alpha threshold"),
     _A("grow", "bool", False, label="grow the cell if needed"),
     _A("pad", "int", 0, min=0, max=256, label="extra padding")],
    "Move every selected cell so a chosen point on the subject (feet, centre, "
    "centroid) lands on the same pixel. This is the one-click fix for an "
    "animation whose subject jitters because the matte bbox moves frame to "
    "frame. The cell only grows if you allow it.")
def op_align(doc, idxs, a):
    thr, ref, anc = a["threshold"], a["reference"], a["anchor"]
    boxes, masks = {}, {}
    for i in idxs:
        b = subject_box(doc.cells[i], thr)
        if b:
            boxes[i] = b
            masks[i] = doc.cells[i][..., 3] > thr
    if not boxes:
        raise ValueError("no subject found in the selection (alpha threshold %d)" % thr)

    def point(i):
        return anchor_point(boxes[i], anc, masks[i])

    # distance from the anchor point to each side of the box
    def rel(i):
        ax, ay = point(i)
        b = boxes[i]
        return (ax - b[0], b[2] + 1 - ax, ay - b[1], b[3] + 1 - ay)

    rels = {i: rel(i) for i in boxes}
    L = max(r[0] for r in rels.values())
    R = max(r[1] for r in rels.values())
    T = max(r[2] for r in rels.values())
    B = max(r[3] for r in rels.values())
    need_w, need_h = int(L + R), int(T + B)
    grew = False

    if need_w > doc.cell_w or need_h > doc.cell_h:
        if not a["grow"]:
            raise ValueError(
                "the aligned subject needs a %dx%d cell but the cells are "
                "%dx%d. Enable 'grow the cell' to enlarge the grid, or align a "
                "different point." % (need_w, need_h, doc.cell_w, doc.cell_h))
        pad = int(a["pad"])
        new_w, new_h = need_w + 2 * pad, need_h + 2 * pad
        # Every cell grows and existing content shifts by the same delta, so
        # nothing moves relative to its own cell -- and _place is required here,
        # because _shift cannot grow an array and would clip instead.
        ox, oy = (new_w - doc.cell_w) // 2, (new_h - doc.cell_h) // 2
        for i in range(doc.n):
            doc.cells[i] = _place(doc.cells[i], ox, oy, new_w, new_h)
            doc.bump(i)
        doc.set_layout(columns=doc.columns, cell_w=new_w, cell_h=new_h)
        boxes = {i: (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy)
                 for i, b in boxes.items()}
        rels = {i: rel(i) for i in boxes}
        L = max(r[0] for r in rels.values())
        R = max(r[1] for r in rels.values())
        T = max(r[2] for r in rels.values())
        B = max(r[3] for r in rels.values())
        grew = True

    # Place the union box centred in the cell. Integer division on purpose: a
    # half-pixel target is unrepresentable, and rounding each cell against it
    # independently is exactly how a pivot ends up one pixel apart.
    tx = (doc.cell_w - (L + R)) // 2 + L
    ty = (doc.cell_h - (T + B)) // 2 + T

    if ref == "first":
        ax, ay = point(idxs[0] if idxs[0] in boxes else min(boxes))
    elif ref == "last":
        ax, ay = point(idxs[-1] if idxs[-1] in boxes else max(boxes))
    else:
        ax, ay = tx, ty

    out = {i: (ax - point(i)[0], ay - point(i)[1]) for i in boxes}

    changed = []
    for i, (dx, dy) in out.items():
        if dx or dy:
            doc.cells[i] = _shift(doc.cells[i], dx, dy)
            doc.bump(i)
            changed.append(i)
    # A grow rewrote every cell (they all shifted onto the new canvas), so the
    # client has to refetch all of them, not just the ones that then moved.
    return list(range(doc.n)) if grew else changed


@op("crop_tight", "Transform", "Crop cells to content",
    [_A("pad", "int", 0, min=0, max=256, label="padding"),
     _A("threshold", "int", 0, min=0, max=254, label="alpha threshold")],
    "Shrink the cell to the union bounding box of the selected cells. Reduces "
    "the sheet size, but changes the pivot -- re-run Align afterwards.",
    full=True)
def op_crop_tight(doc, idxs, a):
    box = None
    for i in idxs:
        b = subject_box(doc.cells[i], a["threshold"])
        if not b:
            continue
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3]))
    if box is None:
        raise ValueError("no subject found in the selection")
    p = int(a["pad"])
    x0 = max(0, box[0] - p)
    y0 = max(0, box[1] - p)
    x1 = min(doc.cell_w, box[2] + 1 + p)
    y1 = min(doc.cell_h, box[3] + 1 + p)
    nw, nh = x1 - x0, y1 - y0
    for i in range(doc.n):
        doc.cells[i] = doc.cells[i][y0:y1, x0:x1].copy()
        doc.bump(i)
    doc.set_layout(columns=doc.columns, cell_w=nw, cell_h=nh)
    return list(range(doc.n))


@op("pad_cell", "Transform", "Grow cell canvas",
    [_A("px", "int", 8, min=1, max=1024, label="pixels per side")],
    "Add transparent border inside every cell. Changes the grid's cell size.",
    full=True)
def op_pad_cell(doc, idxs, a):
    n = int(a["px"])
    nw, nh = doc.cell_w + 2 * n, doc.cell_h + 2 * n
    for i in range(doc.n):
        doc.cells[i] = _place(doc.cells[i], n, n, nw, nh)
        doc.bump(i)
    doc.set_layout(columns=doc.columns, cell_w=nw, cell_h=nh)
    return list(range(doc.n))


@op("resample", "Transform", "Resample all cells",
    [_A("scale", "float", 0.5, min=0.05, max=4.0, step=0.05, label="scale factor")],
    "Scale every cell. Useful to fit a sheet under the 32768 px texture cap "
    "without dropping frames.",
    full=True)
def op_resample(doc, idxs, a):
    from PIL import Image
    s = float(a["scale"])
    nw = max(2, int(round(doc.cell_w * s)) & ~1)
    nh = max(2, int(round(doc.cell_h * s)) & ~1)
    for i in range(doc.n):
        im = Image.fromarray(doc.cells[i], RGBA).resize((nw, nh), Image.LANCZOS)
        doc.cells[i] = np.array(im)
        doc.bump(i)
    doc.set_layout(columns=doc.columns, cell_w=nw, cell_h=nh)
    return list(range(doc.n))


@op("copy_from", "Transform", "Copy one cell into selection",
    [_A("index", "int", 0, min=0, max=100000, label="source cell index")],
    "Overwrite every selected cell with the given cell's pixels.")
def op_copy_from(doc, idxs, a):
    src = int(a["index"])
    if not (0 <= src < doc.n):
        raise ValueError("cell %d does not exist" % src)
    pix = doc.cells[src].copy()
    return _apply(doc, idxs, lambda c: pix.copy())


# --------------------------------------------------------------------------- #
# group: Alpha
# --------------------------------------------------------------------------- #

def _alpha_only(fn):
    """Wrap a float->float alpha mapping into a pure cell operation."""
    def inner(c):
        f = c.astype(np.float32)
        f[..., 3] = np.clip(fn(f[..., 3]), 0, 255)
        return f.astype(np.uint8)
    return inner


@op("alpha_threshold", "Alpha", "Threshold alpha",
    [_A("t", "int", 8, min=0, max=255, label="kill alpha below")],
    "Snap near-transparent pixels to fully transparent. Cleans the faint "
    "fringe a matting model leaves behind, without touching the silhouette.")
def op_alpha_threshold(doc, idxs, a):
    t = float(a["t"])
    return _apply(doc, idxs, _alpha_only(lambda v: np.where(v < t, 0.0, v)))


@op("alpha_levels", "Alpha", "Remap alpha range",
    [_A("lo", "int", 0, min=0, max=255, label="black point"),
     _A("hi", "int", 255, min=0, max=255, label="white point")],
    "Stretch alpha so lo becomes 0 and hi becomes 255. Use it to harden a soft "
    "matte or to soften a clipped one.")
def op_alpha_levels(doc, idxs, a):
    lo, hi = float(a["lo"]), float(a["hi"])
    if hi <= lo:
        raise ValueError("white point must be above the black point")
    return _apply(doc, idxs, _alpha_only(
        lambda v: (v - lo) * 255.0 / (hi - lo)))


@op("alpha_gain", "Alpha", "Scale alpha",
    [_A("gain", "float", 1.25, min=0.0, max=8.0, step=0.05, label="multiplier")])
def op_alpha_gain(doc, idxs, a):
    g = float(a["gain"])
    return _apply(doc, idxs, _alpha_only(lambda v: v * g))


@op("alpha_opacity", "Alpha", "Set opacity",
    [_A("pct", "int", 100, min=0, max=100, label="percent")])
def op_alpha_opacity(doc, idxs, a):
    p = float(a["pct"]) / 100.0
    return _apply(doc, idxs, _alpha_only(lambda v: v * p))


@op("erode_alpha", "Alpha", "Erode alpha",
    [_A("px", "int", 1, min=1, max=64, label="pixels")],
    "Shrink the silhouette. The standard cure for a one-pixel dark fringe "
    "hugging the subject after background removal.")
def op_erode_alpha(doc, idxs, a):
    import cv2
    k = _ellipse(a["px"])

    def fn(c):
        out = c.copy()
        out[..., 3] = cv2.erode(c[..., 3], k)
        return out
    return _apply(doc, idxs, fn)


@op("dilate_alpha", "Alpha", "Dilate alpha",
    [_A("px", "int", 1, min=1, max=64, label="pixels")],
    "Grow the silhouette, for a matte that ate into the subject.")
def op_dilate_alpha(doc, idxs, a):
    import cv2
    k = _ellipse(a["px"])

    def fn(c):
        out = c.copy()
        out[..., 3] = cv2.dilate(c[..., 3], k)
        return out
    return _apply(doc, idxs, fn)


@op("feather_alpha", "Alpha", "Feather alpha",
    [_A("radius", "float", 1.0, min=0.1, max=32.0, step=0.1, label="radius (px)")])
def op_feather_alpha(doc, idxs, a):
    import cv2
    r = float(a["radius"])
    k = int(r * 3) | 1

    def fn(c):
        out = c.copy()
        out[..., 3] = cv2.GaussianBlur(c[..., 3], (k, k), r)
        return out
    return _apply(doc, idxs, fn)


# --------------------------------------------------------------------------- #
# group: Matte repair
# --------------------------------------------------------------------------- #

@op("unpremultiply", "Matte repair", "Un-premultiply RGB",
    [_A("min_alpha", "int", 8, min=0, max=255, label="ignore below alpha"),
     _A("strength", "float", 1.0, min=0.0, max=1.0, step=0.05, label="strength"),
     _A("edges_only", "bool", False, label="edges only (alpha < 255)"),
     _A("clamp", "bool", True, label="clamp to 255")],
    "Divide RGB by alpha. If a sheet was composited over black (or pasted with "
    "the image as its own mask) its RGB is already multiplied by alpha, and "
    "every semi-transparent edge is too dark. This is the exact inverse.")
def op_unpremultiply(doc, idxs, a):
    lo = float(a["min_alpha"])
    s = float(a["strength"])
    edges = a["edges_only"]
    cl = a["clamp"]

    def fn(c):
        f = c.astype(np.float32)
        al = f[..., 3]
        m = al >= lo
        if edges:
            m &= al < 255.0
        if not m.any():
            return c
        fixed = f[..., :3] / np.maximum(al[..., None] / 255.0, 1e-6)
        if s < 1.0:
            fixed = f[..., :3] + (fixed - f[..., :3]) * s
        if cl:
            fixed = np.clip(fixed, 0, 255)
        out = f.copy()
        out[..., :3] = np.where(m[..., None], fixed, f[..., :3])
        return np.clip(out, 0, 255).astype(np.uint8)
    return _apply(doc, idxs, fn)


@op("erase_border_black", "Matte repair", "Erase leaked backdrop",
    [_A("black_max", "int", 4, min=0, max=64, label="near-black threshold"),
     _A("leak_alpha", "int", 200, min=1, max=255, label="opaque threshold"),
     _A("connected", "bool", True, label="only border-connected regions"),
     _A("grow", "int", 0, min=0, max=16, label="grow the erase mask (px)")],
    "Topological, not tonal: a backdrop touches the frame border, the subject's "
    "own dark pixels are enclosed by its body. Clears border-connected "
    "near-black pixels that are still opaque. This is the same repair the "
    "pipeline applies, with an optional mask grow to eat the fringe. Unticking "
    "'border-connected' drops the connectivity test and makes it a tonal black "
    "key: that reaches backdrop the subject enclosed and eats the subject's dark "
    "parts with it. The same switch 'erase a colour' already has.")
def op_erase_border_black(doc, idxs, a):
    bm, la, grow = int(a["black_max"]), int(a["leak_alpha"]), int(a["grow"])
    connected = bool(a["connected"])

    def fn(c):
        near_black = c[..., :3].max(axis=2) < bm
        reach = P.border_reachable(near_black) if connected else near_black
        kill = reach & (c[..., 3] >= la)
        if grow > 0:
            kill = _mask_grow(kill, grow) & (c[..., 3] > 0)
        if not kill.any():
            return c
        out = c.copy()
        out[..., 3] = np.where(kill, 0, out[..., 3])
        return out
    return _apply(doc, idxs, fn)


@op("erase_color", "Matte repair", "Erase a colour",
    [_A("color", "color", "#000000", label="colour to erase"),
     _A("tol", "int", 12, min=0, max=255, label="tolerance"),
     _A("connected", "bool", True, label="only border-connected regions"),
     _A("opaque_only", "bool", False, label="opaque pixels only")],
    "Colour key, chosen with a picker rather than typed as three numbers. With "
    "'border-connected' on, only regions reachable from the frame border are "
    "removed, so a dark patch enclosed by the subject survives -- which is what "
    "makes a keyer usable on line art. The swatch opens the system picker; a "
    "value that is not a hex colour is refused rather than falling back to "
    "black, because black is the one fallback that would erase the subject's "
    "own dark pixels and still report success.")
def op_erase_color(doc, idxs, a):
    import cv2
    tgt = np.array(hex_rgb(a["color"]), np.int16)
    tol = int(a["tol"])

    def fn(c):
        d = np.abs(c[..., :3].astype(np.int16) - tgt).max(axis=2)
        m = d <= tol
        if a["connected"]:
            n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=4)
            if n > 1:
                touch = set(lab[0].tolist()) | set(lab[-1].tolist())
                touch |= set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
                touch.discard(0)
                m = np.isin(lab, np.fromiter(touch, np.int32)) if touch else np.zeros_like(m)
        if a["opaque_only"]:
            m &= c[..., 3] >= 128
        if not m.any():
            return c
        out = c.copy()
        out[..., 3] = np.where(m, 0, out[..., 3])
        return out
    return _apply(doc, idxs, fn)


# --------------------------------------------------------------------------- #
# group: Colour
# --------------------------------------------------------------------------- #

@op("adjust", "Colour", "Brightness / contrast / saturation",
    [_A("brightness", "float", 1.0, min=0.0, max=4.0, step=0.05, label="brightness"),
     _A("contrast", "float", 1.0, min=0.0, max=4.0, step=0.05, label="contrast"),
     _A("saturation", "float", 1.0, min=0.0, max=4.0, step=0.05, label="saturation"),
     _A("gamma", "float", 1.0, min=0.1, max=4.0, step=0.05, label="gamma")],
    "Applied to opaque pixels only, so the transparent backdrop stays provably "
    "(0,0,0) and the backdrop-leak invariant keeps its meaning.")
def op_adjust(doc, idxs, a):
    br, ct = float(a["brightness"]), float(a["contrast"])
    sa, gm = float(a["saturation"]), float(a["gamma"])

    def fn(c):
        if br == ct == sa == gm == 1.0:
            return c
        f = c.astype(np.float32) / 255.0
        x = f[..., :3]
        if gm != 1.0:
            x = np.power(np.clip(x, 0, 1), 1.0 / gm)
        if ct != 1.0:
            x = (x - 0.5) * ct + 0.5
        if br != 1.0:
            x = x * br
        if sa != 1.0:
            lum = (x * np.array([0.299, 0.587, 0.114], np.float32)).sum(axis=2,
                                                                        keepdims=True)
            x = lum + (x - lum) * sa
        x = np.clip(x, 0, 1) * 255.0
        m = (c[..., 3] > 0)[..., None]
        out = f.copy()
        out[..., :3] = np.where(m, x, f[..., :3])
        return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)
    return _apply(doc, idxs, fn)


@op("posterize", "Colour", "Posterize",
    [_A("levels", "int", 6, min=2, max=64, label="levels per channel")],
    "Quantise RGB to N levels. Snaps a soft gradient matte back to a pixel-art "
    "palette.")
def op_posterize(doc, idxs, a):
    n = max(2, int(a["levels"])) - 1

    def fn(c):
        f = c.astype(np.float32)
        q = np.round(f[..., :3] * n / 255.0) * 255.0 / n
        m = (c[..., 3] > 0)[..., None]
        out = f.copy()
        out[..., :3] = np.where(m, q, f[..., :3])
        return np.clip(out, 0, 255).astype(np.uint8)
    return _apply(doc, idxs, fn)


def _nearest_index(rgb, palette):
    """For each row of ``rgb`` (n,3), the index of the nearest palette colour.

    Chunked by a fixed budget of rows x colours so a 512x512 frame against a
    256-entry palette never materialises the whole (n, k) distance matrix.
    """
    P = np.asarray(palette, np.float32)
    step = max(1, 1_000_000 // max(1, len(P)))
    out = np.empty(len(rgb), np.intp)
    for s in range(0, len(rgb), step):
        chunk = np.asarray(rgb[s:s + step], np.float32)
        d = ((chunk[:, None, :] - P[None, :, :]) ** 2).sum(2)
        out[s:s + step] = d.argmin(1)
    return out


def _kmeans_palette(pixels, k, seed=42, iters=24):
    """At most ``k`` representative RGB colours for an (n,3) uint8 sample.

    Deterministic: seeded k-means++ init and Lloyd iterations, so the same
    sheet and count give the same palette every run. A colour-reduction op that
    reshuffles its palette on each click is not one a pixel artist can use.
    """
    px = np.asarray(pixels, np.int32)
    if not len(px):
        return np.zeros((0, 3), np.uint8)
    k = int(min(int(k), len(np.unique(px, axis=0))))
    if k <= 1:
        return np.round(px.mean(0, keepdims=True)).clip(0, 255).astype(np.uint8)
    rng = np.random.default_rng(seed)
    centers = np.empty((k, 3), np.float64)
    centers[0] = px[rng.integers(len(px))]
    d2 = ((px - centers[0]) ** 2).sum(1)
    for j in range(1, k):
        total = d2.sum()
        if total <= 0:
            centers = centers[:j]
            k = j
            break
        centers[j] = px[rng.choice(len(px), p=d2 / total)]
        d2 = np.minimum(d2, ((px - centers[j]) ** 2).sum(1))
    for _ in range(iters):
        labels = _nearest_index(px, centers)
        new = centers.copy()
        for j in range(k):
            m = labels == j
            if m.any():
                new[j] = px[m].mean(0)
        settled = np.allclose(new, centers, atol=0.5)
        centers = new
        if settled:
            break
    return np.unique(np.round(centers).clip(0, 255).astype(np.uint8), axis=0)


def _pixels_of(arrays, cap=120000, seed=42):
    """The opaque RGB pixels of ``arrays``, sampled down to ``cap`` for fitting."""
    chunks = []
    for c in arrays:
        m = c[..., 3] > 0
        if m.any():
            chunks.append(c[..., :3][m])
    if not chunks:
        return np.zeros((0, 3), np.uint8)
    px = np.concatenate(chunks, axis=0)
    if len(px) > cap:
        px = px[np.random.default_rng(seed).choice(len(px), cap, replace=False)]
    return px


def _selection_pixels(doc, sel, cap=120000, seed=42):
    """The opaque RGB pixels of ``sel``, sampled down to ``cap`` for fitting."""
    return _pixels_of([doc.cells[i] for i in sel], cap=cap, seed=seed)


def _limit_palette_fn(palette):
    """A cell -> cell mapper that recolours opaque pixels to ``palette``."""
    def fn(c):
        m = c[..., 3] > 0
        if not m.any():
            return c
        out = c.copy()
        out[..., :3][m] = palette[_nearest_index(c[..., :3][m], palette)]
        return out
    return fn


@op("quantize_colors", "Colour", "Limit palette",
    [_A("colors", "int", 16, min=2, max=256, label="colours"),
     _A("shared", "bool", True, label="one palette for the selection")],
    "Reduce every selected frame to at most N colours (default 16) with k-means. "
    "Only RGB changes: the frame's shape and size and every alpha value are "
    "untouched, and transparent pixels stay transparent. With 'one palette' "
    "ticked the palette is fitted to the whole selection and shared by every "
    "frame, so an animation cannot flicker between near-identical colours.")
def op_quantize_colors(doc, idxs, a):
    sel = sorted({int(i) for i in idxs})
    n = max(2, min(256, int(a["colors"])))
    if a["shared"]:
        pal = _kmeans_palette(_selection_pixels(doc, sel), n)
        if not len(pal):
            doc.op_note = "Limit palette: the selection has no opaque pixels"
            return []
        return _apply(doc, sel, _limit_palette_fn(pal))
    changed = []
    for i in sel:
        pal = _kmeans_palette(_selection_pixels(doc, [i]), n)
        if len(pal):
            changed.extend(_apply(doc, [i], _limit_palette_fn(pal)))
    if not changed:
        doc.op_note = "Limit palette: the selection has no opaque pixels"
    return changed


@op("fill", "Colour", "Fill / tint",
    [_A("color", "color", "#ff0000", label="colour"),
     _A("alpha", "int", 255, min=0, max=255, label="alpha"),
     _A("mode", "choice", "tint", label="mode",
        opts=["tint", "replace", "alpha_only"])],
    "tint recolours opaque pixels and keeps their alpha; replace sets the whole "
    "cell including alpha; alpha_only just sets alpha. The colour comes from the "
    "same picker Erase a colour uses, so a colour is chosen rather than typed as "
    "three channel numbers. The colour is parsed even in alpha_only, where it is "
    "unused: a value the op cannot read is refused rather than ignored, so a typo "
    "never looks like a setting that had no effect.")
def op_fill(doc, idxs, a):
    # Defaults to #ff0000, which is the (255, 0, 0) the three channel boxes used
    # to default to, so the default behaviour is unchanged.
    r, g, b = hex_rgb(a["color"])
    rgb = np.array([r, g, b], np.uint8)

    def fn(c):
        out = c.copy()
        if a["mode"] == "replace":
            out[...] = np.array([r, g, b, a["alpha"]], np.uint8)
        elif a["mode"] == "alpha_only":
            out[..., 3] = a["alpha"]
        else:
            m = c[..., 3] > 0
            out[..., :3] = np.where(m[..., None], rgb, c[..., :3])
        return out
    return _apply(doc, idxs, fn)


# --------------------------------------------------------------------------- #
# group: Frames
# --------------------------------------------------------------------------- #

def _permute(doc, order):
    doc.cells = [doc.cells[i] for i in order]
    doc.orig = [doc.orig[i] for i in order]
    doc.rev += 1
    doc.cell_rev = [doc.rev] * len(order)
    doc.dirty = set(range(len(order)))
    doc.png_cache.clear()
    return list(range(len(order)))


# -- frame scoring, for pick_best ------------------------------------------- #

def _ink(arr):
    """Opaque pixel count. Drops when the matte ate part of the subject."""
    return float((arr[..., 3] > 0).sum())


def _sharpness(arr):
    """Variance of the Laplacian over the subject.

    Not cv2.Laplacian: this is the one place in the editor that would otherwise
    pull in cv2 for a single 5-point stencil, and the absolute value does not
    matter -- only the ranking does. Computed on the alpha-weighted luminance so
    that whatever sits in the transparent region cannot contribute.
    """
    a = arr[..., 3].astype(np.float32)
    if not a.any():
        return 0.0
    lum = (0.299 * arr[..., 0].astype(np.float32)
           + 0.587 * arr[..., 1].astype(np.float32)
           + 0.114 * arr[..., 2].astype(np.float32))
    lum = lum * (a / 255.0)
    lap = (np.roll(lum, 1, 0) + np.roll(lum, -1, 0)
           + np.roll(lum, 1, 1) + np.roll(lum, -1, 1) - 4.0 * lum)
    m = a > 0
    # np.roll wraps, so the outermost ring is differenced against the opposite
    # edge. Excluding it costs nothing and removes the artefact.
    m[0, :] = m[-1, :] = False
    m[:, 0] = m[:, -1] = False
    if not m.any():
        return 0.0
    return float(lap[m].var())


def _unit_scale(v):
    """Scale an axis so its best frame scores 1.0.

    Deliberately NOT min-max to [0,1]. Min-max maps the *worst* frame on the
    axis to exactly 0, so multiplying two min-maxed factors makes every frame
    score 0 the moment the two factors peak on different frames -- the metric
    stops discriminating entirely while still looking like it works. Measured on
    a fixture of 11 crisp frames and 1 smeared one: min-max times min-max scored
    all 12 frames 0.0. Scaling by the max keeps the number readable ("this frame
    is 0.53 of the best on this axis") and cannot collapse.
    """
    hi = max(v)
    if hi <= 1e-9:
        return [0.0] * len(v)
    return [x / hi for x in v]


@op("reverse", "Frames", "Reverse order",
    help="Play the animation backwards.", full=True)
def op_reverse(doc, idxs, a):
    return _permute(doc, list(range(doc.n - 1, -1, -1)))


@op("shift_sequence", "Frames", "Rotate sequence",
    [_A("n", "int", 1, min=-100000, max=100000, label="shift by (frames)")],
    "Roll the frame order -- changes which frame the loop starts on.", full=True)
def op_shift_sequence(doc, idxs, a):
    n = int(a["n"]) % doc.n
    if not n:
        return []
    return _permute(doc, [(i - n) % doc.n for i in range(doc.n)])


@op("keep_range", "Frames", "Keep a frame range",
    [_A("from", "int", 1, min=1, max=1000000, label="from (1-based)"),
     _A("to", "int", 0, min=0, max=1000000, label="to (0 = last)")],
    "Drop everything outside a range -- the way to cut an unusable tail. The "
    "grid is re-derived so it stays full.", full=True)
def op_keep_range(doc, idxs, a):
    lo = max(1, int(a["from"])) - 1
    hi = int(a["to"]) - 1 if a["to"] else doc.n - 1
    hi = min(doc.n - 1, hi)
    if hi < lo:
        raise ValueError("empty range")
    return _set_cells(doc, doc.cells[lo:hi + 1], doc.orig[lo:hi + 1])


@op("drop_frames", "Frames", "Drop selected frames",
    help="Delete the selected cells from the animation and re-derive a full "
         "grid. Select the frames you do not want, then apply. Refuses an empty "
         "selection -- for every other op an empty selection means 'all cells', "
         "and here that would delete the animation.", full=True)
def op_drop_frames(doc, idxs, a):
    drop = set(idxs)
    keep = [i for i in range(doc.n) if i not in drop]
    if len(idxs) >= doc.n:
        raise ValueError(
            "that is all %d frames -- nothing would be left. An empty selection "
            "means 'every cell' for the other operations, so it is refused here "
            "rather than silently deleting the animation. Select the frames to "
            "drop." % doc.n)
    if len(keep) < 2:
        raise ValueError(
            "dropping %d of %d frames would leave %d -- not an animation. "
            "A flipbook needs at least 2 frames; keep one more."
            % (len(drop), doc.n, len(keep)))
    return _set_cells(doc, [doc.cells[i] for i in keep], [doc.orig[i] for i in keep])


@op("insert_hold", "Frames", "Hold a frame",
    [_A("index", "int", 0, min=0, max=1000000, label="cell index (0-based)"),
     _A("n", "int", 2, min=1, max=64, label="hold for N frames")],
    "Duplicate one cell N times. Lengthens a pose without touching the others.",
    full=True)
def op_insert_hold(doc, idxs, a):
    i, n = int(a["index"]), int(a["n"])
    if not (0 <= i < doc.n):
        raise ValueError("cell %d does not exist" % i)
    if n < 2:
        return []
    cells = doc.cells[:i + 1] + [doc.cells[i].copy() for _ in range(n - 1)] + doc.cells[i + 1:]
    orig = doc.orig[:i + 1] + [doc.orig[i].copy() for _ in range(n - 1)] + doc.orig[i + 1:]
    return _set_cells(doc, cells, orig)


@op("dedupe", "Frames", "Drop duplicate frames",
    [_A("threshold", "float", 1.0, min=0.0, max=64.0, step=0.25,
        label="mean |delta| to count as a duplicate"),
     _A("check_loop", "bool", True, label="also compare last vs first")],
    "Remove consecutive frames that are effectively identical, so a held pose "
    "does not burn sheet space.", full=True)
def op_dedupe(doc, idxs, a):
    th = float(a["threshold"])
    keep = [0]
    for i in range(1, doc.n):
        d = np.abs(doc.cells[i].astype(np.int16)
                   - doc.cells[keep[-1]].astype(np.int16)).mean()
        if d > th:
            keep.append(i)
    if a["check_loop"] and len(keep) > 2:
        d = np.abs(doc.cells[keep[-1]].astype(np.int16)
                   - doc.cells[keep[0]].astype(np.int16)).mean()
        if d <= th:
            keep.pop()
    if len(keep) == doc.n:
        return []
    if len(keep) < 2:
        raise ValueError(
            "every frame is within %g of its neighbour, so dedupe would leave "
            "%d frame%s -- not an animation. Raise the threshold tolerance or "
            "turn off the loop check." % (th, len(keep), "" if len(keep) == 1 else "s"))
    return _set_cells(doc, [doc.cells[i] for i in keep], [doc.orig[i] for i in keep])


@op("pick_best", "Frames", "Keep the best N frames",
    [_A("n", "int", 24, min=2, max=1000000, label="keep this many"),
     _A("metric", "choice", "both", opts=["both", "sharpness", "ink"],
        label="score by")],
    "Score every frame and keep the top N, in their original order. "
    "'sharpness' is the variance of the Laplacian over the subject, which falls "
    "on a blurred frame; 'ink' is the opaque pixel count, which falls when the "
    "matte ate part of the subject; 'both' multiplies the two after scaling each "
    "so the best frame on that axis scores 1.0. It scores the whole animation "
    "rather than the selection, because which frames are good is a property of "
    "the sequence -- and it reports the scores it used, so the cut is checkable.",
    full=True)
def op_pick_best(doc, idxs, a):
    keep_n = int(a["n"])
    if keep_n >= doc.n:
        return []
    if keep_n < 2:
        raise ValueError("keep at least 2 frames")
    metric = str(a["metric"])
    if metric not in ("both", "sharpness", "ink"):
        raise ValueError("unknown metric %r" % metric)

    ink = [_ink(c) for c in doc.cells]
    sharp = [_sharpness(c) for c in doc.cells]
    if metric == "ink":
        score = list(ink)
    elif metric == "sharpness":
        score = list(sharp)
    else:
        ni, ns = _unit_scale(ink), _unit_scale(sharp)
        score = [x * y for x, y in zip(ni, ns)]

    # ties break towards the earlier frame, so the result is deterministic
    order = sorted(range(doc.n), key=lambda i: (-score[i], i))[:keep_n]
    keep = sorted(order)
    keepset = set(keep)
    dropped = [i for i in range(doc.n) if i not in keepset]

    _set_cells(doc, [doc.cells[i] for i in keep], [doc.orig[i] for i in keep])

    lo_kept = min(score[i] for i in keep)
    hi_dropped = max(score[i] for i in dropped)
    tied = sum(1 for s in score if abs(s - lo_kept) <= 1e-9)
    doc.op_note = (
        "Keep the best %d frames by %s: dropped %d, lowest kept score %.3f, "
        "highest dropped %.3f%s"
        % (keep_n, metric, len(dropped), lo_kept, hi_dropped,
           "" if hi_dropped < lo_kept else
           " -- %d frames tie on that score, so the cut fell between them and "
           "the earlier ones were kept" % tied))
    return list(range(doc.n))


@op("interpolate", "Frames", "Insert in-between frames",
    [_A("factor", "int", 2, min=2, max=8, label="split each step into N")],
    "Cross-dissolve adjacent frames to raise the frame rate without re-running "
    "the model. The blend happens in premultiplied space, so the alpha ramp "
    "does not darken.", full=True)
def op_interpolate(doc, idxs, a):
    f = int(a["factor"])
    if f < 2:
        return []
    cells, orig = [], []
    for i in range(doc.n):
        cells.append(doc.cells[i])
        orig.append(doc.orig[i])
        if i + 1 < doc.n:
            pa, pb = _premul(doc.cells[i]), _premul(doc.cells[i + 1])
            oa, ob = _premul(doc.orig[i]), _premul(doc.orig[i + 1])
            for k in range(1, f):
                t = k / float(f)
                cells.append(_unpremul(pa * (1 - t) + pb * t))
                orig.append(_unpremul(oa * (1 - t) + ob * t))
    return _set_cells(doc, cells, orig)


@op("reorder", "Frames", "Reorder to a given sequence",
    [_A("order", "text", "", label="comma-separated indices")],
    "Explicit permutation, e.g. 0,3,2,1. Every index must appear once.", full=True)
def op_reorder(doc, idxs, a):
    raw = str(a["order"]).replace(" ", "")
    if not raw:
        raise ValueError("give a comma-separated permutation")
    order = [int(x) for x in raw.split(",") if x != ""]
    if sorted(order) != list(range(doc.n)):
        raise ValueError("the sequence must contain each index 0..%d exactly once"
                         % (doc.n - 1))
    return _set_cells(doc, [doc.cells[i] for i in order], [doc.orig[i] for i in order])


# --------------------------------------------------------------------------- #
# group: Layout
# --------------------------------------------------------------------------- #

@op("set_columns", "Layout", "Set columns",
    [_A("columns", "int", 1, min=1, max=100000, label="columns")],
    "Re-grid the sheet. The column count must divide the frame count.")
def op_set_columns(doc, idxs, a):
    doc.set_layout(columns=int(a["columns"]))
    return []


@op("regrid_auto", "Layout", "Auto column count",
    help="Pick the column count that minimises the largest sheet dimension.")
def op_regrid_auto(doc, idxs, a):
    c, r, w, h = P.auto_columns(doc.n, doc.cell_w, doc.cell_h,
                                doc.meta.get("max_texture", P.MAX_TEXTURE))
    doc.set_layout(columns=c)
    return []


@op("set_cell", "Layout", "Resize cell canvas",
    [_A("cell_w", "int", 512, min=2, max=P.MAX_TEXTURE, label="cell width"),
     _A("cell_h", "int", 640, min=2, max=P.MAX_TEXTURE, label="cell height")],
    "Resample every cell to an exact size. Anchors to the top-left.",
    full=True)
def op_set_cell(doc, idxs, a):
    from PIL import Image
    nw, nh = int(a["cell_w"]), int(a["cell_h"])
    for i in range(doc.n):
        im = Image.fromarray(doc.cells[i], RGBA).resize((nw, nh), Image.LANCZOS)
        doc.cells[i] = np.array(im)
        doc.bump(i)
    doc.set_layout(columns=doc.columns, cell_w=nw, cell_h=nh)
    return list(range(doc.n))


# --------------------------------------------------------------------------- #
# group: Pixel art
# --------------------------------------------------------------------------- #

def detect_pixel_size(pngs, cw, ch, colors=16, palette=None, sample=8):
    """One pixel size for a batch, detected once over a sample of the frames.

    spritefusion-pixel-snapper auto-detects the pixel size *per image*, so
    snapping a selection frame by frame can land consecutive frames on different
    grids: the grid breathes as the content moves and the animation shimmers.
    proper-pixel-art answers the same problem for video by making every global
    decision once and reusing it for all frames; the equivalent here is to
    detect the size on a sample and then pass it to every frame as the override.
    The median over the sample keeps a single odd frame from setting the grid,
    and 0 (let the snapper detect per frame) is returned when that is not
    possible.
    """
    import pixel_snapper
    if not pngs:
        return None
    hi = min(int(cw), int(ch)) // 2
    if hi < 1:
        return None
    k = max(1, min(int(sample), len(pngs)))
    picks = np.unique(np.linspace(0, len(pngs) - 1, k, dtype=int))
    outs = pixel_snapper.snap_pngs([pngs[i] for i in picks], colors, None,
                                   palette)
    factors = []
    for o in outs:
        if o["w"] > 0 and o["h"] > 0:
            factors.append((cw / float(o["w"]) + ch / float(o["h"])) / 2.0)
    if not factors:
        return None
    return int(min(hi, max(1, round(float(np.median(factors))))))


@op("snap_pixels", "Pixel art", "Snap Pixels",
    [_A("colors", "int", 16, min=1, max=256, label="colours"),
     _A("pixel_size", "int", 10, min=0, max=512,
        label="pixel size override (0 = auto)"),
     _A("palette", "text", "", label="palette hex (optional)")],
    "Snap loose, AI-generated pixel art to a perfect grid with "
    "spritefusion-pixel-snapper. Each selected frame is downscaled to the "
    "detected pixel size (or the override) and quantised to a palette, then "
    "scaled back up to the cell with nearest-neighbour -- the frame and the "
    "sheet keep their size, and each grid cell reads as one crisp block. With "
    "auto detection the size is fitted once over a sample of the selection and "
    "shared by every frame, and the frames are then re-quantised to one palette "
    "fitted over the selection, so an animation does not shimmer between grids "
    "or flicker between palettes.")
def op_snap_pixels(doc, idxs, a):
    import pixel_snapper
    if not pixel_snapper.available():
        raise ValueError(
            "spritefusion-pixel-snapper is not available (needs Node.js and the "
            "package's pkg/ folder; set SPRITE_SNAPPER_DIR to its checkout)")
    sel = sorted({int(i) for i in idxs})
    colors = int(a["colors"]) or 16
    px = int(a["pixel_size"]) or None
    palette = (a["palette"] or "").strip() or None
    pngs = [doc.cell_png(i) for i in sel]
    # One grid for the whole selection: auto detects once over a sample and reuses
    # it for every frame, instead of each frame choosing its own grid.
    shared = False
    if px is None:
        px = detect_pixel_size(pngs, doc.cell_w, doc.cell_h, colors, palette)
        shared = px is not None
    outs = pixel_snapper.snap_pngs(pngs, colors, px, palette)
    # The snapper quantizes each frame to its own palette, so the same source
    # colour can drift between frames. Re-quantize the snapped frames to a single
    # palette fitted over the selection so colours cannot flicker. A user-supplied
    # palette already constrains every frame, so it is left alone.
    cw, ch = doc.cell_w, doc.cell_h
    smalls = [_unpng(o["png"]) for o in outs]
    shared_pal = False
    if palette is None and len(sel) > 1:
        pal = _kmeans_palette(_pixels_of(smalls), colors)
        if len(pal):
            smalls = [_limit_palette_fn(pal)(a) for a in smalls]
            shared_pal = True
    grid = None
    for i, small, o in zip(sel, smalls, outs):
        doc.cells[i] = fit_cell(small, cw, ch)
        doc.bump(i)
        if grid is None:
            grid = (o["w"], o["h"])
    note = "Snap Pixels: %d frame%s snapped, cell kept at %dx%d" % (
        len(sel), "" if len(sel) == 1 else "s", cw, ch)
    bits = []
    if shared:
        bits.append("one %dpx grid" % px)
    if shared_pal:
        bits.append("one palette")
    if bits:
        note += ", " + " + ".join(bits) + " shared"
    if grid:
        note += " (grid %dx%d)" % grid
    doc.op_note = note
    return list(sel)


@op("pixelate_mesh", "Pixel art", "Pixelate (mesh)",
    [_A("colors", "int", 0, min=0, max=256,
        label="colours (0 = keep all)"),
     _A("pixel_width", "int", 0, min=0, max=512,
        label="pixel width override (0 = auto)"),
     _A("sample", "int", 8, min=1, max=512,
        label="frames sampled for grid + palette"),
     _A("upscale", "int", 8, min=1, max=8, label="mesh-detection upscale"),
     _A("transparent", "bool", False, label="clear the background colour"),
     _A("palette", "text", "", label="palette hex (optional)")],
    "True-resolution pixel art from noisy AI/web images, with the "
    "proper-pixel-art algorithm: Canny edges -> Hough lines -> a homogenised "
    "mesh, then one representative colour per mesh cell. Fitted to a selection, "
    "the mesh and the palette are detected once over a sample of the frames and "
    "shared by all of them, so an animation resolves to one grid and cannot "
    "jitter frame to frame. The cell size is kept: the true-resolution result is "
    "scaled back up to it with nearest-neighbour. A palette from the picker "
    "replaces the fitted one, the same way it does on Snap Pixels.")
def op_pixelate_mesh(doc, idxs, a):
    import proper_pixel
    if not proper_pixel.available():
        raise ValueError(
            "proper-pixel-art is not available (set SPRITE_PPA_DIR to the "
            "checkout that holds its proper_pixel_art/ folder)")
    sel = sorted({int(i) for i in idxs})
    if not sel:
        return []
    # A named palette IS the colour decision, so the auto quantiser is switched
    # off and the mesh's own representative colours are mapped onto the named
    # palette instead. Leaving `colors` on would quantise twice -- once to a
    # palette fitted from the sample, then again onto the user's -- and the first
    # pass can merge away colours the user asked for, so the result could come
    # back with fewer of their colours than they named.
    palette = (a["palette"] or "").strip()
    pal = hex_palette(palette) if palette else None
    outs, grid = proper_pixel.pixelate_frames(
        [doc.cells[i] for i in sel],
        colors=0 if pal is not None else max(0, min(256, int(a["colors"]))),
        pixel_width=max(0, int(a["pixel_width"])),
        sample=max(1, int(a["sample"])),
        upscale=max(1, int(a["upscale"])),
        transparent=bool(a["transparent"]))
    if pal is not None:
        outs = [_limit_palette_fn(pal)(o) for o in outs]
    changed = []
    for i, out in zip(sel, outs):
        if out.shape == doc.cells[i].shape and not np.array_equal(
                out, doc.cells[i]):
            doc.cells[i] = out
            doc.bump(i)
            changed.append(i)
    note = ("Pixelate (mesh): %d frame%s -> %dx%d true-resolution grid, cell kept "
            "at %dx%d"
            % (len(sel), "" if len(sel) == 1 else "s", grid[0], grid[1],
               doc.cell_w, doc.cell_h))
    if pal is not None:
        note += ", %d-colour palette from the picker" % len(pal)
    doc.op_note = note
    return changed


@op("pixelize_oe", "Pixel art", "Pixelize (outline)",
    [_A("pixel_size", "int", 2, min=1, max=64,
        label="pixel size (blocks per pixel)"),
     _A("thickness", "int", 0, min=0, max=16,
        label="outline expansion (0 = off)"),
     _A("mode", "choice", "k_centroid", label="downsampler",
        opts=["contrast", "k_centroid", "lanczos", "nearest", "bicubic"]),
     _A("colors", "int", 0, min=0, max=256,
        label="colours (0 = keep all)"),
     _A("dither", "choice", "none", label="dither (when quantising)",
        opts=["none", "ordered", "error_diffusion"]),
     _A("sharpen", "choice", "none", label="sharpen",
        opts=["none", "unsharp", "laplacian"]),
     _A("sharpen_factor", "float", 0, min=0.0, max=4.0,
        label="sharpen amount"),
     _A("color_match", "bool", True, label="colour match after expansion"),
     _A("palette", "text", "", label="palette hex (optional)")],
    "One representative colour per pixel_size block, chosen for contrast "
    "rather than averaged: the LAB luminance is outlined first (contrast-aware "
    "outline expansion -- lines thicken where contrast is high, which is what "
    "keeps a one-pixel edge from being averaged away before the grid exists), "
    "then each block takes the pixel nearest its centre, its median, mean, min "
    "or max, and the chroma by median. The pixel size defaults to 2, the "
    "outline expansion to 0 (off) and the sharpen to none, so out of the box "
    "this is a plain block downsample; the outline and sharpen stages are there "
    "when a softened edge needs rescuing. `thickness` of 0 turns the outline "
    "stage off. `k_centroid` and `lanczos` are the package's other "
    "downsamplers; the "
    "rest are plain interpolation modes, which is what `nearest` gives. "
    "`colours` above 1 quantises with k-means and then dithers; a named "
    "`colours` count is a target, not a guarantee, because the colour match "
    "afterwards can reintroduce a shade. A palette from the picker REPLACES "
    "quantising: `colours` is ignored and the pixelized result is mapped onto "
    "the named palette instead, the same way it works on Snap Pixels and "
    "Pixelate (mesh). Unlike Snap Pixels and Pixelate "
    "(mesh) this does not look for a grid the image already has -- it imposes "
    "one. PixelOE is RGB-only, so alpha is carried through untouched and the "
    "colour under a transparent pixel is invented; and the pixel size must "
    "divide the cell, because PixelOE replicate-pads a cell that does not "
    "divide and that padding changes the result. The cell size is kept exactly.")
def op_pixelize_oe(doc, idxs, a):
    import pixeloe_bridge
    if not pixeloe_bridge.available():
        raise ValueError(
            "PixelOE is not available (set PIXELOE_DIR to the checkout that "
            "holds its src/pixeloe/ folder)")
    sel = sorted({int(i) for i in idxs})
    if not sel:
        return []
    ps = int(a["pixel_size"])
    cw, ch = doc.cell_w, doc.cell_h
    # Coercion does not clamp, so an out-of-range size can arrive from a stale
    # page. A size that does not divide the cell is refused by the bridge with
    # the list of sizes that do, which is the reason the user actually needs.
    if ps < 1:
        raise ValueError("pixel size is 1 or more, not %d" % ps)
    # A named palette IS the colour decision, so the k-means quantiser is
    # switched off and the pixelized result is mapped onto the named palette
    # instead -- the same rule and the same reason as Pixelate (mesh). Leaving
    # the quantiser on would quantise twice: once to a palette PixelOE fitted
    # from the image, then again onto the user's, and the first pass can merge
    # away colours the user asked for. A blank palette means "no palette", which
    # is the default, and `hex_palette` refuses an entry it cannot read rather
    # than dropping it.
    palette = (a["palette"] or "").strip()
    pal = hex_palette(palette) if palette else None
    want = max(0, min(256, int(a["colors"])))
    outs = pixeloe_bridge.pixelize_cells(
        [doc.cells[i] for i in sel],
        pixel_size=ps,
        thickness=int(a["thickness"]),
        mode=a["mode"],
        colors=0 if pal is not None else want,
        dither=a["dither"],
        # `none` is a named off-state, not a truthiness test: the control has to
        # offer a labelled option, because an empty-string `choice` renders as a
        # blank row in the dropdown that reads as a bug. A blank or missing
        # value is also treated as off, so a stale page cannot turn sharpening
        # on by accident.
        sharpen=(None if a["sharpen"] in ("", "none") else a["sharpen"]),
        sharpen_factor=float(a["sharpen_factor"]),
        color_match=bool(a["color_match"]))
    if pal is not None:
        # Mapped here rather than inside the bridge: the bridge is a thin
        # wrapper over PixelOE and knows nothing about the editor's palette
        # control; `_limit_palette_fn` is the same helper the other two pixel
        # ops use, so a named palette means one thing everywhere.
        outs = [_limit_palette_fn(pal)(o) for o in outs]
    changed = []
    for i, out in zip(sel, outs):
        if out.shape == doc.cells[i].shape and not np.array_equal(
                out, doc.cells[i]):
            doc.cells[i] = out
            doc.bump(i)
            changed.append(i)
    note = ("Pixelize (outline): %d frame%s -> %dpx blocks, %s downscale, "
            "cell kept at %dx%d"
            % (len(sel), "" if len(sel) == 1 else "s", ps, a["mode"], cw, ch))
    if int(a["thickness"]) <= 0:
        note += ", outline off"
    if pal is not None:
        note += ", %d-colour palette from the picker" % len(pal)
    elif want > 1:
        note += ", %d-colour target + %s dither" % (want, a["dither"])
    doc.op_note = note
    return changed


def _resize_nn(cell, k, up):
    """Nearest-neighbour by a whole factor: replicate, or take every k-th pixel.

    Not `Image.resize(..., NEAREST)`. That is the same arithmetic on paper, but
    it derives the output size from a ratio and samples from a scaled coordinate,
    so a down-then-up is not the identity even when the factor divides the cell
    exactly. `np.repeat` and a stride are exact inverses of each other, and that
    round trip is the property pixel art needs -- it is what the tests pin, and
    it is why a cell can be doubled and halved again without drifting.
    """
    if up:
        return np.repeat(np.repeat(cell, k, axis=0), k, axis=1)
    # Every k-th pixel, taken from the block's first. Whole blocks only: the
    # caller has already refused a cell the factor does not divide, because
    # striding past a partial block would silently drop a row of art.
    return cell[::k, ::k].copy()


@op("resize_pixels", "Pixel art", "Resize (pixels)",
    [_A("mode", "choice", "up", label="direction", opts=["up", "down"]),
     _A("factor", "int", 2, min=2, max=16, label="integer factor")],
    "Scale the art by a whole number with nearest-neighbour, so nothing is "
    "blended and no colour is invented. `up` replaces every pixel with a "
    "factor x factor block -- exact, and `down` by the same factor gives the "
    "original bytes back, so a cell can be doubled and halved again without "
    "drifting. `down` takes each block's first pixel, so it needs a cell size "
    "the factor divides and refuses rather than dropping a row of art. This "
    "changes the cell size, so every frame is resized and the selection is not "
    "consulted; the sheet and the sidecar are written by Save. For a size that "
    "is not a whole multiple, use Resample all cells or Resize cells, which "
    "filter instead.",
    full=True)
def op_resize_pixels(doc, idxs, a):
    k = int(a["factor"])
    # `min=2` already keeps the form from offering 1, and coercion does not
    # clamp, so the op checks it too: factor 1 would report every cell changed
    # and leave the document identical.
    if k < 2:
        raise ValueError("an integer factor is 2 or more, not %d" % k)
    up = a["mode"] == "up"
    cw, ch = doc.cell_w, doc.cell_h
    if up:
        nw, nh = cw * k, ch * k
    else:
        if cw % k or ch % k:
            raise ValueError(
                "cell is %dx%d, which %d does not divide -- nearest-neighbour "
                "down would drop the remainder, so make the cell a multiple of "
                "%d first (Grid: cell size, or Resize cells)"
                % (cw, ch, k, k))
        nw, nh = cw // k, ch // k
    for i in range(doc.n):
        doc.cells[i] = _resize_nn(doc.cells[i], k, up)
        doc.bump(i)
    # set_layout validates the grid and the 32768 texture cap, so a factor that
    # would make the sheet too big refuses here with its own message rather than
    # writing a sheet no engine can load.
    doc.set_layout(columns=doc.columns, cell_w=nw, cell_h=nh)
    doc.op_note = ("Resize (pixels): %d frame%s %s x%d, cell %dx%d -> %dx%d, "
                   "nearest-neighbour"
                   % (doc.n, "" if doc.n == 1 else "s", a["mode"], k,
                      cw, ch, nw, nh))
    return list(range(doc.n))


# --------------------------------------------------------------------------- #
# group: Paint
# --------------------------------------------------------------------------- #

@op("paint", "Paint", "Composite a patch",
    [_A("cell", "int", 0, min=0, max=1000000, label="cell index"),
     _A("x", "int", 0, min=-P.MAX_TEXTURE, max=P.MAX_TEXTURE, label="x"),
     _A("y", "int", 0, min=-P.MAX_TEXTURE, max=P.MAX_TEXTURE, label="y"),
     _A("w", "int", 1, min=1, max=P.MAX_TEXTURE, label="w"),
     _A("h", "int", 1, min=1, max=P.MAX_TEXTURE, label="h"),
     _A("mode", "choice", "blend", label="mode", opts=["blend", "set", "erase"]),
     _A("data", "text", "", label="base64 RGBA")],
    "Used by the pencil and eraser: the browser rasterises a stroke and posts "
    "the affected rectangle. blend is source-over, set overwrites including "
    "alpha, erase multiplies alpha down.")
def op_paint(doc, idxs, a):
    i = int(a["cell"])
    if not (0 <= i < doc.n):
        raise ValueError("cell %d does not exist" % i)
    raw = base64.b64decode(a["data"] or "")
    w, h = int(a["w"]), int(a["h"])
    if a["mode"] == "png":
        # A whole cell sent as its PNG. Used by copy/paste: PNG carries straight
        # (non-premultiplied) RGBA, so a pasted frame is bit-exact, whereas
        # reading the pixels back through a canvas rounds RGB under low alpha.
        from PIL import Image                       # imported lazily, see _png
        im = Image.open(io.BytesIO(raw)).convert(RGBA)
        if im.size != (w, h):
            raise ValueError("png patch is %dx%d, expected %dx%d"
                             % (im.size[0], im.size[1], w, h))
        patch = np.array(im)
    else:
        if len(raw) != w * h * 4:
            raise ValueError("patch is %d bytes, expected %d for %dx%d RGBA"
                             % (len(raw), w * h * 4, w, h))
        patch = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
    x, y = int(a["x"]), int(a["y"])
    c = doc.cells[i]
    H, W = c.shape[:2]
    px0, py0 = max(0, -x), max(0, -y)
    px1, py1 = min(w, W - x), min(h, H - y)
    if px0 >= px1 or py0 >= py1:
        return []
    sx0, sy0 = x + px0, y + py0
    sub = patch[py0:py1, px0:px1]
    if a["mode"] in ("set", "png"):
        merged = sub.copy()
    else:
        dst = c[sy0:sy0 + (py1 - py0), sx0:sx0 + (px1 - px0)]
        sa = sub[..., 3:4].astype(np.float32) / 255.0
        if a["mode"] == "erase":
            merged = dst.copy()
            merged[..., 3] = np.clip(
                dst[..., 3].astype(np.float32) * (1.0 - sa[..., 0]), 0, 255
            ).astype(np.uint8)
        else:
            src_pm = sub[..., :3].astype(np.float32) * sa
            dst_a = dst[..., 3:4].astype(np.float32) / 255.0
            out_a = sa + dst_a * (1.0 - sa)
            out_rgb = np.divide(
                src_pm + dst[..., :3].astype(np.float32) * dst_a * (1.0 - sa),
                np.maximum(out_a, 1e-6))
            merged = np.concatenate(
                [np.clip(out_rgb, 0, 255), np.clip(out_a * 255.0, 0, 255)],
                axis=2).astype(np.uint8)
    out = c.copy()
    out[sy0:sy0 + merged.shape[0], sx0:sx0 + merged.shape[1]] = merged
    if np.array_equal(out, c):
        return []
    doc.cells[i] = out
    doc.bump(i)
    return [i]


# --------------------------------------------------------------------------- #
# group: Meta
# --------------------------------------------------------------------------- #

@op("set_meta", "Meta", "Playback metadata",
    [_A("fps", "int", 24, min=1, max=240, label="fps"),
     _A("play_mode", "choice", "loop", label="play mode",
        opts=["loop", "pingpong", "once"]),
     _A("trim_to", "int", 0, min=0, max=1000000, label="trim to frame (0 = last)"),
     _A("anchor", "choice", "none", label="anchor",
        opts=["none", "bottom-center"]),
     _A("blend", "choice", "straight", label="blend",
        opts=["straight", "additive"]),
     _A("name", "text", "", label="name")],
    "Written straight into the sidecar JSON.")
def op_set_meta(doc, idxs, a):
    for k in ("fps", "play_mode", "trim_to", "anchor", "blend"):
        doc.meta[k] = a[k]
    if a["name"]:
        doc.meta["name"] = a["name"]
    doc.rev += 1
    return []


@op("set_verify", "Meta", "Verification thresholds",
    [_A("black_max", "int", 4, min=0, max=64, label="near-black threshold"),
     _A("leak_alpha", "int", 200, min=1, max=255, label="opaque threshold"),
     _A("bright", "int", 64, min=1, max=255, label="bright threshold"),
     _A("erode_max", "int", 24, min=1, max=4096, label="max hole (px)")])
def op_set_verify(doc, idxs, a):
    doc.meta.update({k: a[k] for k in ("black_max", "leak_alpha", "bright", "erode_max")})
    doc.rev += 1
    return []


# --------------------------------------------------------------------------- #
# op plumbing
# --------------------------------------------------------------------------- #

def _apply(doc, idxs, fn):
    """Run a pure cell operation over the selection, reporting real changes."""
    changed = []
    for i in idxs:
        before = doc.cells[i]
        after = fn(before)
        if after is None or after.shape != before.shape:
            continue
        if not np.array_equal(after, before):
            doc.cells[i] = after
            doc.bump(i)
            changed.append(i)
    return changed


def _set_cells(doc, cells, orig):
    """Replace the whole frame list, re-deriving a valid grid.

    Every index is marked dirty: even when the pixels are the same objects, the
    index->cell mapping changed, so per-index the cell did change.

    Atomic. The frame list is only committed if a valid grid is found for it.
    This used to assign doc.cells first and call set_layout afterwards, so a
    re-grid that was *refused* left the document holding 122 cells against a
    31x4 layout -- n != cols*rows, verify reporting "122 cells (31x4)", and
    save writing a sheet whose sidecar claimed more frames than it had. Every
    frame-pruning op routes through here, so the corruption was one refused
    shrink away from being saved to disk.
    """
    old_cells, old_orig = doc.cells, doc.orig
    old_cell_rev, old_dirty = list(doc.cell_rev), set(doc.dirty)
    old_cols = doc.columns
    cw, ch = doc.cell_w, doc.cell_h
    cap = int(doc.meta.get("max_texture") or P.MAX_TEXTURE)

    def keep_ok(c):
        """Is `c` worth keeping, i.e. both it and the sheet it makes are legal?

        Keeping the document's own column count is a convenience, not a right.
        Testing divisibility alone re-gridded a subset straight back onto a
        count whose sheet is over the cap, and the refusal that followed claimed
        no column count fits while a smaller one plainly did. A document can
        hold an over-cap grid: open a sheet generated under a smaller cap, or
        re-grid up, and its own layout is the illegal one.
        """
        return bool(c) and len(cells) % c == 0 and c * cw <= cap \
            and (len(cells) // c) * ch <= cap

    cols = old_cols if keep_ok(old_cols) else 0
    doc.cells = list(cells)
    doc.orig = list(orig)
    doc.rev += 1
    doc.cell_rev = [doc.rev] * len(cells)
    doc.dirty = set(range(len(cells)))
    doc.png_cache.clear()
    try:
        if cols:
            doc.set_layout(columns=cols)
        else:
            c, r, w, h = P.auto_columns(len(cells), doc.cell_w, doc.cell_h,
                                        doc.meta.get("max_texture", P.MAX_TEXTURE))
            doc.set_layout(columns=c)
    except ValueError as ex:
        # set_layout validates before it touches self.layout, so only the frame
        # list has to go back. doc.rev is deliberately left advanced: the
        # revision is monotonic by design, and rewinding it could let a restored
        # cell share a revision with a client's stale copy.
        doc.cells, doc.orig = old_cells, old_orig
        doc.cell_rev, doc.dirty = old_cell_rev, old_dirty
        doc.png_cache.clear()
        raise ValueError(
            "%s -- %d frames have no column count that fits, so nothing was "
            "changed. Drop or add frames until the count has a workable grid "
            "(a prime count only ever fits 1 x n), or shrink the cell."
            % (ex, len(cells)))
    return list(range(len(cells)))


def coerce_args(spec, given):
    out = {}
    for s in spec["args"]:
        v = (given or {}).get(s["k"], s["d"])
        t = s["t"]
        try:
            if t == "int":
                out[s["k"]] = int(float(v))
            elif t == "float":
                out[s["k"]] = float(v)
            elif t == "bool":
                out[s["k"]] = v if isinstance(v, bool) else str(v).strip().lower() in (
                    "1", "true", "yes", "on")
            else:
                out[s["k"]] = "" if v is None else str(v)
        except (TypeError, ValueError):
            out[s["k"]] = s["d"]
    return out


def run_op(doc, name, sel, args):
    """Apply one operation. Returns (changed_indices, message, undo_depth)."""
    spec = OPS.get(name)
    if spec is None:
        raise ValueError("unknown operation '%s'" % name)
    n = doc.n
    if name in ("paint",):
        idxs = [int((args or {}).get("cell", 0))]
    else:
        idxs = sorted({int(i) for i in (sel or []) if 0 <= int(i) < n}) or list(range(n))
    a = coerce_args(spec, args)

    full = spec["full"] or name in ("rotate", "set_cell", "resample", "pad_cell",
                                    "crop_tight")
    snap_idx = list(range(n)) if full else idxs
    pre = {i: doc.cells[i].copy() for i in snap_idx}
    # The frame ops rewrite `orig` as well as `cells`, so a full entry has to
    # carry both -- restoring cells without their reference leaves len(orig) < n
    # and the next op that indexes orig dies on the tail frames.
    pre_orig = ({i: doc.orig[i].copy() for i in snap_idx if i < len(doc.orig)}
                if full else None)
    layout_before = dict(doc.layout)
    meta_before = dict(doc.meta)

    changed = spec["fn"](doc, idxs, a) or []
    # An op may leave a human-readable note about what it decided -- pick_best
    # reports the scores that drove the cut, because "kept the best 24" with no
    # numbers is a claim the user cannot check. Consume it either way.
    note = getattr(doc, "op_note", None)
    doc.op_note = None

    if not changed and doc.layout == layout_before and doc.meta == meta_before:
        return [], note or "no change", len(doc.undo)

    # `store` must span the PRE-op frame count, not the post-op one. The op is
    # free to change doc.n -- keep_range, dedupe and drop_frames all shrink it --
    # and the undo entry has to hold every pre-op cell because _restore rebuilds
    # `range(entry["n"])`. Deriving store from doc.n *after* the op drops the
    # tail cells and undo then raises KeyError on the first one past the new
    # count. Growing ops (interpolate, insert_hold) only escaped this because
    # `if i in pre` happened to filter the surplus back out again.
    store = sorted(set(changed)) if not full else list(range(n))
    entry = {"cells": {i: _png(pre[i]) for i in store if i in pre},
             "orig": ({i: _png(pre_orig[i]) for i in store if i in (pre_orig or {})}
                      if full else None),
             "layout": layout_before, "meta": meta_before,
             "full": full, "n": n}
    entry["bytes"] = (sum(len(b) for b in entry["cells"].values())
                      + sum(len(b) for b in (entry["orig"] or {}).values()) + 256)
    doc.push_undo(entry)
    doc.journal.append(name)
    return store, (note or _describe(spec, a, store, doc, n)), len(doc.undo)


def _describe(spec, a, changed, doc, n_before=None):
    if not changed:
        return "%s: metadata updated" % spec["label"]
    # A frame-count change is the most important thing that happened, and for a
    # full op `changed` is the pre-op index range -- so without this branch a
    # keep_range that took 124 frames down to 10 reported "124 cells", which
    # reads as though nothing was removed.
    if n_before is not None and n_before != doc.n:
        return "%s: %d -> %d frames" % (spec["label"], n_before, doc.n)
    if len(changed) == doc.n:
        return "%s: all %d cells" % (spec["label"], doc.n)
    return "%s: %d cell%s" % (spec["label"], len(changed),
                              "" if len(changed) == 1 else "s")


# --------------------------------------------------------------------------- #
# verification -- same invariants as the pipeline, adapted to an edited doc
# --------------------------------------------------------------------------- #

def verify_doc(doc):
    """Returns (ok, lines). Honest about what an edit destroys."""
    out, fails = [], []
    L = doc.layout
    n, cols, rows = doc.n, doc.columns, doc.rows
    cw, ch = doc.cell_w, doc.cell_h
    cap = int(doc.meta.get("max_texture") or P.MAX_TEXTURE)
    bm = int(doc.meta.get("black_max", 4))
    la = int(doc.meta.get("leak_alpha", 200))
    br = int(doc.meta.get("bright", 64))
    emax = int(doc.meta.get("erode_max", 24))

    if n != cols * rows:
        fails.append("NOTFULL  frame_count %d != %d x %d" % (n, cols, rows))
    if cols * cw > cap or rows * ch > cap:
        fails.append("TEXTURE  %dx%d over the %d cap" % (cols * cw, rows * ch, cap))

    i1 = i2 = i4 = 0
    leak_px = lost_px = subj_px = 0
    largest = 0
    faint = 0
    modified = 0
    ref_free = 0
    for i in range(n):
        cell = doc.cells[i].astype(np.int16)
        alpha = cell[..., 3]
        if not (alpha > 0).any():
            fails.append("EMPTY  cell %d fully transparent" % i)
            continue
        # I2 -- self-contained, no reference needed
        reach_dark = P.border_reachable(cell[..., :3].max(axis=2) < bm)
        leak = reach_dark & (alpha >= la)
        leak_px += int(leak.sum())
        if leak.any():
            i2 += 1
            if i2 <= 5:
                ys, xs = np.where(leak)
                fails.append("BACKDROP  cell %d: %d px opaque reachable-black "
                             "(e.g. (%d,%d))" % (i, int(leak.sum()),
                                                 int(ys[0]), int(xs[0])))
        faint += int((reach_dark & (alpha > 0) & (alpha < la)).sum())
        # Reference-free premultiply smell: under straight alpha, max(RGB) is
        # routinely above alpha; under premultiplied alpha it never is.
        #
        # Restricted to alpha > 0. A fully transparent pixel carries no colour
        # information at all, and a real matte routinely leaves stray RGB behind
        # in those pixels -- this project's own torch sheet has RGB=2 at alpha=0.
        # Counting them made the test demand perfectly zeroed transparency, which
        # almost no real sheet has, so the line effectively never fired.
        above = int(((cell[..., :3].max(axis=2) > alpha) & (alpha > 0)).sum())
        if above == 0:
            ref_free += 1
        if i in doc.dirty:
            modified += 1
            continue
        # I1 / I4 need the pre-edit reference, so they only cover unmodified cells
        src = doc.orig[i].astype(np.int16)
        m = alpha > 0
        if (m & (np.abs(cell[..., :3] - src[..., :3]).max(axis=2) > 0)).any():
            i1 += 1
            fails.append("PREMULT  cell %d: RGB differs from the loaded cell "
                         "where alpha>0" % i)
        bright = src[..., :3].max(axis=2) > br
        if bright.any():
            lost = bright & (alpha == 0)
            nl = int(lost.sum())
            subj_px += int(bright.sum())
            if nl:
                import cv2
                nlab, _, stats, _ = cv2.connectedComponentsWithStats(
                    lost.astype(np.uint8), 8)
                big = int(stats[1:, 4].max()) if nlab > 1 else 0
                largest = max(largest, big)
                lost_px += nl
                if big >= emax:
                    i4 += 1
                    if i4 <= 5:
                        fails.append("ERODED  cell %d: %d bright px lost alpha "
                                     "(largest hole %d px)" % (i, nl, big))

    out.append("grid      : %d cells (%dx%d of %dx%d) sheet %dx%d"
               % (n, cols, rows, cw, ch, cols * cw, rows * ch))
    out.append("I2 no backdrop leak    failed cells: %d  (%d px opaque reachable-black)"
               % (i2, leak_px))
    if modified:
        out.append("I1 straight alpha      failed cells: %d  (%d cell%s modified "
                   "since load -- no reference, not applicable)"
                   % (i1, modified, "" if modified == 1 else "s"))
        out.append("I4 subject intact      failed cells: %d  (%d of %d bright px "
                   "lost, largest hole %d px; unmodified cells only)"
                   % (i4, lost_px, subj_px, largest))
    else:
        out.append("I1 straight alpha      failed cells: %d" % i1)
        out.append("I4 subject intact      failed cells: %d  (%d of %d bright px "
                   "lost, largest hole %d px; tolerance %d px)"
                   % (i4, lost_px, subj_px, largest, emax))
    if ref_free:
        out.append("premultiply smell     %d cell%s look premultiplied "
                   "(max(RGB) never exceeds alpha). Informational: run "
                   "Un-premultiply if that is not intentional."
                   % (ref_free, "" if ref_free == 1 else "s"))
    out.append("faint fringe (informational, alpha<%d): %d px" % (la, faint))
    if fails:
        out.append("")
        out.extend(fails[:25])
        if len(fails) > 25:
            out.append("... and %d more" % (len(fails) - 25))
    return (len(fails) == 0), out


# --------------------------------------------------------------------------- #
# loading and saving
# --------------------------------------------------------------------------- #

def _find_sidecar(sheet_path):
    """Locate the sidecar for a sheet PNG.

    A sheet is not always named `<stem>_sheet.png` beside `<stem>.json`: the
    walk/ and sprites/ trees use `<name>.json` + `<name>_sheet.png`, and a
    generated run uses `<prefix>.json` + `<prefix>_sheet.png`. Guessing one
    naming convention silently fails on the other, so check the obvious names
    and then ask the directory: any JSON whose `sheet` field names this PNG.
    """
    base = os.path.basename(sheet_path)
    stem, _ = os.path.splitext(sheet_path)
    cands = [stem + ".json"]
    if stem.endswith("_sheet"):
        cands.append(stem[:-len("_sheet")] + ".json")
    for c in cands:
        if os.path.isfile(c):
            return c
    d = os.path.dirname(sheet_path) or "."
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return None
    for n in names:
        if not n.lower().endswith(".json"):
            continue
        p = os.path.join(d, n)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                sc = json.load(fh)
        except Exception:                            # noqa: BLE001
            continue
        if not isinstance(sc, dict):
            continue
        if os.path.basename(str(sc.get("sheet") or "")) == base:
            return p
    return None


def _sheet_for_sidecar(sidecar_path, sc):
    """The PNG a sidecar names, for opening a document *by its JSON*.

    The JSON is the document: it names the sheet, the grid and the clips. So a
    path pointing at it has to open, and the audio has to come with it -- which
    it does, because this resolves the PNG and leaves the sidecar in hand.
    """
    d = os.path.dirname(sidecar_path) or "."
    stem = os.path.splitext(sidecar_path)[0]
    cands = []
    named = str((sc or {}).get("sheet") or "")
    if named:
        cands.append(named if os.path.isabs(named) else os.path.join(d, named))
    cands += [stem + "_sheet.png", stem + ".png"]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "sidecar %s names no sheet that is there (tried %s)"
        % (os.path.basename(sidecar_path), ", ".join(cands)))


def _sidecar_in_folder(folder):
    """The sheet sidecar in a folder, for opening a document by the folder path.

    A run's output folder holds one sheet, and the file that *is* the document in
    there is the JSON -- so pointing at the folder should find it rather than
    asking the user to name it. Prefer a JSON named after the folder (which is
    what a run writes), then any JSON in there that is a sheet sidecar naming a
    PNG that exists. Not recursive: a folder of folders is a list to choose from,
    not something to guess through.
    """
    folder = os.path.normpath(folder)
    named = os.path.join(folder, os.path.basename(folder) + ".json")
    cands = [named] if os.path.isfile(named) else []
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        names = []
    cands += [os.path.join(folder, n) for n in names
              if n.lower().endswith(".json") and os.path.join(folder, n) != named]
    for c in cands:
        try:
            with open(c, "r", encoding="utf-8") as fh:
                sc = json.load(fh)
        except Exception:                        # noqa: BLE001
            continue
        if not isinstance(sc, dict):
            continue
        if not all(k in sc for k in ("columns", "rows",
                                     "frame_width", "frame_height")):
            continue
        try:
            _sheet_for_sidecar(c, sc)
        except FileNotFoundError:
            continue
        return c
    return None


def load_doc(doc_id, sheet_path, sidecar_path=None, columns=None, rows=None):
    from PIL import Image
    if not os.path.exists(sheet_path):
        raise FileNotFoundError("no such sheet: %s" % sheet_path)
    sc = None
    # A folder. A run writes one sheet into a folder of its own, so the folder is
    # the natural thing to paste -- and what it resolves to is the sidecar, which
    # is what carries the grid, the name and the clips.
    if os.path.isdir(sheet_path):
        found = _sidecar_in_folder(sheet_path)
        if not found:
            raise FileNotFoundError(
                "no sheet sidecar in %s -- give the .json, the sheet PNG, or "
                "columns and rows" % sheet_path)
        sheet_path = found
    # Opening by the sidecar, which is what the sheet library hands over: the
    # PNG it names is found from the JSON, not guessed from a path. This used to
    # reach PIL with the .json and die with "cannot identify image file".
    if sheet_path.lower().endswith(".json"):
        with open(sheet_path, "r", encoding="utf-8") as fh:
            sc = json.load(fh)
        if not isinstance(sc, dict):
            raise ValueError("%s is not a sheet sidecar"
                             % os.path.basename(sheet_path))
        sidecar_path = sheet_path
        sheet_path = _sheet_for_sidecar(sheet_path, sc)
    elif sidecar_path and os.path.isfile(sidecar_path):
        with open(sidecar_path, "r", encoding="utf-8") as fh:
            sc = json.load(fh)
    # NOTE: the browser sends columns: 0 for "not given", so this must test
    # falsiness, not `is None`. Testing `is None` made every open-from-the-UI
    # skip sidecar discovery and fail with "no sidecar JSON found".
    #
    # The search runs even when columns and rows WERE given. It used to be gated
    # on `not (columns and rows)`, which meant an explicit grid suppressed sidecar
    # discovery completely: opening a sheet PNG with the columns/rows boxes filled
    # in opened it with no clips at all, and the sidecar's own grid was ignored
    # even when it disagreed with the boxes. Measured on a real run sheet -- the
    # PNG alone returns its 2 clips, the PNG with its own 11x3 grid supplied
    # returns 0. An explicit grid is a fallback for a PNG that has no sidecar, not
    # a request to ignore the one beside it.
    else:
        sidecar_path = _find_sidecar(sheet_path)
        if sidecar_path:
            with open(sidecar_path, "r", encoding="utf-8") as fh:
                sc = json.load(fh)

    sheet = None
    with Image.open(sheet_path) as _im:
        # .convert() copies the pixels; the `with` guarantees the source file
        # handle is closed before we return. On Windows an open handle on
        # src_sheet would make a later overwrite save fail with WinError 32.
        sheet = _im.convert(RGBA)
    W, H = sheet.size

    if sc:
        cols = int(sc.get("columns") or 0)
        rows_n = int(sc.get("rows") or 0)
        cw = int(sc.get("frame_width") or 0)
        ch = int(sc.get("frame_height") or 0)
        if not (cols and rows_n and cw and ch):
            raise ValueError("sidecar %s is missing columns/rows/frame_width/"
                             "frame_height" % sidecar_path)
        if cols * cw != W or rows_n * ch != H:
            raise ValueError(
                "sidecar says %dx%d of %dx%d = %dx%d but the PNG is %dx%d"
                % (cols, rows_n, cw, ch, cols * cw, rows_n * ch, W, H))
    else:
        cols = int(columns or 0)
        rows_n = int(rows or 0)
        if not cols or not rows_n:
            raise ValueError(
                "no sidecar JSON found for %s -- searched %s and every .json in "
                "that folder for one naming this PNG. The editor opens a sheet "
                "through its sidecar; the API can also take explicit columns "
                "and rows."
                % (os.path.basename(sheet_path),
                   os.path.splitext(os.path.basename(sheet_path))[0] + ".json"))
        if W % cols or H % rows_n:
            raise ValueError("%dx%d does not divide evenly into %dx%d"
                             % (W, H, cols, rows_n))
        cw, ch = W // cols, H // rows_n

    arr = np.array(sheet)
    cells = []
    for r in range(rows_n):
        for c in range(cols):
            cells.append(arr[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].copy())

    meta = {
        "fps": int(sc.get("fps", 24)) if sc else 24,
        # Loop is the default. Every sheet generated before that default
        # carries "pingpong" because the old generator wrote it, so a stored
        # pingpong is normalised to loop on open -- otherwise "make loop the
        # default" would not apply to any existing sheet. "once" is left alone;
        # the mode is still per-document and can be changed in the panel.
        "play_mode": "loop" if (not sc or sc.get("play_mode") in (None, "pingpong"))
                     else sc.get("play_mode"),
        "trim_to": 0,
        "anchor": sc.get("anchor", "none") if sc else "none",
        "blend": sc.get("blend", "straight") if sc else "straight",
        "name": (sc.get("name") if sc else None)
                or os.path.splitext(os.path.basename(sheet_path))[0],
        # Legacy input, not written any more: the video a sheet was cut from is
        # named in the `note` now. Read so a sheet that already has it keeps its
        # provenance when it is re-saved.
        "source": sc.get("source", "") if sc else "",
        "matte": sc.get("matte", "edited") if sc else "edited",
        "matte_repair": sc.get("matte_repair", "none") if sc else "none",
        "crop": sc.get("crop") if sc else [0, 0, cw, ch],
        "max_texture": P.MAX_TEXTURE,
        "black_max": P.BLACK_MAX, "leak_alpha": P.LEAK_ALPHA,
        "bright": P.BRIGHT, "erode_max": 24,
    }
    layout = {"columns": cols, "rows": rows_n, "frame_count": len(cells),
              "cell_w": cw, "cell_h": ch,
              "sheet_w": W, "sheet_h": H,
              "crop": meta["crop"], "warnings": []}
    doc = Doc(doc_id, cells, layout, meta,
              src_sheet=sheet_path, src_sidecar=sidecar_path)
    # Audio clips are named relative to the sheet, so a clip that shipped beside
    # it resolves here and stays usable (and re-savable) without re-importing.
    for e in (sc.get("audio") or []) if sc else []:
        rel = e.get("file")
        if not rel:
            continue
        doc.add_audio(int(e.get("frame", 0)),
                      os.path.normpath(os.path.join(os.path.dirname(sheet_path),
                                                    rel)),
                      name=e.get("name") or os.path.basename(rel),
                      volume=float(e.get("volume", 1.0)))
        doc.audio[-1]["frame"] = max(0, min(len(cells) - 1,
                                            doc.audio[-1]["frame"]))
    return doc


def export_subset(doc, frames):
    """A throwaway document holding only `frames`, on a valid grid.

    "Export only the selected frames" is a save-time *view* of the document, not
    an edit to it. The open document is left exactly as it was, so the user can
    export a subset, change the selection, export again, and their undo history
    and dirty set are untouched -- which is the whole point, because the obvious
    implementation (keep_range on the real document) silently destroys the
    frames that were not selected.

    It routes through _set_cells, so a subset gets the same atomic re-grid every
    frame-pruning op gets: the column count is kept when it still divides, an
    auto count is chosen when it does not, and a count with no workable grid
    under the texture cap is refused with the same message instead of writing a
    sheet with blank cells in it.

    Returns (doc_to_save, indices). The indices are sorted and de-duplicated: a
    selection is a set, and the frame order in the exported sheet has to be
    deterministic rather than whatever order the set happened to iterate in.
    """
    idx = sorted({int(i) for i in frames})
    if not idx:
        raise ValueError("no frames are selected, so there is nothing to export")
    bad = [i for i in idx if not (0 <= i < doc.n)]
    if bad:
        raise ValueError(
            "frame %d is not in this document, which has %d frames (0-%d)"
            % (bad[0], doc.n, doc.n - 1))
    if len(idx) == doc.n:
        # The whole document. Re-gridding it here would throw away a layout the
        # user chose deliberately (the layout form, or an earlier re-grid), for
        # no reason at all.
        return doc, list(range(doc.n))

    sub = Doc(doc.id + ":export", [doc.cells[i] for i in idx], doc.layout,
              doc.meta, src_sheet=doc.src_sheet, src_sidecar=doc.src_sidecar)
    # Carry the op history, then record the export itself. The sidecar's
    # `edited.ops` is free-form, so this is provenance without adding a key an
    # engine's loader might not expect.
    sub.journal = list(doc.journal) + [
        "export: %d of %d frames" % (len(idx), doc.n)]
    _set_cells(sub, [doc.cells[i] for i in idx], [doc.orig[i] for i in idx])
    # A subset's audio follows the frames that survived: keep only clips on an
    # exported frame and renumber them to their new index, so the subset's
    # sidecar points at the right frame instead of the original one.
    pos = {old: new for new, old in enumerate(idx)}
    for e in doc.audio:
        if e["frame"] in pos:
            sub.add_audio(pos[e["frame"]], e["path"], name=e["name"],
                          volume=e["volume"])
    return sub, idx


def _locked_hint(path):
    return ("%s is locked by another program (WinError 32). Close it in any "
            "image viewer / game engine / file preview holding it, then save "
            "again." % path)


def _replace_atomic(tmp, dst):
    try:
        os.replace(tmp, dst)
    except PermissionError as ex:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise PermissionError(_locked_hint(dst)) from ex


def _save_sheet_atomic(sheet_img, dst):
    d = os.path.dirname(os.path.abspath(dst)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = dst + ".tmp-%d.png" % os.getpid()
    try:
        sheet_img.save(tmp)
    except PermissionError as ex:
        raise PermissionError(_locked_hint(dst)) from ex
    _replace_atomic(tmp, dst)


def _write_json_atomic(dst, obj):
    d = os.path.dirname(os.path.abspath(dst)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = dst + ".tmp-%d.json" % os.getpid()
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
    except PermissionError as ex:
        raise PermissionError(_locked_hint(dst)) from ex
    _replace_atomic(tmp, dst)


def _copy_atomic(src, dst):
    d = os.path.dirname(os.path.abspath(dst)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = dst + ".tmp-%d" % os.getpid()
    try:
        shutil.copy2(src, tmp)
    except PermissionError as ex:
        raise PermissionError(_locked_hint(dst)) from ex
    _replace_atomic(tmp, dst)


def save_doc(doc, out_dir, name=None, want_gif=True, want_preview=True,
             gif_scale=0.5, gif_bg="checker", emit=None, overwrite=False,
             out_dir_named=True):
    """Write sheet + sidecar + preview + gif. Returns the artifact paths.

    `out_dir_named` says whether the caller actually chose `out_dir` or whether
    the app invented it. It only matters when `overwrite` is set, and it is what
    gives that box a coherent meaning: the box *allows* replacing the loaded
    sheet, and the destination is the loaded files when no destination was named.
    """
    # Refuse to write a document whose grid is not full. _set_cells is atomic so
    # this should be unreachable, but a sheet whose sidecar claims more frames
    # than it contains loads fine and then misbehaves in an engine -- exactly the
    # failure this whole tool exists to prevent, so it is worth the two lines.
    if doc.n != doc.columns * doc.rows:
        raise ValueError(
            "refusing to save: %d frames do not fill a %d x %d grid. The "
            "document is inconsistent -- reload the sheet."
            % (doc.n, doc.columns, doc.rows))
    base = (name or doc.meta.get("name") or "sprite")
    base = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base)[:60]
    sheet_name = base + "_sheet.png"
    sheet_path = os.path.join(out_dir, sheet_name)
    sc_path = os.path.join(out_dir, base + ".json")
    audio_dir = out_dir

    # "allow overwriting the sheet this document was loaded from" has to mean
    # what it says. Reported three times -- the last as "overwriting not working
    # fix taht" -- and the gesture that exposes it is the ordinary one: open a
    # run's sheet, tick the box, leave the output folder BLANK (its own label
    # says "blank = runs/editor_...").
    #
    # A blank folder makes app.py invent `runs/editor_<stamp>_<name>/`, so the
    # save landed in a brand-new folder and the loaded sidecar was never
    # rewritten: the box promised to replace the loaded sheet and replaced
    # nothing. Measured through the running page, ticked, blank folder -- the
    # save reported runs/editor_0921-181431_smoke12/ and both loaded files were
    # byte-identical afterwards.
    #
    # Two things make this more than "always write the source paths":
    #
    #   * The output names are derived from the document's `name`, and the name
    #     *inside* a sidecar need not match the sidecar's filename -- real run
    #     folders do both (runs/*_smoke_api/smoke_api.json is named
    #     "032137_px_00002_front_walk"). So even pointing out_dir at the source
    #     folder is not enough: the derived name misses the loaded file.
    #
    #   * The box is worded as *permission* ("allow..."), and the refusal it
    #     disables is about the destination. A user who typed a folder named a
    #     destination, so that folder is still honoured -- redirecting there
    #     would silently ignore an explicit instruction and replace the original
    #     when they asked for a copy. The box only decides the destination when
    #     the caller named none.
    #
    # The sidecar's `sheet` is made relative to the JSON, because the two are not
    # guaranteed to sit in the same folder (load_doc takes an explicit sidecar
    # path), and a sidecar naming its sheet wrongly is the failure this whole
    # tool exists to prevent.
    if overwrite and doc.src_sheet and not out_dir_named:
        sheet_path = doc.src_sheet
        sc_path = (doc.src_sidecar
                   or os.path.join(os.path.dirname(doc.src_sheet), base + ".json"))
        sheet_name = os.path.relpath(sheet_path, os.path.dirname(sc_path))
        audio_dir = os.path.dirname(sc_path)
        # Everything else follows the sidecar home. The preview player, the gif
        # and the _cells staging used to stay in the invented runs/editor_...
        # folder while only the sheet and the json went back to the source --
        # so every ticked save littered a new folder holding no sheet and no
        # json, which is exactly what this was reported as. Now the whole save
        # lands beside the source and the invented folder is never created.
        out_dir = os.path.dirname(os.path.abspath(sc_path))
        base = os.path.splitext(os.path.basename(sc_path))[0] or base
    os.makedirs(out_dir, exist_ok=True)

    # Refuse to write over the files this document was loaded from.
    #
    # The output folder is a free-text field and the name defaults to the
    # document's own name, so "save it back next to the original" composes
    # exactly the source filename: opening walk/<n>_sheet.png and typing
    # walk/ into that field silently replaces the sheet the whole edit was
    # derived from -- and its sidecar with it. There is no undo for that, and
    # the damage is invisible until an engine reads the new sheet.
    #
    # src_sheet/src_sidecar were already being tracked for exactly this kind of
    # check and simply never consulted. Replacing the original is occasionally
    # what you want, so it stays possible -- but it has to be asked for.
    if not overwrite:
        for what, target, src in (("sheet", sheet_path, doc.src_sheet),
                                  ("sidecar", sc_path, doc.src_sidecar)):
            if src and os.path.abspath(target) == os.path.abspath(src):
                raise ValueError(
                    "refusing to overwrite the %s this document was loaded "
                    "from (%s). Save to a different folder or change the name; "
                    "pass overwrite=true if replacing the original is really "
                    "what you want." % (what, target))

    if emit:
        emit("compose", 0.05, "composing %d cells" % doc.n)
    _save_sheet_atomic(doc.compose(), sheet_path)

    # Ship the sounds with the sheet. The sidecar names each file relative to
    # itself ("audio/003_hit.wav"), so an engine that loads the JSON beside the
    # sheet finds every clip. The staging path the editor uploaded to is private
    # to this server and is not something a player can reach.
    audio_entries = []
    if doc.audio:
        adir = os.path.join(audio_dir, "audio")
        os.makedirs(adir, exist_ok=True)
        used = set()
        for e in sorted(doc.audio, key=lambda x: (x["frame"], x["id"])):
            src = e.get("path")
            if not src or not os.path.isfile(src):
                raise ValueError(
                    "audio clip %r (frame %d) is missing on disk; re-import it "
                    "before saving" % (e.get("name", "clip"), e["frame"]))
            ext = os.path.splitext(e.get("name") or src)[1]
            stem = os.path.splitext(os.path.basename(e.get("name") or src))[0]
            stem = "".join(ch if (ch.isalnum() or ch in "._-") else "_"
                           for ch in stem)[:40] or "clip"
            frame = max(0, min(doc.n - 1, int(e["frame"])))
            fn = "%03d_%s%s" % (frame, stem, ext)
            n = 2
            while fn in used:                     # two clips on the same frame
                fn = "%03d_%s_%d%s" % (frame, stem, n, ext)
                n += 1
            used.add(fn)
            _copy_atomic(src, os.path.join(adir, fn))
            audio_entries.append({
                "frame": frame,
                "file": "audio/" + fn,
                "name": e.get("name") or fn,
                "volume": float(e.get("volume", 1.0)),
            })

    sc = doc.sidecar(sheet_name, audio=audio_entries)
    _write_json_atomic(sc_path, sc)

    arts = {"sheet": sheet_path, "sidecar": sc_path}

    # Reuse the pipeline's preview and gif writers by staging the cells as
    # full-size "frames" and handing them a layout whose crop is the identity.
    tmp = os.path.join(out_dir, "_cells")
    os.makedirs(tmp, exist_ok=True)
    paths = []
    for i, c in enumerate(doc.cells):
        p = os.path.join(tmp, "cell_%04d.png" % i)
        from PIL import Image
        Image.fromarray(c, RGBA).save(p)
        paths.append(p)
    flat = {"cell_w": doc.cell_w, "cell_h": doc.cell_h,
            "columns": doc.columns, "rows": doc.rows,
            "frame_count": doc.n, "crop": [0, 0, doc.cell_w, doc.cell_h],
            "pad_left": 0, "pad_top": 0, "warnings": []}
    cfg = {"fps": sc["fps"], "play_mode": sc["play_mode"],
           "gif_scale": gif_scale, "gif_bg": gif_bg}

    if want_preview:
        if emit:
            emit("preview", 0.75, "writing preview player")
        arts["preview"] = P.write_preview(
            os.path.join(out_dir, base + "_preview.html"), cfg, flat, sc)

    if want_gif:
        if emit:
            emit("gif", 0.8, "writing gif")
        arts["gif"] = P.write_gif(paths, os.path.join(out_dir, base + "_preview.gif"),
                                  cfg, flat, emit=emit)

    if emit:
        emit("done", 1.0, "saved")
    return arts


# The keys the metadata panel owns -- the ones Apply metadata is allowed to
# change in the file. Everything else in the sidecar is carried through
# untouched, which is the whole point of save_meta below.
_META_KEYS = ("name", "fps", "play_mode", "anchor", "blend", "animations", "note")


def save_meta(doc):
    """Write the document's metadata back into the sidecar it was loaded from.

    A targeted update of the file, not a regeneration of it. Regenerating would
    replace `edited.ops` with this session's journal, drop `source` and `matte`,
    and rewrite `crop` -- all of which record how the sheet was made, not what
    the panel was just asked to change. So the existing JSON is read, the
    metadata keys are overwritten, and every other key survives verbatim.
    `audio` in particular is carried through as it stands: the clips are already
    sitting beside the sheet and their paths are already relative to it, so
    re-copying them here would be work with nothing to show for it.

    Refuses rather than writing something wrong when the JSON on disk and the
    document in memory no longer describe the same sheet: the sheet PNG is not
    rewritten here, so a grid change would leave the sidecar claiming a geometry
    the PNG on disk does not have -- and an engine reading that pair gets a
    silently wrong animation, which is the failure this tool exists to prevent.

    Returns the path written.
    """
    src = doc.src_sidecar
    if not src:
        raise ValueError(
            "this document was not opened from a sidecar, so there is no JSON "
            "to write back into. Use Save sheet + sidecar instead.")
    if not os.path.isfile(src):
        raise ValueError("the sidecar this document was loaded from is gone: %s"
                         % src)
    with open(src, "r", encoding="utf-8") as fh:
        try:
            old = json.load(fh)
        except ValueError as ex:
            raise ValueError("the sidecar is not valid JSON (%s); repair it or "
                             "Save to a new file" % ex)
    if not isinstance(old, dict):
        raise ValueError("the sidecar is not a JSON object; refusing to "
                         "overwrite it")

    for key, mine in (("frame_count", doc.n), ("columns", doc.columns),
                      ("rows", doc.rows), ("frame_width", doc.cell_w),
                      ("frame_height", doc.cell_h)):
        if key in old and old[key] != mine:
            raise ValueError(
                "the grid changed (%s is %r in the JSON, %r in the editor) but "
                "this only rewrites the JSON. The sheet PNG on disk still has "
                "the old layout -- use Save sheet + sidecar to write both."
                % (key, old[key], mine))

    # The clips are carried through as the file already has them, so if the
    # document's audio has moved on, writing now would leave the JSON claiming a
    # soundtrack the editor is no longer showing -- and the new clip's file has
    # not been copied beside the sheet, so there is nothing to point at. Save
    # does both of those, so send the user there.
    old_audio = sorted((int(e.get("frame", 0)), e.get("name") or "")
                       for e in (old.get("audio") or []))
    new_audio = sorted((int(e.get("frame", 0)), e.get("name") or "")
                       for e in (doc.audio or []))
    if old_audio != new_audio:
        raise ValueError(
            "the clips changed (%d in the JSON, %d in the editor) but this only "
            "rewrites the metadata. A new clip has to be copied beside the "
            "sheet as well -- use Save sheet + sidecar."
            % (len(old_audio), len(new_audio)))

    fresh = doc.sidecar(old.get("sheet") or os.path.basename(doc.src_sheet or ""),
                        audio=old.get("audio"))
    updated = dict(old)
    for k in _META_KEYS:
        if k in fresh:
            updated[k] = fresh[k]

    # Atomic: the app and any engine polling the file both read it, and a
    # half-written sidecar parses as a corrupt one.
    tmp = src + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(updated, fh, indent=2)
    os.replace(tmp, src)
    return src
