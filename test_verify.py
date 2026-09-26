"""Mutation-test the app's sheet verifier.

A guard never seen to fail is not a guard. Build a real sheet from a real run,
then plant each defect the invariants exist to catch and confirm it is flagged,
while the pristine sheet passes.

    python test_verify.py <run_dir>
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else None
if not RUN:
    print("usage: python test_verify.py <run_dir>")
    sys.exit(2)

side = json.load(open(os.path.join(RUN, [f for f in os.listdir(RUN)
                                         if f.endswith(".json")][0])))
sheet_name = side["sheet"]
sheet_path = os.path.join(RUN, sheet_name)
frames = sorted(os.path.join(RUN, "frames", f)
                for f in os.listdir(os.path.join(RUN, "frames")))
frames = sorted(frames, key=lambda p: p)

cfg = P.merge_cfg({})
layout = {
    "columns": side["columns"], "rows": side["rows"],
    "frame_count": side["frame_count"],
    "cell_w": side["frame_width"], "cell_h": side["frame_height"],
    "crop": side["crop"], "sheet_w": side["columns"] * side["frame_width"],
    "sheet_h": side["rows"] * side["frame_height"],
}

print("sheet :", sheet_path)
print("frames:", len(frames), " layout:", layout["columns"], "x", layout["rows"])
ok, lines = P.verify_sheet(sheet_path, frames, layout, cfg)
print("\n--- pristine ---")
print("\n".join(lines))
print("PASS" if ok else "FAIL")

TMP = os.path.join(RUN, "_mut")
os.makedirs(TMP, exist_ok=True)
base = np.array(Image.open(sheet_path).convert("RGBA"))


def run_case(name, arr, expect_fail=True):
    p = os.path.join(TMP, name + ".png")
    Image.fromarray(arr).save(p)
    ok, lines = P.verify_sheet(p, frames, layout, cfg)
    flagged = not ok
    hits = [l for l in lines if l.startswith(("PREMULT", "BACKDROP", "ERODED",
                                              "GRID", "TEXTURE", "EMPTY", "NOTFULL"))]
    verdict = "OK" if flagged == expect_fail else "*** WRONG ***"
    print("%-18s flagged=%-5s expected=%-5s %s" % (name, flagged, expect_fail, verdict))
    for h in hits[:2]:
        print("      ", h)
    return flagged == expect_fail


cw, ch = layout["cell_w"], layout["cell_h"]
cols = layout["columns"]
results = []

# 1. premultiply: the original shipped bug
a = base[..., 3:4].astype(np.float32) / 255.0
m = base.copy()
m[..., :3] = np.clip(base[..., :3].astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)
results.append(run_case("premultiplied", m))

# 2. backdrop blob: an opaque black square touching the cell border
m = base.copy()
m[ch - 60:ch - 10, 20:80] = (0, 0, 0, 255)
results.append(run_case("backdrop_blob", m))

# 3. erosion: a 30x30 hole punched in a BRIGHT part of the subject.
# Target the brightest opaque pixel, not the centroid -- the centroid of a
# character silhouette lands in the gap between its legs, where punching a hole
# removes nothing bright and therefore tests nothing.
m = base.copy()
cell = m[0:ch, 0:cw]
lum = cell[..., :3].astype(np.int32).max(axis=2)
lum[cell[..., 3] == 0] = -1
by, bx = np.unravel_index(int(np.argmax(lum)), lum.shape)
cell[by - 15:by + 15, bx - 15:bx + 15, 3] = 0
print("   (erosion target: brightest opaque px at (%d,%d), lum %d)" % (by, bx, lum[by, bx]))
results.append(run_case("eroded_30x30", m))

# 4. a 1-px seam -- the model-noise case that must NOT fail
m = base.copy()
cell = m[0:ch, 0:cw]
cell[by - 8:by + 8, bx, 3] = 0
results.append(run_case("seam_1x16", m, expect_fail=False))

# 5. empty cell
m = base.copy()
m[0:ch, 0:cw, 3] = 0
results.append(run_case("empty_cell", m))

# 6. wrong frame_count (full-grid rule)
lay2 = dict(layout); lay2["frame_count"] = layout["frame_count"] + 1
p = os.path.join(TMP, "pristine2.png")
Image.fromarray(base).save(p)
ok2, lines2 = P.verify_sheet(p, frames, lay2, cfg)
print("%-18s flagged=%-5s expected=%-5s %s"
      % ("frame_count+1", not ok2, True, "OK" if not ok2 else "*** WRONG ***"))
results.append(not ok2)

shutil.rmtree(TMP, ignore_errors=True)
print("\n%d/%d mutation cases behaved as expected" % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
