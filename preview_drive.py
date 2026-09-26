"""Drive the generated preview player in a real browser, over CDP.

The player is a *page in its own right* that the pipeline writes for every run,
and nothing tested it. The byte-level checks in test_verify.py look at the sheet,
the op-layer tests look at the document, and ui_drive.py only asserts that the
generator embedded an <iframe class="pv"> -- never that the player inside it
draws the right cell, fills the window, or stops at the last frame.

No model and no video are needed: a synthetic sheet of solid, distinct colours is
written here, so each assertion is about pixels the test chose.

The playback check records every animation tick from inside the page rather than
sampling it from outside. That matters: the bug this guards is playback passing
the last frame, at which point drawImage paints nothing and the canvas silently
goes blank -- a sampled observation can easily miss the moment, a recorded trace
cannot.

    python preview_drive.py [--url http://127.0.0.1:8765] [--port 9222]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

import pipeline as P
from editor_drive import Page, targets

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "runs", "_preview_drive")
CW, CH, COUNT, COLS = 32, 32, 8, 4
COLOUR = [(20 + i * 25, 90, 160, 255) for i in range(COUNT)]

FAILS = []
N = [0]


def ok(cond, label, detail=""):
    N[0] += 1
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", label,
                          "" if cond else "   " + str(detail)))
    if not cond:
        FAILS.append(label)


def build():
    """A sheet of solid cells plus one player page per playback mode."""
    os.makedirs(TMP, exist_ok=True)
    rows = COUNT // COLS
    arr = np.zeros((rows * CH, COLS * CW, 4), np.uint8)
    for i in range(COUNT):
        y, x = (i // COLS) * CH, (i % COLS) * CW
        arr[y:y + CH, x:x + CW] = COLOUR[i]
    Image.fromarray(arr).save(os.path.join(TMP, "player_sheet.png"))
    layout = {"columns": COLS, "rows": rows, "cell_w": CW, "cell_h": CH,
              "frame_count": COUNT}
    pages = {}
    for mode in ("once", "loop", "pingpong"):
        sc = {"name": "player_" + mode, "sheet": "player_sheet.png",
              "columns": COLS, "rows": rows, "frame_width": CW,
              "frame_height": CH, "fps": 24, "play_mode": mode}
        cfg = dict(P.DEFAULT_CFG, fps=24, play_mode=mode)
        out = os.path.join(TMP, mode + "_preview.html")
        P.write_preview(out, cfg, layout, sc)
        pages[mode] = out
    return layout, pages


MEASURE = r"""
(() => {
  const st = document.querySelector('.stage');
  const cv = document.getElementById('cv');
  const r = st.getBoundingClientRect(), c = cv.getBoundingClientRect();
  return {stage: [Math.round(r.width), Math.round(r.height)],
          canvas: [Math.round(c.width), Math.round(c.height)],
          fit: Math.round(fitZoom() * 1000) / 1000,
          scroll: [st.scrollWidth > st.clientWidth,
                   st.scrollHeight > st.clientHeight],
          err: document.getElementById('meta').textContent};
})()
"""

# Runs on every animation tick: the frame index, whether it is playing, the alpha
# of the cell's centre pixel (0 = the canvas is blank) and the button's label.
RECORD = r"""
window.__s = [];
const _orig = window.draw;
window.draw = function () {
  _orig();
  const c = document.getElementById('cv');
  let a = -1;
  if (c.width > 1 && c.height > 1 && state.loaded) {
    a = c.getContext('2d').getImageData(
      Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data[3];
  }
  window.__s.push([state.f, state.playing ? 1 : 0, a,
                   document.getElementById('play').textContent]);
};
"""

CENTRE = r"""
(() => {
  const c = document.getElementById('cv');
  const d = c.getContext('2d').getImageData(
    Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data;
  return [d[0], d[1], d[2], d[3]];
})()
"""


def open_page(page, url):
    page.call("Page.navigate", url=url)
    page.wait_for("document.readyState === 'complete'", "load")
    page.wait_for("typeof CFG === 'object'", "the player to boot", timeout=30)
    page.wait_for("document.getElementById('cv').width > 1 && state.loaded",
                  "the sheet to load", timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--port", type=int, default=9222)
    args = ap.parse_args()

    layout, _pages = build()
    print("synthetic sheet: %d frames of %dx%d on a %dx%d grid"
          % (COUNT, CW, CH, COLS, layout["rows"]))
    print("pages written to %s" % TMP)

    tg = [t for t in targets(args.port) if t["type"] == "page"]
    if not tg:
        print("no page target on port %d -- is Chrome running with "
              "--remote-debugging-port?" % args.port)
        shutil.rmtree(TMP, ignore_errors=True)
        return 2
    page = Page(tg[0]["webSocketDebuggerUrl"])
    page.call("Runtime.enable")
    page.call("Page.enable")
    print("attached to:", tg[0].get("url"))

    print("\n=== 1. the player fills the window it is given ===")
    base = args.url + "/files/_preview_drive/loop_preview.html"
    for w, h in ((1200, 800), (760, 520), (460, 900)):
        page.call("Emulation.setDeviceMetricsOverride", width=w, height=h,
                  deviceScaleFactor=1, mobile=False)
        open_page(page, base)
        m = page.js(MEASURE)
        fill_w = m["canvas"][0] / float(m["stage"][0] - 24)
        fill_h = m["canvas"][1] / float(m["stage"][1] - 24)
        print("    %4dx%-4d stage %-11s canvas %-11s fit %-6s fill %.2fx%.2f"
              % (w, h, m["stage"], m["canvas"], m["fit"], fill_w, fill_h))
        ok("could not load" not in m["err"],
           "%dx%d: the sheet loaded" % (w, h), m["err"][:90])
        # 1.0 on the zoom slider means "as big as the window allows", so one axis
        # has to be full and neither may overflow.
        ok(max(fill_w, fill_h) > 0.9,
           "%dx%d: the cell fills the stage on one axis" % (w, h),
           "%.2f x %.2f" % (fill_w, fill_h))
        ok(min(fill_w, fill_h) <= 1.02,
           "%dx%d: and does not overflow it" % (w, h),
           "%.2f x %.2f" % (fill_w, fill_h))
    page.call("Emulation.setDeviceMetricsOverride", width=900, height=620,
              deviceScaleFactor=1, mobile=False)

    print("\n=== 2. playback honours the last frame in every mode ===")
    for mode in ("once", "loop", "pingpong"):
        open_page(page, args.url + "/files/_preview_drive/" + mode
                  + "_preview.html")
        page.js(RECORD)
        page.js("window.__s = []; document.getElementById('play').click();")
        page.wait_for("window.__s.length > 90", "playback to run", timeout=30)
        snaps = page.js("window.__s")
        frames = [s[0] for s in snaps]
        blank = sum(1 for s in snaps if s[2] == 0)
        label = page.js("document.getElementById('play').textContent")
        cur = int(page.js("document.getElementById('fl').textContent"))
        moved = sum(1 for a, b in zip(frames, frames[1:]) if a != b)
        print("    %-9s %d ticks, %d frame changes, frames %d..%d, blank %d, "
              "button %r" % (mode, len(frames), moved, min(frames), max(frames),
                             blank, label))
        ok(moved > 0, "%s: the frames advance" % mode)
        ok(max(frames) == COUNT - 1,
           "%s: playback reaches the last frame and no further" % mode,
           "max %d, last is %d" % (max(frames), COUNT - 1))
        ok(min(frames) >= 0, "%s: the frame index stays in range" % mode,
           min(frames))
        # The old bug's signature: past the last cell drawImage paints nothing.
        ok(blank == 0, "%s: the canvas is never blank" % mode,
           "%d of %d ticks" % (blank, len(snaps)))
        ok(tuple(page.js(CENTRE)) == tuple(COLOUR[cur]),
           "%s: the drawn cell is the frame the counter names" % mode,
           "%s vs %s" % (page.js(CENTRE), list(COLOUR[cur])))
        if mode == "once":
            ok(frames[-1] == COUNT - 1 and not snaps[-1][1],
               "once: playback stops on the last frame",
               "frame %d, playing %s" % (frames[-1], snaps[-1][1]))
            ok(label == "Play", "once: the button goes back to Play", label)
        if mode == "loop":
            wraps = sum(1 for a, b in zip(frames, frames[1:]) if b < a)
            ok(wraps > 0, "loop: playback wraps back to the first frame", wraps)
            ok(snaps[-1][1] and label == "Pause",
               "loop: and keeps playing", label)
        if mode == "pingpong":
            ups = sum(1 for a, b in zip(frames, frames[1:]) if b > a)
            downs = sum(1 for a, b in zip(frames, frames[1:]) if b < a)
            ok(ups and downs, "pingpong: playback reverses at both ends",
               "up %d, down %d" % (ups, downs))
            ok(snaps[-1][1] and label == "Pause",
               "pingpong: and keeps playing", label)

    print("\n=== 3. the page stays quiet ===")
    ok(not page.errors(), "no console errors while playing",
       "\n        ".join(str(e)[:200] for e in page.errors()[:5]))

    shutil.rmtree(TMP, ignore_errors=True)
    print("\n" + "=" * 66)
    print("%d checks, %d failures" % (N[0], len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("  FAILED:", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
