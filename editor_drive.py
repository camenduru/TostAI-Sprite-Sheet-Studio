"""Drive the real sprite editor in a real browser, over CDP.

The op-layer tests and the HTTP smoke both post JSON directly. Neither touches
the DOM: `runOp` gathering arguments out of form controls, the canvas
pointer handling, the stroke rasteriser, the timeline, the playback loop. That is
exactly the layer where the generator's `<input type="range">` bug lived, so it
is worth its own pass.

Mouse input goes through `Input.dispatchMouseEvent`, NOT a JS-constructed
PointerEvent. A synthetic event carries no active pointer, so
`setPointerCapture` throws `NotFoundError` inside the handler and the drag dies;
and `offsetX`/`offsetY` are only filled in for events the browser actually
routes. Real CDP input avoids both.

    python editor_drive.py [--url http://127.0.0.1:8765] [--port 9222]
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import wave

import websocket

FAILS = []
N = [0]

# Fixture folders this run made. Registered for removal when the process ends,
# however it ends -- including an uncaught exception in the middle of a section.
# A crashed run used to leave its folders on disk, and because the page remembers
# the folder a sheet was opened from, the *next* run's very first check ("a fresh
# profile shows an empty folder picker") then failed against a folder this suite
# had made. Cleanup on exit makes a crash in one section unable to poison the
# next run.
_TMPDIRS = []


def mkdtemp(prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    _TMPDIRS.append(d)
    return d


atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _TMPDIRS])


def ok(cond, label, detail=""):
    N[0] += 1
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", label,
                          "" if cond else "   " + str(detail)))
    if not cond:
        FAILS.append(label)


def targets(port):
    with urllib.request.urlopen("http://127.0.0.1:%d/json" % port, timeout=10) as r:
        return json.load(r)


class Page:
    def __init__(self, ws_url):
        # suppress_origin: a browser started with just
        # `--headless=new --remote-debugging-port=9222` refuses the WebSocket
        # handshake with a 403 unless that origin was passed to
        # --remote-allow-origins, so the documented launch line could not
        # connect. Dropping the Origin header is what the flag allows anyway.
        self.ws = websocket.create_connection(ws_url, timeout=180,
                                             suppress_origin=True)
        self.i = 0
        self.events = []

    def call(self, method, **params):
        self.i += 1
        mid = self.i
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError("%s -> %s" % (method, msg["error"]))
                return msg.get("result", {})
            if "method" in msg:
                self.events.append(msg)

    def js(self, expr, await_promise=False):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True,
                      awaitPromise=await_promise)
        if r.get("exceptionDetails"):
            ex = r["exceptionDetails"]
            d = (ex.get("exception") or {}).get("description") or ex.get("text")
            raise RuntimeError("JS threw: %s" % str(d)[:400])
        return r.get("result", {}).get("value")

    def wait_for(self, expr, label, timeout=90, interval=0.25):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.js(expr):
                    return time.time() - t0
            except RuntimeError:
                pass
            time.sleep(interval)
        raise TimeoutError("timed out waiting for %s" % label)

    # -- real input -------------------------------------------------------- #

    def mouse(self, kind, x, y, button="left", buttons=0, modifiers=0,
              clicks=1):
        self.call("Input.dispatchMouseEvent", type=kind, x=float(x), y=float(y),
                  button=button, buttons=buttons, clickCount=clicks,
                  modifiers=modifiers)

    def click_at(self, x, y, modifiers=0):
        self.mouse("mouseMoved", x, y, "none", 0, modifiers)
        self.mouse("mousePressed", x, y, "left", 1, modifiers)
        self.mouse("mouseReleased", x, y, "left", 0, modifiers)

    def drag(self, pts, modifiers=0):
        self.mouse("mouseMoved", pts[0][0], pts[0][1], "none", 0, modifiers)
        self.mouse("mousePressed", pts[0][0], pts[0][1], "left", 1, modifiers)
        for x, y in pts[1:]:
            self.mouse("mouseMoved", x, y, "left", 1, modifiers)
            time.sleep(0.02)
        self.mouse("mouseReleased", pts[-1][0], pts[-1][1], "left", 0, modifiers)

    # -- real HTML5 drag-and-drop ------------------------------------------ #

    def drag_start(self, x, y):
        """Press on a draggable and move off it, so the browser starts a drag.

        Returns the drag data the browser intercepted, or None if this browser
        will not hand a drag to the protocol (then the caller drives the page's
        own handlers instead). `setInterceptDrags` is what makes Chrome report
        the drag over CDP rather than running it itself; without it
        `Input.dispatchDragEvent` has no drag to attach to.
        """
        try:
            self.call("Input.setInterceptDrags", enabled=True)
        except RuntimeError:
            return None
        n0 = len(self.events)
        self.mouse("mouseMoved", x, y, "none", 0)
        self.mouse("mousePressed", x, y, "left", 1)
        data = None
        for k in range(1, 12):
            # Chrome only starts a drag once the pointer has left the press point
            # by its own threshold, so creep away from it rather than jumping.
            self.mouse("mouseMoved", x + k * 4, y + k * 2, "left", 1)
            time.sleep(0.04)
            data = self.drag_data(n0)
            if data:
                break
        if not data:
            self.mouse("mouseReleased", x + 44, y + 22, "left", 0)
            self._drag_off()
            return None
        return data

    def drag_data(self, since=0):
        for e in self.events[since:]:
            if e.get("method") == "Input.dragIntercepted":
                return e["params"]["data"]
        return None

    def drag_over(self, x, y, data):
        self.call("Input.dispatchDragEvent", type="dragEnter", x=float(x),
                  y=float(y), data=data)
        self.call("Input.dispatchDragEvent", type="dragOver", x=float(x),
                  y=float(y), data=data)

    def drag_drop(self, x, y, data):
        self.call("Input.dispatchDragEvent", type="drop", x=float(x),
                  y=float(y), data=data)
        self.mouse("mouseReleased", x, y, "left", 0)
        self._drag_off()

    def _drag_off(self):
        try:
            self.call("Input.setInterceptDrags", enabled=False)
        except RuntimeError:
            pass

    # -- real keyboard ------------------------------------------------------ #

    _VK = {"b": 66, "e": 69, "v": 86, "h": 72, "i": 73, "z": 90, "a": 65,
           "s": 83}
    # Named keys get their virtual key code from here; anything else is assumed to
    # be a character, which is why a cursor key that is not in this table used to
    # die in ord() with a TypeError instead of pressing a key. The arrows are
    # frame stepping (and, with Alt, the pixel nudge), so they belong here.
    _NAMED = {"Escape": (27, "Escape"), "Enter": (13, "Enter"),
              " ": (32, "Space"),
              "ArrowLeft": (37, "ArrowLeft"), "ArrowUp": (38, "ArrowUp"),
              "ArrowRight": (39, "ArrowRight"), "ArrowDown": (40, "ArrowDown")}

    def key(self, k, ctrl=False):
        """A real key event, dispatched by the browser rather than synthesised.

        `text` must be omitted entirely for a modified or named key: CDP rejects
        `text: null` outright with "Failed to deserialize params.text".
        """
        if k in self._NAMED:
            vk, code = self._NAMED[k]
        elif k.isdigit():
            vk, code = ord(k), "Digit" + k
        else:
            vk, code = self._VK.get(k.lower(), ord(k.upper())), "Key" + k.upper()
        common = dict(key=k, code=code, windowsVirtualKeyCode=vk,
                      nativeVirtualKeyCode=vk, modifiers=2 if ctrl else 0)
        if ctrl or len(k) != 1:
            self.call("Input.dispatchKeyEvent", type="keyDown", **common)
        else:
            self.call("Input.dispatchKeyEvent", type="keyDown", text=k, **common)
        self.call("Input.dispatchKeyEvent", type="keyUp", **common)

    def errors(self):
        out = []
        for e in self.events:
            m = e.get("method")
            if m == "Runtime.exceptionThrown":
                d = e["params"]["exceptionDetails"]
                out.append((d.get("exception") or {}).get("description")
                           or d.get("text") or "exception")
            elif m == "Runtime.consoleAPICalled" and e["params"].get("type") == "error":
                out.append(" ".join(str(a.get("value", a.get("description", "")))
                                    for a in e["params"].get("args", [])))
        return out


# JS helpers injected once. `S`, `SHEET` and `view` are top-level `const`/`let`
# in a classic script, so they live in the global lexical scope and are readable
# from Runtime.evaluate -- but they are NOT properties of window.
HELPERS = r"""
window.__d = {
  ready: () => document.readyState === 'complete' && !!S.doc && S.img.size === S.doc.n,
  sel: () => [...S.sel].sort((a,b)=>a-b),
  rev: () => S.doc.rev,
  cellrev: i => S.doc.cell_rev[i],
  // IDLE, not busy, despite the name: true when nothing is in flight. Waiting on
  // it is the normal pattern -- `__d.busy() && <the outcome>` -- and inverting it
  // to `!__d.busy()` waits for a moment that has usually already passed.
  idle: () => S.busy === 0 && S.stroke.length === 0,
  busy: () => S.busy === 0 && S.stroke.length === 0,
  cellScreen: i => {
    const r = view.getBoundingClientRect();
    const L = SHEET();
    const col = i % L.cols, row = Math.floor(i / L.cols);
    const sx = col * L.cw + L.cw / 2, sy = row * L.ch + L.ch / 2;
    return {x: r.left + S.view.tx + sx * S.view.s,
            y: r.top + S.view.ty + sy * S.view.s};
  },
  // Only cells whose centre is actually inside the canvas can be clicked. At a
  // fitted zoom the sheet is wider than the viewport, so "cell 9" may be off
  // screen -- clicking it would land on the side panel and do nothing.
  visibleCells: n => {
    const r = view.getBoundingClientRect();
    const out = [];
    for (let i = 0; i < S.doc.n && out.length < n; i++){
      const p = window.__d.cellScreen(i);
      if (p.x > r.left + 2 && p.x < r.right - 2 &&
          p.y > r.top + 2 && p.y < r.bottom - 2) out.push(i);
    }
    return out;
  },
  sheetScreen: (sx, sy) => {
    const r = view.getBoundingClientRect();
    return {x: r.left + S.view.tx + sx * S.view.s,
            y: r.top + S.view.ty + sy * S.view.s};
  },
  setView: (s, tx, ty) => { S.view.s = s; S.view.tx = tx; S.view.ty = ty; draw(); },
  hashCell: async i => {
    const r = await fetch('/api/editor/' + S.doc.id + '/cell/' + i + '.png',
                          {cache: 'no-store'});
    const b = new Uint8Array(await r.arrayBuffer());
    let h = 0; for (let k = 0; k < b.length; k++) h = (h * 31 + b[k]) >>> 0;
    return h + ':' + b.length;
  },
  // The alpha box of a cell as the CLIENT has it decoded -- the pixels that are
  // actually on screen, not a re-fetch. Every other helper here asks the server
  // what the document says; this one asks the page what it is showing, which is
  // the only way to catch a surface that was never re-rendered after an edit.
  cellBox: i => {
    const im = S.img.get(i);
    if (!im) return null;
    const c = document.createElement('canvas');
    c.width = im.width; c.height = im.height;
    const g = c.getContext('2d');
    g.drawImage(im, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
    for (let y = 0; y < c.height; y++){
      for (let x = 0; x < c.width; x++){
        if (d[(y * c.width + x) * 4 + 3] > 0){
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
    }
    return x1 < 0 ? null : [x0, y0, x1, y1];
  },
  // The distinct opaque colours of a cell, as the client has decoded them. The
  // companion to cellBox for anything about *which* colours are on screen rather
  // than where they are -- a palette has to be measured as a set, not a bbox.
  cellColours: i => {
    const im = S.img.get(i);
    if (!im) return null;
    const c = document.createElement('canvas');
    c.width = im.width; c.height = im.height;
    const g = c.getContext('2d');
    g.drawImage(im, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const seen = new Set();
    for (let k = 0; k < d.length; k += 4){
      if (d[k + 3] > 0){
        seen.add((d[k] << 16) | (d[k + 1] << 8) | d[k + 2]);
      }
    }
    return [...seen].map(v => v.toString(16).padStart(6, '0').toUpperCase());
  },
  // The decoded size of a cell as the CLIENT has it, which is the only place a
  // change of cell size is visible without trusting S.doc.layout. A resize that
  // updated the document but never reloaded the cell would report the new
  // layout and still show the old pixels.
  cellSize: i => {
    const im = S.img.get(i);
    return im ? [im.width, im.height] : null;
  },
  // A hash of a cell's decoded RGBA plus its size. For any op that has to be
  // byte-exact -- a resize by a whole factor, a down after an up -- comparing
  // two of these is the end-to-end statement: same pixels on screen, not merely
  // same numbers in the payload.
  cellSig: i => {
    const im = S.img.get(i);
    if (!im) return null;
    const c = document.createElement('canvas');
    c.width = im.width; c.height = im.height;
    const g = c.getContext('2d');
    g.drawImage(im, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let h = 0;
    for (let k = 0; k < d.length; k++) h = (h * 31 + d[k]) >>> 0;
    return c.width + 'x' + c.height + ':' + h;
  },
  recentDirs: () => [...document.querySelectorAll('#recent div[data-k]')]
      .map(d => d.title),
  recentNames: () => [...document.querySelectorAll('#recent div[data-k]')]
      .map(d => d.querySelector('.n').textContent),
  opNode: label => [...document.querySelectorAll('#ops details.op')]
      .find(d => d.querySelector('summary').textContent.trim() === label),
  opInputs: label => {
    const d = window.__d.opNode(label);
    return d ? [...d.querySelectorAll('.opbody input, .opbody select')] : [];
  },
  applyOp: label => {
    const d = window.__d.opNode(label);
    if (!d) return 'no such op node: ' + label;
    d.open = true;
    const b = [...d.querySelectorAll('.opbody button')].find(x => x.textContent === 'Apply');
    if (!b) return 'no Apply button';
    b.click();
    return 'clicked';
  },
  previewInk: () => {
    const c = document.getElementById('prev');
    const g = c.getContext('2d');
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let k = 3; k < d.length; k += 4) if (d[k] > 8) n++;
    return n;
  },
  timelineSel: () => document.querySelectorAll('#strip canvas.sel').length,
  stripCount: () => document.querySelectorAll('#strip canvas').length,
};
'ok'
"""


def open_by_name(page, frag, cols=0, rows=0):
    """Open a document by a name fragment, resolved through the sheet route.

    Not a `window.__d` helper on purpose. The page is reloaded in section 16, and
    `__d` does not survive a reload -- a helper would work in the early sections
    and then fail with "Cannot read properties of undefined" in every section
    after it. This is self-contained, so it works wherever it is called.

    The picker it replaces used to list the server's sheet library and the driver
    clicked a row. It lists recently opened folders now, which is empty in a fresh
    profile -- so a section that needs a *specific* sheet asks the route that knows
    where the sheets are. Returns a string, so a miss is a named assertion rather
    than a 60-second wait that says nothing.
    """
    js = ("(async () => {"
          " const d = await fetch('/api/sheets', {cache: 'no-store'})"
          "   .then(r => r.json());"
          " const s = (d.sheets || []).find(x =>"
          "   (x.name || '').includes(%r) || (x.sheet || '').includes(%r));"
          " if (!s) return 'no sheet matching ' + %r;"
          " return (await openSheet(s.sidecar || s.sheet, %d, %d))"
          "   ? 'opened' : 'failed';"
          " })()" % (frag, frag, frag, cols, rows))
    return page.js(js, await_promise=True)


def settle(page, n=2):
    """Wait n real rendering frames.

    Needed after anything that *queues* work for the rendering step rather than
    doing it synchronously. A scroll event is the case that matters: assigning
    `element.scrollTop` moves the content immediately but dispatches the `scroll`
    event during the next rendering update, so `document.body.offsetHeight` --
    which forces layout but not event delivery -- is not enough. Section 16q's
    first draft measured straight after the scroll and reported "still open" for
    a listener that had simply not run yet.
    """
    page.js("new Promise(r => { let k = %d;"
            " const step = () => (--k > 0) ? requestAnimationFrame(step) : r(1);"
            " requestAnimationFrame(step); })" % n, await_promise=True)


# Heights 30, 24, 20, 38 across the four frames -- the 18 px spread a loop jumps
# on. The widths differ too, so the op's `measure` control has a real choice to
# make rather than two identical answers.
PULSE_BOXES = [(20, 20, 40, 50), (22, 22, 42, 46), (12, 20, 52, 40),
               (21, 14, 41, 52)]


def write_pulse_sheet(folder):
    """Write the reported sheet to disk and return its sidecar path.

    The fixture has to be wrong in exactly the way the report described before
    the fix can be shown to correct it, so it is drawn here rather than reached
    for out of the sheet library -- no sheet in there is guaranteed to still have
    a size spread after someone edits it.
    """
    import numpy as np
    from PIL import Image
    arr = np.zeros((128, 128, 4), np.uint8)
    for i, (x0, y0, x1, y1) in enumerate(PULSE_BOXES):
        row, col = divmod(i, 2)
        cell = np.zeros((64, 64, 4), np.uint8)
        cell[y0:y1, x0:x1] = (230, 230, 230, 255)
        cell[y0 + 2, x0 + 2] = (255, 0, 0, 255)     # marker, as in the op tests
        arr[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = cell
    Image.fromarray(arr).save(os.path.join(folder, "pulse_sheet.png"))
    side = os.path.join(folder, "pulse.json")
    with open(side, "w", encoding="utf-8") as fh:
        json.dump({"name": "pulse", "sheet": "pulse_sheet.png",
                   "frame_width": 64, "frame_height": 64, "columns": 2,
                   "rows": 2, "frame_count": 4, "fps": 24}, fh)
    return side


def write_mesh_sheet(folder):
    """Write a noisy nearest-upscaled sprite sheet and return its sidecar path.

    2x2 of 192x192, each frame an 12x12 sprite upscaled 16x and jittered -- the
    input the mesh detector is built for. At this size the auto pixel width
    resolves, so the section that uses it exercises the default path a user hits
    rather than the override.
    """
    import numpy as np
    from PIL import Image
    arr = np.zeros((384, 384, 4), np.uint8)
    for i in range(4):
        true, size = 12, 192
        small = np.zeros((true, true, 4), np.uint8)
        yy, xx = np.mgrid[0:true, 0:true]
        small[((xx - true // 2) ** 2 + (yy - true // 2) ** 2)
              < (true // 2 - 1) ** 2] = (80, 150, 220, 255)
        small[(i + 2):(i + 4), 3:6] = (220, 80, 80, 255)
        big = np.array(Image.fromarray(small).resize((size, size),
                                                     Image.NEAREST))
        rng = np.random.default_rng(i)
        n = rng.integers(-16, 17, big.shape[:2]).astype(np.int16)
        big[..., :3] = np.clip(big[..., :3].astype(np.int16) + n[..., None],
                               0, 255).astype(np.uint8)
        row, col = divmod(i, 2)
        arr[row * 192:(row + 1) * 192, col * 192:(col + 1) * 192] = big
    Image.fromarray(arr).save(os.path.join(folder, "mesh_sheet.png"))
    side = os.path.join(folder, "mesh.json")
    with open(side, "w", encoding="utf-8") as fh:
        json.dump({"name": "mesh", "sheet": "mesh_sheet.png",
                   "frame_width": 192, "frame_height": 192, "columns": 2,
                   "rows": 2, "frame_count": 4, "fps": 24}, fh)
    return side


def write_resize_sheet(folder):
    """Write a 2x2 sheet of 64x64 hard-edged cells and return its sidecar path.

    Deliberately the worst case for any filter: a one-pixel checkerboard of two
    saturated colours in the top-left 16x16 of every cell. Nearest-neighbour
    scaling turns each source pixel into a 2x2 block of the same colour, so the
    checkerboard survives as a checkerboard; anything that interpolates has to
    invent a third colour, and the driver measures both halves of that claim --
    the block shape and the colour set. The rest of the cell is transparent, so
    a stray colour cannot hide in a flat field.
    """
    import numpy as np
    from PIL import Image
    arr = np.zeros((128, 128, 4), np.uint8)
    yy, xx = np.mgrid[0:64, 0:64]
    near = (xx < 16) & (yy < 16)
    cell = np.zeros((64, 64, 4), np.uint8)
    cell[near & ((xx + yy) % 2 == 0)] = (0, 0, 255, 255)        # blue
    cell[near & ((xx + yy) % 2 == 1)] = (255, 255, 0, 255)       # yellow
    for i in range(4):
        row, col = divmod(i, 2)
        arr[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = cell
    Image.fromarray(arr).save(os.path.join(folder, "resize_sheet.png"))
    side = os.path.join(folder, "resize.json")
    with open(side, "w", encoding="utf-8") as fh:
        json.dump({"name": "resize", "sheet": "resize_sheet.png",
                   "frame_width": 64, "frame_height": 64, "columns": 2,
                   "rows": 2, "frame_count": 4, "fps": 24}, fh)
    return side


def write_transform_sheet(folder):
    """Write a 2x2 sheet of 64x64 cells, each with a 16x16 white block at (24,24).

    Off-centre and smaller than the cell, so the box the gizmo draws is
    distinguishable from the cell rectangle and a 90-degree rotation is
    observable as a swapped aspect ratio.
    """
    import numpy as np
    from PIL import Image
    arr = np.zeros((128, 128, 4), np.uint8)
    cell = np.zeros((64, 64, 4), np.uint8)
    cell[24:40, 24:40] = (230, 230, 230, 255)
    cell[26, 26] = (255, 0, 0, 255)          # marker pixel for exact-position checks
    for i in range(4):
        row, col = divmod(i, 2)
        arr[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = cell
    Image.fromarray(arr).save(os.path.join(folder, "transform_sheet.png"))
    side = os.path.join(folder, "transform.json")
    with open(side, "w", encoding="utf-8") as fh:
        json.dump({"name": "transform", "sheet": "transform_sheet.png",
                   "frame_width": 64, "frame_height": 64, "columns": 2,
                   "rows": 2, "frame_count": 4, "fps": 24}, fh)
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--port", type=int, default=9222)
    args = ap.parse_args()

    tg = [t for t in targets(args.port) if t["type"] == "page"]
    if not tg:
        print("no page target on port %d" % args.port)
        return 2
    page = Page(tg[0]["webSocketDebuggerUrl"])
    page.call("Runtime.enable")
    page.call("Page.enable")

    print("\n=== 1. load the editor in a real browser ===")
    # The editor remembers the last sheet it opened and reopens it on load.
    # Clear that first and reload, so the run starts from a known state instead
    # of inheriting whatever document the previous run left behind.
    page.call("Page.navigate", url=args.url + "/editor")
    page.wait_for("document.readyState === 'complete'", "document load")
    page.js("try { localStorage.removeItem('sprite.lastSheet'); } catch(e) {}")
    # ...and the recently-opened-folders list. This one is easy to forget because
    # it is a list rather than a single value: leave it, and the picker is
    # pre-populated by every earlier run in the same browser profile, so the
    # "empty state" and "the folder it came from is the only entry" checks fail
    # against history rather than against the code.
    page.js("try { localStorage.removeItem('sprite.recentFolders'); } catch(e) {}")
    # ...and the remembered audio library folder, for the same reason: boot()
    # re-lists it, so a leftover path from an earlier run in this browser profile
    # would race section 16j's own listing and could render its rows.
    page.js("try { localStorage.removeItem('sprite.audioLibDir'); } catch(e) {}")
    page.call("Page.navigate", url=args.url + "/editor")
    page.wait_for("document.readyState === 'complete'", "document load")
    # The picker lists recently opened folders, and boot() has just cleared them,
    # so a fresh profile must show the empty state rather than a blank box.
    page.wait_for("!!document.getElementById('recent')", "the folder picker",
                  timeout=30)
    _empty = page.js("document.getElementById('recent').textContent.trim()")
    ok(len(_empty) > 0 and page.js(
        "document.querySelectorAll('#recent div[data-k]').length") == 0,
       "a fresh profile shows an empty folder picker that says so", _empty)
    n_ops = page.js("document.querySelectorAll('#ops details.op').length")
    ok(n_ops == 43, "the op palette rendered every registered op", n_ops)
    ok(page.js("document.querySelectorAll('#ops details.opgroup').length") == 9,
       "grouped into 9 sections")
    first_groups = page.js(
        "[...document.querySelectorAll('#ops details.opgroup > summary')]"
        ".slice(0, 2).map(s => s.textContent)")
    ok(first_groups == ["Pixel art", "Transform"],
       "Pixel art is listed above Transform", first_groups)
    _folded = page.js(
        "[...document.querySelectorAll('#ops details.opgroup')]"
        ".filter(d => ['Transform', 'Matte repair'].includes("
        "d.querySelector('summary').textContent))"
        ".map(d => d.open)")
    ok(_folded == [False, False],
       "Transform and Matte repair start folded", _folded)
    ok(page.js("(() => { const d = [...document.querySelectorAll("
               "'#ops details.opgroup')].find(d => d.querySelector('summary')"
               ".textContent === 'Pixel art'); return !!(d && d.open); })()"),
       "Pixel art starts open, since it leads")

    # The panel layout. A clip is attached to a frame while the sheet is open, so
    # the audio panels live in the left column under Operations, beside the sheet
    # they annotate: the library first and the attach controls under it, which is
    # the order they had before the library was folded into the Audio panel --
    # reported as "put under audio librarry like back then". The library is folded
    # and Audio is open: the folder is chosen once and then the panel is in the
    # way, whereas the attach controls are used on every clip. Everything in that
    # column starts folded except Audio; the palette is long, and the picker is
    # used once a session. Reopening any of them, or putting Audio back on the
    # right, is what mutant J reverts.
    _left = page.js(
        "[...document.querySelectorAll('.main > .col:not(.right) > details.panel')]"
        ".map(d => [d.querySelector('summary').textContent.trim(), d.open])")
    ok(_left == [["Open a sheet", False], ["Operations", False],
                 ["Audio Library", False], ["Audio", True]],
       "the left column is the picker, the palette and the audio library over "
       "the audio panel, with only Audio open", _left)
    # Exact, not a membership test: the right column lost its Op history panel
    # (the journal is still on S.doc and still asserted further down), and an
    # exact list is what notices a panel appearing or going missing here.
    _right = page.js(
        "[...document.querySelectorAll('.main > .col.right > details.panel > "
        "summary')].map(s => s.textContent.trim())")
    ok(_right == ["Playback", "Timeline", "Grid", "Verification",
                  "Export"],
       "the right column is playback, timeline, grid, verification and export, "
       "with no audio panel and no op history in it", _right)

    page.js(HELPERS)
    ok(page.js("typeof window.__d.ready") == "function", "helpers installed")

    print("\n=== 2. open a sheet by name, and watch the folder picker record it ===")
    # Resolved through the sheet route, not clicked out of the picker: the picker
    # lists the folders this profile has visited, and it has visited none.
    _r = open_by_name(page, "walk")
    ok(_r == "opened", "the walk sheet opened by name", _r)
    page.wait_for("!!(S.doc) && S.doc.n > 0", "the document to open", timeout=60)
    ok(page.js("S.doc.n") == 124, "opened 124 frames", page.js("S.doc.n"))
    page.wait_for("S.img.size === S.doc.n", "every cell PNG to load", timeout=120)
    ok(page.js("S.img.size") == 124, "all 124 cells loaded into the client cache")
    # Opening is what records the folder, and the entry is named after the folder
    # rather than the sheet inside it -- which is the whole point of the list.
    _dirs = page.js("window.__d.recentDirs()")
    _names = page.js("window.__d.recentNames()")
    ok(len(_dirs) == 1 and _dirs[0].lower().endswith("walk"),
       "the folder it came from is now the only entry in the picker", _dirs)
    ok(_names == [os.path.basename(_dirs[0])] if _dirs else False,
       "and the entry is labelled with the folder, not the sheet", _names)
    ok(page.js("document.getElementById('docname').textContent").count("124") == 1,
       "the header shows the frame count",
       page.js("document.getElementById('docname').textContent"))
    ok(page.js("document.getElementById('dlSheet').href").endswith("/sheet.png"),
       "the download link points at the composed sheet")
    # Opening by name hands over the sidecar, not the PNG: the JSON is the
    # document (it names the sheet, the grid and the clips), so opening it is what
    # brings a saved sheet's sounds back with it.
    ok(page.js("document.getElementById('sheetPath').value").endswith(".json"),
       "the sheet route opens the sheet's sidecar, which is what carries its "
       "sounds", page.js("document.getElementById('sheetPath').value"))

    print("\n=== 3. real mouse clicks select cells ===")
    page.js("window.__d.setView(0.25, 4, 4)")
    vis = page.js("window.__d.visibleCells(8)")
    ok(len(vis) >= 6, "found %d cells inside the canvas to click" % len(vis), vis)
    c_a, c_b, c_c = vis[1], vis[4], vis[2]
    ok(page.js("document.getElementById('loopSelLabel').textContent") ==
       "play selection only",
       "with nothing selected the play box carries no count")

    p = page.js("window.__d.cellScreen(%d)" % c_a)
    page.click_at(p["x"], p["y"])
    ok(page.js("window.__d.sel()") == [c_a], "clicking a cell selects just it",
       page.js("window.__d.sel()"))
    ok(page.js("document.getElementById('selinfo').textContent").startswith("1 of"),
       "the selection readout updated",
       page.js("document.getElementById('selinfo').textContent"))
    ok(page.js("S.cur") == c_a, "clicking also moves the playback cursor",
       page.js("S.cur"))

    p = page.js("window.__d.cellScreen(%d)" % c_b)
    page.click_at(p["x"], p["y"], modifiers=8)          # shift
    ok(page.js("window.__d.sel()") == list(range(c_a, c_b + 1)),
       "shift-click extends a contiguous range", page.js("window.__d.sel()"))

    p = page.js("window.__d.cellScreen(%d)" % c_c)
    page.click_at(p["x"], p["y"], modifiers=2)          # ctrl
    ok(page.js("window.__d.sel()") ==
       [i for i in range(c_a, c_b + 1) if i != c_c],
       "ctrl-click toggles one cell out", page.js("window.__d.sel()"))

    # --- the range must mean the same thing on both selection surfaces ------
    # The check above could not tell a frame range from a grid rectangle apart,
    # because it picks cells that share a row -- and those two are the same thing
    # only while they share a row. On this 31-column sheet cell 4 is row 0 col 4
    # and cell 99 is row 3 col 6: the rectangle between them is 12 cells, the
    # frames between them are 96. The canvas drew the rectangle, the timeline
    # drew the range, and both were called "shift-click". Reported as "when user
    # shift select 4 to 99 shift select should select all frames between 4 and
    # 99 also 4 and 99".
    page.js("fitView()")          # all 124 cells on screen, so both are clickable
    n_all = page.js("S.doc.n")
    page.js("S.sel.clear(); S.anchorCell = 0; syncSel()")
    p4 = page.js("window.__d.cellScreen(4)")
    p99 = page.js("window.__d.cellScreen(99)")
    page.click_at(p4["x"], p4["y"])
    ok(page.js("window.__d.sel()") == [4],
       "clicking cell 4 selects just it", page.js("window.__d.sel()"))
    page.click_at(p99["x"], p99["y"], modifiers=8)      # shift
    ok(page.js("window.__d.sel()") == list(range(4, 100)),
       "shift-clicking 99 after 4 selects every frame from 4 to 99, both ends "
       "included",
       "%d cells: %s..." % (page.js("S.sel.size"),
                            page.js("window.__d.sel()")[:8]))
    ok(page.js("document.getElementById('loopSelLabel').textContent") ==
       "play selection only (96)",
       "the play-selection box names how many frames are selected",
       page.js("document.getElementById('loopSelLabel').textContent"))
    ok(page.js("S.sel.size") == 96,
       "96 frames, not the 12-cell rectangle between them",
       page.js("S.sel.size"))

    page.js("S.sel.clear(); S.anchorCell = 0; syncSel()")
    page.click_at(p99["x"], p99["y"])
    page.click_at(p4["x"], p4["y"], modifiers=8)
    ok(page.js("window.__d.sel()") == list(range(4, 100)),
       "and the same set when the two clicks are the other way round (99 then 4)",
       page.js("window.__d.sel()")[:8])

    # the anchor holds on the cell that was *plain*-clicked (99 here), so a later
    # shift-click ranges from there rather than from the shift-click's target
    page.js("S.sel.clear(); syncSel()")     # keep the anchor, drop the selection
    ok(page.js("S.anchorCell") == 99,
       "a shift-click leaves the anchor on the plain-clicked cell, not on the "
       "cell it ranged to", page.js("S.anchorCell"))
    page.click_at(p4["x"], p4["y"], modifiers=8)
    ok(page.js("window.__d.sel()") == list(range(4, 100)),
       "so the next shift-click ranges from that anchor again",
       page.js("window.__d.sel()")[:8])

    page.js("S.sel.clear(); S.anchorCell = 4; syncSel()")
    page.js("""
      (() => {
        const s = document.getElementById('strip');
        s.children[99].dispatchEvent(new MouseEvent('click',
          {bubbles: true, shiftKey: true}));
        return true;
      })()
    """)
    ok(page.js("window.__d.sel()") == list(range(4, 100)),
       "and the timeline's shift-click gives the identical set, from the same "
       "helper", page.js("window.__d.sel()")[:8])

    # An anchor left over from a longer sheet points past the end of this one.
    # Nothing out of range may reach the selection: it would be sent to the
    # server as a cell index this document does not have.
    page.js("S.sel.clear(); S.anchorCell = 9999; syncSel()")
    page.click_at(p99["x"], p99["y"], modifiers=8)
    sel = page.js("window.__d.sel()")
    ok(sel and all(0 <= k < n_all for k in sel),
       "an anchor past the end of the document cannot select a cell that is not "
       "there", "%s of %d" % (sel[:8], n_all))
    page.js("S.sel.clear(); S.anchorCell = 0; syncSel()")
    page.js("window.__d.setView(0.25, 4, 4)")

    print("\n=== 4. a marquee drag starting ON a cell selects a block ===")
    # This is the case the naive implementation gets wrong: committing to
    # "click" on pointerdown makes it impossible to rubber-band from inside the
    # sheet, which is where you always start.
    page.js("S.sel.clear(); syncSel()")
    a = page.js("window.__d.sheetScreen(10, 10)")
    # rows 0..2, so the end point has to be past y = 2*640, not before it
    b = page.js("window.__d.sheetScreen(3 * 512 - 10, 3 * 640 - 10)")
    page.drag([(a["x"], a["y"]), ((a["x"]+b["x"])/2, (a["y"]+b["y"])/2),
               (b["x"], b["y"])])
    sel = page.js("window.__d.sel()")
    ok(sel == [0, 1, 2, 31, 32, 33, 62, 63, 64],
       "the marquee selected the 3x3 block 0-2 / 31-33 / 62-64", sel)
    ok(page.js("window.__d.timelineSel()") == 9,
       "the timeline highlights the same 9 cells", page.js("window.__d.timelineSel()"))

    # and a press-release with no movement must still be a click, not an empty
    # marquee that wipes the selection
    p = page.js("window.__d.cellScreen(%d)" % c_b)
    page.click_at(p["x"], p["y"])
    ok(page.js("window.__d.sel()") == [c_b],
       "a zero-movement press is still a click", page.js("window.__d.sel()"))

    # a ctrl-drag adds to the selection it started from
    page.js("S.sel.clear(); S.sel.add(0); S.anchorCell = 0; syncSel()")
    a = page.js("window.__d.sheetScreen(600, 10)")
    b = page.js("window.__d.sheetScreen(1000, 200)")
    page.drag([(a["x"], a["y"]), (b["x"], b["y"])], modifiers=2)
    sel = page.js("window.__d.sel()")
    ok(0 in sel and len(sel) > 1,
       "a ctrl-drag adds to the existing selection", sel)

    print("\n=== 5. an op form drives a real operation ===")
    # Section 3/4 already proved the selection paths, so pin a known selection
    # here and assert the op's message against it.
    page.js("S.sel = new Set([0,1,2,31,32,33,62,63,64]); S.anchorCell = 0; syncSel()")
    n_sel = page.js("S.sel.size")
    ok(page.js("!!window.__d.opNode('Nudge')"), "the Nudge op form exists")
    h_before = page.js("window.__d.hashCell(0)", await_promise=True)
    rev_before = page.js("window.__d.rev()")
    # the DOM path: type into the form's own controls, then press its Apply
    page.js("""
      (() => {
        const inp = window.__d.opInputs('Nudge');
        inp[0].value = 6; inp[0].dispatchEvent(new Event('input', {bubbles:true}));
        inp[1].value = 4; inp[1].dispatchEvent(new Event('input', {bubbles:true}));
        return inp.length;
      })()
    """)
    ok(page.js("window.__d.opInputs('Nudge').length") == 3,
       "Nudge renders 3 controls (dx, dy, wrap)",
       page.js("window.__d.opInputs('Nudge').length"))
    page.js("window.__d.applyOp('Nudge')")
    page.wait_for("window.__d.busy() && S.doc.rev > %d" % rev_before,
                  "the Nudge op to complete", timeout=60)
    ok(page.js("window.__d.rev()") > rev_before, "the document revision advanced")
    h_after = page.js("window.__d.hashCell(0)", await_promise=True)
    ok(h_after != h_before, "cell 0's PNG really changed on the server",
       "%s -> %s" % (h_before, h_after))
    ok(page.js("window.__d.cellrev(0)") > 0, "cell 0's revision bumped")
    ok(page.js("window.__d.cellrev(20)") == 0,
       "an unselected cell kept its revision", page.js("window.__d.cellrev(20)"))
    msg = page.js("document.getElementById('status').textContent")
    ok(msg == "Nudge: %d cells" % n_sel,
       "the status line reports exactly the %d selected cells" % n_sel, msg)
    ok(page.js("document.getElementById('undo').disabled") is False,
       "the undo button became enabled")
    ok(page.js("(S.doc.journal || []).join('\\n')").strip() == "offset",
       "the op history recorded it", page.js("S.doc.journal"))

    print("\n=== 6. the undo button restores the bytes ===")
    page.js("document.getElementById('undo').click()")
    page.wait_for("window.__d.busy() && document.getElementById('undo').disabled === true",
                  "undo to complete", timeout=60)
    h_undone = page.js("window.__d.hashCell(0)", await_promise=True)
    ok(h_undone == h_before, "undo restored cell 0's exact PNG bytes",
       "%s vs %s" % (h_undone, h_before))
    page.js("document.getElementById('redo').click()")
    page.wait_for("window.__d.busy() && document.getElementById('redo').disabled === true",
                  "redo to complete", timeout=60)
    h_redone = page.js("window.__d.hashCell(0)", await_promise=True)
    ok(h_redone == h_after, "redo put the edit back", "%s vs %s" % (h_redone, h_after))

    print("\n=== 7. the pencil paints through the canvas ===")
    page.js("""
      (() => {
        document.getElementById('toolPencil').click();
        document.getElementById('brushSize').value = 30;
        document.getElementById('brushAlpha').value = 255;
        document.getElementById('brushColor').value = '#ff00ff';
        return S.tool;
      })()
    """)
    ok(page.js("S.tool") == "pencil", "the pencil tool is active", page.js("S.tool"))
    ok(page.js("document.getElementById('toolPencil').classList.contains('on')"),
       "the toolbar highlights it")
    # clear the selection first so the timeline readout is unambiguous
    page.js("S.sel.clear(); syncSel()")
    page.js("window.__d.setView(0.25, 4, 4)")
    p0 = page.js("window.__d.sheetScreen(300, 320)")     # inside cell 0
    p1 = page.js("window.__d.sheetScreen(560, 320)")     # across into cell 1
    h0 = page.js("window.__d.hashCell(0)", await_promise=True)
    h1 = page.js("window.__d.hashCell(1)", await_promise=True)
    page.drag([(p0["x"], p0["y"]),
               ((p0["x"]+p1["x"])/2, p0["y"]),
               (p1["x"], p1["y"])])
    page.wait_for("window.__d.busy()", "the stroke to commit", timeout=90)
    n0 = page.js("window.__d.hashCell(0)", await_promise=True)
    n1 = page.js("window.__d.hashCell(1)", await_promise=True)
    ok(n0 != h0, "the stroke painted cell 0", "%s -> %s" % (h0, n0))
    ok(n1 != h1, "the stroke crossed into cell 1", "%s -> %s" % (h1, n1))
    ok(page.js("S.stroke.length") == 0, "the stroke buffer was flushed")
    # Read from the document, not from a panel: the op history panel was removed
    # from the UI, but the server still journals every op and the page still holds
    # it on S.doc, which is what the check is actually about.
    ok(page.js("(S.doc.journal || []).join('\\n')").strip().split("\n")[-1]
       == "paint", "the op history recorded paint", page.js("S.doc.journal"))
    ok(page.js("S.tool") == "pencil", "the pencil is still active after the stroke")

    # --- is the mark VISIBLE? ------------------------------------------------
    # Everything above only proves the bytes changed, and that is not enough. The
    # brush box used to be read as sheet pixels: on this 15872-px sheet at the
    # fitted zoom, the default 6 meant 0.32 px on screen. The stroke landed, the
    # hash changed, the journal recorded "paint" -- and the user saw nothing, and
    # reported that the pencil and eraser "are not working". A test that asserts
    # the hash moved cannot tell that apart from working software.
    #
    # So: fit the view (the zoom a user actually gets), use the DEFAULT brush, and
    # measure the mark in screen pixels.
    page.js("fitView()")
    page.js("""
      (() => {
        document.getElementById('brushSize').value = 6;
        document.getElementById('brushAlpha').value = 255;
        document.getElementById('brushColor').value = '#ff0000';
        document.getElementById('toolPencil').click();
        return true;
      })()
    """)
    zoom = page.js("S.view.s")
    r_sheet = page.js("brushRadius()")
    r_screen = r_sheet * zoom
    ok(abs(r_screen - 3.0) < 0.01,
       "the brush box means screen pixels: 6 -> a 3px screen radius at any zoom",
       "zoom %.4f: %s sheet px = %.3f screen px" % (zoom, r_sheet, r_screen))

    try:
        import io
        import numpy as np
        from PIL import Image
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        did = page.js("S.doc.id")

        def cell_px(i):
            # args.url, not a hardcoded port: the sheet's own cell route has to be
            # the server under test, or running the suite against a second
            # instance (a recycled server with new code, say) fetches the cell
            # from the first one and 404s on a document it has never seen.
            u = args.url + "/api/editor/%s/cell/%d.png" % (did, i)
            with opener.open(u, timeout=60) as fh:
                return np.array(Image.open(io.BytesIO(fh.read())).convert("RGBA"))

        b0 = cell_px(0)
        q0 = page.js("window.__d.sheetScreen(200, 320)")
        q1 = page.js("window.__d.sheetScreen(300, 320)")
        page.drag([(q0["x"], q0["y"]), (q1["x"], q1["y"])])
        page.wait_for("window.__d.busy()", "the visibility stroke to commit",
                      timeout=90)
        b1 = cell_px(0)
        d = (b0 != b1).any(axis=2)
        if not d.any():
            ok(False, "the default brush left a mark at all")
        else:
            ys, xs = np.where(d)
            thick = int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
            ok(thick * zoom >= 3.0,
               "the mark is visible at the fitted zoom, not a hairline",
               "%d sheet px thick = %.2f screen px" % (thick, thick * zoom))
            print("      zoom %.4f, brush 6px -> %.1f sheet px radius, mark %d "
                  "sheet px = %.2f screen px thick"
                  % (zoom, r_sheet, thick, thick * zoom))
    except ImportError:
        print("      (PIL/numpy missing -- could not measure the mark in screen px)")

    # --- is the mark in the PAGE, not just on the server? --------------------
    # Every check above reads the stroke back over HTTP: the hash route and this
    # section's own `cell_px` both fetch the cell from the server. That is why
    # neither of them could see the report that this section exists for. The cell
    # URL used to be constant, and a re-load of a URL the browser has already
    # decoded is answered from that bitmap -- so the stroke was on the server,
    # the journal recorded it, the hashes all moved, and the sheet, the preview
    # panel and the timeline went on painting the cell from before the edit. The
    # user's report ("pencil eraser and clear frame not working") is exactly
    # that, and it is invisible to anything that reads the cell over HTTP.
    # So: look at the page's own pixels, in all three places that draw them.
    page.js("S.cur = 0; renderPreview(); markTimelineCurrent(); draw();")
    seen = page.js("""
      (() => {
        const red = (d) => { let n = 0;
          for (let k = 0; k < d.length; k += 4)
            if (d[k] > 150 && d[k+1] < 110 && d[k+2] < 110) n++;
          return n; };
        const L = SHEET();
        const canvas = red(view.getContext('2d').getImageData(
          Math.round(S.view.tx) + 2, Math.round(S.view.ty) + 2,
          Math.round(L.cw * S.view.s) - 4, Math.round(L.ch * S.view.s) - 4).data);
        const p = document.getElementById('prev');
        const preview = red(p.getContext('2d')
          .getImageData(0, 0, p.width, p.height).data);
        const t = document.querySelectorAll('#strip canvas')[0];
        const thumb = red(t.getContext('2d')
          .getImageData(0, 0, t.width, t.height).data);
        return {canvas: canvas, preview: preview, thumb: thumb,
                src: (S.img.get(0) || {}).src || ''};
      })()
    """)
    ok(seen["canvas"] > 0,
       "the sheet canvas draws the stroke, not the cell it had already decoded",
       seen)
    ok(seen["preview"] > 0, "so does the frame preview panel", seen)
    ok(seen["thumb"] > 0, "and the timeline thumbnail", seen)
    ok("rev=" in seen["src"],
       "the cell URL carries its revision, so a reload cannot be answered stale",
       seen["src"])

    print("\n=== 7b. clear / copy / paste whole frames ===")
    did = page.js("S.doc.id")

    def cell_bytes(i):
        u = args.url + "/api/editor/%s/cell/%d.png" % (did, i)
        with urllib.request.urlopen(u, timeout=60) as fh:
            return fh.read()

    # clear frame 0 and confirm the server pixels are gone, not just that a
    # button exists
    page.js("setTool('select'); S.sel = new Set([0]); syncSel()")
    page.js("document.getElementById('clearFrame').click()")
    page.wait_for("window.__d.busy() && document.getElementById('status')"
                  ".textContent.startsWith('cleared')", "the clear", timeout=120)
    try:
        import io
        import numpy as np
        from PIL import Image
        ca = np.array(Image.open(io.BytesIO(cell_bytes(0))).convert("RGBA"))
        ok(not ca[..., 3].any(), "clear emptied every opaque pixel of frame 0")
    except ImportError:
        ok(True, "clear ran (PIL missing -- bytes not measured)")

    # copy frame 1, paste it into frame 3: the paste must be byte-exact, which
    # is why paste sends the cell PNG rather than canvas-read pixels
    src1 = cell_bytes(1)
    page.js("S.sel = new Set([1]); syncSel()")
    page.js("(async () => { await copyCells(); })()")
    page.wait_for("CLIP.length === 1", "the copy", timeout=60)
    page.js("S.sel = new Set([3]); syncSel()")
    page.js("pasteCells()")
    page.wait_for("window.__d.busy() && document.getElementById('status')"
                  ".textContent.startsWith('pasted')", "the paste", timeout=180)
    ok(cell_bytes(3) == src1, "pasting frame 1 into frame 3 is byte-exact")

    # Undo both edits, so the later verify section sees the document it saw
    # before this section existed (an emptied frame legitimately fails I4).
    for _ in range(2):
        page.js("document.getElementById('undo').click()")
        page.wait_for("window.__d.busy()", "the undo", timeout=120)

    print("\n=== 8. the timeline and the playback loop ===")
    ok(page.js("window.__d.stripCount()") == 124,
       "the timeline has a thumbnail per frame", page.js("window.__d.stripCount()"))
    ok(page.js("window.__d.previewInk()") > 0,
       "the preview canvas has ink on it", page.js("window.__d.previewInk()"))
    page.js("document.querySelectorAll('#strip canvas')[40].click()")
    ok(page.js("S.cur") == 40, "clicking a thumbnail moves the frame", page.js("S.cur"))
    ok(page.js("document.getElementById('frameLabel').textContent") == "frame 40 / 123",
       "the frame label updated",
       page.js("document.getElementById('frameLabel').textContent"))
    ok(page.js("document.getElementById('scrub').value") == "40",
       "the scrubber followed")

    page.js("document.getElementById('play').click()")
    t = page.wait_for("S.cur !== 40", "playback to advance", timeout=20)
    ok(True, "playback advanced off frame 40 in %.1fs" % t)
    page.js("document.getElementById('play').click()")
    frozen = page.js("S.cur")
    time.sleep(1.0)
    ok(page.js("S.cur") == frozen, "pause actually pauses", page.js("S.cur"))

    # 8b. "play selection only" is not only a playback mode: with it on, the seek
    #     has no business landing outside the frames that are played. The scrub
    #     becomes a position in the playback order -- for a selection of {0, 23} a
    #     slider spanning 0..23 spends 22 of its 24 stops on frames the preview
    #     never shows, which is what was reported: play the selection, and the
    #     seek still scrubs the whole sheet.
    print("\n=== 8b. the seek stays inside the played frames ===")
    page.js("document.getElementById('selNone').click()")
    page.js("""(() => {
      S.sel.clear(); S.sel.add(0); S.sel.add(23); S.anchorCell = 0; S.cur = 0;
      document.getElementById('loopSel').checked = false;
      syncSel(); renderPreview(); draw(); return S.cur;
    })()""")
    off = page.js("({max: Number(document.getElementById('scrub').max), "
                  "value: Number(document.getElementById('scrub').value), cur: S.cur})")
    ok(off["max"] == 123 and off["value"] == off["cur"],
       "unticked: the scrub spans the document and its value is the frame index", off)
    rect = page.js("(() => {const r = document.getElementById('loopSel')"
                   ".getBoundingClientRect();"
                   "return [r.left + r.width/2, r.top + r.height/2];})()")
    page.click_at(rect[0], rect[1])
    on = page.js("({checked: document.getElementById('loopSel').checked, "
                 "max: Number(document.getElementById('scrub').max), "
                 "value: Number(document.getElementById('scrub').value), cur: S.cur})")
    ok(on["checked"] is True, "a real click ticks the box", on)
    ok(on["max"] == 1, "ticked: the scrub spans the two played frames, not 124", on)
    ok(on["cur"] == 0, "and it is pointing at the first of them", on)

    # every stop the slider can reach has to be a played frame -- the sweep is what
    # makes this a claim about the control rather than about two sample drags
    sweep = page.js("""(() => {
      const el = document.getElementById('scrub'), out = [];
      const max = Number(el.max);
      for (let v = 0; v <= max; v++){
        el.value = String(v);
        el.dispatchEvent(new Event('input', {bubbles: true}));
        out.push([v, S.cur]);
      }
      return out;
    })()""")
    ok(all(cur in (0, 23) for _v, cur in sweep),
       "no stop of the scrub reaches a frame that is not played", sweep)
    ok([cur for _v, cur in sweep] == [0, 23],
       "and the stops are the played frames, in playback order", sweep)

    # a real drag to the far end of the slider must land on the last PLAYED frame,
    # not on frame 123 -- the mouse is where the bug showed
    srect = page.js("(() => {const r = document.getElementById('scrub')"
                    ".getBoundingClientRect();"
                    "return [r.left, r.top + r.height/2, r.right, r.width];})()")
    page.mouse("mouseMoved", srect[0] + 2, srect[1], "none", 0)
    page.mouse("mousePressed", srect[0] + 2, srect[1], "left", 1)
    page.mouse("mouseMoved", srect[2] - 1, srect[1], "left", 1)
    page.mouse("mouseReleased", srect[2] - 1, srect[1], "left", 0)
    drag_end = page.js("({cur: S.cur, value: Number(document.getElementById('scrub').value)})")
    ok(drag_end["cur"] == 23,
       "dragging the scrub to the end lands on the last played frame", drag_end)

    # and the arrow keys walk the selection rather than the document
    page.js("document.activeElement && document.activeElement.blur()")
    page.key("ArrowLeft")
    ok(page.js("S.cur") == 0, "ArrowLeft steps to the previous played frame", page.js("S.cur"))
    page.key("ArrowRight")
    ok(page.js("S.cur") == 23, "ArrowRight steps to the next one", page.js("S.cur"))
    page.key("ArrowRight")
    ok(page.js("S.cur") == 23, "and stops at the end instead of walking into the "
       "unplayed frames", page.js("S.cur"))

    # playback itself only ever lands on played frames
    page.js("document.getElementById('play').click()")
    time.sleep(1.2)
    seen = set()
    for _ in range(12):
        seen.add(page.js("S.cur"))
        time.sleep(0.06)
    page.js("document.getElementById('play').click()")
    ok(seen <= {0, 23}, "playback of the selection never shows another frame", seen)

    # unticking it gives the document back
    page.click_at(rect[0], rect[1])
    back = page.js("({checked: document.getElementById('loopSel').checked, "
                   "max: Number(document.getElementById('scrub').max), "
                   "value: Number(document.getElementById('scrub').value), cur: S.cur})")
    ok(back["max"] == 123 and back["value"] == back["cur"],
       "unticking it hands the whole document back to the scrub", back)
    page.js("document.activeElement && document.activeElement.blur()")

    # 8c. The player's fps and speed are the player's own. "player should have
    #     own fps and speed": they drive the preview and reach nothing else. The
    #     document's fps is the Grid panel's, and that is the one the sidecar
    #     carries. They used to be one control -- the tick read #fps -- so setting
    #     the document's frame rate silently changed how the preview played, and
    #     there was no way to preview a 24 fps sheet at 12 without editing the
    #     document's own rate.
    #
    #     Nothing here opens another document: section 10 asserts on the cells
    #     the earlier sections edited, and a reopen would throw them away. The
    #     "opening a sheet hands it back" half of the behaviour is section 16h.
    print("\n=== 8c. the player owns its fps and speed ===")
    panels = page.js("""(() => {
      const f = id => { const el = document.getElementById(id);
        const p = el && el.closest('details.panel');
        return p ? p.querySelector('summary').textContent.trim() : null; };
      return {playFps: f('playFps'), playSpeed: f('playSpeed'), fps: f('fps')};
    })()""")
    ok(panels["playFps"] == "Playback" and panels["playSpeed"] == "Playback",
       "the player's fps and speed both live in the Playback panel", panels)
    ok(panels["fps"] == "Grid",
       "and the document's fps lives in the Grid panel, on its own", panels)

    # On open the player follows the document, so a sheet previews at its own
    # rate. Compared against S.doc.meta.fps rather than the literal 24, so a
    # hardcoded default that happened to differ would fail here.
    _follow = page.js("({own: S.playFpsOwn, "
                      "player: Number(document.getElementById('playFps').value), "
                      "doc: S.doc.meta.fps})")
    ok(_follow["own"] is False and _follow["player"] == _follow["doc"],
       "opening a sheet starts the player at the document's own fps", _follow)

    _own = page.js("""(() => {
      const el = document.getElementById('playFps');
      el.value = '77';
      el.dispatchEvent(new Event('input', {bubbles: true}));
      return [Number(el.value), S.playFpsOwn];
    })()""")
    ok(_own == [77, True], "typing in the player's fps takes it over", _own)

    # Changing the DOCUMENT's fps must not drag the player with it. Driven
    # through the op route rather than the Apply metadata button on purpose: that
    # button now writes the sidecar (section 16h), and this document's sidecar is
    # a real asset in the user's walk/ folder.
    page.js("""(async () => {
      const r = await api('/api/editor/' + S.doc.id + '/op',
        {name: 'set_meta', sel: [],
         args: {fps: 17, play_mode: 'loop', trim_to: 0, anchor: 'none',
                blend: 'straight', name: S.doc.meta.name}});
      applyDoc(r.doc); renderLayout();
      return r.doc.meta.fps;
    })()""", await_promise=True)
    _dec = page.js("({doc: S.doc.meta.fps, "
                   "grid: Number(document.getElementById('fps').value), "
                   "player: Number(document.getElementById('playFps').value)})")
    ok(_dec["doc"] == 17 and _dec["grid"] == 17,
       "the document's fps really changed, so the next check is not vacuous",
       _dec)
    ok(_dec["player"] == 77,
       "and the player's fps did not move with it -- the two are independent",
       _dec)

    def runs_out(secs=1.0):
        """Play from frame 0 in `once` mode; report whether it reached the end.

        A boolean rather than a frame count on purpose. A count depends on how
        long the sleep actually took and wraps at 124, so it is a flaky thing to
        assert on -- the first version of this measured S.cur and would have
        failed whenever the sleep overshot. "Did it get through the sheet" has an
        enormous margin either way: at 240 fps the 124 frames take about half a
        second, at 1 fps they take two minutes.
        """
        page.js("""(() => {
          document.getElementById('playMode').value = 'once';
          S.cur = 0; S.acc = 0; S.lastT = 0; S.dir = 1; S.playing = true;
          document.getElementById('play').textContent = 'Pause';
          return true;
        })()""")
        time.sleep(secs)
        done = page.js("S.playing") is False
        page.js("S.playing = false; "
                "document.getElementById('play').textContent = 'Play'")
        return done

    # The rate has to come from the player, not from the document. Both halves
    # put the document's fps and the player's fps at opposite ends, so a tick that
    # read the wrong one fails both.
    page.js("""(() => {
      document.getElementById('playFps').value = '240';
      document.getElementById('playSpeed').value = '1';
      document.getElementById('fps').value = '1';
      return true;
    })()""")
    ok(runs_out(),
       "at 240 player fps the preview plays the whole sheet through, even "
       "though the document says 1 fps")

    page.js("""(() => {
      document.getElementById('playFps').value = '1';
      document.getElementById('playSpeed').value = '1';
      document.getElementById('fps').value = '240';
      return true;
    })()""")
    ok(not runs_out(),
       "and at 1 player fps it does not, even though the document says 240 fps")

    # Speed is a multiplier on the player's fps, and is still the player's own.
    # 60 fps alone does not clear 124 frames in a second; 60 x4 does.
    page.js("""(() => {
      document.getElementById('playFps').value = '60';
      document.getElementById('playSpeed').value = '1';
      document.getElementById('fps').value = '1';
      return true;
    })()""")
    ok(not runs_out(),
       "60 player fps at 1x does not get through the sheet in a second")
    page.js("document.getElementById('playSpeed').value = '4'")
    ok(runs_out(),
       "and the same 60 at 4x does -- speed multiplies the player's fps")

    # Put the document's own fps back. The sidecar is not written anywhere in
    # this section, so this is only about leaving the rest of the run as it was
    # found -- section 13 and 14 save this document to a copy.
    page.js("""(async () => {
      const r = await api('/api/editor/' + S.doc.id + '/op',
        {name: 'set_meta', sel: [],
         args: {fps: 24, play_mode: 'loop', trim_to: 0, anchor: 'none',
                blend: 'straight', name: S.doc.meta.name}});
      applyDoc(r.doc); renderLayout();
      return r.doc.meta.fps;
    })()""", await_promise=True)
    page.js("""(() => {
      document.getElementById('playMode').value = 'loop';
      document.getElementById('playSpeed').value = '1';
      S.playing = false;
      document.getElementById('play').textContent = 'Play';
      return true;
    })()""")
    ok(page.js("S.doc.meta.fps") == 24,
       "the document's fps is put back where it was found",
       page.js("S.doc.meta.fps"))

    print("\n=== 9. selection helpers and zoom ===")
    page.js("document.getElementById('selAll').click()")
    ok(page.js("S.sel.size") == 124, "All selects every frame", page.js("S.sel.size"))
    page.js("document.getElementById('selNone').click()")
    ok(page.js("S.sel.size") == 0, "None clears it")
    page.js("document.getElementById('selOdd').click()")
    ok(page.js("S.sel.size") == 62, "Odd selects 62 of 124", page.js("S.sel.size"))
    page.js("document.getElementById('selInvert').click()")
    ok(page.js("S.sel.size") == 62, "Invert swaps to the other 62")
    page.js("document.getElementById('selEven').click()")
    ok(page.js("window.__d.sel()[0]") == 0, "Even starts at frame 0")
    page.js("document.getElementById('zoom100').click()")
    ok(abs(page.js("S.view.s") - 1) < 1e-6, "1:1 sets zoom to 100%", page.js("S.view.s"))
    page.js("document.getElementById('fit').click()")
    ok(page.js("S.view.s") < 1, "Fit zooms back out", page.js("S.view.s"))

    print("\n=== 10. verification from the UI (on an edited document) ===")
    page.js("document.getElementById('verify2').click()")
    page.wait_for("['PASSED','FAILED'].includes(document.getElementById('vbadge').textContent)",
                  "the verify badge", timeout=120)
    badge = page.js("document.getElementById('vbadge').textContent")
    vlog = page.js("document.getElementById('vlog').textContent")
    ok(badge in ("PASSED", "FAILED"), "the verify badge rendered", badge)
    ok("I2 no backdrop leak" in vlog, "the report includes I2")
    ok("modified" in vlog and "not applicable" in vlog,
       "and declares that I1/I4 lost their reference for the edited cells")
    # The premultiply-smell line is conditional -- it only prints when some cell
    # actually looks premultiplied, and this sheet's cells are straight alpha, so
    # it is absent here. test_editor.py section 17 covers the branch itself on
    # constructed documents (a matte that must not trip it, a premultiplied cell
    # that must, and the stray-transparent-RGB regression); this driver asserts
    # the informational tail that IS present on a real sheet.
    ok("faint fringe" in vlog, "and prints the informational fringe count")
    print("      badge: %s" % badge)
    for line in vlog.splitlines()[:5]:
        print("      " + line)

    print("\n=== 11. grid changes: the texture cap, both ways ===")
    # The walk sheet is 124 cells of 512x640, so at the 32768 px cap both halves
    # of the rule are reachable here: 4 columns is 2048 x 19840 and fits, 2
    # columns is 1024 x 39680 and does not. Asserting only the refusal would not
    # tell a raised cap apart from one that is not enforced at all.
    page.js("document.getElementById('cols').value = 2")
    page.js("document.getElementById('applyCols').click()")
    page.wait_for("document.getElementById('status').textContent.includes('texture limit')",
                  "the texture-cap refusal to be surfaced", timeout=30)
    ok(True, "a re-grid over the texture cap is refused, with the reason in #status")
    ok(page.js("S.doc.layout.columns") == 31,
       "the refused re-grid left the layout untouched",
       page.js("S.doc.layout.columns"))
    ok(page.js("document.getElementById('cols').value") == "31",
       "and the form was not left showing the rejected value")

    # The grid the old 16384 cap refused, now applied -- that is what raising the
    # cap means in practice, and it is the half a refusal-only test cannot see.
    page.js("document.getElementById('cols').value = 4")
    page.js("document.getElementById('applyCols').click()")
    page.wait_for("window.__d.busy() && S.doc.layout.columns === 4",
                  "the 4-column re-grid", timeout=60)
    ok(page.js("S.doc.layout.rows") == 31,
       "2048 x 19840 is under the cap, so 4 columns applies where 2 is refused",
       page.js("S.doc.layout.columns"))

    page.js("document.getElementById('autoCols').click()")
    page.wait_for("window.__d.busy() && S.doc.layout.columns === 31",
                  "regrid_auto", timeout=60)
    ok(page.js("S.doc.layout.columns") == 31,
       "auto columns puts it back to 31 -- the grid that keeps the sheet smallest",
       page.js("S.doc.layout.columns"))

    # now a sheet where a re-grid is actually possible: 16 frames of 100x192
    _r = open_by_name(page, "A01_torch")
    ok(_r == "opened", "the torch sheet opened by name", _r)
    page.wait_for("!!(S.doc) && S.doc.n === 16", "the torch sheet to open",
                  timeout=60)
    page.wait_for("S.img.size === S.doc.n", "its cells to load", timeout=60)
    ok(page.js("S.doc.layout.columns") == 8, "the torch sheet opens as 8x2",
       page.js("S.doc.layout.columns"))
    ok(page.js("document.getElementById('docname').textContent").startswith("A01_torch"),
       "the header switched to the new sheet",
       page.js("document.getElementById('docname').textContent"))
    ok(page.js("window.__d.stripCount()") == 16, "the timeline rebuilt for 16 cells",
       page.js("window.__d.stripCount()"))

    page.js("document.getElementById('cols').value = 4")
    page.js("document.getElementById('applyCols').click()")
    page.wait_for("window.__d.busy() && S.doc.layout.columns === 4",
                  "the 4-column re-grid", timeout=60)
    ok(page.js("S.doc.layout.rows") == 4, "4 columns gives 4 rows",
       page.js("S.doc.layout.rows"))
    ok(page.js("S.img.size") == 16,
       "the cell cache survived a pure re-grid (indices are unchanged)",
       page.js("S.img.size"))
    ok(page.js("window.__d.stripCount()") == 16, "the timeline still has 16 cells")
    ok(page.js("document.getElementById('dims').textContent").startswith("400"),
       "the sheet dimensions updated to 400 px wide",
       page.js("document.getElementById('dims').textContent"))

    # an impossible column count on THIS sheet must also be refused
    page.js("document.getElementById('cols').value = 3")
    page.js("document.getElementById('applyCols').click()")
    page.wait_for("document.getElementById('status').textContent.includes('divide')",
                  "the full-grid refusal", timeout=30)
    ok(True, "a column count that does not divide the frame count is refused")
    ok(page.js("S.doc.layout.columns") == 4,
       "and the layout is still the last valid one",
       page.js("S.doc.layout.columns"))
    # Same guard as above, on the other refusal path: a refusal must restore the
    # form, whichever reason it was refused for.
    ok(page.js("document.getElementById('cols').value") == "4",
       "and the form was restored to the last valid value here too",
       page.js("document.getElementById('cols').value"))

    print("\n=== 12. pruning frames from the real UI ===")
    # Section 11 left the torch sheet at 16 frames on a 4x4 grid.
    #
    # The tool has to be set explicitly: section 7 selected the pencil and the
    # tool persists, so a "click" here would paint a stroke instead of selecting
    # a cell. Nothing in the product is wrong -- painting when the pencil is
    # active is the whole point -- but the driver must not inherit a precondition
    # from a section it happens to run after.
    page.js("document.getElementById('toolSelect').click()")
    ok(page.js("S.tool") == "select", "the select tool is active", page.js("S.tool"))
    page.js("document.getElementById('fit').click()")
    ok(page.js("S.doc.n") == 16, "starting from the 16-frame torch sheet",
       page.js("S.doc.n"))

    vis = page.js("window.__d.visibleCells(6)")
    ok(len(vis) >= 3, "found %d clickable cells" % len(vis), vis)
    drop_a, drop_b = vis[0], vis[2]
    p = page.js("window.__d.cellScreen(%d)" % drop_a)
    page.click_at(p["x"], p["y"])
    p = page.js("window.__d.cellScreen(%d)" % drop_b)
    page.click_at(p["x"], p["y"], modifiers=2)          # ctrl -> add
    ok(page.js("window.__d.sel()") == sorted([drop_a, drop_b]),
       "two cells selected with real clicks", page.js("window.__d.sel()"))

    ok(page.js("window.__d.applyOp('Drop selected frames')") == "clicked",
       "the Drop selected frames form applied")
    page.wait_for("window.__d.busy() && S.doc.n === 14",
                  "the two frames to be dropped", timeout=60)
    ok(page.js("S.doc.n") == 14, "the sheet went from 16 to 14 frames",
       page.js("S.doc.n"))
    ok(page.js("S.doc.layout.columns * S.doc.layout.rows") == 14,
       "and the grid stayed full",
       "%s x %s" % (page.js("S.doc.layout.columns"), page.js("S.doc.layout.rows")))
    msg = page.js("document.getElementById('status').textContent")
    ok("16 -> 14 frames" in msg, "the status reports the frame-count change", msg)
    ok(page.js("window.__d.stripCount()") == 14,
       "the timeline rebuilt for 14 frames", page.js("window.__d.stripCount()"))

    # Undo across a frame-count change: this is the path that used to raise
    # KeyError server-side, because the snapshot was sized from the post-op count.
    page.js("document.getElementById('undo').click()")
    page.wait_for("window.__d.busy() && S.doc.n === 16",
                  "undo to restore the dropped frames", timeout=60)
    ok(page.js("S.doc.n") == 16, "undo brought the dropped frames back",
       page.js("S.doc.n"))
    ok(page.js("window.__d.stripCount()") == 16, "and the timeline with them")

    # An empty selection means "every cell" for every other op. Here that would
    # delete the animation, so it has to be refused.
    page.js("S.sel.clear(); syncSel()")
    page.js("window.__d.applyOp('Drop selected frames')")
    page.wait_for("document.getElementById('status').textContent"
                  ".includes('nothing would be left')",
                  "the empty-selection refusal", timeout=30)
    ok(page.js("S.doc.n") == 16,
       "an empty selection is refused and nothing is dropped", page.js("S.doc.n"))

    # pick_best, through its own form
    page.js("""
      (() => {
        const inp = window.__d.opInputs('Keep the best N frames');
        inp[0].value = 8; inp[0].dispatchEvent(new Event('input', {bubbles:true}));
        return inp.length;
      })()
    """)
    ok(page.js("window.__d.opInputs('Keep the best N frames').length") == 2,
       "pick_best renders 2 controls (N, metric)",
       page.js("window.__d.opInputs('Keep the best N frames').length"))
    page.js("window.__d.applyOp('Keep the best N frames')")
    page.wait_for("window.__d.busy() && S.doc.n === 8",
                  "pick_best to run", timeout=90)
    ok(page.js("S.doc.n") == 8, "pick_best kept 8 of 16", page.js("S.doc.n"))
    msg = page.js("document.getElementById('status').textContent")
    ok("lowest kept score" in msg,
       "the status reports the scores the cut was made on", msg)
    print("      " + msg)
    ok(page.js("window.__d.stripCount()") == 8, "the timeline followed",
       page.js("window.__d.stripCount()"))

    print("\n=== 13. the save panel, and its overwrite box ===")
    # The server refuses to write over the sheet this document was loaded from,
    # so the escape hatch has to exist in the page -- a refusal with no way past
    # it is a dead end. What matters most is that the box defaults OFF.
    ok(page.js("!!document.getElementById('overwrite')"),
       "the save panel has an overwrite opt-in")
    ok(page.js("document.getElementById('overwrite').checked") is False,
       "and it is unchecked by default, so the safe path is the default one")
    ok("loaded from" in page.js(
        "document.getElementById('overwrite').parentElement.textContent"),
       "and it says what it would overwrite")

    # Drive a real save down the normal path (blank output folder, gif off to
    # keep it quick). This exercises doSave() end to end, including the fact
    # that it reads the new box without throwing.
    page.js("document.getElementById('wantGif').checked = false")
    page.js("document.getElementById('outDir').value = ''")
    page.js("document.getElementById('save2').click()")
    page.wait_for(
        "document.getElementById('saveout').textContent.includes('saved to')"
        " || !!document.querySelector('#saveout .err')",
        "the save to report back", timeout=180)
    out = page.js("document.getElementById('saveout').textContent")
    ok("saved to" in out, "a save with a blank folder succeeds", out[:200])
    ok("editor_" in out,
       "and lands in the app's runs folder, not next to the source", out[:200])

    print("\n=== 13b. export only the selected frames, then edit them ===")
    # The two halves of this option are "write fewer frames" and "let me carry on
    # editing those frames". The second is why the result panel carries an edit
    # action at all -- otherwise a subset export leaves you holding a file and
    # the original document, with no way back into what you just wrote.
    #
    # The document here is the 8 frames pick_best left; a 3-frame subset of it has
    # no 4-wide grid, so this also exercises the re-grid on the export path.
    n_all = page.js("S.doc.n")
    page.js("S.sel.clear(); syncSel()")
    ok(page.js("document.getElementById('onlySel').disabled") is True,
       "the export box is disabled while nothing is selected")
    ok("nothing selected" in page.js(
        "document.getElementById('onlySelLabel').textContent"),
       "and says so, rather than silently meaning 'export everything'",
       page.js("document.getElementById('onlySelLabel').textContent"))

    page.js("S.sel = new Set([1, 2, 3]); syncSel()")
    ok(page.js("document.getElementById('onlySel').disabled") is False,
       "selecting frames enables it")

    # The label must not promise something the box is not doing. With the option
    # OFF the count belongs to the *selection*, not to the save, and the hint
    # beside the button says what the save will actually write -- in both states.
    # Reported as "i select 12 to 66 check only the selected 55 of 100 frames ...
    # but when i click edit not 55 frames full 100 frames returned": the count in
    # the label was read as a description of Save, the box was left unticked (it
    # is off by default), Save wrote all 100, and edit reopened all 100.
    page.js("document.getElementById('onlySel').checked = false")
    page.js("document.getElementById('onlySel')"
            ".dispatchEvent(new Event('change'))")
    lab_off = page.js("document.getElementById('onlySelLabel').textContent")
    ok("selected)" in lab_off,
       "with the box off, the label talks about the selection rather than about "
       "what will be written", lab_off)
    hint_off = page.js("document.getElementById('saveHint').textContent")
    ok(("all %d frames" % n_all) in hint_off,
       "and the hint beside Save says it would write every frame", hint_off)

    page.js("document.getElementById('onlySel').checked = true")
    page.js("document.getElementById('onlySel')"
            ".dispatchEvent(new Event('change'))")
    lab_on = page.js("document.getElementById('onlySelLabel').textContent")
    ok(("3 of %d" % n_all) in lab_on,
       "ticking it states the count as what the save will write", lab_on)
    hint_on = page.js("document.getElementById('saveHint').textContent")
    ok("the 3 selected frames" in hint_on and "not the other" in hint_on,
       "and the hint switches to the subset", hint_on)

    page.js("document.getElementById('outDir').value = ''")
    id_before = page.js("S.doc.id")
    page.js("document.getElementById('save2').click()")
    page.wait_for(
        "document.getElementById('saveout').textContent.includes('saved to')"
        " || !!document.querySelector('#saveout .err')",
        "the subset save to report back", timeout=180)
    out = page.js("document.getElementById('saveout').textContent")
    ok("saved to" in out, "the subset export succeeds", out[:200])
    ok(("3 of %d frames" % n_all) in out,
       "and the panel reports how many frames it actually wrote", out[:220])
    ok(page.js("S.doc.n") == n_all,
       "the open document still has every one of its frames", page.js("S.doc.n"))
    ok(page.js("S.doc.id") == id_before,
       "and it is still the same document, not a replaced one")

    # Every artifact is named, and each is a link when the app can serve it.
    names = page.js("[...document.querySelectorAll("
                    "'#saveout a.navlink, #saveout span.chip')]"
                    ".map(e => e.textContent)")
    ok(names == ["sheet", "sidecar", "preview"],
       "the panel names every artifact it wrote (gif was switched off)", names)

    ok(page.js("!!document.getElementById('editSaved')"),
       "and offers an edit action on the result")
    ok(page.js("document.getElementById('editSaved').textContent")
       == "edit the 3 saved frames",
       "which names the frame count, so it cannot be read as 'open the sheet I "
       "came from'",
       page.js("document.getElementById('editSaved').textContent"))
    page.js("document.getElementById('editSaved').click()")
    page.wait_for("S.busy === 0 && !!S.doc && S.doc.n === 3"
                  " && S.img.size === S.doc.n",
                  "the exported subset to open for editing", timeout=180)
    ok(page.js("S.doc.n") == 3,
       "clicking edit reopens the frames that were written, as the document",
       page.js("S.doc.n"))
    ok(page.js("S.doc.id") != id_before,
       "as its own document, so the one it was exported from is not what is "
       "being edited")
    box = page.js("document.getElementById('sheetPath').value")
    ok(box.endswith(".json") and os.path.isfile(box),
       "and the open-by-path box names the sidecar of the file that was written",
       box)
    ok(os.path.dirname(box) == os.path.dirname(page.js("S.doc.source")),
       "in the folder the export wrote to", (box, page.js("S.doc.source")))
    ok(page.js("S.doc.n") == page.js("S.doc.layout.columns")
       * page.js("S.doc.layout.rows"),
       "and the reopened sheet is on a full grid")
    # An open clears the selection -- section 14 depends on that being true, and
    # the export box has to follow it rather than hold a stale count.
    ok(page.js("S.sel.size") == 0,
       "the reopen cleared the selection, as any open does", page.js("S.sel.size"))
    ok(page.js("document.getElementById('onlySel').disabled") is True,
       "so the export box went back to disabled with it")
    ok(page.js("document.getElementById('onlySel').checked") is False,
       "and unticked, so the next save is not silently a subset")
    ok(page.js("document.getElementById('onlySelLabel').textContent")
       == "export only the selected frames (nothing selected)",
       "and the label matches",
       page.js("document.getElementById('onlySelLabel').textContent"))

    print("\n=== 14. the save refusal, driven through the page ===")
    # The refusal is the point of the guard, and only a real browser can show
    # that it reaches the panel instead of being swallowed.
    #
    # Done on a COPY, opened through the real open-by-path box. The op-layer
    # suite mutation-reverts this very guard (`save-clobbers-source`), so
    # pointing the output folder at a real sheet would let that mutation destroy
    # a real asset.
    n_now = page.js("S.doc.n")
    src_png = page.js("S.doc.source")
    box_now = page.js("document.getElementById('sheetPath').value")
    ok(bool(src_png) and os.path.isfile(src_png) and src_png.endswith(".png"),
       "the open document names the sheet it was loaded from", src_png)
    ok(bool(box_now) and os.path.isfile(box_now) and box_now.endswith(".json"),
       "and the open-by-path box names that sheet's sidecar", box_now)

    tmp = mkdtemp(prefix="drive14_")
    copy_sheet = os.path.join(tmp, "copy_sheet.png")
    # The PNG, not the box: the box holds the sidecar now, and the refusal this
    # section tests is about a *sheet* being overwritten.
    shutil.copyfile(src_png, copy_sheet)
    # A sidecar beside it, naming the copy. This used to be unnecessary -- the
    # section filled in the columns/rows boxes and opened the bare PNG with an
    # explicit grid -- but those boxes are gone from the panel, and the editor
    # opens a sheet through its sidecar. Same grid, same name, no clips: only the
    # sheet file differs, which is what the refusal below is about.
    with open(box_now, encoding="utf-8") as fh:
        copy_sc = json.load(fh)
    copy_sc["sheet"] = "copy_sheet.png"
    copy_sc["name"] = "copy"
    copy_sc.pop("audio", None)
    with open(os.path.join(tmp, "copy.json"), "w", encoding="utf-8") as fh:
        json.dump(copy_sc, fh)
    before = open(copy_sheet, "rb").read()

    # Wait on the document ID, not the frame count. The copy has the same frame
    # count as the sheet it came from, so `S.doc.n === n_now` is true *before*
    # the open lands -- the wait passes immediately, the old document stays
    # loaded, and the "refusal" never fires because there is nothing to collide
    # with. That is exactly how the first version of this section failed.
    id_before = page.js("S.doc.id")
    # The folder, so the sidecar in it is what supplies the grid. Opening the PNG
    # directly would work too -- the sidecar is found beside it -- but the folder
    # is the path a run leaves you with.
    page.js("document.getElementById('sheetPath').value = %s" % json.dumps(tmp))
    page.js("document.getElementById('openPath').click()")
    page.wait_for("!!S.doc && S.doc.id !== %s" % json.dumps(id_before),
                  "the copy to open through the open-by-path box", timeout=180)
    ok(page.js("S.doc.id") != id_before,
       "the copy opened through the open-by-path box (a new document id)",
       page.js("S.doc.id"))
    ok(page.js("S.doc.n") == n_now,
       "and it carries the same frame count as the sheet it was copied from",
       page.js("S.doc.n"))

    # Save it back over itself: same folder, same name.
    page.js("document.getElementById('metaName').value = 'copy'")
    page.js("document.getElementById('outDir').value = %s" % json.dumps(tmp))
    page.js("document.getElementById('overwrite').checked = false")
    page.js("document.getElementById('save2').click()")
    page.wait_for("!!document.querySelector('#saveout .err')",
                  "the refusal to reach the panel", timeout=180)
    msg = page.js("document.querySelector('#saveout .err').textContent")
    ok("refusing to overwrite the sheet" in msg,
       "the refusal reaches the panel, not just the server", msg[:220])
    ok("overwrite" in msg, "and it names the opt-in", msg[:220])
    print("      %s" % msg[:150])
    ok(open(copy_sheet, "rb").read() == before,
       "the sheet it would have destroyed is byte-identical")

    # ...and the box is what gets past it.
    #
    # Edit the document first, so "the copy changed" means the write landed
    # rather than "the bytes happened to differ". Section 13b leaves the document
    # byte-identical to the sheet it was opened from -- that is what a subset
    # export followed by "edit" produces -- and re-saving an unedited document
    # over its own file is byte-identical by definition, which would fail this
    # check for a reason that has nothing to do with the overwrite box.
    page.js("S.sel = new Set([0]); syncSel()")
    _r = page.js("window.__d.applyOp('Flip horizontal')")
    ok(_r == "clicked", "the pre-save edit applied through its real form", _r)
    # __d.busy() is an IDLE predicate (S.busy === 0 && no stroke buffered),
    # despite the name -- so it is true here, and the wait is for it to stay true
    # *with* the op recorded. `!__d.busy()` would be waiting for busy.
    page.wait_for("window.__d.busy() && S.doc.journal.length > 0",
                  "the edit to land", timeout=120)
    ok(page.js("S.doc.journal.length") > 0,
       "the document now has an edit in it to write",
       page.js("S.doc.journal"))

    # Re-type the name: applyDoc() re-renders the meta panel from the document on
    # every op, so the flip above discarded what was typed into it, and the save
    # would have written "<stem>_sheet_sheet.png" instead of the file under test.
    # Re-typing is what a user does anyway (name the file, then save). The
    # clobbering itself is a separate, pre-existing quirk -- it is noted in the
    # README rather than changed here.
    page.js("document.getElementById('metaName').value = 'copy'")

    page.js("document.getElementById('overwrite').checked = true")
    page.js("document.getElementById('save2').click()")
    page.wait_for(
        "document.getElementById('saveout').textContent.includes('saved to')"
        " || !!document.querySelector('#saveout .err')",
        "the overwrite save to report back", timeout=180)
    out2 = page.js("document.getElementById('saveout').textContent")
    ok("saved to" in out2, "ticking the box gets the save through", out2[:200])
    ok(open(copy_sheet, "rb").read() != before,
       "and it really did replace the copy -- the box is wired, not decorative")

    # The case above passes even when the box is broken, because the name box
    # holds 'copy' and the files are copy.json / copy_sheet.png: the derived
    # output name happens to be the loaded file. That is exactly how "when allow
    # overwriting the sheet this document was loaded from checked not writing on
    # same json" survived -- a real run folder has a sidecar whose filename does
    # not match the `name` inside it (smoke_api.json, name
    # "032137_px_00002_front_walk"). Rename the document so the derived name
    # cannot coincide, and then check the bytes on disk rather than the panel's
    # report.
    #
    # The output folder is CLEARED first, and that is the point rather than a
    # detail: blank is the box's own default ("blank = runs/editor_..."), and it
    # is the gesture that was reported. With a blank folder the server invented
    # runs/editor_<stamp>_<name>/ and wrote a brand-new pair there, so the ticked
    # save never went near the files it was loaded from. Measured on the running
    # server before the fix: dir=runs/editor_0921-181431_smoke12, and both loaded
    # files byte-identical afterwards.
    _sc_path = os.path.join(tmp, "copy.json")
    # Edit again first, and this is the same trap as the comment above one step
    # further on: the case above just saved this document over these very files,
    # so saving it again would be byte-identical *by definition* and "the bytes
    # differ" would fail for a reason that has nothing to do with where the write
    # went. The op re-renders the meta panel, so the name is re-typed below it.
    page.js("window.__d.applyOp('Flip vertical')")
    page.wait_for("window.__d.busy() && S.doc.journal.length > 1",
                  "the second edit to land", timeout=120)
    _sheet_bytes = open(copy_sheet, "rb").read()
    _sc_bytes = open(_sc_path, "rb").read()
    page.js("document.getElementById('outDir').value = ''")
    page.js("document.getElementById('metaName').value = 'renamed-not-the-file'")
    page.js("document.getElementById('overwrite').checked = true")
    page.js("document.getElementById('saveout').innerHTML = ''")
    page.js("document.getElementById('save2').click()")
    page.wait_for(
        "document.getElementById('saveout').textContent.includes('saved to')"
        " || !!document.querySelector('#saveout .err')",
        "the renamed overwrite save to report", timeout=180)
    _out3 = page.js("document.getElementById('saveout').textContent")
    ok("saved to" in _out3,
       "a ticked save goes through even when the name box is not the filename",
       _out3[:200])
    ok(open(copy_sheet, "rb").read() != _sheet_bytes,
       "and it replaced the sheet it was loaded from, whatever the name box says")
    ok(open(_sc_path, "rb").read() != _sc_bytes,
       "and the sidecar JSON it was loaded from, not a new one beside it")
    _stray = [f for f in os.listdir(tmp) if f.startswith("renamed-not-the-file")]
    ok(not _stray, "and nothing was written under the new name", _stray)

    # ...but a folder the user *did* name still wins. The box is worded as
    # permission ("allow..."), and the refusal it disables is about the
    # destination -- so redirecting to the source here would silently ignore an
    # explicit instruction and replace the original when a copy was asked for.
    # The two cases together are the whole rule, which is why both are pinned.
    _elsewhere = os.path.join(tmp, "elsewhere")
    os.makedirs(_elsewhere, exist_ok=True)
    _sheet_bytes2 = open(copy_sheet, "rb").read()
    page.js("document.getElementById('outDir').value = %s"
            % json.dumps(_elsewhere))
    page.js("document.getElementById('metaName').value = 'copy'")
    page.js("document.getElementById('overwrite').checked = true")
    page.js("document.getElementById('saveout').innerHTML = ''")
    page.js("document.getElementById('save2').click()")
    page.wait_for(
        "document.getElementById('saveout').textContent.includes('saved to')"
        " || !!document.querySelector('#saveout .err')",
        "the copy-to-a-named-folder save to report", timeout=180)
    ok(os.path.isfile(os.path.join(_elsewhere, "copy_sheet.png")),
       "ticked with a folder named, the copy lands in that folder",
       os.listdir(_elsewhere))
    ok(open(copy_sheet, "rb").read() == _sheet_bytes2,
       "and the sheet it was loaded from is left alone")
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n=== 15. the keyboard: shortcuts survive a toolbar box ===")
    # Setting the brush size is the first thing anyone does before drawing, and
    # it leaves focus in a number box. The handler used to bail out for ANY
    # input, so B and E then did nothing at all -- the other half of "the eraser
    # and pencil are not working", and invisible to every other layer: the
    # buttons work, the tools work, only the advertised key is dead.
    #
    # Set the precondition rather than inheriting it: this section runs after one
    # that opens a fresh document, so the undo stack is empty and a ctrl+z test
    # would prove nothing. Paint a stroke first.
    page.js("fitView()")
    page.js("setTool('pencil')")
    cc = page.js("window.__d.cellScreen(0)")
    page.drag([(cc["x"], cc["y"]), (cc["x"] + 12, cc["y"] + 12)])
    page.wait_for("window.__d.busy()", "the precondition stroke", timeout=90)
    ok(page.js("S.doc.undo") >= 1, "there is an undo entry to spend",
       page.js("S.doc.undo"))

    page.js("document.activeElement && document.activeElement.blur()")
    page.js("setTool('select')")
    page.key("b")
    ok(page.js("S.tool") == "pencil", "b picks the pencil from the page body",
       page.js("S.tool"))

    # now the case that was broken: focus sitting in the brush size box
    r = page.js("(() => { const e = document.getElementById('brushSize');"
                " const b = e.getBoundingClientRect();"
                " return {x: b.left + b.width/2, y: b.top + b.height/2}; })()")
    page.click_at(r["x"], r["y"])
    ok(page.js("document.activeElement.id") == "brushSize",
       "clicking the size box really does take focus",
       page.js("document.activeElement.id"))
    page.js("setTool('select')")
    page.key("e")
    ok(page.js("S.tool") == "erase",
       "e still picks the eraser while the size box has focus",
       page.js("S.tool"))
    page.key("b")
    ok(page.js("S.tool") == "pencil",
       "and b still picks the pencil", page.js("S.tool"))

    page.js("setTool('pencil')")
    u0 = page.js("S.doc.undo")
    page.key("z", ctrl=True)
    page.wait_for("window.__d.busy()", "the undo shortcut to settle", timeout=60)
    ok(page.js("S.doc.undo") == u0 - 1,
       "ctrl+z undoes while the size box has focus",
       "%s -> %s" % (u0, page.js("S.doc.undo")))

    page.key("Escape")
    ok(page.js("document.activeElement.tagName") != "INPUT",
       "Escape hands the keyboard back", page.js("document.activeElement.tagName"))

    # ...and the negative that keeps the fix honest: a free-text field must still
    # swallow letters, or typing a path with a b in it would switch tools.
    #
    # The field has to be visible for focus() to take -- it lives in a collapsed
    # <details>, and focusing a hidden element silently does nothing, which is
    # how the first version of this check "failed" for the wrong reason.
    page.js("document.getElementById('outDir').closest('details').open = true")
    ok(page.js("document.getElementById('outDir').offsetParent !== null"),
       "the text field is visible, so it can take focus")
    page.js("document.getElementById('outDir').focus()")
    ok(page.js("document.activeElement.id") == "outDir",
       "a text field takes focus", page.js("document.activeElement.id"))
    page.js("setTool('select')")
    page.key("b")
    ok(page.js("S.tool") == "select",
       "a letter typed into a text field is NOT a shortcut", page.js("S.tool"))
    page.js("document.getElementById('outDir').blur()")

    print("\n=== 16. a document the server has forgotten ===")
    # The server keeps open documents in memory, so restarting it invalidates
    # every id an open page is holding. The page kept its cached cell images and
    # still looked perfectly healthy -- toolbar, sheet, timeline -- while every
    # single write died as an unhandled rejection. The pencil and eraser looked
    # broken and nothing else did. This is the one layer that can see it: both
    # other suites open a fresh document against a live server.
    #
    # Simulated by pointing the page at an id the server does not know, which is
    # exactly what a restart leaves behind. (A real restart was used to find it.)
    page.wait_for("S.busy === 0", "the previous section to settle", timeout=90)
    # Open a sheet that still exists first. Section 14 leaves the path of a temp
    # copy it has since deleted in the box, and "reopen a sheet that is gone" is
    # a different case -- the second half of this section.
    _r = open_by_name(page, "walk")
    ok(_r == "opened", "a live walk sheet opened by name", _r)
    page.wait_for("!!S.doc && S.busy === 0 && S.img.size === S.doc.n",
                  "a live sheet to lose", timeout=180)
    live = page.js("document.getElementById('sheetPath').value")
    ok(bool(live) and os.path.isfile(live),
       "a sheet that still exists is in the open-by-path box", live)

    page.js("S.doc.id = 'forgotten_by_the_server'")
    page.js("document.getElementById('toast').textContent = ''")
    n_errs = len(page.errors())
    page.js("setTool('pencil')")
    page.js("document.getElementById('brushSize').value = 30")
    c = page.js("window.__d.cellScreen(0)")
    page.drag([(c["x"], c["y"]), (c["x"] + 15, c["y"] + 15)])
    # S.doc goes non-null as soon as applyDoc runs, which is well before the
    # reopen has finished loading cells -- so waiting on the id alone reads the
    # status line mid-load ("loading cells 42/124") and the assertion fails for a
    # reason that has nothing to do with recovery. Wait for the whole thing.
    page.wait_for("!recovering && !!S.doc"
                  " && S.doc.id !== 'forgotten_by_the_server'"
                  " && S.busy === 0 && S.img.size === S.doc.n",
                  "the page to notice the dead document and reopen it",
                  timeout=180)
    ok(page.js("S.doc.id") != "forgotten_by_the_server",
       "the page noticed the dead document and reopened the sheet",
       page.js("S.doc.id"))
    st = page.js("document.getElementById('status').textContent")
    ok(st.startswith("the server restarted"),
       "and says what happened in the status line, not a 2s toast", st[:110])
    ok("Reopened" in st, "and that it did reopen, rather than only promising to",
       st[:160])
    ok(page.js("S.img.size") == page.js("S.doc.n"),
       "the reopened sheet is fully loaded", page.js("S.img.size"))
    ok(page.js("S.anchorCell") == 0,
       "and the shift-select anchor was reset with the new document, so it "
       "cannot range from an index of the sheet that was lost",
       page.js("S.anchorCell"))
    h0 = page.js("window.__d.hashCell(0)", await_promise=True)
    page.js("setTool('pencil')")
    c = page.js("window.__d.cellScreen(0)")
    page.drag([(c["x"], c["y"]), (c["x"] + 15, c["y"] + 15)])
    page.wait_for("window.__d.busy()", "the post-recovery stroke", timeout=90)
    ok(page.js("window.__d.hashCell(0)", await_promise=True) != h0,
       "and a stroke lands again once it has reopened")
    ok(len(page.errors()) == n_errs,
       "with no unhandled rejection anywhere in that",
       page.errors()[n_errs:n_errs + 1])

    # --- and when the sheet it would reopen is gone -------------------------
    # A moved or deleted file is not hypothetical: the open-by-path box is free
    # text, and section 14 has just demonstrated that the box outlives the file
    # it names. The page must not claim to be reopening something it has given
    # up on, and it must stay usable afterwards.
    gone = os.path.join(tempfile.gettempdir(), "drive16_sheet_that_is_gone.png")
    if os.path.isfile(gone):
        os.remove(gone)
    page.js("document.getElementById('sheetPath').value = %s" % json.dumps(gone))
    page.js("S.doc.id = 'forgotten_by_the_server'")
    page.js("document.getElementById('status').textContent = ''")
    n_errs = len(page.errors())
    page.js("setTool('pencil')")
    c = page.js("window.__d.cellScreen(0)")
    page.drag([(c["x"], c["y"]), (c["x"] + 15, c["y"] + 15)])
    # wait on recovering, not just on S.doc: S.doc is nulled part-way through and
    # the status line is written after the failed open returns.
    page.wait_for("!recovering && !S.doc",
                  "the failed reopen to finish", timeout=180)
    st = page.js("document.getElementById('status').textContent")
    ok(st.startswith("the server restarted"),
       "a sheet that is gone still gets the explanation", st[:160])
    ok("could not be reopened" in st,
       "and the status admits the reopen failed instead of claiming it", st[:200])
    ok(page.js("document.getElementById('docname').textContent") == "no sheet open",
       "and no document is claimed to be open",
       page.js("document.getElementById('docname').textContent"))
    ok(len(page.errors()) == n_errs,
       "with no unhandled rejection in the failed reopen either",
       page.errors()[n_errs:n_errs + 1])

    # The Audio panel in exactly that state, because "no sheet open" is not a
    # state a panel is allowed to leave blank. Reported as "audio file list
    # disapeard": the page had lost its document, and `renderAudio` cleared the
    # box and returned *without* writing even the "no sounds attached" line it
    # uses for a document that has no clips -- so an empty list and a broken list
    # looked identical. The rows matter as much as the text: the panel also has to
    # forget the lost document's clips rather than go on listing them, and only
    # the clips assertion can tell those two apart.
    _al = page.js("""(() => {
      const l = document.getElementById('audioList');
      return {kids: l.children.length, text: l.textContent.trim(),
              rows: l.querySelectorAll('.audiorow').length};
    })()""")
    ok(_al["kids"] > 0 and bool(_al["text"]),
       "with no sheet open the Audio panel says so instead of rendering a blank "
       "box", _al)
    ok(_al["rows"] == 0,
       "and it does not keep listing the clips of the document it lost", _al)
    # still usable: the page must recover after a reopen has failed
    _r = open_by_name(page, "walk")
    ok(_r == "opened", "a live walk sheet opened by name", _r)
    page.wait_for("!!S.doc && S.busy === 0 && S.img.size === S.doc.n",
                  "the editor to be usable again", timeout=180)
    ok(page.js("S.doc.n") > 0,
       "and the editor still opens a sheet after a failed recovery",
       page.js("S.doc.n"))

    print("\n=== 16b. the black theme ===")
    # The theme is page-level, so the only honest place to test it is here: the
    # attribute has to be right on load (a light flash is the bug it exists to
    # avoid), the choice has to survive a reload, and the CANVAS is a separate
    # claim -- it reads one colour back out of the CSS in JS, so the page can be
    # entirely dark while the sheet area is still painted light.
    page.call("Page.navigate", url=args.url + "/editor")
    page.wait_for("document.readyState === 'complete'", "the reload for 16b")
    page.wait_for("typeof setTheme === 'function'", "the theme hooks", timeout=30)

    def stage_px():
        # Nothing is open after the reload, so the whole canvas is the stage.
        return page.js("(() => { const d = view.getContext('2d')"
                       ".getImageData(1, 1, 1, 1).data;"
                       " return [d[0], d[1], d[2], d[3]].join(','); })()")

    # The same switch is what the generator's button drives, through the one
    # localStorage key: the check for that is in _probe_theme_boot.py's table.
    page.js("setTheme('light')")
    px_light = stage_px()
    page.js("document.getElementById('theme').click()")
    ok(page.js("document.documentElement.dataset.theme") == "dark",
       "the theme button switches the page to the black theme",
       page.js("document.documentElement.dataset.theme"))
    ok(page.js("document.getElementById('theme').textContent") == "Light",
       "and its label names the theme it switches to, not the one in force",
       page.js("document.getElementById('theme').textContent"))
    ok(page.js("localStorage.getItem('sprite.theme')") == "dark",
       "and the choice is remembered for the generator too")
    px_dark = stage_px()
    # The expected value is read out of the CSS rather than written here as a
    # triple. The claim is that the canvas FOLLOWS --stage; a literal would
    # instead freeze the palette, and it did -- it still expected the old dark
    # value after the palette was moved onto TostUI's colours.
    _stage_css = page.js(
        "getComputedStyle(document.documentElement)"
        ".getPropertyValue('--stage').trim()")
    _h = _stage_css.lstrip("#")
    _stage_rgb = ",".join(str(int(_h[i:i + 2], 16)) for i in (0, 2, 4))
    ok(px_dark != px_light and px_dark.startswith(_stage_rgb),
       "the sheet area repaints to the dark --stage -- the canvas reads the "
       "variable, not a colour baked into draw()",
       "%s -> %s (--stage %s)" % (px_light, px_dark, _stage_css))

    # The stored choice has to win on load, before any click: set by the inline
    # script in <head>, so there is no light flash to repaint away.
    page.call("Page.navigate", url=args.url + "/editor")
    page.wait_for("document.readyState === 'complete'", "the reload")
    page.wait_for("typeof setTheme === 'function'", "the theme hooks", timeout=30)
    ok(page.js("document.documentElement.dataset.theme") == "dark",
       "the stored theme is applied on load, with no click and no flash",
       page.js("document.documentElement.dataset.theme"))
    ok(page.js("document.getElementById('theme').textContent") == "Light",
       "and the button agrees with the attribute it did not set")
    ok(stage_px() == px_dark,
       "the stage is black from the first frame after the reload",
       "%s vs %s" % (stage_px(), px_dark))

    # ...and back, so the rest of the session (and the next run) starts light.
    page.js("document.getElementById('theme').click()")
    ok(page.js("document.documentElement.dataset.theme") == "light"
       and stage_px() == px_light,
       "switching back restores the light stage",
       "%s vs %s" % (stage_px(), px_light))

    print("\n=== 16c. Snap Pixels ===")
    # The panel is the one op with its own UI, and it is gated on a selection.
    # This is also the only check that the live preview (server op -> PNG ->
    # canvas) and the pixel-perfect viewport zoom actually reach the page.
    _r = open_by_name(page, "walk")
    ok(_r == "opened", "a sheet for the snap test", _r)
    page.wait_for("!!S.doc && S.img.size === S.doc.n", "the sheet for the snap test",
                  timeout=120)
    cw0 = page.js("S.doc.layout.cell_w")
    ch0 = page.js("S.doc.layout.cell_h")

    SNAP_NODE = ("[...document.querySelectorAll('#ops details.op')]"
                 ".find(x => x.querySelector('summary').textContent.trim() === "
                 "'Snap Pixels')")
    CROP = "d.querySelector('.snapview canvas')"
    page.js("window.__snap = { node: %s }; 'ok'" % SNAP_NODE)
    # The side panels are folded by default now, and a closed <details> has no
    # layout: the "fitted preview fits the panel" check further down would compare
    # 0 to 0 and pass without measuring anything. Open the chain this op lives in
    # first, so the measurements below are of a laid-out element.
    page.js("(() => { let el = window.__snap.node;"
            " while (el && el !== document.body){"
            " if (el.tagName === 'DETAILS') el.open = true;"
            " el = el.parentElement; } return true; })()")
    ok(page.js("SNAP_GROUP && SNAP_GROUP.style.display === 'none'"),
       "the Snap Pixels group is hidden with nothing selected")
    page.js("S.sel = new Set([0]); syncSel()")
    ok(page.js("SNAP_GROUP && SNAP_GROUP.style.display !== 'none'"),
       "selecting a frame reveals it")
    defaults = page.js(
        "(() => { const d = window.__snap.node;"
        " return [...d.querySelectorAll('.opbody input[type=number]')]"
        ".map(i => i.value).join(','); })()")
    ok(defaults == "16,10", "colours and pixel size default to 16 and 10", defaults)

    # The palette picker: Auto + the console presets + import, writing its choice
    # into the hidden form value the op actually sends.
    _pal = page.js(
        "(() => { const d = window.__snap.node;"
        " const b = d.querySelector('.palbtn');"
        " return b ? {label: b.querySelector('.pname').textContent,"
        " swatch: !!b.querySelector('canvas')} : null; })()")
    ok(_pal and _pal["label"] == "Auto (k-means)" and _pal["swatch"],
       "the palette picker defaults to Auto (k-means)", _pal)
    page.js("window.__snap.node.querySelector('.palbtn').click(); 'ok'")
    page.wait_for("(() => { const p = [...document.querySelectorAll('.palpanel')]"
                  ".find(x => !x.hidden);"
                  " return !!p && p.querySelectorAll('.palrow').length > 5; })()",
                  "the palette dropdown to open")
    _nrows = page.js("(() => { const p = [...document.querySelectorAll('.palpanel')]"
                     ".find(x => !x.hidden);"
                     " return p.querySelectorAll('.palrow').length; })()")
    ok(_nrows and _nrows > 20, "it lists Auto plus the console presets", _nrows)
    _pick = page.js(
        "(() => { const p = [...document.querySelectorAll('.palpanel')]"
        ".find(x => !x.hidden);"
        " const r = [...p.querySelectorAll('.palrow')].find(x =>"
        " x.querySelector('span').textContent === 'PICO-8');"
        " r.click();"
        " return {label: window.__snap.node.querySelector('.pname').textContent,"
        " value: window.__snap.node.querySelector('input[type=hidden]').value,"
        " open: !p.hidden}; })()")
    ok(_pick["label"] == "PICO-8" and _pick["open"] is False
       and len(_pick["value"].split(",")) == 16,
       "picking a preset sets the palette and closes the panel", _pick)
    page.js("window.__snap.node.querySelector('.palbtn').click(); 'ok'")
    page.js("(() => { const p = [...document.querySelectorAll('.palpanel')]"
            ".find(x => !x.hidden);"
            " const r = [...p.querySelectorAll('.palrow')].find(x =>"
            " x.querySelector('span').textContent === 'Auto (k-means)');"
            " r.click(); return true; })()")
    ok(page.js("window.__snap.node.querySelector('input[type=hidden]').value") == "",
       "choosing Auto clears the fixed palette")

    have_snapper = True
    try:
        page.wait_for("(() => { const d = window.__snap.node;"
                      " return %s.width > 1; })()" % CROP,
                      "the snapped preview", timeout=60)
    except TimeoutError:
        have_snapper = False
    if not have_snapper:
        print("  skip  spritefusion-pixel-snapper/Node is not available here")
    else:
        pw = page.js("window.__snap.node.querySelector('.snapview canvas').width")
        ph = page.js("window.__snap.node.querySelector('.snapview canvas').height")
        ok(0 < pw < cw0,
           "the preview is the snapped frame (%dpx from %dpx)" % (pw, cw0),
           (pw, ph))
        ok(page.js("SNAP.fit === true"),
           "the preview opens fitted, not at the pixel-size zoom")
        ok(page.js("(() => { const v = SNAP.view;"
                   " return v.scrollWidth <= v.clientWidth + 1 &&"
                   " v.scrollHeight <= v.clientHeight + 1; })()"),
           "the fitted preview fits the panel with no scrollbars")
        info = page.js("(() => { const d = window.__snap.node;"
                       " const i = d.querySelector('.snapinfo');"
                       " const b = d.querySelector('.snapbar');"
                       " return (i ? i.textContent : '') + '|' +"
                       " (i && i.getBoundingClientRect().top >="
                       "  b.getBoundingClientRect().bottom); })()")
        ok("grid" in info and "kept at" in info and "frame" in info
           and info.endswith("|true"),
           "the snapped grid and the kept frame size are written under the "
           "controls", info)
        z0 = page.js("Math.round(SNAP.zoom * 100)")
        step = ("(() => { const d = window.__snap.node;"
                " const b = [...d.querySelectorAll('.snapbar button')]"
                ".find(x => x.textContent === '%s'); b.click(); return true; })()")
        page.js(step % "+100%")
        z1 = page.js("Math.round(SNAP.zoom * 100)")
        ok(z1 == z0 + 100, "one zoom step changes it by exactly 100%",
           (z0, z1))
        page.js(step % "Fit")
        zf = page.js("Math.round(SNAP.zoom * 100)")
        ok(page.js("SNAP.fit === true") and zf == z0,
           "Fit returns to the fitted view, not a smaller one", (z0, zf))
        rev0 = page.js("S.doc.cell_rev[0]")
        s0 = page.js("S.view.s")
        page.js("(() => { const d = window.__snap.node;"
                " const b = [...d.querySelectorAll('.opbody button')]"
                ".find(x => x.textContent === 'Snap Pixels'); b.click();"
                " return true; })()")
        page.wait_for("S.busy === 0 &&"
                      " /Snap Pixels/.test(document.getElementById('status')"
                      ".textContent)",
                      "the snap op to land", timeout=180)
        cw1 = page.js("S.doc.layout.cell_w")
        ch1 = page.js("S.doc.layout.cell_h")
        rev1 = page.js("S.doc.cell_rev[0]")
        ok(cw1 == cw0 and ch1 == ch0,
           "the snap kept the document's cell size, so the frame and sheet do "
           "not change size", (cw1, ch1, cw0, ch0))
        ok(rev1 != rev0, "the selected frame really was resnapped", (rev0, rev1))
        ok(page.js("S.view.s") == s0,
           "the view is left where the user had it, not re-zoomed",
           (s0, page.js("S.view.s")))
        page.js("document.getElementById('undo').click()")
        page.wait_for("S.busy === 0 && S.doc.cell_rev[0] != %d" % rev1,
                      "undo of the snap", timeout=60)
        ok(page.js("S.doc.layout.cell_w") == cw0
           and page.js("S.doc.layout.cell_h") == ch0,
           "undo keeps the same cell size and restores the frame")

    print("\n=== 16d. Audio ===")
    import wave as _wv
    _adir = mkdtemp(prefix="drive_audio_")
    try:
        _wav = os.path.join(_adir, "blip.wav")
        with _wv.open(_wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 8)
        page.js("S.cur = 0; S.sel = new Set([0]); syncSel()")
        ok(page.js("document.getElementById('audioTarget').textContent") ==
           "target: 1 selected frame",
           "the audio target names the selection",
           page.js("document.getElementById('audioTarget').textContent"))
        # A real file into the real input, so the page's own change handler runs.
        page.call("DOM.enable")
        _root = page.call("DOM.getDocument", depth=-1)["root"]["nodeId"]
        _nid = page.call("DOM.querySelector", nodeId=_root,
                         selector="#audioFile")["nodeId"]
        ok(_nid > 0, "the audio file input is in the page")
        page.call("DOM.setFileInputFiles", nodeId=_nid, files=[_wav])
        page.wait_for("(S.doc.audio || []).length > 0",
                      "the clip to attach", timeout=60)
        ok(page.js("document.querySelectorAll('#audioList .audiorow').length")
           == 1, "the clip is listed once uploaded",
           page.js("document.querySelectorAll('#audioList .audiorow').length"))
        ok(page.js("document.querySelector('#audioList .fr').textContent")
           == "frame 0", "and it names its frame",
           page.js("document.querySelector('#audioList .fr').textContent"))
        ok("blip.wav" in page.js(
               "document.getElementById('frameLabel').textContent"),
           "the playback label names the current frame's sound",
           page.js("document.getElementById('frameLabel').textContent"))
        ok("\u266a" in page.js(
               "document.querySelectorAll('#strip canvas')[0].title"),
           "the timeline marks the frame that has a sound",
           page.js("document.querySelectorAll('#strip canvas')[0].title"))
        # ...and the sheet marks it too, which is the surface the user is looking
        # at while working on a frame. The README claimed this check for a long
        # time and it did not exist: it went with the old audio-library section
        # and was never restored, which is why mutant K came back MISSED from a
        # sweep while the table said it was caught. Measured as a comparison
        # between the frame that has the clip and one that does not, because an
        # absolute count of cyan could be satisfied by anything else on the
        # canvas -- and read off the page's own canvas, since "the document has
        # the clip" and "the sheet says so" are different claims.
        #
        # `__d` is not available here: 16b reloaded the page and the helpers are
        # not put back until 16g, so the view is set through the page's own state
        # (which is what the helper does anyway, and what 16c and 16d already do).
        _v_was = page.js("[S.view.s, S.view.tx, S.view.ty]")
        page.js("S.view.s = 60 / SHEET().cw; S.view.tx = 0; S.view.ty = 0; draw()")
        _mark = page.js("""(() => {
          const c = document.getElementById('view');
          const g = c.getContext('2d');
          const dpr = Math.max(1, window.devicePixelRatio || 1);
          const L = SHEET(), s = S.view.s, tx = S.view.tx, ty = S.view.ty;
          const near = (i) => {
            const col = i % L.cols, row = Math.floor(i / L.cols);
            const x1 = (col + 1) * L.cw * s + tx, y0 = row * L.ch * s + ty;
            const X0 = Math.max(0, Math.round((x1 - 18) * dpr));
            const Y0 = Math.max(0, Math.round(y0 * dpr));
            const X1 = Math.min(c.width, Math.round(x1 * dpr));
            const Y1 = Math.min(c.height, Math.round((y0 + 18) * dpr));
            if (X1 <= X0 || Y1 <= Y0) return -1;
            const px = g.getImageData(X0, Y0, X1 - X0, Y1 - Y0).data;
            let n = 0;
            for (let k = 0; k < px.length; k += 4){
              if (px[k] === 14 && px[k+1] === 116 && px[k+2] === 144) n++;
            }
            return n;
          };
          return {legible: s * L.cw > 34 && s * L.ch > 20,
                  marked: near(0), plain: near(1)};
        })()""")
        ok(_mark["legible"],
           "the sheet is zoomed enough for the corner marks to be drawn at all",
           _mark)
        ok(_mark["marked"] > 0 and _mark["plain"] == 0,
           "and the frame with a sound is marked on the sheet while the one "
           "without is not", _mark)
        page.js("S.view.s = %r; S.view.tx = %r; S.view.ty = %r; draw()"
                % tuple(_v_was))
        ok("here" in page.js(
               "document.querySelector('#audioList .audiorow').className"),
           "the row for the current frame is highlighted",
           page.js("document.querySelector('#audioList .audiorow').className"))
        # The volume control. It changes a clip *after* it is attached, and it is
        # the number save_doc writes into the sidecar -- so the panel has to read
        # it back as well as write it, or the control is a display that silently
        # resets. 12b of the smoke suite checks the sidecar that comes out.
        _vs0 = page.js("""(() => {const s = document.querySelector(
          '#audioList .volrow input[type=range]');
          return s ? [s.value, s.parentNode.querySelector('.volval').textContent]
                   : null;})()""")
        ok(_vs0 == ["100", "100%"],
           "an attached clip starts at full volume", _vs0)
        page.js("""(() => {const s = document.querySelector(
          '#audioList .volrow input[type=range]');
          s.value = '40';
          s.dispatchEvent(new Event('input', {bubbles: true}));
          s.dispatchEvent(new Event('change', {bubbles: true}));
          return 'ok';})()""")
        page.wait_for("S.doc.audio[0].volume === 0.4",
                      "the volume to reach the document", timeout=60)
        ok(page.js("document.querySelector('#audioList .volval').textContent")
           == "40%", "and the readout follows the slider",
           page.js("document.querySelector('#audioList .volval').textContent"))
        page.js("renderAudio()")
        ok(page.js("document.querySelector('#audioList .volrow "
                   "input[type=range]').value") == "40",
           "and the panel renders the stored volume rather than resetting to "
           "full", page.js("document.querySelector('#audioList .volrow "
                           "input[type=range]').value"))
        page.js("document.querySelector('#audioList .audiorow button:last-child'"
                ").click(); 'ok'")
        page.wait_for("(S.doc.audio || []).length === 0",
                      "the clip to be removed", timeout=30)
        ok(page.js("document.querySelectorAll('#audioList .audiorow').length")
           == 0, "removing a clip clears it from the list")
        page.js("S.sel.clear(); syncSel()")
    finally:
        shutil.rmtree(_adir, ignore_errors=True)

    # 16e used to cover the audio library: point the editor at a folder of
    # clips, list them, drag one onto a frame, attach it by path. The panel is
    # gone, and the drag-and-drop went with it -- every drop target was gated
    # on the library's own drag state, so there was no other drag source. 16d
    # still covers attaching a clip through Add sound, listing it, marking its
    # frame and removing it, and 16f covers the clips surviving a save and a
    # reopen, which is the behaviour the library section was there to protect.
    # The report this section exists for: "when i load folder audios are not
    # loaded". A run's output folder is the natural thing to paste, and the
    # sidecar inside it is NOT named after the folder, so this pins the fallback
    # that finds it as well as the clips travelling with it. The folder is built
    # on disk in exactly the shape save_doc writes: a sheet, a sidecar naming it
    # relative, and the clip in audio/ beside them.
    _fdir = mkdtemp(prefix="drive_folder_")
    try:
        # A document has to be open to build the fixture from. 16e ended on a
        # reload, and boot() only reopens what was remembered -- which an earlier
        # section may have pointed at a temp path it then deleted.
        if not page.js("!!S.doc"):
            _r = open_by_name(page, "walk")
            ok(_r == "opened", "a sheet to build the folder fixture from", _r)
            page.wait_for("S.doc && document.getElementById('status')"
                          ".textContent.indexOf('opened') === 0",
                          "a sheet to build the fixture from", timeout=120)
        _lay = page.js("S.doc.layout")
        _fname = "folderopen"
        _fsheet = os.path.join(_fdir, _fname + "_sheet.png")
        with urllib.request.urlopen("%s/api/editor/%s/sheet.png"
                                    % (args.url, page.js("S.doc.id"))) as _r:
            with open(_fsheet, "wb") as fh:
                fh.write(_r.read())
        os.makedirs(os.path.join(_fdir, "audio"), exist_ok=True)
        with _wv.open(os.path.join(_fdir, "audio", "blip.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 8)
        with open(os.path.join(_fdir, _fname + ".json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"name": _fname, "sheet": _fname + "_sheet.png",
                       "frame_width": _lay["cell_w"],
                       "frame_height": _lay["cell_h"],
                       "columns": _lay["columns"], "rows": _lay["rows"],
                       "frame_count": _lay["frame_count"],
                       "audio": [{"frame": 0, "file": "audio/blip.wav",
                                  "name": "blip.wav", "volume": 1.0}]}, fh)

        page.js("document.getElementById('sheetPath').value = %s"
                % json.dumps(_fdir))
        page.js("document.getElementById('openPath').click()")
        page.wait_for("S.doc && document.getElementById('status').textContent"
                      ".indexOf('opened') === 0", "the folder to open",
                      timeout=120)
        ok(page.js("S.doc.n") == _lay["frame_count"],
           "opening the folder opens the sheet the sidecar in it names",
           page.js("S.doc.n"))
        ok(page.js("(S.doc.audio || []).length") == 1,
           "and the clip that sidecar names comes with it",
           page.js("S.doc.audio"))
        ok(page.js("document.querySelectorAll('#audioList .audiorow').length")
           == 1, "the Audio panel lists it",
           page.js("document.querySelectorAll('#audioList .audiorow').length"))
        ok(page.js("document.querySelector('#audioList .fr').textContent")
           == "frame 0", "on the frame the sidecar put it on",
           page.js("document.querySelector('#audioList .fr').textContent"))
        ok("\u266a" in page.js(
               "document.querySelectorAll('#strip canvas')[0].title"),
           "and the timeline marks that frame",
           page.js("document.querySelectorAll('#strip canvas')[0].title"))
        ok(page.js("document.getElementById('sheetPath').value")
           .endswith(_fname + ".json"),
           "the box names the sidecar the folder resolved to, not the folder",
           page.js("document.getElementById('sheetPath').value"))
    finally:
        shutil.rmtree(_fdir, ignore_errors=True)

    print("\n=== 16g. the folder picker: newest first, no duplicates, ten at most ===")
    # 16f ends on a reload, and __d does not survive one -- reinstall before the
    # first helper call rather than after the section has already failed.
    page.js(HELPERS)
    # The picker replaced the server's sheet-library listing. It is a shortlist
    # rather than a history -- one entry per folder, newest first, ten at most --
    # and those three rules ARE the feature, so each gets its own check.
    from PIL import Image as _PIL
    _rt = mkdtemp(prefix="drive_recent_")

    def _mkfolder(name):
        """A folder holding a 2x2 sheet and the sidecar that names it."""
        d = os.path.join(_rt, name)
        os.makedirs(d, exist_ok=True)
        _PIL.new("RGBA", (64, 64), (10, 20, 30, 255)).save(
            os.path.join(d, "s_sheet.png"))
        with open(os.path.join(d, "s.json"), "w", encoding="utf-8") as fh:
            json.dump({"name": name, "sheet": "s_sheet.png", "columns": 2,
                       "rows": 2, "frame_width": 32, "frame_height": 32}, fh)
        return d

    def _open_folder(p):
        """Type a folder into the path box and click Open, like a user."""
        page.js("document.getElementById('sheetPath').value = %s"
                % json.dumps(p))
        page.js("document.getElementById('openPath').click()")
        # The box is rewritten to the sidecar the folder resolved to, so that is
        # the signal the open landed -- not the frame count, which every fixture
        # here shares.
        page.wait_for("document.getElementById('sheetPath').value"
                      ".toLowerCase().endsWith('s.json')",
                      "the folder to resolve to its sidecar", timeout=120)
        page.wait_for("window.__d.busy()", "the open to settle", timeout=120)

    try:
        # From nothing, so "the list is exactly what this section opened" is a
        # real claim instead of one that depends on earlier sections.
        page.js("try { localStorage.removeItem('sprite.recentFolders'); }"
                " catch(e) {}")
        page.js("renderRecent()")
        ok(page.js("window.__d.recentDirs().length") == 0,
           "the picker starts empty once its list is cleared",
           page.js("window.__d.recentDirs()"))

        _a = _mkfolder("alpha")
        _b = _mkfolder("beta")
        _open_folder(_a)
        ok(page.js("window.__d.recentDirs()") == [_a],
           "opening a folder puts that folder in the picker",
           page.js("window.__d.recentDirs()"))
        ok(page.js("window.__d.recentNames()") == ["alpha"],
           "labelled with the folder, not with the sheet inside it",
           page.js("window.__d.recentNames()"))

        _open_folder(_b)
        ok(page.js("window.__d.recentDirs()") == [_b, _a],
           "a second folder goes on top -- newest first",
           page.js("window.__d.recentDirs()"))

        _open_folder(_a)
        ok(page.js("window.__d.recentDirs()") == [_a, _b],
           "reopening the first moves it back to the top rather than listing it "
           "twice", page.js("window.__d.recentDirs()"))

        # Eleven more, so the cap has to bite: alpha and beta are long gone.
        _many = [_mkfolder("f%02d" % i) for i in range(11)]
        for d in _many:
            _open_folder(d)
        _list = page.js("window.__d.recentDirs()")
        ok(len(_list) == 10, "the list stops at ten", len(_list))
        ok(_list[0].lower() == _many[-1].lower(),
           "with the newest at the top", (_list[0], _many[-1]))
        ok(_a.lower() not in [x.lower() for x in _list],
           "and the oldest dropped off the end", _list)

        # Click the LAST row -- the oldest of the ten -- so this cannot pass by
        # reopening the document that happens to be open already.
        _target = _list[9]
        page.js("document.querySelectorAll('#recent div[data-k]')[9].click()")
        page.wait_for("document.getElementById('sheetPath').value"
                      ".toLowerCase().endsWith('s.json')",
                      "the clicked row to open", timeout=120)
        page.wait_for("window.__d.busy()", "that open to settle", timeout=120)
        ok(os.path.normcase(page.js("document.getElementById('sheetPath').value"))
           == os.path.normcase(os.path.join(_target, "s.json")),
           "clicking a row opens that folder",
           page.js("document.getElementById('sheetPath').value"))

        # localStorage, not page state: a reload has to bring the list back.
        # The set, not the order -- boot() reopens the last sheet, which moves
        # that folder to the top, and that is correct behaviour, not a failure.
        _before = sorted(x.lower() for x in page.js("window.__d.recentDirs()"))
        page.call("Page.navigate", url=args.url + "/editor")
        page.wait_for("document.readyState === 'complete'", "document load")
        page.wait_for("!!document.getElementById('recent')", "the picker",
                      timeout=30)
        page.js(HELPERS)          # __d does not survive a reload
        ok(sorted(x.lower() for x in page.js("window.__d.recentDirs()"))
           == _before, "and the list survives a reload",
           page.js("window.__d.recentDirs()"))
    finally:
        shutil.rmtree(_rt, ignore_errors=True)

    print("\n=== 16h. Apply metadata writes into the sidecar it was loaded from ===")
    # "when i click apply metadata it should write in current json". The op route
    # is a pure change to the in-memory document -- nothing else in the page
    # persists until Save -- so metadata used to sit in memory until the user
    # also pressed Save, which rewrites the PNG, the GIF and the preview player
    # as well.
    #
    # On a throwaway copy, never on the walk sheet: this section's whole point is
    # that a button writes a file, and the walk sheet's sidecar is a real asset.
    page.js(HELPERS)              # 16g ends on a reload
    from PIL import Image as _PIL2
    _mt = mkdtemp(prefix="drive_meta_")
    _msheet = os.path.join(_mt, "m_sheet.png")
    _msc = os.path.join(_mt, "m.json")
    _PIL2.new("RGBA", (64, 64), (10, 20, 30, 255)).save(_msheet)
    with open(_msc, "w", encoding="utf-8") as fh:
        json.dump({"name": "meta", "sheet": "m_sheet.png", "columns": 2,
                   "rows": 2, "frame_width": 32, "frame_height": 32,
                   "fps": 30,
                   # Keys the panel does not own. They are what separates a
                   # targeted update from a regeneration, so they are planted
                   # here and asserted below.
                   "source": "pipeline/frame_%04d.png",
                   "matte": "edited",
                   "edited": {"ops": ["offset"], "revision": 3},
                   "note": "old note"}, fh)
    _png_before = open(_msheet, "rb").read()
    _sc_before = json.load(open(_msc, encoding="utf-8"))

    # Take the player's fps over first, so the open below has something to hand
    # back. This is the other half of section 8c: there, touching the box claimed
    # it; here, opening a sheet gives it back.
    page.js("""(() => {
      const el = document.getElementById('playFps');
      el.value = '99';
      el.dispatchEvent(new Event('input', {bubbles: true}));
      return S.playFpsOwn;
    })()""")
    page.js("document.getElementById('sheetPath').value = %s" % json.dumps(_mt))
    page.js("document.getElementById('openPath').click()")
    page.wait_for("!!S.doc && S.doc.meta.fps === 30",
                  "the fixture to open", timeout=120)
    page.wait_for("window.__d.busy()", "the open to settle", timeout=120)
    _hb = page.js("({own: S.playFpsOwn, "
                  "player: Number(document.getElementById('playFps').value), "
                  "doc: S.doc.meta.fps})")
    ok(_hb["own"] is False and _hb["player"] == 30,
       "opening a sheet hands the player's fps back to the document", _hb)

    # The write itself, driven by the real button.
    page.js("document.getElementById('fps').value = '13'")
    page.js("document.getElementById('metaName').value = 'meta-renamed'")
    page.js("document.getElementById('applyMeta').click()")
    page.wait_for("document.getElementById('status').textContent"
                  ".indexOf('metadata written to') === 0",
                  "Apply metadata to report the write", timeout=120)
    _sc_after = json.load(open(_msc, encoding="utf-8"))
    ok(_sc_after.get("fps") == 13,
       "the fps typed into the panel is in the JSON on disk",
       _sc_after.get("fps"))
    ok(_sc_after.get("name") == "meta-renamed",
       "and so is the name", _sc_after.get("name"))
    ok("13.00 fps" in _sc_after.get("note", ""),
       "the note was rewritten from the new fps", _sc_after.get("note"))
    ok(_sc_after.get("source") == _sc_before.get("source"),
       "a key the panel does not own is carried through, not regenerated",
       _sc_after.get("source"))
    ok(_sc_after.get("edited") == _sc_before.get("edited")
       and _sc_after.get("matte") == _sc_before.get("matte"),
       "and the record of how the sheet was made survives the write",
       (_sc_after.get("edited"), _sc_after.get("matte")))
    ok(open(_msheet, "rb").read() == _png_before,
       "the sheet PNG was not touched -- only the JSON was written")
    _st = page.js("document.getElementById('status').textContent")
    ok(os.path.normcase(_st).endswith(os.path.normcase(_msc)),
       "and the status line names the file it wrote", _st[:200])

    # A grid change cannot be written this way: the PNG is not rewritten, so the
    # JSON would describe a sheet that is not the one on disk. The refusal has to
    # reach the panel rather than only the server.
    page.js("document.getElementById('cols').value = '1'")
    page.js("document.getElementById('applyCols').click()")
    page.wait_for("S.doc.layout.columns === 1", "the re-grid to land",
                  timeout=120)
    page.js("document.getElementById('applyMeta').click()")
    page.wait_for("document.getElementById('status').textContent"
                  ".indexOf('metadata NOT written') === 0",
                  "the refusal to reach the panel", timeout=120)
    _ref = page.js("document.getElementById('status').textContent")
    ok("the grid changed" in _ref,
       "a grid change is refused, and the panel says why", _ref[:200])
    ok(json.load(open(_msc, encoding="utf-8")).get("fps") == 13,
       "and the refusal left the file alone rather than half-writing it")
    shutil.rmtree(_mt, ignore_errors=True)

    print("\n=== 16i. a checkbox has to be tickable by its own words ===")
    # The reported bug: "allow overwriting the sheet this document was loaded
    # from when checked not working". The box itself was fine -- measured with a
    # real click, ticked it saves over the source and unticked it refuses. What
    # was broken is that the words beside it were a <span> in a <div>, so
    # clicking the text did nothing at all, and the text is what a user aims at.
    # Tick nothing, press Save, get the refusal, and the opt-in looks broken.
    #
    # Nothing caught it because every other interaction with these controls --
    # here and in the sections above -- sets `.checked` in JavaScript. A JS
    # assignment proves the handler reads the box; it says nothing about whether
    # a mouse can reach it. So this section clicks the words.
    page.js(HELPERS)

    def click_words(i):
        """Scroll the row into view, click its text, return the box's state.

        Addressed by index, not by id: the op registry builds its `bool` controls
        in JavaScript and gives them no id at all, so `getElementById("")` finds
        nothing. Indexing also brings those generated rows under this guard, which
        is the point -- Nudge's *wrap around* is built by the same code path and
        had the same dead label.

        Scrolled first on purpose: the columns scroll internally
        (`.col{overflow:auto}`) while the document does not, so the Export panel's
        rows start below the fold and a click at their un-scrolled coordinates
        lands on nothing -- which reads as "the label does not work" for a reason
        that has nothing to do with labels.

        And the enclosing panels are opened first, for the same reason one step
        earlier: a row inside a folded `<details>` has no box at all, so its rect
        is all zeros and the click goes to (0, 0). A user opens the panel before
        clicking in it, so this does too. Leaving this out failed ten op-form rows
        and all three Export rows -- none of which is a dead label.

        Finally the Snap Pixels gate. The Pixel art group is `display:none` until
        a document is open AND a frame is selected (`updateSnapGate`), and
        `display:none` ignores `open` -- so unfolding ancestors can never lay that
        group out, however many times it is tried. Its checkbox then has a zero
        rect and the click lands on the header. That is the page working as
        designed (the panel tells you to select something first), not a dead
        label, and it is the row that made this section fail for a day's worth of
        the wrong reason. The gate is satisfied here the way a user satisfies it.
        """
        if page.js("""(() => {
              const g = document.querySelectorAll('.chk')[%d]
                          .closest('.opgroup');
              return !!g && getComputedStyle(g).display === 'none';
            })()""" % i):
            page.js("if (!S.sel || !S.sel.size){ S.sel = new Set([0]); syncSel(); }")
            page.js("updateSnapGate()")
            time.sleep(0.2)
        page.js("""(() => {
          let el = document.querySelectorAll('.chk')[%d];
          while (el){
            if (el.tagName === 'DETAILS') el.open = true;
            el = el.parentElement;
          }
          return true;
        })()""" % i)
        page.js("document.querySelectorAll('.chk')[%d]"
                ".scrollIntoView({block: 'center'})" % i)
        time.sleep(0.25)
        pt = page.js("""(() => {
          const row = document.querySelectorAll('.chk')[%d];
          const sp = row.querySelector('span') || row;
          const r = sp.getBoundingClientRect();
          return [r.left + r.width/2, r.top + r.height/2];
        })()""" % i)
        page.click_at(pt[0], pt[1])
        return box_checked(i)

    def words_diag(i):
        """What a click at the row's text would actually hit.

        Returned as the failure detail, so a miss says *why* rather than leaving
        "the label does not work" as the only clue -- which is exactly the wrong
        conclusion to draw when the cause is a panel that is not laid out yet.
        The `gated` field is the one that took longest to find: a row inside the
        hidden Pixel art group has no box at all, so it looks identical to a dead
        label until you ask which ancestor is display:none.
        """
        return page.js("""(() => {
          const row = document.querySelectorAll('.chk')[%d];
          const sp = row.querySelector('span') || row;
          const r = sp.getBoundingClientRect();
          const x = r.left + r.width/2, y = r.top + r.height/2;
          const hit = document.elementFromPoint(x, y);
          const g = row.closest('.opgroup');
          return {rect: [Math.round(r.left), Math.round(r.top),
                         Math.round(r.width), Math.round(r.height)],
                  viewport: [window.innerWidth, window.innerHeight],
                  hit: hit ? (hit.tagName + '#' + (hit.id || '')) : null,
                  rowTag: row.tagName, disabled: row.querySelector(
                    'input[type="checkbox"]').disabled,
                  gated: !!g && getComputedStyle(g).display === 'none',
                  sel: S.sel ? S.sel.size : null};
        })()""" % i)

    def box_checked(i):
        return page.js("""(() => {
          const row = document.querySelectorAll('.chk')[%d];
          const b = row.querySelector('input[type="checkbox"]');
          return b ? b.checked : null;
        })()""" % i)

    def set_box(i, val):
        page.js("""(() => {
          const row = document.querySelectorAll('.chk')[%d];
          const b = row.querySelector('input[type="checkbox"]');
          b.checked = %s;
          b.dispatchEvent(new Event('change', {bubbles: true}));
          return b.checked;
        })()""" % (i, json.dumps(val)))

    _rows = page.js("""(() => {
      const out = [];
      document.querySelectorAll('.chk').forEach((row, i) => {
        const box = row.querySelector('input[type="checkbox"]');
        if (!box) return;
        out.push({i: i, id: box.id || "(no id)", tag: row.tagName,
                  disabled: box.disabled, checked: box.checked,
                  text: (row.querySelector('span') || {}).textContent || ''});
      });
      return out;
    })()""")
    _enabled = [r for r in _rows if not r["disabled"]]
    ok(len(_enabled) >= 5,
       "the panel's checkboxes are all there to be tested", _rows)
    _wrong = [(r["id"], r["tag"]) for r in _rows if r["tag"] != "LABEL"]
    ok(not _wrong,
       "every checkbox row is a <label>, so its words belong to the control",
       _wrong)

    # The gated rows, named rather than skipped. The Pixel art group is hidden
    # until a frame is selected, so Snap Pixels' own checkbox genuinely cannot be
    # clicked before then -- but "it is behind a gate" must not become a general
    # excuse for any row that will not click, so the gate is identified, its
    # membership asserted, and its release asserted too.
    _gate_js = """(() => {
      const out = [];
      document.querySelectorAll('.chk').forEach((row, i) => {
        const g = row.closest('.opgroup');
        if (g && getComputedStyle(g).display === 'none')
          out.push({i: i, group: (g.querySelector('summary') || {}).textContent || ''});
      });
      return out;
    })()"""
    _gated = page.js(_gate_js)
    ok(all(r["group"] == "Pixel art" for r in _gated),
       "the only rows that cannot be clicked are the ones the page gates behind a "
       "selection", _gated)
    page.js("S.sel = new Set([0]); syncSel(); updateSnapGate()")
    time.sleep(0.25)
    _still = page.js(_gate_js)
    ok(not _still,
       "and selecting a frame opens that gate, which is what the panel's own "
       "hint tells the user to do", _still)

    for _r in _enabled:
        _was = box_checked(_r["i"])
        _now = click_words(_r["i"])
        _d = None if _now != _was else words_diag(_r["i"])
        ok(_now != _was,
           "clicking the words %r toggles %s"
           % (_r["text"][:38], _r["id"]), (_was, _now, _d))
        # Put it back, so nothing downstream inherits a flipped box.
        set_box(_r["i"], _was)

    # The one that was reported, end to end: tick it by its words alone.
    _ow_i = next(r["i"] for r in _enabled if r["id"] == "overwrite")
    ok(click_words(_ow_i) is True,
       "the overwrite box can be ticked by clicking its words")
    set_box(_ow_i, False)

    print("\n=== 16j. the audio library: list a folder, drag a clip onto a frame ===")
    # Reported as "Audio Librarry disapeard". The panel had been removed
    # deliberately, on the argument that Add sound is a file dialog and needs no
    # folder; that turned out to be wrong for how it is actually used, and the
    # server routes had survived the removal the whole time. So this section is
    # about the panel and, above all, about the **drag**: it is the one gesture
    # neither of the other suites can reach, because both of them post JSON and
    # never touch a drop target.
    lib = mkdtemp(prefix="drive16j_")

    def _wav(name, n=8, sub=None):
        d = os.path.join(lib, sub) if sub else lib
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        with wave.open(p, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * n)
        return p

    _wav("hit.wav", 8000)
    _wav("step.wav", 8000)
    _wav("blip.wav", 8000)
    _wav("boss_death.wav", 8000, sub="deep")   # a clip below the top level
    _wav(".hidden.wav")                        # dotfile: not listed
    # One second each, not eight samples: the audition check below waits for the
    # clip to load and then clicks stop, and a 1 ms clip has already ended by
    # then -- onended clears the player, and the check would race the fixture.
    with open(os.path.join(lib, "notes.txt"), "w") as fh:
        fh.write("not audio\n")

    ok(page.js("!!document.getElementById('libDir')"
               " && !!document.getElementById('libGo')"),
       "the Audio panel has the library controls")

    # The panel is folded by default ("audio librarry default folded"), so it has
    # to be opened before anything in it can be laid out: a row inside a
    # display:none body has a zero rect and can be neither clicked nor dragged.
    # A real click on the heading, which also proves the heading is what opens it.
    _head = page.js("""(() => {const d = [...document.querySelectorAll(
      '.main > .col:not(.right) > details.panel')]
      .find(d => d.querySelector('summary').textContent.trim() === 'Audio Library');
      const r = d.querySelector('summary').getBoundingClientRect();
      return [r.left + r.width/2, r.top + r.height/2, d.open];})()""")
    ok(_head[2] is False,
       "the Audio Library panel starts folded, so the folder does not push the "
       "panels below it down the column", _head)
    page.click_at(_head[0], _head[1])
    time.sleep(0.3)
    ok(page.js("""[...document.querySelectorAll(
      '.main > .col:not(.right) > details.panel')]
      .find(d => d.querySelector('summary').textContent.trim() === 'Audio Library')
      .open""") is True,
       "and clicking its heading opens it")

    page.js("document.getElementById('libDir').value = %s" % json.dumps(lib))
    page.js("document.getElementById('libGo').click()")
    page.wait_for("document.querySelectorAll('#libRows .audiorow').length > 0",
                  "the library listing", timeout=120)
    # Waited on this section's own folder, not just on "some rows are there". A
    # folder remembered from an earlier session is listed at boot, and its rows
    # can arrive before this section's listing does -- so a `rows > 0` wait can be
    # satisfied by the stale listing, and every check below would then be
    # measuring the wrong folder. That is not hypothetical: a probe that left its
    # own folder in localStorage made this section fail on "the clip it fetched is
    # the one on the row that was clicked", because the row path was read from the
    # stale listing and the click landed on the real one. It went unnoticed for
    # three runs because the two fixture folders hold the same clip names, which
    # is exactly the shape of fixture that cannot tell two listings apart. The
    # rows' own `title` is the absolute path, so it is what the wait reads.
    _libkey = os.path.normcase(lib).lower()
    page.wait_for("""[...document.querySelectorAll('#libRows .audiorow')]
      .some(r => r.title.toLowerCase().indexOf(%s) >= 0)""" % json.dumps(_libkey),
                  "this section's own folder to be the one listed", timeout=120)
    _rels = page.js("""[...document.querySelectorAll('#libRows .audiorow')]
      .map(r => r.querySelector('.nm').textContent)""")
    _want = sorted(["blip.wav", "hit.wav", "step.wav",
                    os.path.join("deep", "boss_death.wav")])
    ok(sorted(_rels) == _want,
       "the listing finds every clip, including the one in a subfolder", _rels)
    _joined = " ".join(_rels)
    ok("notes.txt" not in _joined and "hidden" not in _joined,
       "and skips what is not audio, and what is hidden", _rels)
    ok(page.js("""[...document.querySelectorAll('#libRows .audiorow')]
                  .every(r => r.draggable)""") is True,
       "every listed clip is a drag source")
    # No size badge, and the size is gone from the tooltip too: "remove file
    # sizes at audio library". The path stays -- two clips can share a name in
    # different subfolders and the listing shows the relative one.
    ok(page.js("""[...document.querySelectorAll('#libRows .audiorow')]
                  .every(r => !/\\d+\\s*k\\b/.test(r.textContent)
                           && r.title.indexOf('KB') < 0)""") is True,
       "and no row shows a file size",
       page.js("[...document.querySelectorAll('#libRows .audiorow')]"
               ".map(r => [r.textContent, r.title])"))

    # --- play and stop on a listed clip --------------------------------------
    # Reported as "put plan and stop button on listted audio files". The listing
    # hands the page an absolute path and nothing else, so auditioning needs a
    # route that serves those bytes -- and the proof it works is the browser
    # decoding them, not the page saying it is playing.
    ok(page.js("""[...document.querySelectorAll('#libRows .audiorow')]
                  .every(r => r.querySelectorAll('button').length >= 2)""")
       is True,
       "every listed clip carries a play button and a stop button")
    _rowpath = page.js("LIB.items[0].path")
    _pbtn = page.js("""(() => {const b = document.querySelectorAll(
      '#libRows .audiorow')[0].querySelector('button');
      const r = b.getBoundingClientRect();
      return [r.left + r.width/2, r.top + r.height/2];})()""")
    page.click_at(_pbtn[0], _pbtn[1])
    page.wait_for("!!LIBPLAY.el && LIBPLAY.el.readyState >= 1",
                  "the clip to be fetched and decoded", timeout=60)
    _pl = page.js("""({rs: LIBPLAY.el.readyState,
      marked: document.querySelectorAll('#libRows .audiorow')[0]
              .classList.contains('playing'),
      src: decodeURIComponent(LIBPLAY.el.src)})""")
    ok(_pl["rs"] >= 1 and _pl["marked"] is True,
       "clicking play fetches that clip and the browser decodes it, so a listed "
       "clip can be auditioned", _pl)
    ok(os.path.normcase(_rowpath) in os.path.normcase(_pl["src"]),
       "and the clip it fetched is the one on the row that was clicked",
       (_rowpath, _pl["src"][-70:]))
    _sbtn = page.js("""(() => {const b = document.querySelectorAll(
      '#libRows .audiorow')[0].querySelectorAll('button')[1];
      const r = b.getBoundingClientRect();
      return [r.left + r.width/2, r.top + r.height/2];})()""")
    page.click_at(_sbtn[0], _sbtn[1])
    page.wait_for("!LIBPLAY.el", "the stop to land", timeout=30)
    ok(page.js("document.querySelectorAll('#libRows .audiorow')[0]"
               ".classList.contains('playing')") is False,
       "and stop clears it, so the mark means a clip is sounding right now")

    LIB_MIME = "application/x-sprite-clip"

    def drag_row_to(row_i, tx, ty, mime=LIB_MIME, payload=None):
        """Drag a library row onto a point, with the browser running the drag.

        The press and the travel are real mouse input, so the *browser* starts the
        drag and `Input.dragIntercepted` hands back the payload the page's own
        `dragstart` built. That is what makes this a test of the drop target
        rather than of a fabricated event.

        Two things cost real time here and both look identical to a dead drop
        handler, so they are worth keeping written down:

          * `dragEnter` is required. `dragOver` and `drop` on their own produce no
            drop at all -- measured, 0 drops.
          * the mouse has to travel all the way to the target. Chromium tracks the
            drag's position from those moves, so a mouse that stops short leaves
            the drag somewhere else and the drop is delivered there.

        `mime`/`payload` replace the drag's own data, which is how the
        not-our-drag case below is built.

        A row that is not there returns `([], None)` rather than dying on
        `undefined.getBoundingClientRect()`. The caller's `LIB_MIME in types`
        check then fails with an empty list, which is a labelled failure about
        the drag; a traceback out of here would instead report BROKEN, which
        means "the harness could not run" and hides the check that did its job.
        """
        if not page.js("document.querySelectorAll('#libRows .audiorow').length"):
            return [], None
        src = page.js(
            """(() => {const r = document.querySelectorAll('#libRows .audiorow')[%d]
               .getBoundingClientRect();
               return [r.left + r.width/2, r.top + r.height/2];})()""" % row_i)
        # Mark, not clear: section 17 reads this buffer to decide the console
        # stayed clean, so emptying it here would throw away every earlier
        # section's errors and leave that check passing on a short list.
        _ev0 = len(page.events)
        page.call("Input.setInterceptDrags", enabled=True)
        page.mouse("mouseMoved", src[0], src[1], "none", 0)
        page.mouse("mousePressed", src[0], src[1], "left", 1)
        for k in range(1, 7):
            page.mouse("mouseMoved", src[0] + (tx - src[0]) * k / 6.0,
                       src[1] + (ty - src[1]) * k / 6.0, "left", 1)
            time.sleep(0.06)
        page.mouse("mouseMoved", tx, ty, "left", 1)
        time.sleep(0.2)
        got = [e for e in page.events[_ev0:]
               if e.get("method") == "Input.dragIntercepted"]
        if not got:
            page.call("Input.setInterceptDrags", enabled=False)
            page.mouse("mouseReleased", tx, ty, "left", 0)
            return [], None
        data = got[-1]["params"]["data"]
        types = [i.get("mimeType") for i in data.get("items", [])]
        if mime != LIB_MIME:
            data = dict(data)
            data["items"] = [{"mimeType": mime, "data": payload or ""}]
        for kind in ("dragEnter", "dragOver", "drop"):
            page.call("Input.dispatchDragEvent", type=kind, x=float(tx),
                      y=float(ty), data=data)
            time.sleep(0.2)
        page.mouse("mouseReleased", tx, ty, "left", 0)
        page.call("Input.setInterceptDrags", enabled=False)
        time.sleep(0.6)
        return types, data

    # --- a sheet of this section's own, with a known frame count -------------
    # 16h leaves a document behind, but it has just been re-gridded to a single
    # column, so it holds two frames -- and the first version of this section said
    # `_tgt = 4` and died on `undefined.getBoundingClientRect()`. The target is
    # derived from the open document rather than assumed, and the document is this
    # section's own so a change over there cannot move the target out from under
    # it here.
    _jt = mkdtemp(prefix="drive16j_doc_")
    from PIL import Image as _PIL2
    _PIL2.new("RGBA", (96, 64), (12, 34, 56, 255)).save(
        os.path.join(_jt, "j_sheet.png"))
    with open(os.path.join(_jt, "j.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": "j", "sheet": "j_sheet.png", "columns": 3,
                   "rows": 2, "frame_width": 32, "frame_height": 32,
                   "fps": 12}, fh)
    _id_before = page.js("S.doc ? S.doc.id : null")
    page.js("document.getElementById('sheetPath').value = %s" % json.dumps(_jt))
    page.js("document.getElementById('openPath').click()")
    page.wait_for("!!S.doc && S.doc.id !== %s" % json.dumps(_id_before),
                  "the section's own fixture to open", timeout=180)
    page.wait_for("window.__d.busy()", "the open to settle", timeout=120)
    _nfr = page.js("S.doc.n")
    ok(_nfr == 6, "the sheet this section drops onto has six frames", _nfr)

    # --- onto a frame in the timeline ---------------------------------------
    # A frame that is not 0 and not the current one, so "it landed on the frame it
    # was dropped on" cannot be satisfied by a handler that ignores the target and
    # attaches to S.cur instead. Derived, and asserted, so it cannot silently
    # point past the end of the strip again.
    _tgt = min(4, _nfr - 1)
    ok(_tgt >= 1,
       "there is a frame to drop onto that is neither the first nor the current "
       "one", (_nfr, _tgt))
    page.js("S.cur = 0; syncSel(); renderPreview()")
    ok(page.js("S.cur") != _tgt,
       "the current frame is deliberately not the drop target, so the two cannot "
       "be confused", (page.js("S.cur"), _tgt))
    # Bring the target frame into view before aiming at it. The strip is
    # `overflow-x:auto` and 6 frames do not fit in it (scrollWidth 311 against a
    # clientWidth of 240), so frame 4's *rect* runs past the scrollport: its
    # centre lands ~9px inside the right edge, and a drag hovering there makes
    # Chromium autoscroll the strip. Measured, the strip ran to its maximum --
    # scrollLeft 71 of 71 -- while the pointer dwelt, which slid the aimed-at
    # frame 71px away and ended the drag with `dragend` and no `drop` at all:
    # the handler never ran, `attachFromLibrary` was never called, and every
    # check below failed exactly as if the drop target were dead code. Scrolling
    # the frame into view first puts the aim point well clear of the autoscroll
    # band, which is also what a user does before dropping onto a frame they
    # cannot see.
    page.js("""(() => {const c = document.querySelectorAll('#strip canvas')[%d];
      if (c) c.scrollIntoView({block: 'nearest', inline: 'center'});})()""" % _tgt)
    time.sleep(0.3)
    _xy = page.js(
        """(() => {const c = document.querySelectorAll('#strip canvas')[%d];
           const r = c.getBoundingClientRect();
           return [r.left + r.width/2, r.top + r.height/2];})()""" % _tgt)
    _dragged = page.js("""(() => {const r = document.querySelectorAll('#libRows .audiorow')[1];
      return r.querySelector('.nm').textContent;})()""")
    _n_before = page.js("(S.doc.audio || []).length")
    # What the drop aimed at, and what the drag actually did. A drop that goes
    # nowhere is completely silent -- the strip's handler returns without a word
    # when `closest('#strip canvas')` finds nothing, and a point outside the
    # viewport or under another element hit-tests to something that is not the
    # canvas at all. Both look exactly like a dead handler from the outside, so
    # the aim point, the strip's scroll state, and the events the drag produced
    # are recorded here and reported when the attach fails. The recorder is
    # document-wide and in the capture phase, so a drop that lands outside the
    # strip is seen too -- "the strip saw nothing" and "the drop went somewhere
    # else" are different diagnoses.
    _aim = page.js("""(() => {
      const [x, y] = %s;
      const e = document.elementFromPoint(x, y);
      const cv = e && e.closest && e.closest('#strip canvas');
      const st = document.getElementById('strip');
      const sr = st.getBoundingClientRect();
      window.__drops = [];
      const rec = (ev) => {
        const t = ev.target;
        window.__drops.push({
          t: ev.type, tag: t.tagName,
          canvas: Number(t.dataset ? t.dataset.i : NaN),
          at: [Math.round(ev.clientX), Math.round(ev.clientY)],
          scrollL: Math.round(st.scrollLeft),
        });
      };
      for (const t of ['dragenter', 'dragover', 'drop', 'dragleave', 'dragend'])
        document.addEventListener(t, rec, true);
      return {at: e ? e.tagName + '.' + String(e.className || '').slice(0, 16) : null,
              frame: cv ? Number(cv.dataset.i) : null,
              strip: [Math.round(sr.left), Math.round(sr.top),
                      Math.round(sr.width), Math.round(sr.height)],
              edge: Math.round(sr.right - x),
              scrollLeft: Math.round(st.scrollLeft),
              scrollWidth: st.scrollWidth, clientWidth: st.clientWidth,
              playing: !!S.playing, win: [innerWidth, innerHeight]};
    })()""" % json.dumps([_xy[0], _xy[1]]))
    # The aim point is a check of its own, because a miss here is what made this
    # section look like a dead drop target: every assertion below reads the
    # document, and a drag that never landed leaves it untouched.
    ok(_aim["frame"] == _tgt,
       "the drop aims at the frame it means to, and that frame is the element "
       "under the pointer rather than one beside it", (_aim, _tgt))
    _sl_before = page.js("Math.round(document.getElementById('strip').scrollLeft)")
    _types, _data = drag_row_to(1, _xy[0], _xy[1])
    _saw = page.js("window.__drops")
    _toast = page.js("document.getElementById('toast').textContent")
    _sl_after = page.js("Math.round(document.getElementById('strip').scrollLeft)")
    # Compact on purpose: this goes into five failure details, and a wall of JSON
    # is what stops a diagnostic being read. `canvas` arrives as None when the
    # target has no `data-i` (and NaN does not survive JSON), so it is formatted
    # rather than interpolated.
    _trail = " -> ".join(
        "%s:%s%s@%s sl=%s" % (e["t"], e["tag"],
                              "[%d]" % e["canvas"]
                              if isinstance(e["canvas"], int) else "",
                              e["at"], e["scrollL"])
        for e in (_saw or []))
    _diag = "aim=%s at=%s | %s | toast=%r" % (
        [round(v) for v in _xy], _aim, _trail or "(no drag events at all)",
        _toast)
    # The guard on the mechanism rather than on the symptom. The autoscroll is
    # what ends the drag, so a strip that moved under the pointer means this
    # section is testing Chromium's autoscroll and not the drop handler -- and
    # saying that out loud is the difference between a check that fails for a
    # reason and one that fails with a shrug.
    ok(_sl_before == _sl_after,
       "the strip did not scroll under the drag, so the frame that was aimed at "
       "is still the frame the drop landed on", (_sl_before, _sl_after, _diag))
    ok(LIB_MIME in _types,
       "the browser's own drag carries the library's private type, so the drop "
       "target decides from the payload rather than from a module-level flag",
       (_types, _diag))
    _rows = page.js("""[...document.querySelectorAll('#audioList .audiorow')]
      .map(r => r.textContent)""")
    ok(len(_rows) == _n_before + 1,
       "dropping it on a frame attaches it", (_n_before, _rows, _diag))
    _fr = page.js("(S.doc.audio || []).length"
                  " ? S.doc.audio[S.doc.audio.length-1].frame : -1")
    ok(_fr == _tgt,
       "and it landed on the frame it was dropped on, not on frame 0",
       (_fr, _tgt, _diag))
    ok(_dragged in " ".join(_rows),
       "the clip that was dragged is the clip that was attached",
       (_dragged, _rows, _diag))
    ok("\u266a" in page.js("document.querySelectorAll('#strip canvas')[%d].title"
                           % _tgt),
       "and the timeline marks that frame, so the drop is visible where it landed",
       (page.js("document.querySelectorAll('#strip canvas')[%d].title" % _tgt),
        _diag))

    # --- onto a cell of the sheet -------------------------------------------
    # Through the same screen->sheet mapping the pointer tools use, so a drop
    # lands on the frame a click there would have selected.
    _pt = page.js("""(() => {const r = document.getElementById('view')
      .getBoundingClientRect(); return [r.left + r.width*0.3, r.top + r.height*0.3];})()""")
    _want_cell = page.js("""(() => {const r = document.getElementById('view')
      .getBoundingClientRect();
      const [sx, sy] = screenToSheet(r.width*0.3, r.height*0.3);
      return cellAt(sx, sy);})()""")
    ok(_want_cell >= 0,
       "the point chosen on the sheet really is over a cell", _want_cell)
    _n2 = page.js("(S.doc.audio || []).length")
    _f2_before = page.js("(S.doc.audio || []).map(a => a.frame)")
    drag_row_to(0, _pt[0], _pt[1])
    _f2 = page.js("(S.doc.audio || []).map(a => a.frame)")
    ok(page.js("(S.doc.audio || []).length") == _n2 + 1,
       "a drag onto a sheet cell attaches too",
       page.js("(S.doc.audio || []).map(a => a.name + '@' + a.frame)"))
    # Looked up by frame, not read off the end of the list. `audio_payload` sorts
    # by (frame, id), so the last element is the *highest* frame and not the
    # newest clip -- which means `audio[length-1]` only answered this question
    # while the timeline drop above was failing to attach anything at frame 4.
    # Fixing that drop made this check fail with (4, 0): the sheet's clip was
    # there, at frame 0, with the frame-4 clip sitting behind it.
    ok(_want_cell in _f2 and _want_cell not in _f2_before,
       "to the frame of the cell it was dropped on", (_want_cell, _f2))

    # --- a listing that is no longer the one asked for is dropped ------------
    # boot() starts restoreLibDir() without awaiting it, so the remembered
    # folder's listing can still be in flight when the user types a folder and
    # clicks Go. If that first response lands second it replaces the listing the
    # user asked for with the one they did not: the panel shows one folder while
    # LIB.items hands out another folder's paths. This suite hit exactly that --
    # 16j read a row path belonging to the previous run's folder while the panel
    # showed this run's -- and it survived three runs because both fixture
    # folders hold the same clip names, which is the one shape of fixture that
    # cannot tell two listings apart. The first response is *held open* here, so
    # the wrong order happens by construction rather than by luck, which is what
    # makes this a check rather than a coin toss.
    _stale = mkdtemp(prefix="drive16j_stale_")
    with wave.open(os.path.join(_stale, "stale_only.wav"), "wb") as _w:
        _w.setnchannels(1)
        _w.setsampwidth(2)
        _w.setframerate(8000)
        _w.writeframes(b"\x00\x00" * 8000)
    _real = os.path.normcase(lib).lower()
    _fake = os.path.normcase(_stale).lower()
    # The first `audio_lib?` request is held open for longer than the whole
    # sequence takes, so whichever way the two responses race, the held one
    # arrives second. `api` is a top-level function declaration, so it is a
    # property of the global object and assigning over it really does redirect
    # loadLib's own call.
    _hold_first = """(() => {
      let held = false;
      window.api = (p, b, m) => {
        if (!held && String(p).indexOf('audio_lib?') >= 0){
          held = true;
          return new Promise(res => setTimeout(
            () => res(window.__api0(p, b, m)), 1200));
        }
        return window.__api0(p, b, m);
      };
      return 'ok';
    })()"""
    page.js("window.__api0 = window.api")
    page.js(_hold_first)
    page.js("document.getElementById('libDir').value = %s" % json.dumps(_stale))
    page.js("void loadLib({quiet: true})")            # held open, deliberately
    page.js("document.getElementById('libDir').value = %s" % json.dumps(lib))
    page.js("void loadLib()")
    page.wait_for("LIB.dir && LIB.items.length && LIB.items.every("
                  "i => i.path.toLowerCase().indexOf(%s) >= 0)" % json.dumps(_real),
                  "the newest listing to win", timeout=30)
    time.sleep(1.6)                                   # let the held one land
    ok(os.path.normcase(page.js("LIB.dir")).lower() == _real,
       "a listing that is no longer the one asked for is dropped, so the panel "
       "and the box cannot disagree about which folder is listed",
       (page.js("LIB.dir"), lib, _stale))
    _rows_now = page.js("""[...document.querySelectorAll('#libRows .audiorow')]
      .map(r => r.title)""")
    ok(_rows_now and all(_real in r.lower() for r in _rows_now),
       "and the rows on screen are the ones asked for, not the held-back "
       "listing", _rows_now)
    ok(not any(_fake in r.lower() for r in _rows_now),
       "with nothing from the superseded folder left in the panel", _rows_now)

    # The same rule for a superseded request that *fails*: its error must not
    # wipe the listing the newer request just filled, nor put "not a folder" in
    # the note under a box that no longer names that folder.
    page.js("window.api = window.__api0")
    page.js(_hold_first)
    page.js("document.getElementById('libDir').value = %s"
            % json.dumps(os.path.join(lib, "gone")))
    page.js("void loadLib({quiet: true})")            # held open, and doomed
    page.js("document.getElementById('libDir').value = %s" % json.dumps(lib))
    page.js("void loadLib()")
    page.wait_for("LIB.dir && LIB.items.length", "the good listing", timeout=30)
    time.sleep(1.6)
    _note_now = page.js("document.getElementById('libNote').textContent")
    ok(page.js("document.querySelectorAll('#libRows .audiorow').length") > 0
       and "not a folder" not in _note_now,
       "and a superseded request that fails leaves the panel and the note "
       "alone, rather than blaming the folder that is no longer in the box",
       (page.js("document.querySelectorAll('#libRows .audiorow').length"),
        _note_now))
    page.js("window.api = window.__api0")
    # Put the panel back to listing this section's own folder before anything
    # below drags out of it. A page that mishandles the superseded request leaves
    # the panel empty, and the drag below would then die on
    # `undefined.getBoundingClientRect()` -- which reports the mutant as BROKEN,
    # a harness that could not run, when in fact the check above caught it and
    # said so. Restoring the state this section found keeps the two apart.
    page.js("document.getElementById('libDir').value = %s" % json.dumps(lib))
    page.js("void loadLib()")
    page.wait_for("document.querySelectorAll('#libRows .audiorow').length > 0",
                  "the panel to be listing this section's folder again",
                  timeout=30)

    # --- a drag that is not the library's is ignored -------------------------
    # This is the guard for the design. The removed panel's drop targets consulted
    # a module-level "a library drag is in progress" flag, so anything that set
    # the flag could attach a clip, and when the panel went nothing set it and the
    # targets became dead code. Deciding from the payload cannot be orphaned.
    #
    # The payload is a *real clip path*, not a nonsense one: a foreign drag that
    # carries something unattachable is refused by the server anyway, so it would
    # pass whether or not the page checks the drag type -- the check would be
    # vacuous. Carrying a clip that would attach if it were accepted is what makes
    # "nothing was attached" mean the page turned it away.
    _n3 = page.js("(S.doc.audio || []).length")
    drag_row_to(0, _xy[0], _xy[1], mime="text/plain",
                payload=os.path.join(lib, "hit.wav"))
    ok(page.js("(S.doc.audio || []).length") == _n3,
       "a drag carrying only text/plain attaches nothing, even when it names a "
       "clip that would attach",
       page.js("(S.doc.audio || []).map(a => a.name + '@' + a.frame)"))

    # --- a folder that is not there -----------------------------------------
    page.js("document.getElementById('libDir').value = %s"
            % json.dumps(os.path.join(lib, "nope")))
    page.js("document.getElementById('libGo').click()")
    page.wait_for("document.getElementById('libNote').textContent"
                  ".indexOf('not a folder') >= 0",
                  "the refusal to reach the note", timeout=120)
    ok(page.js("document.querySelectorAll('#libRows .audiorow').length") == 0,
       "a folder that is not there lists nothing and says why",
       page.js("document.getElementById('libNote').textContent"))

    # --- the list scrolls inside the panel -----------------------------------
    # Reported as "put inside a scroll". A folder can hold hundreds of clips and
    # the column is a strip beside the sheet, so the list gets a fixed height and
    # scrolls rather than pushing the Audio panel off the bottom of the column.
    for _k in range(30):
        _wav("bulk%02d.wav" % _k, 4)
    page.js("document.getElementById('libDir').value = %s" % json.dumps(lib))
    page.js("document.getElementById('libGo').click()")
    page.wait_for("document.querySelectorAll('#libRows .audiorow').length > 20",
                  "the bulk listing", timeout=120)
    _n_bulk = page.js("document.querySelectorAll('#libRows .audiorow').length")
    _scr = page.js("""(() => {const h = document.getElementById('libRows');
      const cs = getComputedStyle(h);
      return {oy: cs.overflowY, sh: h.scrollHeight, ch: h.clientHeight};})()""")
    ok(_scr["oy"] in ("auto", "scroll") and _scr["sh"] > _scr["ch"],
       "a folder with more clips than fit scrolls inside the panel instead of "
       "pushing the panels below it down the column", (_scr, _n_bulk))
    # Scrolling to the end has to actually bring the last row into view. A
    # clipped list also has scrollHeight > clientHeight, and what separates the
    # two is whether the rows past the fold can be reached at all.
    page.js("document.getElementById('libRows').scrollTop = 1e6")
    time.sleep(0.2)
    _reach = page.js("""(() => {const h = document.getElementById('libRows');
      const rows = h.querySelectorAll('.audiorow');
      const r = rows[rows.length-1].getBoundingClientRect();
      const hr = h.getBoundingClientRect();
      return [r.top >= hr.top - 1 && r.bottom <= hr.bottom + 1,
              Math.round(h.scrollTop)];})()""")
    ok(_reach[0] is True and _reach[1] > 0,
       "and scrolling to the end brings the last clip into view, so the rows past "
       "the fold are reachable rather than clipped", _reach)

    # --- the folder is remembered -------------------------------------------
    # Written on a successful list, read back at boot. The read is the half that
    # is easy to skip, and a key nobody reads is not a remembered folder.
    _key = page.js("localStorage.getItem('sprite.audioLibDir')")
    ok(_key and os.path.normcase(_key) == os.path.normcase(lib),
       "listing a folder remembers it", (_key, lib))
    page.call("Page.navigate", url=args.url + "/editor")
    page.wait_for("document.readyState === 'complete'", "document load")
    page.wait_for("document.querySelectorAll('#libRows .audiorow').length > 20",
                  "the remembered folder to be listed at boot", timeout=180)
    ok(os.path.normcase(page.js("document.getElementById('libDir').value"))
       == os.path.normcase(lib),
       "and a reload puts it back in the box and lists it again",
       page.js("document.getElementById('libDir').value"))
    ok(page.js("document.querySelectorAll('#libRows .audiorow').length")
       == _n_bulk,
       "with the same clips in it",
       page.js("document.querySelectorAll('#libRows .audiorow').length"))

    # A remembered folder that will not open is forgotten rather than retried on
    # every visit -- the rule the last-open sheet already follows. Waited on the
    # note and not on the empty box: the box starts empty on a fresh load, so
    # waiting for that would pass before boot had even tried the folder.
    page.js("localStorage.setItem('sprite.audioLibDir', %s)"
            % json.dumps(os.path.join(lib, "gone")))
    page.call("Page.navigate", url=args.url + "/editor")
    page.wait_for("document.readyState === 'complete'", "document load")
    page.wait_for("document.getElementById('libNote').textContent"
                  ".indexOf('not a folder') >= 0",
                  "the remembered folder to be tried and refused", timeout=180)
    ok(page.js("localStorage.getItem('sprite.audioLibDir')") is None,
       "a remembered folder that no longer opens is forgotten rather than retried "
       "on every visit",
       page.js("localStorage.getItem('sprite.audioLibDir')"))
    # The key goes and the path stays. That is the pair the sheet's own failed
    # reopen already does -- boot() removes LAST_SHEET_KEY and leaves the path in
    # `sheetPath`, for the same reason: the error is what tells the user what
    # failed, and the path is what lets them fix it. This check used to assert the
    # box was emptied, which the page has never done and its own comment argues
    # against ("The path stays in the box so the failure is visible while the user
    # is looking at it"), so the check was measuring a behaviour that was never
    # implemented rather than a regression. It asserts the visible half now.
    ok(os.path.normcase(page.js("document.getElementById('libDir').value"))
       == os.path.normcase(os.path.join(lib, "gone")),
       "and the path stays in the box, so the folder that failed is still on "
       "screen to be corrected while the key that would retry it is gone",
       page.js("document.getElementById('libDir').value"))

    shutil.rmtree(lib, ignore_errors=True)
    shutil.rmtree(_jt, ignore_errors=True)

    print("\n=== 16k. a server older than the page has to say so ===")
    # The page is re-read from disk on every request while the server's Python is
    # frozen at start, so a server started before a change serves the old body of
    # a route that exists. That looks exactly like a broken feature from here, and
    # it was reported as one three times ("overwriting not working", "when i click
    # play button it says can not play and volume setting says http 404"). The
    # server compares its own load time against the files on disk, and this makes
    # it stale *on purpose*: touching an mtime changes nothing about the running
    # code, which is what makes it a safe positive control.
    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor.py")
    _st0 = os.stat(_src)

    def _reload_editor(what):
        page.call("Page.navigate", url=args.url + "/editor")
        page.wait_for("document.readyState === 'complete'", "document load")
        page.wait_for("!!document.getElementById('stale')", "the footer badge",
                      timeout=30)

    try:
        _reload_editor("the first look")
        page.wait_for("document.getElementById('stale').title"
                      ".indexOf('server pid') >= 0",
                      "the freshness check to report", timeout=60)
        ok(page.js("document.getElementById('stale').classList"
                   ".contains('on')") is False,
           "a server running the code on disk shows no staleness badge")
        ok(page.js("getComputedStyle(document.getElementById('stale')).display")
           == "none",
           "and the badge is genuinely off screen, not merely empty",
           page.js("getComputedStyle(document.getElementById('stale')).display"))

        os.utime(_src, (_st0.st_atime, _st0.st_mtime + 5))
        _reload_editor("the stale look")
        page.wait_for("document.getElementById('stale').classList"
                      ".contains('on')",
                      "the badge to appear", timeout=60)
        _bad = page.js("""({txt: document.getElementById('stale').textContent,
          disp: getComputedStyle(document.getElementById('stale')).display,
          title: document.getElementById('stale').title})""")
        ok("stale" in _bad["txt"] and _bad["disp"] != "none",
           "a server whose code has changed since it started says so on screen",
           _bad)
        ok("editor.py" in _bad["title"] and "Restart" in _bad["title"],
           "and names the file that changed and what to do about it",
           _bad["title"][:200])
    finally:
        os.utime(_src, (_st0.st_atime, _st0.st_mtime))

    _reload_editor("the stamp put back")
    page.wait_for("document.getElementById('stale').title"
                  ".indexOf('server pid') >= 0",
                  "the freshness check to report again", timeout=60)
    ok(page.js("document.getElementById('stale').classList.contains('on')")
       is False,
       "and putting the stamp back clears it, so the badge is not permanent")

    print("\n=== 16l. Match content size: select all, one click, no more pulsing ===")
    # The report: "some art inside cells are little small some little big this
    # couse a jump in the loop ... user will select all frames and app will auto
    # resize all at sime size". test_editor.py proves the op and the HTTP smoke
    # proves the route. What neither can reach is the *gesture*: the palette node
    # the user presses, the controls it generates, the All button in the timeline,
    # and -- the part unique to this suite -- the size measured on the client's
    # own decoded pixels, which is what is actually on screen. 16k reloaded the
    # page, so the helpers go back in first.
    page.js(HELPERS)
    ok(page.js("typeof window.__d.opNode") == "function",
       "the helpers are back after the reload")

    _node = page.js("""(() => {
      const d = window.__d.opNode('Match content size');
      if (!d) return null;
      const g = d.closest('details.opgroup');
      return {group: g ? g.querySelector('summary').textContent : null,
              help: (d.querySelector('summary').title || '').length};
    })()""")
    ok(_node is not None and _node["group"] == "Transform",
       "the palette has a Match content size node in Transform", _node)
    ok(_node and _node["help"] > 200,
       "and it carries the help text the summary tooltip shows",
       _node and _node["help"])

    _ctrl = page.js("""(() => {
      const inp = window.__d.opInputs('Match content size');
      return {tags: inp.map(e => e.tagName), vals: inp.map(e => e.value)};
    })()""")
    ok(_ctrl["tags"] == ["SELECT", "SELECT", "INPUT", "SELECT", "INPUT",
                         "SELECT"],
       "the form generated a control per argument, in order", _ctrl["tags"])
    ok(_ctrl["vals"] == ["height", "median", "0", "bottom-center", "0",
                         "bilinear"],
       "with the defaults the registry declares, so the one-click case needs no "
       "setup", _ctrl["vals"])

    _mdir = mkdtemp(prefix="drive16l_")
    try:
        _mopen = page.js("openSheet(%r)" % write_pulse_sheet(_mdir),
                         await_promise=True)
        ok(_mopen is True, "the pulsing sheet opened", _mopen)
        page.wait_for("!!(S.doc) && S.img.size === S.doc.n",
                      "every cell to load", timeout=60)
        _n16 = page.js("S.doc.n")
        _lay16 = page.js("S.doc.layout")

        def _boxes():
            return page.js("[...Array(S.doc.n).keys()]"
                           ".map(i => window.__d.cellBox(i))")

        def _hs(bs):
            return [None if b is None else b[3] - b[1] + 1 for b in bs]

        def _piv(bs):
            """bottom-center as `anchor_point` defines it: the box's middle
            column, one below its last opaque row. Restated rather than imported
            because the page has no copy of the rule -- the HTTP smoke is the
            suite that measures this against the server's own function, and it
            exists partly because a hand-rolled restatement of it was wrong."""
            return [None if b is None
                    else (b[0] + (b[2] - b[0] + 1) // 2, b[3] + 1) for b in bs]

        _b0 = _boxes()
        ok(_hs(_b0) == [30, 24, 20, 38],
           "the client decoded the reported sheet: heights 30/24/20/38",
           _hs(_b0))

        # The gesture, in the order the user makes it: All, then Apply.
        page.js("document.getElementById('selAll').click()")
        ok(page.js("S.sel.size") == _n16,
           "the All button selects every frame", page.js("S.sel.size"))
        _rev16 = page.js("window.__d.rev()")
        _ink16 = page.js("window.__d.previewInk()")
        _r16 = page.js("window.__d.applyOp('Match content size')")
        ok(_r16 == "clicked", "the node's own Apply button was pressed", _r16)
        page.wait_for("window.__d.busy() && S.doc.rev > %d" % _rev16,
                      "the op to complete", timeout=60)

        _b1 = _boxes()
        ok(max(_hs(_b1)) - min(_hs(_b1)) == 0,
           "the spread is gone -- every frame's subject is the same size on "
           "screen", "%s -> %s" % (_hs(_b0), _hs(_b1)))
        ok(_piv(_b1) == _piv(_b0),
           "and every subject still stands on the pixel it stood on, so the "
           "size fix did not become a position jump",
           "%s vs %s" % (_piv(_b1), _piv(_b0)))
        ok(page.js("S.doc.layout") == _lay16 and page.js("S.doc.n") == _n16,
           "the cell size, the grid and the frame count are untouched",
           page.js("S.doc.layout"))
        _msg16 = page.js("document.getElementById('status').textContent")
        ok("Match content size:" in _msg16 and "-> 27 px" in _msg16,
           "the status line under the canvas reports what it did", _msg16)
        print("      " + _msg16)
        # The preview panel is the surface that kept stale pixels the last time a
        # re-render was missed (see refreshCells). Frame 0's subject shrank from
        # 30 rows to 27, so its ink must have dropped -- if renderPreview() never
        # ran, the count would be exactly what it was.
        _ink16b = page.js("window.__d.previewInk()")
        ok(_ink16b < _ink16,
           "and the frame preview really redrew, rather than keeping the pixels "
           "it had", "%d -> %d" % (_ink16, _ink16b))

        # Undo is the way back, and it has to restore the sizes *and* the pivots.
        page.js("document.getElementById('undo').click()")
        page.wait_for("window.__d.busy() && document.getElementById('undo')"
                      ".disabled === true", "undo to complete", timeout=60)
        ok(_boxes() == _b0,
           "undo puts the original pixels back, size and position both",
           _hs(_boxes()))

        # A selection with nothing to measure has to say so rather than divide by
        # zero -- and the message has to land where the user is looking.
        page.js("S.sel = new Set([1]); syncSel()")
        _rev16b = page.js("window.__d.rev()")
        page.js("window.__d.applyOp('Match content size')")
        page.wait_for("window.__d.busy()", "the second op to settle",
                      timeout=60)
        _msg16b = page.js("document.getElementById('status').textContent")
        ok(page.js("window.__d.rev()") == _rev16b and "already" in _msg16b,
           "one frame is its own reference, so it reports that and changes "
           "nothing", _msg16b)
    finally:
        shutil.rmtree(_mdir, ignore_errors=True)

    print("\n=== 16m. Erase a colour takes its colour from a picker ===")
    # Reported as a request, not a bug: "matte repair erase a color add a color
    # picker". The op used to declare three int channels, which the generator
    # turns into three spin boxes -- so the user had to know the colour's numbers
    # to key it. The registry now declares one `color` arg, which the generator
    # already knew how to render as <input type="color">; this is the only suite
    # that can see whether that actually reaches the page as a picker.
    page.js(HELPERS)
    # Matte repair starts folded, exactly as a user finds it.
    page.js("(() => { const g = [...document.querySelectorAll('#ops details.opgroup')]"
            ".find(d => d.querySelector('summary').textContent === 'Matte repair');"
            " if (g) g.open = true; })()")
    _ec = page.js("""(() => {
      const d = window.__d.opNode('Erase a colour');
      if (!d) return null;
      const g = d.closest('details.opgroup');
      const inp = window.__d.opInputs('Erase a colour');
      return {group: g ? g.querySelector('summary').textContent : null,
              n: inp.length,
              tags: inp.map(e => e.tagName + (e.type ? ':' + e.type : '')),
              color: inp[0] ? inp[0].value : null};
    })()""")
    ok(_ec is not None and _ec["group"] == "Matte repair",
       "the palette has an Erase a colour node in Matte repair", _ec)
    ok(_ec and _ec["n"] == 4,
       "the form renders four controls where it used to render six", _ec)
    ok(_ec and _ec["tags"][0] == "INPUT:color",
       "the first of which is a real colour picker, not three number boxes",
       _ec and _ec["tags"])
    ok(_ec and _ec["color"] == "#000000",
       "defaulting to black, so the default behaviour is unchanged",
       _ec and _ec["color"])

    # The control has to be wired, not merely present. The subject in the sheet
    # already open is (230,230,230) with a red marker pixel inside it, so keying
    # #e6e6e6 must take the block and leave the marker -- which is what makes this
    # a colour key and not "erase everything".
    #
    # 'only border-connected regions' has to come OFF for this fixture, and that
    # is the op working rather than a nuisance: the block sits inside the frame
    # without touching its edge, so the topological guard protects it -- the same
    # property section 5 of the op-layer suite pins. A user keying an enclosed
    # colour untick that box, which is what this does.
    page.js("S.sel = new Set([0]); syncSel()")
    _b16 = page.js("window.__d.cellBox(0)")
    _rev16m = page.js("window.__d.rev()")
    _set16 = page.js("""(() => {
      const inp = window.__d.opInputs('Erase a colour');
      inp[0].value = '#e6e6e6';
      inp[0].dispatchEvent(new Event('input', {bubbles: true}));
      inp[1].value = 0;
      inp[1].dispatchEvent(new Event('input', {bubbles: true}));
      inp[2].checked = false;
      inp[2].dispatchEvent(new Event('change', {bubbles: true}));
      return inp[0].value + '/' + inp[1].value + '/' + inp[2].checked;
    })()""")
    ok(_set16 == "#e6e6e6/0/false",
       "the picker accepts the colour the user chose, and the connectivity box "
       "is unticked", _set16)
    _app16 = page.js("window.__d.applyOp('Erase a colour')")
    ok(_app16 == "clicked", "and its Apply button runs the op", _app16)
    page.wait_for("window.__d.busy() && S.doc.rev > %d" % _rev16m,
                  "the colour key to complete", timeout=60)
    _b16b = page.js("window.__d.cellBox(0)")
    ok(_b16 is not None and _b16[2] - _b16[0] > 1,
       "the subject was there before the key", _b16)
    ok(_b16b is not None and _b16b[2] - _b16b[0] == 0 and _b16b[3] - _b16b[1] == 0,
       "keying the subject's own colour leaves only the marker pixel behind, so "
       "the picker's value is the colour that was erased", "%s -> %s" % (_b16, _b16b))
    _msg16m = page.js("document.getElementById('status').textContent")
    ok(_msg16m == "Erase a colour: 1 cell",
       "and the status line names the op and the one frame it changed", _msg16m)

    print("\n=== 16n. undo leaves the zoom and the pan alone ===")
    # The report: "when undo do not change the zoom and position". Both handlers
    # used to end in fitView(), which is a pure function of the sheet's pixel size
    # and the viewport -- so with the sheet unchanged it recomputes the view that
    # is already there, and the only thing it can actually alter is a zoom or pan
    # the user set by hand. Undo is pressed *while looking at a pixel*, so being
    # thrown back to the fitted view every time makes that impossible. The fix is
    # refitView(was): re-fit only when the sheet's own pixel size moved.
    #
    # This is the only suite that can see it. test_editor.py never loads the page
    # and the HTTP smoke talks to the routes, so a view transform -- which exists
    # only in the browser -- is invisible to both.
    #
    # The check has two halves on purpose. "Undo does not move the view" alone is
    # satisfied by never touching the view, which would strand the sheet off
    # screen after undoing a re-grid -- so the second half requires a re-fit when
    # the sheet's pixel size really does move. It is the *distinction* that is
    # being pinned, not the absence of movement.
    page.js(HELPERS)
    _zdir = mkdtemp(prefix="drive16n_")
    try:
        _zopen = page.js("openSheet(%r)" % write_pulse_sheet(_zdir),
                         await_promise=True)
        ok(_zopen is True, "a sheet to zoom into is open", _zopen)
        page.wait_for("!!(S.doc) && S.img.size === S.doc.n",
                      "every cell to load", timeout=60)

        # A view the user set by hand: 200% and panned well off centre. The
        # fitted view of this 128x128 sheet is centred, so being off centre is
        # exactly what makes the assertion below able to tell the two apart --
        # without it, "the view did not change" would also pass if the view had
        # happened to already be the fitted one.
        page.js("window.__d.setView(2.0, 40, 90)")
        _s0 = page.js("S.view.s")
        _tx0 = page.js("S.view.tx")
        _ty0 = page.js("S.view.ty")
        _off = page.js("Math.abs(S.view.tx - (view.width / DPR"
                       " - SHEET().w * S.view.s) / 2)")
        ok(_off > 1,
           "the hand-set view is off centre, so it is not the fitted view and "
           "the checks below can tell the two apart", _off)

        # A pixel-only op first. match_size rewrites pixels and leaves the layout
        # alone, so nothing about the sheet's own pixel size moves.
        page.js("document.getElementById('selAll').click()")
        _zrev = page.js("window.__d.rev()")
        page.js("window.__d.applyOp('Match content size')")
        page.wait_for("window.__d.busy() && S.doc.rev > %d" % _zrev,
                      "the op to land", timeout=60)
        _zv1 = (page.js("S.view.s"), page.js("S.view.tx"), page.js("S.view.ty"))
        ok(_zv1 == (_s0, _tx0, _ty0),
           "applying an op does not move the view either", _zv1)

        _zbox = page.js("window.__d.cellBox(0)")
        page.js("document.getElementById('undo').click()")
        page.wait_for("window.__d.busy() && document.getElementById('undo')"
                      ".disabled === true", "the undo to land", timeout=60)
        _zv2 = (page.js("S.view.s"), page.js("S.view.tx"), page.js("S.view.ty"))
        ok(_zv2 == (_s0, _tx0, _ty0),
           "and undo leaves the zoom and the pan exactly where the user put "
           "them", "%s vs %s" % (_zv2, (_s0, _tx0, _ty0)))
        ok(page.js("window.__d.cellBox(0)") != _zbox,
           "even though it really did put the pixels back, so the view was kept "
           "by the undo and not by nothing having happened",
           "%s -> %s" % (_zbox, page.js("window.__d.cellBox(0)")))

        # Now the other half of the distinction: an undo that *does* move the
        # sheet's pixel size has to re-fit, or the sheet lands off screen. This is
        # what makes the check above "keep the view unless the sheet moved"
        # rather than "never touch the view".
        _lay0 = page.js("S.doc.layout")
        page.js("document.getElementById('selAll').click()")
        _zrev2 = page.js("window.__d.rev()")
        page.js("(() => { const inp = window.__d.opInputs('Set columns');"
                " inp[0].value = 4;"
                " inp[0].dispatchEvent(new Event('input', {bubbles: true}));"
                " return inp[0].value; })()")
        _zapp = page.js("window.__d.applyOp('Set columns')")
        ok(_zapp == "clicked", "Set columns runs", _zapp)
        page.wait_for("window.__d.busy() && S.doc.rev > %d" % _zrev2,
                      "the re-grid to land", timeout=60)
        ok(page.js("S.doc.layout.columns") == 4,
           "the sheet is four columns wide now, so its pixel size really moved",
           page.js("S.doc.layout"))

        def _fit_state():
            """Whether the view is the fitted one, stated as the two properties a
            user would notice -- the whole sheet is on screen, and it is centred
            -- rather than by recomputing fitView's formula, which would only
            test the restatement against itself."""
            return page.js("""(() => {
              const vw = view.width / DPR, vh = view.height / DPR;
              const {w, h} = SHEET();
              return {fits: S.view.s * w <= vw + 1 && S.view.s * h <= vh + 1,
                      off: Math.abs(S.view.tx - (vw - w * S.view.s) / 2),
                      s: S.view.s};
            })()""")

        _fit1 = _fit_state()
        ok(_fit1["fits"] and _fit1["off"] < 0.001,
           "re-gridding re-fits the view, because the sheet's pixel size moved",
           _fit1)

        page.js("document.getElementById('undo').click()")
        page.wait_for("window.__d.busy() && S.doc.layout.columns == 2",
                      "the re-grid undo to land", timeout=60)
        _fit2 = _fit_state()
        ok(_fit2["fits"] and _fit2["off"] < 0.001,
           "and so does undoing it -- the same kind of change, in reverse",
           _fit2)
        ok(page.js("S.doc.layout") == _lay0 and page.js("S.doc.n") == 4,
           "with the layout and the frame count back where they were",
           page.js("S.doc.layout"))
    finally:
        shutil.rmtree(_zdir, ignore_errors=True)

    print("\n=== 16o. pixelate (mesh) gets the same palette the snapper has ===")
    # The request: "add same palette from snap pixel to pixelate (mesh)". The op
    # layer proves the parsing and the mapping, and the HTTP smoke proves the
    # route carries it; what neither can see is the control -- whether the second
    # panel really renders the *same* picker, or a lookalike that writes a
    # different value. That is the whole request, so it is checked here.
    page.js(HELPERS)
    _pal_nodes = page.js("""(() => {
      const grab = label => {
        const d = window.__d.opNode(label);
        if (!d) return null;
        const b = d.querySelector('.palbtn');
        return {node: !!d, btn: !!b,
                label: b ? b.querySelector('.pname').textContent : null,
                swatch: b ? !!b.querySelector('canvas') : false,
                hidden: !!d.querySelector('input[type=hidden]')};
      };
      return {snap: grab('Snap Pixels'), mesh: grab('Pixelate (mesh)')};
    })()""")
    ok(_pal_nodes["mesh"] and _pal_nodes["mesh"]["node"],
       "the Pixelate (mesh) node is in the palette", _pal_nodes["mesh"])
    ok(_pal_nodes["mesh"] and _pal_nodes["mesh"]["btn"],
       "and it renders a palette picker where it used to render a text box",
       _pal_nodes["mesh"])
    ok(_pal_nodes["mesh"] and _pal_nodes["mesh"]["label"] == "Auto (k-means)",
       "defaulting to Auto, so the fitted palette is still the default",
       _pal_nodes["mesh"] and _pal_nodes["mesh"]["label"])
    ok(_pal_nodes["mesh"] and _pal_nodes["mesh"]["swatch"]
       and _pal_nodes["mesh"]["hidden"],
       "with the same swatch and the same hidden value Snap Pixels has",
       _pal_nodes["mesh"])
    ok(_pal_nodes["snap"] and _pal_nodes["snap"]["btn"]
       and _pal_nodes["snap"]["label"] == _pal_nodes["mesh"]["label"],
       "and it is the same control, not a lookalike: both start on the same "
       "name", (_pal_nodes["snap"] or {}).get("label"))

    # Opening it on the mesh node has to offer the same list. Guarded on the
    # button existing, so a page that lost the control reports labelled failures
    # above instead of dying here on a null dereference -- a traceback is a
    # failure too, but a cruder one that hides the checks it skipped.
    _opened = page.js(
        "(() => { const b = window.__d.opNode('Pixelate (mesh)')"
        ".querySelector('.palbtn'); if (!b) return 'no picker';"
        " b.click(); return 'clicked'; })()")
    ok(_opened == "clicked", "the mesh panel's picker can be opened", _opened)
    if _opened == "clicked":
        page.wait_for(
            "(() => { const p = [...document.querySelectorAll('.palpanel')]"
            ".find(x => !x.hidden);"
            " return !!p && p.querySelectorAll('.palrow').length > 5; })()",
            "the mesh palette dropdown to open")
        _mesh_pick = page.js(
            "(() => { const p = [...document.querySelectorAll('.palpanel')]"
            ".find(x => !x.hidden);"
            " if (!p) return null;"
            " const rows = [...p.querySelectorAll('.palrow')];"
            " const r = rows.find(x =>"
            " x.querySelector('span').textContent === 'PICO-8');"
            " if (!r) return {rows: rows.length, label: null, value: null,"
            "                 open: !p.hidden};"
            " r.click();"
            " const d = window.__d.opNode('Pixelate (mesh)');"
            " return {rows: rows.length,"
            "   label: d.querySelector('.pname').textContent,"
            "   value: d.querySelector('input[type=hidden]').value,"
            "   open: !p.hidden}; })()")
        ok(_mesh_pick and _mesh_pick["rows"] > 20,
           "it lists Auto plus the console presets, the same list",
           _mesh_pick and _mesh_pick["rows"])
        ok(_mesh_pick and _mesh_pick["label"] == "PICO-8"
           and _mesh_pick["open"] is False
           and len((_mesh_pick["value"] or "").split(",")) == 16,
           "picking a preset on the mesh panel writes the palette into the value "
           "the op sends, and closes the panel", _mesh_pick)
    else:
        _mesh_pick = None

    # ...and that value has to be what the op actually colours with. This is the
    # end of the request: a palette picked in the panel, and pixels on screen in
    # exactly those colours.
    _pico = page.js("""(() => {
      const d = window.__d.opNode('Pixelate (mesh)');
      const h = d && d.querySelector('input[type=hidden]');
      if (!h) return [];
      return h.value.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
    })()""")
    ok(_pico,
       "the picker left a palette in the form for the op to send", _pico)
    _odir = mkdtemp(prefix="drive16o_")
    try:
        _oopen = page.js("openSheet(%r)" % write_mesh_sheet(_odir),
                         await_promise=True)
        ok(_oopen is True, "a sheet the mesh detector can resolve is open", _oopen)
        page.wait_for("!!(S.doc) && S.img.size === S.doc.n",
                      "every cell to load", timeout=90)
        page.js("document.getElementById('selAll').click()")
        _orev = page.js("window.__d.rev()")
        _oapp = page.js("window.__d.applyOp('Pixelate (mesh)')")
        ok(_oapp == "clicked", "the node's own Apply button was pressed", _oapp)
        page.wait_for("window.__d.busy() && S.doc.rev > %d" % _orev,
                      "the mesh pixelate to complete", timeout=180)
        _omsg = page.js("document.getElementById('status').textContent")
        ok("palette from the picker" in _omsg,
           "the status line says the palette came from the picker", _omsg)
        print("      " + _omsg)
        _oseen = page.js("[...Array(S.doc.n).keys()]"
                         ".reduce((acc, i) =>"
                         " acc.concat(window.__d.cellColours(i)), [])")
        _oseen = set(_oseen)
        _ostray = sorted(_oseen - set(_pico))
        ok(_oseen and not _ostray,
           "and the pixels on screen are in the colours that were picked, and "
           "no others", _ostray[:4])
    finally:
        shutil.rmtree(_odir, ignore_errors=True)

    print("\n=== 16p. fill / tint gets the same picker ===")
    # The last op on three channel boxes, and the one with no tests at all until
    # now -- so nothing failed while it stayed on r/g/b after erase_color moved.
    # Rather than a third hand-written copy of "the control is a picker", this
    # asks the page's own registry which ops declare a colour and requires every
    # one of them to render a picker showing the colour it declared. A new op
    # that declares a colour is covered without anyone remembering to add it
    # here, and an op that quietly goes back to channel boxes fails.
    page.js(HELPERS)
    _survey = page.js("""(() => {
      const want = OPSCHEMA.filter(o => o.args.some(a => a.t === 'color'));
      return want.map(o => {
        const a = o.args.find(x => x.t === 'color');
        const d = window.__d.opNode(o.label);
        if (!d) return {name: o.name, label: o.label, found: false};
        const cs = [...d.querySelectorAll('input[type=color]')];
        return {name: o.name, label: o.label, found: true,
                arg: a.k, declared: a.d, colors: cs.length,
                value: cs[0] ? cs[0].value : null};
      });
    })()""")
    ok(_survey and len(_survey) >= 2,
       "more than one op declares a colour, so this is a survey rather than a "
       "special case", _survey)
    _bad = [s for s in (_survey or [])
            if not s.get("found") or s.get("colors") != 1]
    ok(not _bad,
       "every op that declares a colour renders exactly one picker", _bad)
    ok(any(s.get("name") == "fill" for s in (_survey or [])),
       "and fill is one of them, which is the op this section is about",
       [s.get("name") for s in (_survey or [])])

    # ...and the picker on Fill / tint has to reach the op, not just exist.
    # Colour starts folded, exactly as a user finds it.
    page.js("(() => { const g = [...document.querySelectorAll('#ops details.opgroup')]"
            ".find(d => d.querySelector('summary').textContent === 'Colour');"
            " if (g) g.open = true; })()")
    _fset = page.js("""(() => {
      const inp = window.__d.opInputs('Fill / tint');
      const c = inp.find(e => e.type === 'color');
      if (!c) return null;
      c.value = '#13ff38';
      c.dispatchEvent(new Event('input', {bubbles: true}));
      return c.value;
    })()""")
    ok(_fset == "#13ff38", "the fill panel's picker takes a colour", _fset)
    page.js("document.getElementById('selAll').click()")
    _frev = page.js("window.__d.rev()")
    _fapp = page.js("window.__d.applyOp('Fill / tint')")
    ok(_fapp == "clicked", "and its Apply button runs the op", _fapp)
    # Wait for the op to settle rather than for the revision to advance. If the
    # op refuses, a rev-based wait can never be satisfied and the section dies on
    # a timeout that says nothing -- and "the op refused" is exactly the failure
    # a wrong argument name produces, so it is the one that must stay labelled.
    page.wait_for("window.__d.busy()", "the fill to settle", timeout=60)
    _frev_after = page.js("window.__d.rev()")
    ok(_frev_after > _frev, "and the op actually ran", (_frev, _frev_after))
    _fcols = page.js("[...Array(S.doc.n).keys()].reduce((acc, i) =>"
                     " acc.concat(window.__d.cellColours(i)), [])")
    ok(_fcols and set(_fcols) == {"13FF38"},
       "and every opaque pixel on screen is the colour that was picked",
       sorted(set(_fcols or []))[:4])
    _fmsg = page.js("document.getElementById('status').textContent")
    ok("Fill / tint" in _fmsg, "with the status line naming the op", _fmsg)

    # The declared default is a property of a *freshly built* page, so it has to
    # be read from one. The survey above deliberately does not assert it: by the
    # time this section runs, 16m has already set Erase a colour's picker to
    # #e6e6e6 and applied it, and the control keeps what the user chose -- the
    # first version of this check compared that against the registry's #000000
    # and reported a bug in the page. Reloading is what makes "the default"
    # mean anything, and nothing after this point needs the open document.
    #
    # The remembered sheet is cleared first, exactly as section 1 does: every
    # fixture this run opened lived in a temp dir that has since been deleted, so
    # leaving the path set would make boot() reopen a file that is not there and
    # put a 404 in the console that section 17 would rightly fail on.
    page.js("try { localStorage.removeItem('sprite.lastSheet'); } catch(e) {}")
    page.call("Page.navigate", url=args.url + "/editor")
    page.wait_for("document.readyState === 'complete'", "the reload")
    page.wait_for("document.querySelectorAll('#ops details.op').length > 0",
                  "the op palette to rebuild", timeout=60)
    page.js(HELPERS)
    _defaults = page.js("""(() => {
      return OPSCHEMA.filter(o => o.args.some(a => a.t === 'color')).map(o => {
        const a = o.args.find(x => x.t === 'color');
        const d = window.__d.opNode(o.label);
        const c = d ? d.querySelector('input[type=color]') : null;
        return {name: o.name, declared: a.d, shown: c ? c.value : null};
      });
    })()""")
    _wrong = [s for s in (_defaults or []) if s.get("shown") != s.get("declared")]
    ok(_defaults and not _wrong,
       "and on a freshly loaded page every colour picker starts on the colour "
       "its op declares", _wrong)
    _ffresh = [s for s in (_defaults or []) if s.get("name") == "fill"]
    ok(_ffresh and _ffresh[0]["declared"] == "#ff0000",
       "with fill defaulting to the red the three channel boxes used to",
       _ffresh)

    print("\n=== 16q. the palette dropdown while the player is running ===")
    # The report: "when play buton active pallet drop downs not opening".
    #
    # The panel is `position:fixed`, measured off the button's rect, and closed
    # by a capture-phase window scroll listener. That listener used to close on
    # *any* scroll outside the panel. The one thing in this page that scrolls by
    # itself is markTimelineCurrent() -- `cur.scrollIntoView()` on the timeline
    # strip, which runs only while S.playing. So the click did open the panel and
    # the strip's own auto-scroll shut it again before it could be seen, which is
    # why the report names the play button specifically.
    #
    # The fix narrows the listener: only a scroll that could have moved the
    # button closes it -- the document, or an ancestor of the button. Both halves
    # are asserted here, because either one alone is satisfiable by a broken
    # listener: "always close" passes the second half, and "never close" passes
    # the first. The pre-fix listener was run against this section and fails the
    # playback half, which is what makes it a check rather than a description.
    #
    # The setup has to reach a *laid-out* button or none of this means anything.
    # Two things stand in the way, and the earlier scratch probe tripped over
    # both: the Operations panel is folded at boot, and updateSnapGate() hides
    # the whole Pixel art group unless a frame is selected. With neither, the
    # picker has a zero rect and the probe was opening a dropdown attached to a
    # button no user could have clicked.
    page.js(HELPERS)
    _q = open_by_name(page, "walk")
    ok(_q == "opened", "16q reopens the walk sheet", _q)
    page.wait_for("!!(S.doc) && S.img.size === S.doc.n", "the walk cells",
                  timeout=180)

    _gate_off = page.js("SNAP_GROUP && SNAP_GROUP.style.display")
    ok(_gate_off == "none",
       "with nothing selected the Pixel art group is hidden, so the picker is "
       "not a control at all", _gate_off)

    # Selecting a frame is what a user does before reaching for the palette, and
    # it is what puts the button on screen.
    page.js("S.sel.add(0); S.anchorCell = 0; syncSel()")
    page.js("[...document.querySelectorAll('.col > details.panel')]"
            ".find(d => d.querySelector('summary').textContent.trim()"
            " === 'Operations').open = true")
    settle(page)

    _q_geo = page.js("""(() => {
      const btn = window.__d.opNode('Snap Pixels').querySelector('.palbtn');
      const b = btn.getBoundingClientRect();
      const col = document.querySelector('.col');
      return {h: Math.round(b.height), y: Math.round(b.top),
              colH: col.scrollHeight, colC: col.clientHeight};
    })()""")
    ok(_q_geo and _q_geo["h"] > 0,
       "selecting a frame lays the picker out, so it is a real target",
       _q_geo)
    ok(_q_geo and _q_geo["colH"] > _q_geo["colC"] + 4,
       "and the ops column now overflows, so it can really scroll the button",
       _q_geo)

    # -- the half that must still close ------------------------------------ #
    # A scroll of the button's own column moves it out from under the fixed
    # panel, so the panel has to go. Without this the fix would be "never close
    # on scroll" and the dropdown would float away from its button.
    page.js("window.__d.opNode('Snap Pixels').querySelector('.palbtn').click()")
    ok(page.js("(() => { const p = [...document.querySelectorAll('.palpanel')]"
               ".find(x => !x.hidden); return !!p; })()"),
       "the picker opens on a click", None)
    _q_before = page.js("Math.round(window.__d.opNode('Snap Pixels')"
                        ".querySelector('.palbtn').getBoundingClientRect().top)")
    _q_scrolled = page.js("""(() => {
      const btn = window.__d.opNode('Snap Pixels').querySelector('.palbtn');
      let e = btn.parentElement;
      while (e && e !== document.documentElement){
        const st = getComputedStyle(e);
        if ((st.overflowY === 'auto' || st.overflowY === 'scroll') &&
            e.scrollHeight > e.clientHeight + 4){
          e.scrollTop = e.scrollTop + 60;
          return e.scrollTop;
        }
        e = e.parentElement;
      }
      return 'none';
    })()""")
    settle(page)
    _q_after = page.js("Math.round(window.__d.opNode('Snap Pixels')"
                       ".querySelector('.palbtn').getBoundingClientRect().top)")
    ok(_q_scrolled not in (None, "none", 0),
       "its own column really scrolls", _q_scrolled)
    ok(_q_after != _q_before,
       "which really moves the button out from under the panel",
       "%s -> %s" % (_q_before, _q_after))
    ok(not page.js("(() => { const p = [...document.querySelectorAll('.palpanel')]"
                   ".find(x => !x.hidden); return !!p; })()"),
       "so the panel closes: the fix narrowed the listener, it did not remove it")

    # The listener's predicate is a single `t.contains(btn)` -- the button moves
    # when one of its own ancestors scrolls, and the viewport is one of those,
    # because a viewport scroll reports `document` as its target and
    # `document.contains(btn)` is true. That is measured here rather than
    # asserted in a comment, because the two are only the same claim if the
    # event really does arrive with `document` as its target.
    #
    # It cannot be reached without help: `body{overflow:hidden}` and
    # `.app{height:100vh}` mean the viewport has nothing to scroll, which is why
    # the check below states that first and then forces the condition. An
    # explicit `t === document || t === document.scrollingElement` pair used to
    # sit in the predicate; the second clause was dead, since the target is
    # `document` and `document.scrollingElement` is `<html>`.
    _q_view = page.js("""(() => {
      const d = document.documentElement;
      return {can: d.scrollHeight > d.clientHeight + 4,
              scrollingElement: document.scrollingElement.tagName};
    })()""")
    ok(_q_view and not _q_view["can"],
       "as shipped the viewport cannot scroll at all, so that path needs forcing",
       _q_view)
    page.js("""(() => {
      document.documentElement.style.overflow = 'auto';
      document.body.style.overflow = 'visible';
      const pad = document.createElement('div');
      pad.id = '__qpad';
      pad.style.height = '3000px';
      document.body.appendChild(pad);
      window.__qt = [];
      window.addEventListener('scroll', (e) => {
        const t = e.target;
        window.__qt.push(t === document ? 'document'
          : (t === document.scrollingElement ? 'document.scrollingElement'
             : (t.id || t.tagName)));
      }, true);
      return 'ok';
    })()""")
    page.js("window.__qt.length = 0")
    page.js("window.__d.opNode('Snap Pixels').querySelector('.palbtn').click()")
    _q_doc_open = page.js("(() => { const p ="
                          " [...document.querySelectorAll('.palpanel')]"
                          ".find(x => !x.hidden); return !!p; })()")
    page.js("document.scrollingElement.scrollTop = 400")
    settle(page)
    _q_doc_targets = page.js("window.__qt") or []
    ok(_q_doc_open, "with the viewport forced scrollable the picker still opens")
    ok(page.js("document.scrollingElement.scrollTop") > 0,
       "and the document really scrolls", page.js(
           "document.scrollingElement.scrollTop"))
    ok("document" in _q_doc_targets,
       "arriving as a scroll of the document, which is what the predicate sees",
       _q_doc_targets[:4])
    ok(not page.js("(() => { const p = [...document.querySelectorAll('.palpanel')]"
                   ".find(x => !x.hidden); return !!p; })()"),
       "so a viewport scroll closes the panel too, without a clause of its own")
    # Put the viewport back: the playback half below waits on the strip, and a
    # page with a 3000px spacer in it is not the page under test.
    page.js("""(() => {
      const pad = document.getElementById('__qpad');
      if (pad) pad.remove();
      document.documentElement.style.overflow = '';
      document.body.style.overflow = '';
      document.scrollingElement.scrollTop = 0;
      return 'ok';
    })()""")
    settle(page)

    # -- the half that must not close --------------------------------------- #
    # Order matters and getting it wrong is how this stage first reported a
    # failure that was not one. There is a second close path -- the ordinary
    # click-outside-to-close:
    #
    #     document.addEventListener("click", () => { panel.hidden = true; });
    #
    # The palette button calls e.stopPropagation() so its own click does not
    # reach that listener, but a click on Play does. So playback is started
    # FIRST and the picker opened second; opening the picker and then pressing
    # Play measures the click-outside path, not the scroll path.
    page.js("if (S.playing){ document.getElementById('play').click(); }")
    page.js("document.getElementById('play').click()")
    page.wait_for("S.playing === true", "playback to start", timeout=10)
    page.js("""(() => {
      window.__qscroll = [];
      window.addEventListener('scroll', (e) => {
        if (window.__qscroll.length > 60) return;
        const t = e.target;
        window.__qscroll.push(t === document ? 'document'
          : (t.id || t.className || t.tagName));
      }, true);
      return 'ok';
    })()""")
    _q_y_play = page.js("Math.round(window.__d.opNode('Snap Pixels')"
                        ".querySelector('.palbtn').getBoundingClientRect().top)")
    page.js("window.__d.opNode('Snap Pixels').querySelector('.palbtn').click()")
    ok(page.js("(() => { const p = [...document.querySelectorAll('.palpanel')]"
               ".find(x => !x.hidden); return !!p; })()"),
       "while playing, the picker still opens on a click")

    page.wait_for("(() => { const s = document.getElementById('strip');"
                  " return s.scrollLeft > 0; })()",
                  "the timeline to follow the playhead", timeout=30)
    settle(page)
    _q_targets = page.js("window.__qscroll") or []
    _q_strip = page.js("Math.round(document.getElementById('strip').scrollLeft)")
    ok(_q_strip > 0, "and the strip really does auto-scroll while playing",
       _q_strip)
    ok("strip" in _q_targets,
       "which the page sees as a scroll of the strip, not of the column",
       _q_targets[:6])
    ok(page.js("(() => { const p = [...document.querySelectorAll('.palpanel')]"
               ".find(x => !x.hidden); return !!p; })()"),
       "the picker stays OPEN through the strip's auto-scroll (the reported bug)")
    ok(page.js("Math.round(window.__d.opNode('Snap Pixels')"
               ".querySelector('.palbtn').getBoundingClientRect().top)")
       == _q_y_play,
       "and the button never moved, which is why the close was spurious",
       "%s -> %s" % (_q_y_play,
                     page.js("Math.round(window.__d.opNode('Snap Pixels')"
                             ".querySelector('.palbtn').getBoundingClientRect()"
                             ".top)")))
    page.js("if (S.playing){ document.getElementById('play').click(); }")
    page.wait_for("S.playing !== true", "playback to stop", timeout=10)

    print("\n=== 16r. resize (pixels): a whole-number scale, exact on the "
          "client's own pixels ===")
    # The request: "add resize function", and the answer to *which* resize was
    # the pixel-art one -- nearest-neighbour, whole factors. The three size
    # controls that already existed all filter, and the user's own wording for
    # the gap was "nearest-neighbour for antialiazed?".
    #
    # test_editor.py section 30 pins the arithmetic on numpy and the HTTP smoke
    # pins the route; neither can see the two things that only exist in a
    # browser: whether the palette really renders the new op with its two
    # controls, and what the CLIENT has decoded after it runs. The second is the
    # point -- "nearest-neighbour" is a claim about pixels on screen, so the
    # pixels on screen are what this measures, on a fixture that punishes any
    # filter.
    page.js(HELPERS)
    _r_node = page.js("""(() => {
      const d = window.__d.opNode('Resize (pixels)');
      if (!d) return null;
      const g = d.closest('details.opgroup');
      return {group: g ? g.querySelector('summary').textContent : null,
              help: (d.querySelector('summary').title || '').length};
    })()""")
    ok(_r_node is not None and _r_node["group"] == "Pixel art",
       "the palette has a Resize (pixels) node in Pixel art", _r_node)
    ok(_r_node and _r_node["help"] > 200,
       "and it carries the help text the summary tooltip shows",
       _r_node and _r_node["help"])

    _r_ctrl = page.js("""(() => {
      const inp = window.__d.opInputs('Resize (pixels)');
      return {tags: inp.map(e => e.tagName), vals: inp.map(e => e.value),
              opts: inp[0] ? [...inp[0].options].map(o => o.value) : []};
    })()""")
    ok(_r_ctrl["tags"] == ["SELECT", "INPUT"],
       "the form generated a control per argument, in order", _r_ctrl["tags"])
    ok(_r_ctrl["vals"] == ["up", "2"],
       "defaulting to a doubling, so the common case needs no setup",
       _r_ctrl["vals"])
    ok(_r_ctrl["opts"] == ["up", "down"],
       "and the direction select offers both whole-number ways",
       _r_ctrl["opts"])

    _rdir = mkdtemp(prefix="drive16r_")
    try:
        _ropen = page.js("openSheet(%r)" % write_resize_sheet(_rdir),
                         await_promise=True)
        ok(_ropen is True, "a 4-frame 64x64 checkerboard sheet opened", _ropen)
        page.wait_for("!!(S.doc) && S.img.size === S.doc.n",
                      "every cell to load", timeout=60)
        ok(page.js("S.doc.layout.cell_w") == 64
           and page.js("S.doc.layout.cell_h") == 64,
           "opening at 64x64", page.js("S.doc.layout"))
        _r_cols0 = page.js("[...new Set([...Array(S.doc.n).keys()]"
                           ".reduce((a, i) =>"
                           " a.concat(window.__d.cellColours(i)), []))].sort()")
        ok(_r_cols0 == ["0000FF", "FFFF00"],
           "the client decoded the fixture as two hard colours, which is what "
           "makes the colour check below mean something", _r_cols0)
        _r_size0 = page.js("[...Array(S.doc.n).keys()]"
                           ".map(i => window.__d.cellSize(i))")
        _r_sig0 = page.js("[...Array(S.doc.n).keys()]"
                          ".map(i => window.__d.cellSig(i))")
        ok(_r_size0 == [[64, 64]] * 4, "every decoded cell is 64x64", _r_size0)

        # The view is forced to a zoom the sheet does not fit in. Without that
        # the refit asserted below is unobservable: fitView is a pure function of
        # the sheet's pixel size and the viewport, so a sheet that already fits
        # is refitted to exactly where it already was. This is the precondition
        # the check needs, not a claim about the product.
        page.js("window.__d.setView(8, 0, 0)")
        _r_pre = page.js("""(() => {
          const {w, h} = SHEET();
          const vw = view.width / DPR, vh = view.height / DPR;
          return {w, h, vw: Math.round(vw), vh: Math.round(vh), s: S.view.s,
                  over: w * S.view.s > vw - 40 || h * S.view.s > vh - 40};
        })()""")
        ok(_r_pre["over"],
           "the forced zoom really puts the sheet off screen, so the refit has "
           "something to do", _r_pre)

        page.js("document.getElementById('selAll').click()")
        ok(page.js("S.sel.size") == 4, "the All button selects every frame",
           page.js("S.sel.size"))
        _r_rev = page.js("window.__d.rev()")
        _r_app = page.js("window.__d.applyOp('Resize (pixels)')")
        ok(_r_app == "clicked", "the node's own Apply button was pressed", _r_app)
        page.wait_for("window.__d.busy() && S.doc.rev > %d" % _r_rev,
                      "the resize to complete", timeout=60)

        _r_msg = page.js("document.getElementById('status').textContent")
        ok("Resize (pixels):" in _r_msg and "64x64 -> 128x128" in _r_msg
           and "nearest-neighbour" in _r_msg,
           "the status line names the op, the size change and the method",
           _r_msg)
        print("      " + _r_msg)
        _r_lay = page.js("S.doc.layout")
        ok(_r_lay["cell_w"] == 128 and _r_lay["cell_h"] == 128
           and _r_lay["columns"] == 2 and _r_lay["rows"] == 2,
           "the document's own cell size and grid followed the pixels", _r_lay)
        ok(page.js("S.doc.n") == 4, "and no frame was dropped",
           page.js("S.doc.n"))
        _r_size1 = page.js("[...Array(S.doc.n).keys()]"
                           ".map(i => window.__d.cellSize(i))")
        _r_sig1 = page.js("[...Array(S.doc.n).keys()]"
                          ".map(i => window.__d.cellSig(i))")
        ok(_r_size1 == [[128, 128]] * 4,
           "and the client decoded every cell at the new size -- a payload that "
           "says 128 while the cell cache still holds 64 is exactly the failure "
           "this is here for", _r_size1)

        # The direct statement of "nearest-neighbour": each source pixel became a
        # 2x2 block of one colour. Measured in the page over the decoded pixels,
        # because the alternative -- a filter that happens to preserve the
        # extreme colours -- would pass a colour-set check on some fixtures.
        _r_blocks = page.js("""(() => {
          const out = {blocks: 0, bad: 0, first: null};
          for (let i = 0; i < S.doc.n; i++){
            const im = S.img.get(i);
            if (!im) return {error: 'cell ' + i + ' is not decoded'};
            const c = document.createElement('canvas');
            c.width = im.width; c.height = im.height;
            const g = c.getContext('2d');
            g.drawImage(im, 0, 0);
            const d = g.getImageData(0, 0, c.width, c.height).data;
            const px = (x, y) => { const k = (y * c.width + x) * 4;
              return d[k] + ',' + d[k+1] + ',' + d[k+2] + ',' + d[k+3]; };
            for (let y = 0; y + 1 < c.height; y += 2){
              for (let x = 0; x + 1 < c.width; x += 2){
                out.blocks++;
                const a = px(x, y);
                if (px(x + 1, y) !== a || px(x, y + 1) !== a
                    || px(x + 1, y + 1) !== a){
                  out.bad++;
                  if (!out.first)
                    out.first = [i, x, y, a, px(x+1, y), px(x, y+1),
                                 px(x+1, y+1)];
                }
              }
            }
          }
          return out;
        })()""")
        ok(_r_blocks and _r_blocks.get("bad") == 0 and _r_blocks.get("blocks"),
           "every pixel on screen is a 2x2 block of a single colour -- the art "
           "was replicated, not filtered", _r_blocks)
        _r_cols1 = page.js("[...new Set([...Array(S.doc.n).keys()]"
                           ".reduce((a, i) =>"
                           " a.concat(window.__d.cellColours(i)), []))].sort()")
        ok(_r_cols1 == _r_cols0,
           "and no colour appeared that was not already on screen: a filter "
           "would have invented the blends between blue and yellow", _r_cols1)

        _r_post = page.js("""(() => {
          const {w, h} = SHEET();
          const vw = view.width / DPR, vh = view.height / DPR;
          return {w, h, s: S.view.s,
                  fits: w * S.view.s <= vw - 39 && h * S.view.s <= vh - 39};
        })()""")
        ok(_r_post["s"] < _r_pre["s"],
           "the view refitted, rather than leaving the doubled sheet at the zoom "
           "it was at", "%s -> %s" % (_r_pre["s"], _r_post["s"]))
        ok(_r_post["fits"], "so the whole sheet is on screen again", _r_post)

        # Undo is the way back, and it has to bring the pixels with it rather
        # than only the layout numbers.
        page.js("document.getElementById('undo').click()")
        page.wait_for("window.__d.busy() && S.doc.layout.cell_w === 64",
                      "undo to complete", timeout=60)
        _r_undo_size = page.js("[...Array(S.doc.n).keys()]"
                               ".map(i => window.__d.cellSize(i))")
        ok(_r_undo_size == _r_size0, "undo puts the 64x64 cells back",
           _r_undo_size)
        ok(page.js("[...Array(S.doc.n).keys()]"
                   ".map(i => window.__d.cellSig(i))") == _r_sig0,
           "with the original pixels, byte for byte, not a re-filtered "
           "approximation", None)

        page.js("document.getElementById('redo').click()")
        page.wait_for("window.__d.busy() && S.doc.layout.cell_w === 128",
                      "redo to complete", timeout=60)
        ok(page.js("[...Array(S.doc.n).keys()]"
                   ".map(i => window.__d.cellSize(i))") == _r_size1
           and page.js("[...Array(S.doc.n).keys()]"
                       ".map(i => window.__d.cellSig(i))") == _r_sig1,
           "redo doubles it to the same pixels it had", None)

        # The round trip is the property the op exists for: a user can scale up
        # to work on the art and scale back down without drifting.
        page.js("window.__d.opInputs('Resize (pixels)')[0].value = 'down'")
        page.js("window.__d.applyOp('Resize (pixels)')")
        page.wait_for("window.__d.busy() && S.doc.layout.cell_w === 64",
                      "the down to complete", timeout=60)
        _r_msg2 = page.js("document.getElementById('status').textContent")
        ok("down x2" in _r_msg2, "the second Apply ran the other direction",
           _r_msg2)
        ok(page.js("[...Array(S.doc.n).keys()]"
                   ".map(i => window.__d.cellSig(i))") == _r_sig0,
           "up x2 then down x2 returns the original pixels on screen, so the two "
           "are exact inverses rather than near-misses", None)

        # A factor the cell size does not divide is refused rather than
        # truncated -- striding past a partial block would silently drop a row of
        # art -- and the refusal has to reach the user, not only the server log.
        page.js("window.__d.opInputs('Resize (pixels)')[1].value = '3'")
        _r_rev3 = page.js("window.__d.rev()")
        _r_lay3 = page.js("S.doc.layout")
        page.js("window.__d.applyOp('Resize (pixels)')")
        page.wait_for("document.getElementById('status').textContent"
                      ".includes('does not divide')",
                      "the refusal to be surfaced", timeout=60)
        _r_msg3 = page.js("document.getElementById('status').textContent")
        ok("multiple of 3" in _r_msg3,
           "3 does not divide 64, so it is refused with the reason and the way "
           "out in the status line", _r_msg3)
        print("      " + _r_msg3)
        ok(page.js("window.__d.rev()") == _r_rev3,
           "and nothing was recorded: a refusal is not an edit",
           page.js("window.__d.rev()"))
        ok(page.js("S.doc.layout") == _r_lay3,
           "the document is exactly where it was", page.js("S.doc.layout"))
    finally:
        shutil.rmtree(_rdir, ignore_errors=True)

    print("\n=== 16s. transform box: the gizmo behind the transform_content "
          "op ===")
    page.js(HELPERS)
    _s_node = page.js("""(() => {
      const d = window.__d.opNode('Transform (box)');
      if (!d) return null;
      const g = d.closest('details.opgroup');
      return {group: g ? g.querySelector('summary').textContent : null,
              help: (d.querySelector('summary').title || '').length};
    })()""")
    ok(_s_node is not None and _s_node["group"] == "Transform",
       "the palette has a Transform (box) node in Transform", _s_node)
    ok(_s_node and _s_node["help"] > 200,
       "and it carries the help text the summary tooltip shows",
       _s_node and _s_node["help"])
    _s_ctrl = page.js("""(() => {
      const inp = window.__d.opInputs('Transform (box)');
      return {tags: inp.map(e => e.tagName), vals: inp.map(e => e.value)};
    })()""")
    ok(_s_ctrl["tags"] == ["INPUT", "INPUT", "INPUT", "INPUT",
                           "SELECT", "INPUT", "SELECT"],
       "the form generated a control per argument, in order", _s_ctrl["tags"])
    ok(_s_ctrl["vals"] == ["0", "0", "1", "0", "center", "0", "nearest"],
       "defaulting to the identity", _s_ctrl["vals"])

    _sdir = mkdtemp(prefix="drive16s_")
    try:
        _sopen = page.js("openSheet(%r)" % write_transform_sheet(_sdir),
                         await_promise=True)
        ok(_sopen is True, "the 4-frame transform fixture opened", _sopen)
        page.wait_for("!!(S.doc) && S.img.size === S.doc.n",
                      "every cell to load", timeout=60)
        page.js("S.view.s = 4; S.view.tx = 0; S.view.ty = 0; "
                "S.tool = 'select'; S.sel.clear(); S.cur = 0; draw()")
        ok(page.js("S.xform") is False,
           "the box is off until asked for")
        page.js("document.getElementById('toolBox').click()")
        ok(page.js("S.xform") is True, "the Box button turns it on")
        ok(page.js("document.getElementById('toolBox').classList.contains('on')"),
           "and shows as on")
        _s_box0 = page.js("window.__d.cellBox(0)")
        ok(_s_box0 == [24, 24, 39, 39],
           "the content box is the 16x16 block the fixture drew", _s_box0)
        ok(_s_box0 != [0, 0, 63, 63],
           "which is not the cell rectangle", _s_box0)

        # A scale: drag the se corner to double its distance from the opposite.
        _s_rev = page.js("window.__d.rev()")
        _s_h = page.js("""(() => {
          const r = view.getBoundingClientRect();
          const g = xformPoints();
          return {se: [r.left + g.pts.se[0], r.top + g.pts.se[1]],
                  rot: [r.left + g.pts.rot[0], r.top + g.pts.rot[1]],
                  pivot: g.L.pivot};
        })()""")
        _s_se = _s_h["se"]
        _s_tgt = page.js("""(() => {
          const r = view.getBoundingClientRect();
          const {cw, ch, cols} = SHEET(); const i = 0;
          const se = xformPoints().pts.se;
          const [sx, sy] = screenToSheet(se[0], se[1]);
          const lx = sx - (i % cols) * cw, ly = sy - Math.floor(i / cols) * ch;
          const p = """ + json.dumps(_s_h["pivot"]) + """;
          const dx = lx - p[0], dy = ly - p[1];
          const tx = p[0] + 2 * dx, ty = p[1] + 2 * dy;
          const t = cellPtScreen(i, tx, ty);
          return [r.left + t[0], r.top + t[1]];
        })()""")
        page.drag([_s_se,
                   [(_s_se[0] + _s_tgt[0]) / 2, (_s_se[1] + _s_tgt[1]) / 2],
                   _s_tgt])
        page.wait_for("window.__d.rev() > %d" % _s_rev,
                      "the scale op to land", timeout=60)
        ok(page.js("window.__d.rev()") > _s_rev,
           "the release sent an op")
        _s_msg = page.js("document.getElementById('status').textContent")
        ok("Transform (box)" in _s_msg and "scale" in _s_msg,
           "the status line names the op and the scale", _s_msg)
        _s_box1 = page.js("window.__d.cellBox(0)")
        _s_area1 = (_s_box1[2] - _s_box1[0] + 1) * (_s_box1[3] - _s_box1[1] + 1)
        ok(_s_area1 > 256,
           "the art grew -- the box's drag was not a no-op", _s_box1)

        # Undo restores.
        _s_rev2 = page.js("window.__d.rev()")
        page.js("document.getElementById('undo').click()")
        page.wait_for("window.__d.rev() !== %d" % _s_rev2,
                      "undo to complete", timeout=60)
        ok(page.js("window.__d.cellBox(0)") == [24, 24, 39, 39],
           "undo restores the original box")

        # Rotate: drag the rot handle to a point 90 degrees clockwise on screen.
        _s_rev3 = page.js("window.__d.rev()")
        _s_rot = page.js("""(() => {
          const r = view.getBoundingClientRect();
          const g = xformPoints();
          return {start: [r.left + g.pts.rot[0], r.top + g.pts.rot[1]],
                  pivot: g.L.pivot};
        })()""")
        _s_rtgt = page.js("""(() => {
          const r = view.getBoundingClientRect();
          const {cw, ch, cols} = SHEET(); const i = 0;
          const [sx, sy] = screenToSheet(
            """ + str(_s_rot["start"][0]) + """ - r.left,
            """ + str(_s_rot["start"][1]) + """ - r.top);
          const lx = sx - (i % cols) * cw, ly = sy - Math.floor(i / cols) * ch;
          const p = """ + json.dumps(_s_rot["pivot"]) + """;
          const dx = lx - p[0], dy = ly - p[1];
          const tx = p[0] + dy, ty = p[1] - dx;
          const t = cellPtScreen(i, tx, ty);
          return [r.left + t[0], r.top + t[1]];
        })()""")
        page.drag([_s_rot["start"], _s_rtgt])
        page.wait_for("window.__d.rev() > %d" % _s_rev3,
                      "the rotate op to land", timeout=60)
        _s_box2 = page.js("window.__d.cellBox(0)")
        ok(_s_box2 == [24, 24, 39, 39],
           "a quarter turn of a square gives the same box back -- the aspect "
           "cannot swap when it is already square", _s_box2)
        # The pivot is still the box's centre, which is what "rotates about the
        # centre" means in the only measurable way.
        _s_cx2 = _s_box2[0] + ((_s_box2[2] - _s_box2[0] + 1) >> 1)
        _s_cy2 = _s_box2[1] + ((_s_box2[3] - _s_box2[1] + 1) >> 1)
        ok(_s_cx2 == 32 and _s_cy2 == 32,
           "and the centre stayed exactly where it was", (_s_cx2, _s_cy2))

        # Esc abandons a drag without sending anything.
        _s_rev4 = page.js("window.__d.rev()")
        _s_h2 = page.js("""(() => {
          const r = view.getBoundingClientRect();
          const g = xformPoints();
          return [r.left + g.pts.se[0], r.top + g.pts.se[1]];
        })()""")
        page.mouse("mouseMoved", _s_h2[0], _s_h2[1], "none", 0)
        page.mouse("mousePressed", _s_h2[0], _s_h2[1], "left", 1)
        page.mouse("mouseMoved", _s_h2[0] + 20, _s_h2[1] + 20, "left", 1)
        page.key("Escape")
        page.mouse("mouseReleased", _s_h2[0] + 20, _s_h2[1] + 20, "left", 0)
        time.sleep(0.5)
        ok(page.js("window.__d.rev()") == _s_rev4,
           "Escape abandons a drag without sending an op",
           page.js("window.__d.rev()"))
        ok(page.js("window.__d.cellBox(0)") == [24, 24, 39, 39],
           "and the art is untouched")

        # The pencil is not swallowed by the box.
        page.js("document.getElementById('toolPencil').click()")
        ok(page.js("S.tool") == "pencil", "the pencil tool selected")
        _s_in = page.js("""(() => {
          const r = view.getBoundingClientRect();
          const g = xformPoints();
          const p = cellPtScreen(g.L.cell, g.L.pivot[0], g.L.pivot[1]);
          return [r.left + p[0], r.top + p[1]];
        })()""")
        page.mouse("mouseMoved", _s_in[0], _s_in[1], "none", 0)
        page.mouse("mousePressed", _s_in[0], _s_in[1], "left", 1)
        _s_kind = page.js("drag ? drag.kind : null")
        page.mouse("mouseReleased", _s_in[0], _s_in[1], "left", 0)
        ok(_s_kind == "paint",
           "under the pencil a press inside the frame is a stroke, not a "
           "transform", _s_kind)
    finally:
        shutil.rmtree(_sdir, ignore_errors=True)

    print("\n=== 17. no console errors over the whole session ===")
    errs = page.errors()
    ok(not errs, "the browser console stayed clean",
       "\n        ".join(str(e)[:200] for e in errs[:5]))

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
