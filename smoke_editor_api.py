"""Integration smoke for the editor HTTP surface.

Opens a real sheet through the API, runs real operations, verifies, undoes, and
saves -- then checks the files on disk. This is the only check that exercises
the wiring (routes, doc registry, cell PNG cache, artifact paths); the op-layer
checks in test_editor.py never touch HTTP.

Requires the server:  python app.py --port 8765
Run:                  python smoke_editor_api.py
                      SPRITE_URL=http://127.0.0.1:8766 python smoke_editor_api.py

The URL is overridable so a change to a Python route can be tested against a
second server without restarting (and losing the open documents on) the one
that is already running.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request

BASE = os.environ.get("SPRITE_URL", "http://127.0.0.1:8765")
HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)

# This machine runs a local proxy that intercepts 127.0.0.1 and answers 502.
# curl needs --noproxy '*'; urllib needs an explicit empty ProxyHandler.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

FAILS = []
N = [0]


def ok(cond, label, detail=""):
    N[0] += 1
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", label,
                          "" if cond else "   " + str(detail)))
    if not cond:
        FAILS.append(label)


def req(path, body=None, method=None, raw=False):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers,
                               method=method or ("POST" if data else "GET"))
    with OPENER.open(r, timeout=180) as fh:
        b = fh.read()
    return b if raw else json.loads(b.decode("utf-8"))


# --------------------------------------------------------------------------- #
print("\n=== 1. the editor page and the op registry ===")
# --------------------------------------------------------------------------- #
html = req("/editor", raw=True).decode("utf-8", "replace")
ok("<canvas id=\"view\"" in html, "the editor page serves the viewport canvas")
ok("/api/editor/" in html, "the editor page talks to the editor API")

ops = req("/api/editor/ops")["ops"]
ok(len(ops) == 44, "the op registry exposes 44 operations", len(ops))
groups = sorted({o["group"] for o in ops})
ok(len(groups) == 9, "in 9 groups", groups)
ok(all("args" in o and "help" in o for o in ops),
   "every op carries its argument schema and help text")
ok(all(o["args"] for o in ops if o["name"] in
       ("offset", "align", "erase_border_black", "set_meta", "match_size")),
   "the configurable ops declare their arguments")

# The page is re-read from disk on every request while the server's Python is
# frozen at start, so a server started before a change serves the *old body of a
# route that exists* -- nothing 404s, nothing errors, and a fix that is on disk
# simply is not there. From the page that is indistinguishable from broken code,
# and it was reported as a bug three times ("overwriting not working", "when i
# click play button it says can not play and volume setting says http 404"). The
# route below compares the server's own load time against the files on disk, and
# the check makes it stale *on purpose*: touching an mtime changes nothing about
# the running code, which is exactly what makes it a safe positive control.
_v = req("/api/editor/version")
ok(_v["ok"] and _v["stale"] == [] and _v["pid"] > 0 and "started" in _v,
   "a server started from the current code does not call itself stale", _v)
# The badge prints `started` as the server's own start time, and the footer is
# the one place a user looks to decide whether their server is current -- so it
# has to be the boot time, not a file mtime dressed up as one. It was the mtime
# of the *oldest watched file*, which read as a start time only when the code had
# just been written, and which announced a two-day-old start for a server booted
# minutes earlier. The invariant that catches it: a fresh server (nothing stale)
# must have started no earlier than the newest file it loaded.
_watched = [os.path.join(HERE, "app.py"), os.path.join(HERE, "editor.py")]
_newest_src = max(os.path.getmtime(p) for p in _watched if os.path.isfile(p))
_started = time.mktime(time.strptime(_v["started"], "%Y-%m-%d %H:%M:%S"))
ok(_started + 1 >= _newest_src,
   "and its reported start time is when the process started, not a file mtime",
   "started %s vs newest source %s"
   % (_v["started"],
      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_newest_src))))
_app_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
_st0 = os.stat(_app_py)
try:
    os.utime(_app_py, (_st0.st_atime, _st0.st_mtime + 5))
    _v2 = req("/api/editor/version")
finally:
    os.utime(_app_py, (_st0.st_atime, _st0.st_mtime))
ok(_v2["stale"] == ["app.py"],
   "and a module written since it started is named", _v2)
ok(req("/api/editor/version")["stale"] == [],
   "with the stamp put back it is fresh again, so the report is not sticky")

# --------------------------------------------------------------------------- #
print("\n=== 2. the sheet library finds real sheets ===")
# --------------------------------------------------------------------------- #
lib = req("/api/sheets")["sheets"]
ok(len(lib) > 0, "the library found sheets", len(lib))
walk = [s for s in lib if s["group"] == "walk"]
ok(len(walk) >= 1, "it found the walk sheet", len(walk))
target = walk[0] if walk else lib[0]
print("      target: %s  (%dx%d grid, %s frames)"
      % (target["sheet"], target["columns"], target["rows"],
         target["frame_count"]))
ok(os.path.isfile(target["sheet"]), "the reported sheet path exists")

# --------------------------------------------------------------------------- #
print("\n=== 3. three ways to open a document ===")
# --------------------------------------------------------------------------- #
# The sidecar has to be discovered, and the naming conventions in this
# workspace differ: walk/ and sprites/ use <name>.json + <name>_sheet.png,
# a generated run uses <prefix>.json + <prefix>_sheet.png.
for label, payload in [
        ("by path alone", {"sheet": target["sheet"]}),
        ("with an explicit sidecar", {"sheet": target["sheet"],
                                      "sidecar": target["sidecar"]}),
        # the exact payload the browser sends: columns 0 means "not given"
        ("with the browser payload (columns:0, rows:0)",
         {"sheet": target["sheet"], "columns": 0, "rows": 0}),
]:
    dd = req("/api/editor/open", payload)["doc"]
    ok(dd["n"] == target["frame_count"] and
       dd["layout"]["cell_w"] == target["frame_width"],
       "open %s -> %d frames of %dx%d" % (label, dd["n"],
                                          dd["layout"]["cell_w"],
                                          dd["layout"]["cell_h"]))
    req("/api/editor/%s/close" % dd["id"], {})

# the document the rest of the run works on
d = req("/api/editor/open", {"sheet": target["sheet"]})["doc"]
did = d["id"]
ok(len(d["cell_rev"]) == d["n"], "every cell has a revision")
ok(d["undo"] == 0 and d["redo"] == 0, "a fresh document has empty history")
ok(d["dirty"] == [], "a fresh document has no modified cells")

# --------------------------------------------------------------------------- #
print("\n=== 4. cells are served as PNGs ===")
# --------------------------------------------------------------------------- #
png = req("/api/editor/%s/cell/0.png" % did, raw=True)
ok(png[:8] == b"\x89PNG\r\n\x1a\n", "cell 0 comes back as a PNG",
   png[:8])
ok(len(png) > 100, "the PNG has real content", "%d bytes" % len(png))
big = req("/api/editor/%s/cell/%d.png" % (did, d["n"] - 1), raw=True)
ok(big[:8] == b"\x89PNG\r\n\x1a\n", "the last cell comes back too")
try:
    req("/api/editor/%s/cell/%d.png" % (did, d["n"] + 5))
    ok(False, "an out-of-range cell is refused")
except urllib.error.HTTPError as ex:
    ok(ex.code == 404, "an out-of-range cell is refused", ex.code)

sheet = req("/api/editor/%s/sheet.png" % did, raw=True)
ok(sheet[:8] == b"\x89PNG\r\n\x1a\n", "the composed sheet is a PNG")
print("      composed sheet: %d bytes" % len(sheet))

# --------------------------------------------------------------------------- #
print("\n=== 5. verify before any edit ===")
# --------------------------------------------------------------------------- #
v = req("/api/editor/%s/verify" % did)
ok(v["ok"], "verify ran")
print("      " + "\n      ".join(v["lines"][:6]))

# --------------------------------------------------------------------------- #
print("\n=== 6. run real operations ===")
# --------------------------------------------------------------------------- #
sel = [0, 1, 2, 3]
r = req("/api/editor/%s/op" % did,
        {"name": "offset", "sel": sel, "args": {"dx": 3, "dy": -2, "wrap": False}})
ok(r["ok"], "offset succeeded")
ok(sorted(r["changed"]) == sel, "offset reported exactly the selected cells",
   r["changed"])
ok(r["doc"]["undo"] == 1, "one undo level recorded", r["doc"]["undo"])
ok(sorted(r["doc"]["dirty"]) == sel, "the four cells are marked modified",
   r["doc"]["dirty"])
rev_after = r["doc"]["cell_rev"]
ok(all(rev_after[i] != d["cell_rev"][i] for i in sel),
   "the changed cells got new revisions, so the client cache invalidates")
ok(all(rev_after[i] == d["cell_rev"][i] for i in range(4, d["n"])),
   "untouched cells kept their revision")

# a second op on a different axis
r2 = req("/api/editor/%s/op" % did,
         {"name": "alpha_gain", "sel": [0], "args": {"gain": 0.5}})
ok(r2["ok"], "alpha_gain succeeded")
ok(r2["changed"] == [0], "it reported only cell 0", r2["changed"])
ok(r2["doc"]["undo"] == 2, "two undo levels", r2["doc"]["undo"])

# a no-op must not push history
r3 = req("/api/editor/%s/op" % did,
         {"name": "offset", "sel": [0], "args": {"dx": 0, "dy": 0, "wrap": False}})
ok(r3["msg"] == "no change", "a no-op reports no change", r3["msg"])
ok(r3["doc"]["undo"] == 2, "a no-op pushes no undo level", r3["doc"]["undo"])

# a bad op name is refused without killing the document
r4 = req("/api/editor/%s/op" % did, {"name": "no_such_op", "sel": [], "args": {}})
ok(r4["ok"] is False and "unknown operation" in r4["error"],
   "an unknown op is refused", r4.get("error"))

# an op that must refuse: set_columns to a non-divisor
r5 = req("/api/editor/%s/op" % did,
         {"name": "set_columns", "sel": [], "args": {"columns": 7}})
ok(r5["ok"] is False, "set_columns(7) is refused")
if r5["ok"] is False:
    ok("divide" in r5["error"], "and says why", r5["error"])

# --------------------------------------------------------------------------- #
print("\n=== 7. verify after edits is honest about what it can still check ===")
# --------------------------------------------------------------------------- #
v2 = req("/api/editor/%s/verify" % did)
ok(v2["ok"], "verify still runs after edits")
joined = "\n".join(v2["lines"])
ok("I2 no backdrop leak" in joined, "I2 (self-contained) is still reported")
ok("modified" in joined or "no reference" in joined,
   "it states that I1/I4 no longer have a reference for the edited cells")
print("      " + "\n      ".join(v2["lines"][:6]))

# --------------------------------------------------------------------------- #
print("\n=== 8. undo / redo round-trip through HTTP ===")
# --------------------------------------------------------------------------- #
before = req("/api/editor/%s/cell/0.png" % did, raw=True)
u = req("/api/editor/%s/undo" % did, {})
ok(u["restored"], "undo reports a restore")
ok(0 in u["changed"], "undo reports cell 0 as changed", u["changed"])
ok(u["doc"]["undo"] == 1 and u["doc"]["redo"] == 1,
   "the entry moved from the undo to the redo stack",
   "%s/%s" % (u["doc"]["undo"], u["doc"]["redo"]))
after = req("/api/editor/%s/cell/0.png" % did, raw=True)
ok(after != before, "the cell PNG really changed on undo (the guard is live)")
rd = req("/api/editor/%s/redo" % did, {})
ok(rd["restored"], "redo reports a restore")
back = req("/api/editor/%s/cell/0.png" % did, raw=True)
ok(back == before, "redo returns the exact same PNG bytes",
   "%d vs %d bytes" % (len(back), len(before)))

# undo twice more, past the start
req("/api/editor/%s/undo" % did, {})
req("/api/editor/%s/undo" % did, {})
u2 = req("/api/editor/%s/undo" % did, {})
ok(u2["restored"] is False, "undo past the start is a clean no-op")

# --------------------------------------------------------------------------- #
print("\n=== 9. save writes real files ===")
# --------------------------------------------------------------------------- #
s = req("/api/editor/%s/save" % did,
        {"name": "smoke_api", "want_gif": True, "want_preview": True,
         "gif_scale": 0.5, "gif_bg": "checker"})
ok(s["ok"], "save succeeded", s.get("error"))
ok(s["served"], "the output landed inside the runs dir")
d_out = s["dir"]
print("      wrote to %s" % d_out)
for key in ("sheet", "sidecar", "preview", "gif"):
    ok(key in s["artifacts"], "the %s artifact was produced" % key)
for key, rel in s["artifacts"].items():
    p = os.path.join(d_out, os.path.basename(rel)) if not os.path.isabs(rel) else rel
    ok(os.path.isfile(p), "%s exists on disk" % key,
       "%s (%s)" % (p, rel))

sc_path = os.path.join(d_out, "smoke_api.json")
if os.path.isfile(sc_path):
    with open(sc_path, "r", encoding="utf-8") as fh:
        sc = json.load(fh)
    ok(sc["frame_count"] == d["n"], "the sidecar records the frame count")
    ok(sc["columns"] * sc["rows"] == sc["frame_count"],
       "the sidecar grid is full")
    ok(sc.get("edited", {}).get("ops"),
       "the sidecar records the edit history", sc.get("edited"))
    ok(sc["blend"] == "straight", "the sidecar keeps straight alpha")

# the saved sheet must be the layout the doc claims
try:
    from PIL import Image
    sp = os.path.join(d_out, "smoke_api_sheet.png")
    if os.path.isfile(sp):
        im = Image.open(sp)
        want = (d["layout"]["columns"] * d["layout"]["cell_w"],
                d["layout"]["rows"] * d["layout"]["cell_h"])
        ok(im.size == want, "the saved sheet matches the grid",
           "%s vs %s" % (im.size, want))
        ok(im.mode == "RGBA", "the saved sheet is RGBA", im.mode)
except ImportError:
    pass

# and the app can serve it back
served = req("/files/" + s["artifacts"]["sheet"], raw=True)
ok(served[:8] == b"\x89PNG\r\n\x1a\n", "the app serves the saved sheet back")

# --------------------------------------------------------------------------- #
print("\n=== 9b. save refuses to overwrite the sheet it was loaded from ===")
# --------------------------------------------------------------------------- #
# The output folder is a free-text field and the name defaults to the document's
# own name, so typing the source folder composes the source filename exactly.
# `src_sheet` was tracked for this and never consulted.
#
# Deliberately done on a COPY. The real walk sheet lives in the workspace, and a
# mutation run of the op-layer suite reverts this very guard -- pointing this at
# the real asset would let that mutation destroy it.
_tmp9b = tempfile.mkdtemp(prefix="smoke9b_")
_src_dir = os.path.join(_tmp9b, "src")
os.makedirs(_src_dir, exist_ok=True)
_src_sheet = os.path.join(_src_dir, "smoke_src_sheet.png")
_src_side = os.path.join(_src_dir, "smoke_src.json")
shutil.copyfile(target["sheet"], _src_sheet)
shutil.copyfile(target["sidecar"], _src_side)
_before = open(_src_sheet, "rb").read()

db = req("/api/editor/open", {"sheet": _src_sheet})["doc"]
ok(db["n"] == target["frame_count"],
   "a copy of the sheet opens for the clobber test", db["n"])

# Change the pixels, or "the file differs" would be true for the wrong reason
# when the overwrite finally succeeds.
req("/api/editor/%s/op" % db["id"], {"name": "flip_h", "sel": [], "args": {}})

try:
    r = req("/api/editor/%s/save" % db["id"],
            {"name": "smoke_src", "out_dir": _src_dir,
             "want_gif": False, "want_preview": False})
    ok(False, "saving over the loaded sheet is refused", r)
except urllib.error.HTTPError as ex:
    body = json.loads(ex.read().decode("utf-8"))
    ok(ex.code == 400, "saving over the loaded sheet is refused with 400", ex.code)
    ok("refusing to overwrite the sheet" in body.get("error", ""),
       "and the message names the file it would have destroyed",
       body.get("error"))
    ok("overwrite=true" in body.get("error", ""),
       "and names the opt-in", body.get("error"))
ok(open(_src_sheet, "rb").read() == _before,
   "the source sheet is byte-identical after the refusal")

# the route must actually thread `overwrite` through, or the refusal is a wall
r2 = req("/api/editor/%s/save" % db["id"],
         {"name": "smoke_src", "out_dir": _src_dir, "overwrite": True,
          "want_gif": False, "want_preview": False})
ok(r2["ok"], "overwrite:true gets through the route", r2.get("error"))
ok(open(_src_sheet, "rb").read() != _before,
   "and does replace the file -- so the route wires it, it is not ignored")
req("/api/editor/%s/close" % db["id"], {})
shutil.rmtree(_tmp9b, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 9c. export only the selected frames ===")
# --------------------------------------------------------------------------- #
# The route has to build the subset server-side and leave the open document
# alone. The document is the thing worth checking: a correct sheet plus a
# destroyed document is exactly the failure mode this option invites.
import tempfile as _tempfile  # noqa: E402
import shutil as _shutil  # noqa: E402

_tmp9c = _tempfile.mkdtemp(prefix="smoke9c_")
d9c = req("/api/editor/open", {"sheet": target["sheet"]})["doc"]
n_all = d9c["n"]
_sel = [3, 1, 2, 0, 5, 4]
try:
    r = req("/api/editor/%s/save" % d9c["id"],
            {"name": "subset", "out_dir": _tmp9c, "frames": _sel,
             "want_gif": False, "want_preview": False})
    ok(r["ok"], "saving a subset succeeded", r.get("error"))
    ok(r["frames"] == len(_sel),
       "the route reports how many frames it wrote", r.get("frames"))
    ok(r["of"] == n_all, "and how many the document has", r.get("of"))

    _sp = os.path.join(_tmp9c, "subset_sheet.png")
    _scp = os.path.join(_tmp9c, "subset.json")
    with open(_scp, "r", encoding="utf-8") as fh:
        sc9 = json.load(fh)
    ok(sc9["frame_count"] == len(_sel),
       "the written sidecar claims only the selected frames",
       sc9["frame_count"])
    try:
        from PIL import Image as _I
        im9 = _I.open(_sp)
        ok(im9.size == (sc9["columns"] * sc9["frame_width"],
                        sc9["rows"] * sc9["frame_height"]),
           "and the sheet's pixels match that grid", im9.size)
        ok(im9.size[0] * im9.size[1]
           < n_all * sc9["frame_width"] * sc9["frame_height"],
           "and the sheet really is smaller than the whole document's",
           "%s vs %d frames" % (im9.size, n_all))
    except ImportError:
        pass

    # The document must be untouched. There is no doc-info GET route, so ask the
    # server to compose the LIVE document and check its size: a subset export
    # that damaged the document would compose a smaller sheet.
    import io as _io  # noqa: E402
    try:
        from PIL import Image as _I2
        _whole = req("/api/editor/%s/sheet.png" % d9c["id"], raw=True)
        _im9 = _I2.open(_io.BytesIO(_whole))
        ok(_im9.size == (d9c["layout"]["columns"] * d9c["layout"]["cell_w"],
                         d9c["layout"]["rows"] * d9c["layout"]["cell_h"]),
           "the live document still composes to its full sheet after the subset "
           "export", "%s vs %s frames" % (_im9.size, n_all))
    except ImportError:
        pass
    _kept = [i for i in range(n_all) if i not in _sel][-1]
    _cell = req("/api/editor/%s/cell/%d.png" % (d9c["id"], _kept), raw=True)
    ok(_cell[:8] == b"\x89PNG\r\n\x1a\n",
       "and a frame that was not selected is still served (cell %d)" % _kept)

    # `frames: []` means "no subset asked for", NOT "export nothing". The box is
    # disabled while the selection is empty, so the UI cannot produce the
    # dangerous case, and a client that never sends the field must keep working.
    # Assert the outcome, not merely the absence of an error.
    _all = req("/api/editor/%s/save" % d9c["id"],
               {"name": "all", "out_dir": _tmp9c, "frames": [],
                "want_gif": False, "want_preview": False})
    ok(_all["frames"] == n_all,
       "an empty frames list exports the whole document, as an absent one does",
       _all.get("frames"))

    # ...and an out-of-range index must be refused by name, not silently dropped.
    try:
        req("/api/editor/%s/save" % d9c["id"],
            {"name": "oob", "out_dir": _tmp9c, "frames": [0, n_all]})
        ok(False, "a frame past the end is refused")
    except urllib.error.HTTPError as ex:
        body = json.loads(ex.read().decode("utf-8"))
        ok(ex.code == 400, "a frame past the end is refused with 400", ex.code)
        ok("is not in this document" in body.get("error", ""),
           "and the message names the frame", body.get("error"))
finally:
    req("/api/editor/%s/close" % d9c["id"], {})
    _shutil.rmtree(_tmp9c, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 10. a second document is independent ===")
# --------------------------------------------------------------------------- #
d2 = req("/api/editor/open", {"sheet": target["sheet"]})["doc"]
ok(d2["id"] != did, "a new document gets a new id")
ok(d2["undo"] == 0 and d2["dirty"] == [],
   "the new document does not inherit the first one's edits")
req("/api/editor/%s/close" % d2["id"], {})
try:
    req("/api/editor/%s/verify" % d2["id"])
    ok(False, "a closed document is gone")
except urllib.error.HTTPError as ex:
    ok(ex.code == 404, "a closed document is gone", ex.code)

# --------------------------------------------------------------------------- #
print("\n=== 11. bad input is refused with a usable message ===")
# --------------------------------------------------------------------------- #
# (a) not an image at all
try:
    req("/api/editor/open", {"sheet": os.path.join(HERE, "editor.html")})
    ok(False, "opening a non-image is refused")
except urllib.error.HTTPError as ex:
    body = ex.read().decode("utf-8", "replace")
    ok(ex.code == 400, "opening a non-image is refused with 400", ex.code)
    ok("cannot identify image" in body,
       "and names the real problem", body[:200])

# (b) a real image with no sidecar anywhere -> the grid is genuinely unknown
import tempfile
tmp = tempfile.mkdtemp(prefix="ss_nosidecar_")
try:
    from PIL import Image as _I
    lonely = os.path.join(tmp, "lonely_sheet.png")
    _I.new("RGBA", (64, 64), (0, 0, 0, 0)).save(lonely)
    try:
        req("/api/editor/open", {"sheet": lonely})
        ok(False, "an image with no sidecar is refused")
    except urllib.error.HTTPError as ex:
        body = ex.read().decode("utf-8", "replace")
        ok(ex.code == 400, "an image with no sidecar is refused with 400", ex.code)
        ok("sidecar" in body and "columns" in body,
           "and names the sidecar it looked for and the columns/rows fallback",
           body[:260])
    # ...and succeeds when the grid is supplied. The editor no longer offers the
    # boxes that used to do this, so this is an API-level capability now -- but it
    # is what keeps a bare sheet openable at all, so it stays covered.
    dd = req("/api/editor/open", {"sheet": lonely, "columns": 2, "rows": 2})["doc"]
    ok(dd["n"] == 4 and dd["layout"]["cell_w"] == 32,
       "the same image opens when columns/rows are given",
       "%s cells of %s" % (dd["n"], dd["layout"]["cell_w"]))
    req("/api/editor/%s/close" % dd["id"], {})
finally:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 12. per-frame audio: upload, serve, save, delete ===")
# --------------------------------------------------------------------------- #
import io as _io                                              # noqa: E402
import wave                                                   # noqa: E402


def _raw(path, data=b"", method="POST"):
    r = urllib.request.Request(BASE + path, data=data, method=method)
    with OPENER.open(r, timeout=60) as fh:
        return json.loads(fh.read().decode("utf-8"))


def _wav(n=8):
    b = _io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x00\x00" * n)
    return b.getvalue()


atmp = tempfile.mkdtemp(prefix="ss_audio_")
try:
    from PIL import Image as _I2
    _asheet = os.path.join(atmp, "a_sheet.png")
    _I2.new("RGBA", (64, 64), (0, 0, 0, 0)).save(_asheet)
    with open(os.path.join(atmp, "a.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": "a", "sheet": "a_sheet.png", "frame_width": 32,
                   "frame_height": 32, "columns": 2, "rows": 2,
                   "frame_count": 4, "fps": 24}, fh)
    adoc = req("/api/editor/open", {"sheet": _asheet})["doc"]
    adid = adoc["id"]
    ok(adoc["audio"] == [], "a fresh document has no audio", adoc["audio"])

    try:
        _raw("/api/editor/%s/audio?frame=0&name=x.wav" % adid, b"")
        ok(False, "an empty audio upload is refused with 400")
    except urllib.error.HTTPError as ex:
        ok(ex.code == 400, "an empty audio upload is refused with 400", ex.code)

    up = _raw("/api/editor/%s/audio?frame=2&name=hit.wav" % adid, _wav())
    ok(up["ok"] and up["frame"] == 2 and len(up["audio"]) == 1,
       "uploading a clip attaches it to the named frame", up)
    aid = up["id"]
    ok(up["audio"][0]["url"].endswith("/audio/%d" % aid),
       "the payload gives the page a URL to play it", up["audio"][0])

    got = req("/api/editor/%s/audio/%d" % (adid, aid), raw=True)
    ok(got == _wav(), "the clip is served back byte-for-byte", len(got))

    out12 = os.path.join(atmp, "out")
    saved = req("/api/editor/%s/save" % adid,
                {"name": "a", "out_dir": out12, "want_gif": False,
                 "want_preview": False})
    ok(saved["ok"], "saving a document that has audio succeeds", saved)
    with open(saved["artifacts"]["sidecar"], encoding="utf-8") as fh:
        scj = json.load(fh)
    ok(len(scj.get("audio", [])) == 1 and scj["audio"][0]["frame"] == 2,
       "the saved sidecar lists the clip on its frame", scj.get("audio"))
    afile = os.path.join(os.path.dirname(saved["artifacts"]["sidecar"]),
                         scj["audio"][0]["file"])
    ok(os.path.isfile(afile), "and the clip file ships beside the sheet", afile)

    dl = _raw("/api/editor/%s/audio/%d" % (adid, aid), method="DELETE")
    ok(dl["ok"] and dl["removed"] and dl["audio"] == [],
       "deleting a clip removes it from the document", dl)

    try:
        req("/api/editor/%s/audio/9999" % adid, raw=True)
        ok(False, "fetching a clip that does not exist is a 404")
    except urllib.error.HTTPError as ex:
        ok(ex.code == 404, "fetching a clip that does not exist is a 404", ex.code)

    req("/api/editor/%s/close" % adid, {})
finally:
    shutil.rmtree(atmp, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 12b. the audio library: a folder listed, a clip attached by path ===")
# --------------------------------------------------------------------------- #


def _refused(path, body=None):
    """The JSON body of a refusal. `req` raises on 4xx and the refusal is the
    thing being checked here, so it has to be caught and read."""
    try:
        return req(path, body)
    except urllib.error.HTTPError as ex:
        return json.loads(ex.read().decode("utf-8"))


ltmp = tempfile.mkdtemp(prefix="ss_alib_")
try:
    os.makedirs(os.path.join(ltmp, "sub", "deep"))
    os.makedirs(os.path.join(ltmp, ".hidden"))
    _hit = os.path.join(ltmp, "hit.wav")
    for _p, _n in ((_hit, 8), (os.path.join(ltmp, "sub", "foot.ogg"), 16),
                   (os.path.join(ltmp, "sub", "deep", "boom.flac"), 32),
                   (os.path.join(ltmp, ".hidden", "no.wav"), 8),
                   (os.path.join(ltmp, ".secret.wav"), 8)):
        with open(_p, "wb") as fh:
            fh.write(_wav(_n))
    open(os.path.join(ltmp, "notes.txt"), "w").write("not audio")
    open(os.path.join(ltmp, "empty.wav"), "wb").close()

    lib = req("/api/editor/audio_lib?dir=" + urllib.parse.quote(ltmp))
    rels = sorted(a["rel"] for a in lib["audio"])
    ok(rels == sorted(["hit.wav", "empty.wav",
                       os.path.join("sub", "deep", "boom.flac"),
                       os.path.join("sub", "foot.ogg")]),
       "the folder's clips are listed, subfolders included", rels)
    ok(not any("hidden" in r or "secret" in r or "notes" in r for r in rels),
       "hidden files, hidden folders and non-audio files are left out")
    _hitrow = next(a for a in lib["audio"] if a["rel"] == "hit.wav")
    ok(_hitrow["bytes"] == os.path.getsize(_hit)
       and _hitrow["ext"] == "wav" and os.path.isabs(_hitrow["path"]),
       "each entry carries an absolute path, its extension and its size", _hitrow)
    ok(lib["truncated"] is False,
       "and a folder this small is not reported truncated")

    ok(len(req("/api/editor/audio_lib?dir="
               + urllib.parse.quote('"%s"' % ltmp))["audio"]) == 4,
       "a path pasted with quotes around it still lists")
    ok(_refused("/api/editor/audio_lib?dir=").get("error")
       == "give a folder to list", "a blank folder is refused")
    ok("not a folder" in _refused(
        "/api/editor/audio_lib?dir=" + urllib.parse.quote(ltmp + "_nope")
    ).get("error"), "a folder that does not exist is refused")
    ok("not a folder" in _refused(
        "/api/editor/audio_lib?dir=" + urllib.parse.quote(_hit)
    ).get("error"), "a file is not a folder")

    from PIL import Image as _I3
    _bsheet = os.path.join(ltmp, "lib_sheet.png")
    _I3.new("RGBA", (64, 64), (0, 0, 0, 0)).save(_bsheet)
    with open(os.path.join(ltmp, "lib.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": "lib", "sheet": "lib_sheet.png", "frame_width": 32,
                   "frame_height": 32, "columns": 2, "rows": 2,
                   "frame_count": 4, "fps": 24}, fh)
    ldoc = req("/api/editor/open", {"sheet": _bsheet})["doc"]
    lid = ldoc["id"]

    r = req("/api/editor/%s/audio/from_path" % lid, {"frame": 3, "path": _hit})
    ok(r["ok"] and r["frame"] == 3 and len(r["audio"]) == 1,
       "a library clip attaches to the frame it was dropped on", r)
    ok(r["audio"][0]["name"] == "hit.wav",
       "and keeps its own name", r["audio"][0])
    ok(req("/api/editor/%s/audio/%d" % (lid, r["id"]), raw=True)
       == open(_hit, "rb").read(),
       "the staged copy is byte-identical to the library file")

    # Auditioning. The listing hands the page a path and nothing else, so it
    # cannot play a library clip without a route that serves those bytes -- and
    # the refusal set is the one the attach route applies, because this is the
    # other half of the same panel rather than a second way in.
    _q = "/api/editor/audio_lib/file?path=" + urllib.parse.quote(_hit)
    ok(req(_q, raw=True) == open(_hit, "rb").read(),
       "the audition route serves a listed clip's bytes")
    with OPENER.open(BASE + _q, timeout=30) as _fh:
        _ct = _fh.headers.get("Content-Type")
    ok("audio" in (_ct or ""),
       "served as audio, so the browser will actually play it", _ct)
    ok("not an audio file" in _refused(
        "/api/editor/audio_lib/file?path="
        + urllib.parse.quote(os.path.join(ltmp, "notes.txt"))).get("error"),
       "a file that is not audio is refused")
    ok("that clip is empty" in _refused(
        "/api/editor/audio_lib/file?path="
        + urllib.parse.quote(os.path.join(ltmp, "empty.wav"))).get("error"),
       "an empty clip is refused rather than served as a silent file")
    ok("no such file" in _refused(
        "/api/editor/audio_lib/file?path="
        + urllib.parse.quote(os.path.join(ltmp, "gone.wav"))).get("error"),
       "and a path that is not there is refused")
    ok("no such file" in _refused(
        "/api/editor/audio_lib/file?path="
        + urllib.parse.quote(ltmp)).get("error"),
       "a folder is not a clip")

    r2 = req("/api/editor/%s/audio/from_path" % lid,
             {"frame": 9999, "path": os.path.join(ltmp, "sub", "foot.ogg")})
    ok(r2["frame"] == ldoc["n"] - 1,
       "a frame past the end is clamped to the last frame", r2["frame"])

    # The volume control. It is the number save_doc writes into the sidecar, so
    # the route is what has to move it -- and a value out of range is refused
    # rather than quietly clamped, because a wrong number must not read as a
    # quiet clip.
    _aid = r["audio"][0]["id"]
    _vp = "/api/editor/%s/audio/%d/volume" % (lid, _aid)
    v = req(_vp, {"volume": 0.35})
    ok(v["ok"] and v["volume"] == 0.35 and v["audio"][0]["volume"] == 0.35,
       "a clip's volume can be set, and comes back in the payload", v)
    ok("between 0 and 1" in _refused(_vp, {"volume": 1.5}).get("error"),
       "a volume above full is refused, not clamped")
    ok("between 0 and 1" in _refused(_vp, {"volume": -0.2}).get("error"),
       "and so is one below silence")
    ok("must be a number" in _refused(_vp, {"volume": "loud"}).get("error"),
       "a volume that is not a number is refused")
    ok(_refused("/api/editor/%s/audio/999999/volume" % lid,
                {"volume": 0.5}).get("error") == "no such clip",
       "an unknown clip is refused")
    ok("unknown document" in _refused(
        "/api/editor/nope/audio/%d/volume" % _aid,
        {"volume": 0.5}).get("error"),
       "and so is an unknown document")

    # A clip that came from the library has to ship exactly like an uploaded one:
    # that is the whole reason it is staged rather than referenced in place.
    outb = os.path.join(ltmp, "out")
    savedb = req("/api/editor/%s/save" % lid,
                 {"name": "lib", "out_dir": outb, "want_gif": False,
                  "want_preview": False})
    ok(savedb["ok"], "a document whose clips came from the library saves", savedb)
    with open(savedb["artifacts"]["sidecar"], encoding="utf-8") as fh:
        scb = json.load(fh)
    ok([e["frame"] for e in scb.get("audio", [])] == [3, ldoc["n"] - 1],
       "the sidecar lists both clips on their frames", scb.get("audio"))
    ok(all(os.path.isfile(os.path.join(
               os.path.dirname(savedb["artifacts"]["sidecar"]), e["file"]))
           for e in scb["audio"]),
       "and both clip files ship beside the sheet", scb.get("audio"))
    ok(scb["audio"][0]["volume"] == 0.35,
       "and the volume set in the panel is the number the sidecar carries",
       scb["audio"][0])

    # Opening by the sidecar. The JSON is the document -- it names the sheet, the
    # grid and the clips -- and it is what the sheet library hands over, so this
    # is the route the panel actually uses.
    byj = req("/api/editor/open",
              {"sheet": savedb["artifacts"]["sidecar"]})["doc"]
    ok(byj["n"] == ldoc["n"] and len(byj["audio"]) == 2,
       "opening the sidecar JSON brings the document and its clips",
       (byj["n"], byj["audio"]))
    ok([e["frame"] for e in byj["audio"]] == [3, ldoc["n"] - 1],
       "on the frames they were saved on", byj["audio"])

    # Opening by the folder. The path box takes whatever the user typed, and
    # typing the folder the sheet lives in is at least as natural as naming the
    # JSON inside it, so the folder is resolved to the sidecar in it. The run
    # directory is the real test: it holds the sidecar, the sheet, the clips and
    # a preview page, and the sidecar is not named after the folder.
    _rundir = os.path.dirname(savedb["artifacts"]["sidecar"])
    bydir = req("/api/editor/open", {"sheet": _rundir})["doc"]
    ok(bydir["n"] == ldoc["n"] and len(bydir["audio"]) == 2,
       "opening the folder that holds the sidecar brings the same document",
       (bydir["n"], bydir["audio"]))
    ok(os.path.normcase(bydir["sidecar"])
       == os.path.normcase(savedb["artifacts"]["sidecar"]),
       "and says which sidecar it found, so the box can show it",
       bydir.get("sidecar"))
    req("/api/editor/%s/close" % bydir["id"], {})

    # A folder with no sidecar in it is not a document, and must not be guessed
    # at from the sheet sitting there: the grid would be invented.
    _empty = os.path.join(ltmp, "nosidecar")
    os.makedirs(_empty, exist_ok=True)
    shutil.copyfile(_bsheet, os.path.join(_empty, "lonely.png"))
    ok("sidecar" in _refused("/api/editor/open",
                             {"sheet": _empty}).get("error"),
       "a folder holding a sheet but no sidecar is refused, not guessed")

    orphan = os.path.join(ltmp, "orphan.json")
    with open(orphan, "w", encoding="utf-8") as fh:
        json.dump({"name": "orphan", "sheet": "not_here.png", "frame_width": 32,
                   "frame_height": 32, "columns": 2, "rows": 2}, fh)
    ok("names no sheet that is there" in _refused(
        "/api/editor/open", {"sheet": orphan}).get("error"),
       "a sidecar whose sheet is missing is refused by name, not guessed")
    req("/api/editor/%s/close" % byj["id"], {})

    ok("no such file" in _refused(
        "/api/editor/%s/audio/from_path" % lid,
        {"frame": 0, "path": os.path.join(ltmp, "nope.wav")}).get("error"),
       "a clip that is not there is refused")
    ok("not an audio file" in _refused(
        "/api/editor/%s/audio/from_path" % lid,
        {"frame": 0, "path": os.path.join(ltmp, "notes.txt")}).get("error"),
       "a file that is not audio is refused")
    ok("empty" in _refused(
        "/api/editor/%s/audio/from_path" % lid,
        {"frame": 0, "path": os.path.join(ltmp, "empty.wav")}).get("error"),
       "an empty clip is refused")
    ok("numbers" in _refused(
        "/api/editor/%s/audio/from_path" % lid,
        {"frame": "abc", "path": _hit}).get("error"),
       "a frame that is not a number is refused")
    ok(_refused("/api/editor/nope/audio/from_path",
                {"frame": 0, "path": _hit}).get("error") == "unknown document",
       "an unknown document is refused")

    _big = os.path.join(ltmp, "huge.wav")
    with open(_big, "wb") as fh:
        fh.write(b"\x00" * (65 * 1024 * 1024))
    ok("cap is 64 MB" in _refused(
        "/api/editor/%s/audio/from_path" % lid,
        {"frame": 0, "path": _big}).get("error"),
       "a clip over the size cap is refused")
    os.remove(_big)

    req("/api/editor/%s/close" % lid, {})
finally:
    shutil.rmtree(ltmp, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 13. match content size: one click, every frame the same size ===")
# --------------------------------------------------------------------------- #
# The report this exists for: "some art inside cells are little small some little
# big this couse a jump in the loop ... user will select all frames and app will
# auto resize all at sime size". test_editor.py proves the op does that to a
# document; this section proves the *route* carries it -- selection in, changed
# cells out, a scoped undo entry, and the cell PNGs the page re-fetches all
# showing what the op layer showed. The measurements below come from the bytes
# the browser is about to draw, not from anything the server says about itself.

import io as _io13                                                # noqa: E402
import re as _re13                                                # noqa: E402
import numpy as _np13                                             # noqa: E402
from PIL import Image as _I13                                     # noqa: E402
import editor as _E13                                             # noqa: E402

# Heights 30, 24, 20, 38 -- the 18 px spread a loop jumps on. The widths differ
# too, so `measure` has a real choice and `measure=width` is not the same test
# twice.
_MB13 = [(20, 20, 40, 50), (22, 22, 42, 46), (12, 20, 52, 40), (21, 14, 41, 52)]


def _sheet_raw(folder, name, cells):
    """Write a 2x2 sheet of 64x64 RGBA frames and return its sidecar path.

    Takes whole frames rather than boxes, so a fixture can put whatever it likes
    in one: a subject, a backdrop band, or nothing.
    """
    arr = _np13.zeros((128, 128, 4), _np13.uint8)
    for i, cell in enumerate(cells):
        row, col = divmod(i, 2)
        arr[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = cell
    # No mode argument: the array's shape already says RGBA, and passing it
    # explicitly is deprecated in Pillow 11 and warns on every run.
    _I13.fromarray(arr).save(os.path.join(folder, name + "_sheet.png"))
    side = os.path.join(folder, name + ".json")
    with open(side, "w", encoding="utf-8") as fh:
        json.dump({"name": name, "sheet": name + "_sheet.png",
                   "frame_width": 64, "frame_height": 64, "columns": 2,
                   "rows": 2, "frame_count": 4, "fps": 24}, fh)
    return side


def _sheet_of(folder, name, boxes):
    """The same sheet from subject boxes. `None` is a blank frame, which is the
    case a select-all run always ends up hitting."""
    cells = []
    for b in boxes:
        cell = _np13.zeros((64, 64, 4), _np13.uint8)
        if b is not None:
            x0, y0, x1, y1 = b
            cell[y0:y1, x0:x1] = (230, 230, 230, 255)
            cell[y0 + 2, x0 + 2] = (255, 0, 0, 255)   # marker, as in the op tests
        cells.append(cell)
    return _sheet_raw(folder, name, cells)


def _cell_arr(doc_id, i):
    raw = req("/api/editor/%s/cell/%d.png" % (doc_id, i), raw=True)
    return _np13.array(_I13.open(_io13.BytesIO(raw)).convert("RGBA"))


def _alpha_box(arr, thr=0):
    ys, xs = _np13.where(arr[..., 3] > thr)
    if not len(ys):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _heights(doc_id, n):
    out = []
    for i in range(n):
        b = _alpha_box(_cell_arr(doc_id, i))
        out.append(None if b is None else b[3] - b[1] + 1)
    return out


def _anchors(doc_id, n, anchor="bottom-center"):
    """The pivot the op promises to hold, measured on the served pixels.

    This calls the editor's own `anchor_point` rather than restating the rule.
    The first version of this check restated it as "the bbox's middle column,
    last opaque row" and reported a one-pixel drift on two frames -- because
    `bottom-center` is `(x0 + w // 2, y1 + 1)`, and the bbox middle column is
    `x0 + (w - 1) // 2`. Those two agree or differ by one depending on whether
    the subject's *width* is even or odd, and scaling changes the parity. So the
    proxy was measuring a point the op never claimed to hold. A check that
    restates a rule it is supposed to verify can only ever test itself.
    """
    out = []
    for i in range(n):
        arr = _cell_arr(doc_id, i)
        b = _alpha_box(arr)
        out.append(None if b is None
                   else _E13.anchor_point(b, anchor, arr[..., 3] > 0))
    return out


_ops13 = {o["name"]: o for o in req("/api/editor/ops")["ops"]}
ok("match_size" in _ops13 and _ops13["match_size"]["group"] == "Transform",
   "the registry serves match_size as a Transform op",
   _ops13.get("match_size", {}).get("group"))
ok([a["k"] for a in _ops13["match_size"]["args"]]
   == ["measure", "reference", "target", "anchor", "threshold", "resample"],
   "with its controls in the order the form renders them",
   [a["k"] for a in _ops13["match_size"]["args"]])

_t13 = tempfile.mkdtemp(prefix="ss_match_")
try:
    d13 = req("/api/editor/open", {"sheet": _sheet_of(_t13, "pulse", _MB13)})["doc"]
    _id13, _n13, _lay13 = d13["id"], d13["n"], dict(d13["layout"])
    ok(_n13 == 4 and _lay13["cell_w"] == 64 and _lay13["columns"] == 2,
       "the reported sheet opens as 4 frames of 64x64 in a 2x2 grid",
       (_n13, _lay13["cell_w"], _lay13["columns"]))

    _h0, _a0 = _heights(_id13, _n13), _anchors(_id13, _n13)
    ok(max(_h0) - min(_h0) == 18,
       "the served cells really are the pulsing sheet: an 18 px spread", _h0)

    r13 = req("/api/editor/%s/op" % _id13,
              {"name": "match_size", "sel": list(range(_n13)), "args": {}})
    ok(r13["ok"], "the op runs over the real route", r13.get("error"))
    ok(sorted(r13["changed"]) == [0, 1, 2, 3],
       "and reports every selected cell as changed", r13["changed"])
    ok(r13["doc"]["undo"] == 1, "as one undo level", r13["doc"]["undo"])
    ok(sorted(r13["doc"]["dirty"]) == [0, 1, 2, 3],
       "every frame is marked modified, so the save knows what to write",
       r13["doc"]["dirty"])

    _h1, _a1 = _heights(_id13, _n13), _anchors(_id13, _n13)
    ok(max(_h1) - min(_h1) == 0,
       "the spread is gone -- every frame's subject is the same size",
       "%s -> %s" % (_h0, _h1))
    ok(_a1 == _a0,
       "and every subject stands on the pixel it stood on, so the size fix did "
       "not become a position jump", "%s vs %s" % (_a1, _a0))
    ok(r13["doc"]["layout"] == _lay13 and r13["doc"]["n"] == _n13,
       "the cell size, the grid and the frame count are untouched",
       r13["doc"]["layout"])
    ok("Match content size:" in r13["msg"] and "-> 27 px" in r13["msg"],
       "the status line the page shows names the reference size it derived",
       r13["msg"])
    # The panel prints the size it achieved. That number is the only thing the
    # user can check the result against without opening a frame, so it has to be
    # the size the pixels really have -- a report that rounds up would be worse
    # than no report.
    _now13 = [int(x) for x in
              _re13.search(r"now (\d+)(?:-(\d+))?", r13["msg"]).groups() if x]
    ok(_now13 == sorted({min(_h1), max(_h1)}),
       "and the size it claims it achieved is the size the pixels actually "
       "have", "reported %s, measured %s" % (_now13, sorted(set(_h1))))
    print("      " + r13["msg"])

    # Scoped undo. The entry covers the four cells rather than the document, so
    # undoing it has to put the original sizes *and* positions back.
    u13 = req("/api/editor/%s/undo" % _id13, {})
    ok(sorted(u13["changed"]) == [0, 1, 2, 3] and u13["doc"]["undo"] == 0,
       "undo reports the four cells back and empties the undo stack",
       (u13["changed"], u13["doc"]["undo"]))
    ok(_heights(_id13, _n13) == _h0 and _anchors(_id13, _n13) == _a0,
       "undo restores the original sizes and the original positions",
       _heights(_id13, _n13))

    # A partial selection must leave the rest alone, byte for byte -- the user
    # selects all frames, but the op is scoped to the selection and a silent
    # "fix them all anyway" would be a different feature.
    _before13 = [_cell_arr(_id13, i) for i in range(_n13)]
    r13b = req("/api/editor/%s/op" % _id13,
               {"name": "match_size", "sel": [0, 1], "args": {}})
    ok(r13b["ok"] and set(r13b["changed"]) <= {0, 1},
       "a two-frame selection never reports a cell outside it as changed",
       r13b["changed"])
    _after13 = [_cell_arr(_id13, i) for i in range(_n13)]
    ok(all((_before13[i] == _after13[i]).all() for i in (2, 3)),
       "and the frames outside it come back byte-identical")
    _h13b = _heights(_id13, _n13)
    ok(_h13b[2:] == _h0[2:] and _h13b[0] != _h0[0],
       "the two inside it really moved while the two outside did not, so the "
       "comparison above is not passing for no reason",
       "%s vs %s" % (_h13b, _h0))

    # The controls have to reach the op, not merely be rendered by the form. Two
    # of them are observable from outside the server: the pivot, because a
    # different anchor holds a different point, and the filter, because nearest
    # lands on the reference size exactly where the interpolating default came
    # back one pixel over -- the residual the status line above honestly called
    # "now 28".
    req("/api/editor/%s/undo" % _id13, {})
    _tl0 = _anchors(_id13, _n13, "top-left")
    _bc0 = _anchors(_id13, _n13, "bottom-center")
    r13e = req("/api/editor/%s/op" % _id13,
               {"name": "match_size", "sel": list(range(_n13)),
                "args": {"anchor": "top-left", "resample": "nearest"}})
    ok(r13e["ok"], "the op runs with its controls set", r13e.get("error"))
    ok(_anchors(_id13, _n13, "top-left") == _tl0,
       "anchor=top-left holds the top-left corner instead of the feet",
       "%s vs %s" % (_anchors(_id13, _n13, "top-left"), _tl0))
    ok(_anchors(_id13, _n13, "bottom-center") != _bc0,
       "and the feet really did move, so the pivot is wired to the control "
       "rather than fixed",
       "%s vs %s" % (_anchors(_id13, _n13, "bottom-center"), _bc0))
    ok(_heights(_id13, _n13) == [27] * _n13,
       "resample=nearest lands every frame on the reference size exactly",
       _heights(_id13, _n13))

    # What it refuses to hide: a frame with no subject, and a selection that is
    # all subject-less.
    _id13b = req("/api/editor/open",
                 {"sheet": _sheet_of(_t13, "gappy",
                                     [_MB13[0], None, _MB13[3], None])})["doc"]["id"]
    r13c = req("/api/editor/%s/op" % _id13b,
               {"name": "match_size", "sel": [0, 1, 2, 3], "args": {}})
    ok(r13c["ok"] and "2 frames had no subject" in r13c["msg"],
       "blank frames are skipped and counted in the status line", r13c["msg"])
    ok(sorted(r13c["changed"]) == [0, 2],
       "and only the frames that had something to scale are rewritten",
       r13c["changed"])
    req("/api/editor/%s/close" % _id13b, {})

    _id13c = req("/api/editor/open",
                 {"sheet": _sheet_of(_t13, "void",
                                     [None, None, None, None])})["doc"]["id"]
    r13d = req("/api/editor/%s/op" % _id13c,
               {"name": "match_size", "sel": [0, 1, 2, 3], "args": {}})
    ok(r13d["ok"] is False and "no subject" in r13d["error"],
       "a selection with nothing to measure is refused with a reason the page "
       "can show", r13d.get("error"))
    ok(r13d["doc"]["undo"] == 0,
       "and the refusal left no undo entry behind", r13d["doc"]["undo"])
    req("/api/editor/%s/close" % _id13c, {})

    req("/api/editor/%s/close" % _id13, {})
finally:
    shutil.rmtree(_t13, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 13b. erase a colour: the picker's value is the colour that goes ===")
# --------------------------------------------------------------------------- #
# The control is a colour picker now, so the wire carries one hex string instead
# of three channel numbers. That is a change to a route's payload, which is the
# kind of thing only this suite can see: the op layer calls the op directly and
# the browser suite drives the page's own form, so neither would notice a route
# that dropped or mangled the value on the way through.
_ec13 = next(o for o in req("/api/editor/ops")["ops"] if o["name"] == "erase_color")
ok([a["k"] for a in _ec13["args"]] == ["color", "tol", "connected", "opaque_only"],
   "the registry serves one colour control where the three channels were",
   [a["k"] for a in _ec13["args"]])
ok(_ec13["args"][0]["t"] == "color" and _ec13["args"][0]["d"] == "#000000",
   "of the type the page renders as a picker, defaulting to black as before",
   _ec13["args"][0])

_t13b = tempfile.mkdtemp(prefix="ss_key_")
try:
    _BACKDROP = (19, 255, 56, 255)          # the walk clip's green, not #00ff00
    _cells13 = []
    for _ in range(4):
        _c = _np13.zeros((64, 64, 4), _np13.uint8)
        _c[0:10, :] = _BACKDROP             # a backdrop band, border-connected
        _c[16:48, 16:48] = (255, 255, 255, 255)
        _cells13.append(_c)
    _id13b = req("/api/editor/open",
                 {"sheet": _sheet_raw(_t13b, "key", _cells13)})["doc"]["id"]

    def _alpha(doc_id, i, y, x):
        return int(_cell_arr(doc_id, i)[y, x, 3])

    ok(_alpha(_id13b, 0, 2, 2) == 255 and _alpha(_id13b, 0, 30, 30) == 255,
       "the fixture starts with a green band and a white subject")

    r13b = req("/api/editor/%s/op" % _id13b,
               {"name": "erase_color", "sel": [0, 1, 2, 3],
                "args": {"color": "#13ff38", "tol": 4, "connected": False,
                         "opaque_only": False}})
    ok(r13b["ok"], "the op runs with a colour from the picker", r13b.get("error"))
    ok(_alpha(_id13b, 0, 2, 2) == 0,
       "and erases the colour the picker named, not black and not a default")
    ok(_alpha(_id13b, 0, 30, 30) == 255,
       "while the subject it was keyed against survives")
    ok(sorted(r13b["changed"]) == [0, 1, 2, 3],
       "every selected frame is reported changed", r13b["changed"])

    # A colour the picker cannot produce is refused at the route rather than
    # silently keyed as black -- black is the one fallback that would take out
    # the subject's own dark pixels and still report success.
    _bad13 = req("/api/editor/%s/op" % _id13b,
                 {"name": "erase_color", "sel": [0], "args": {"color": "oops"}})
    ok(_bad13["ok"] is False and "hex colour" in _bad13["error"],
       "an unreadable colour is refused with a reason the page can show",
       _bad13.get("error"))
    ok(_bad13["doc"]["undo"] == 1,
       "and the refusal pushed no undo entry", _bad13["doc"]["undo"])

    # The other channels still work, so the refusal above is about the value and
    # not about the op having stopped reading its arguments at all.
    r13c = req("/api/editor/%s/op" % _id13b,
               {"name": "erase_color", "sel": [1],
                "args": {"color": "#ffffff", "tol": 4, "connected": False,
                         "opaque_only": False}})
    ok(r13c["ok"] and _alpha(_id13b, 1, 30, 30) == 0,
       "a second colour through the same control erases that colour instead",
       r13c.get("changed"))
    ok(_alpha(_id13b, 1, 2, 2) == 0,
       "and the frame it had already been keyed on stays keyed")
    req("/api/editor/%s/close" % _id13b, {})
finally:
    shutil.rmtree(_t13b, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 13c. pixelate (mesh) takes the same palette from the picker ===")
# Reported as a request: "add same palette from snap pixel to pixelate (mesh)".
# The op layer proves the parsing and the mapping; this proves the palette
# survives the route and comes back as the colours in the served cell PNGs --
# which is the only place the user can see them.
# --------------------------------------------------------------------------- #
import proper_pixel as _ppa13c

if not _ppa13c.available():
    print("  skip  proper-pixel-art checkout not available")
else:
    _t13c = tempfile.mkdtemp(prefix="ss_meshpal_")

    def _blocky13c(seed, size=64, true=8):
        """A noisy nearest-upscaled sprite: what the mesh detector targets."""
        small = _np13.zeros((true, true, 4), _np13.uint8)
        yy, xx = _np13.mgrid[0:true, 0:true]
        small[((xx - true // 2) ** 2 + (yy - true // 2) ** 2)
              < (true // 2 - 1) ** 2] = (80, 150, 220, 255)
        small[(seed + 2):(seed + 4), 2:4] = (220, 80, 80, 255)
        big = _np13.array(_I13.fromarray(small).resize(
            (size, size), _I13.NEAREST))
        rng = _np13.random.default_rng(seed)
        n = rng.integers(-14, 15, big.shape[:2]).astype(_np13.int16)
        big[..., :3] = _np13.clip(
            big[..., :3].astype(_np13.int16) + n[..., None],
            0, 255).astype(_np13.uint8)
        return big

    try:
        _id13c = req("/api/editor/open", {"sheet": _sheet_raw(
            _t13c, "meshpal", [_blocky13c(s) for s in range(4)])}
            )["doc"]["id"]

        def _cols13c(i):
            arr = _cell_arr(_id13c, i)
            m = arr[..., 3] > 0
            return {tuple(int(v) for v in px)
                    for px in arr[..., :3][m].reshape(-1, 3)}

        # The registry's own control, as the page receives it.
        _spec13c = [o for o in req("/api/editor/ops")["ops"]
                    if o["name"] == "pixelate_mesh"][0]
        ok([a["k"] for a in _spec13c["args"]][-1] == "palette"
           and [a["t"] for a in _spec13c["args"]][-1] == "text",
           "the route serves pixelate (mesh) with the palette control",
           [(a["k"], a["t"]) for a in _spec13c["args"]])

        # The frame that is not selected, measured before the op: the palette
        # must not reach it either.
        _untouched13c = _cell_arr(_id13c, 3).copy()
        _hexes13c = ["#101820", "#e8f0f8", "#4c9ad0", "#dc5050"]
        _allowed13c = {tuple(int(v) for v in row)
                       for row in _E13.hex_palette(",".join(_hexes13c))}
        # pixel_width is set rather than auto: this sheet is 64x64 because
        # _sheet_raw writes 2x2 of 64x64, and the mesh detector finds nothing to
        # measure in a source that small -- it returns a degenerate 1x1 grid and
        # a fully transparent result. A real input is a full-size AI image, which
        # is what the auto path is for; naming the width is the control the op
        # offers for exactly this case, and it keeps the mesh out of the way of
        # what this section is actually testing.
        r13c = req("/api/editor/%s/op" % _id13c,
                   {"name": "pixelate_mesh", "sel": [0, 1, 2],
                    "args": {"colors": 16, "pixel_width": 8, "sample": 8,
                             "upscale": 2, "transparent": False,
                             "palette": ",".join(_hexes13c)}})
        ok(r13c["ok"], "the op runs over the route with a picked palette",
           r13c.get("error"))
        _seen13c = set()
        for _i in range(3):
            _seen13c |= _cols13c(_i)
        ok(_seen13c and _seen13c <= _allowed13c,
           "and the served pixels use the palette the picker sent, and "
           "nothing else", sorted(_seen13c - _allowed13c)[:4])
        ok("palette from the picker" in r13c["msg"],
           "with the status line saying where the palette came from",
           r13c["msg"])
        ok(_np13.array_equal(_cell_arr(_id13c, 3), _untouched13c),
           "and the unselected frame is left byte-identical, palette or not")

        # A palette entry the picker cannot produce is refused at the route, not
        # silently dropped -- the op layer pins the parser, this pins that the
        # refusal reaches the caller with a reason the page can show.
        _undo13c = r13c["doc"]["undo"]
        _bad13c = req("/api/editor/%s/op" % _id13c,
                      {"name": "pixelate_mesh", "sel": [0],
                       "args": {"palette": "#101820,oops"}})
        ok(_bad13c["ok"] is False and "hex colour" in _bad13c["error"],
           "an unreadable palette entry is refused with a reason",
           _bad13c.get("error"))
        ok(_bad13c["doc"]["undo"] == _undo13c,
           "and the refusal pushed no undo entry", _bad13c["doc"]["undo"])
        req("/api/editor/%s/close" % _id13c, {})
    finally:
        shutil.rmtree(_t13c, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 13d. fill / tint takes its colour from the same picker ===")
# The last op that still asked for three channel numbers, and the one with no
# tests at all before this -- so nothing failed while it stayed on r/g/b. The op
# layer pins the modes and the refusal; this pins the wire format, which is what
# a page actually sends.
# --------------------------------------------------------------------------- #
_t13d = tempfile.mkdtemp(prefix="ss_fill_")
try:
    _cells13d = []
    for _ in range(4):
        _c = _np13.zeros((64, 64, 4), _np13.uint8)
        _c[16:48, 16:48] = (230, 230, 230, 255)
        _cells13d.append(_c)
    _id13d = req("/api/editor/open",
                 {"sheet": _sheet_raw(_t13d, "fill", _cells13d)})["doc"]["id"]

    _spec13d = [o for o in req("/api/editor/ops")["ops"]
                if o["name"] == "fill"][0]
    ok([a["k"] for a in _spec13d["args"]] == ["color", "alpha", "mode"]
       and _spec13d["args"][0]["t"] == "color"
       and _spec13d["args"][0]["d"] == "#ff0000",
       "the route serves fill with one colour control defaulting to red",
       [(a["k"], a["t"], a["d"]) for a in _spec13d["args"]])

    r13d = req("/api/editor/%s/op" % _id13d,
               {"name": "fill", "sel": [0, 1],
                "args": {"color": "#13ff38", "alpha": 255, "mode": "tint"}})
    ok(r13d["ok"], "the op runs with a colour from the picker", r13d.get("error"))
    _arr13d = _cell_arr(_id13d, 0)
    ok(tuple(int(v) for v in _arr13d[30, 30]) == (19, 255, 56, 255),
       "and the served pixels are the colour the picker named",
       tuple(int(v) for v in _arr13d[30, 30]))
    ok(int(_arr13d[0, 0, 3]) == 0,
       "with the transparent region left transparent, so it is a tint")
    ok(sorted(r13d["changed"]) == [0, 1],
       "and only the selected frames are reported changed", r13d["changed"])

    # The refusal has to reach the caller too, not just the op layer.
    _undo13d = r13d["doc"]["undo"]
    _bad13d = req("/api/editor/%s/op" % _id13d,
                  {"name": "fill", "sel": [0], "args": {"color": "oops"}})
    ok(_bad13d["ok"] is False and "hex colour" in _bad13d["error"],
       "an unreadable colour is refused at the route with a reason",
       _bad13d.get("error"))
    ok(_bad13d["doc"]["undo"] == _undo13d,
       "and the refusal pushed no undo entry", _bad13d["doc"]["undo"])
    req("/api/editor/%s/close" % _id13d, {})
finally:
    shutil.rmtree(_t13d, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 13e. pixelize (outline): PixelOE over the route ===")
# The op layer pins the pixel-size guard, the alpha carry and the blockiness;
# this pins the two things only the wire can show -- that the op is in the
# served registry (so the page builds its controls), and that a run over HTTP
# keeps the cell size and reports the refusal as a message rather than a 500.
# --------------------------------------------------------------------------- #
try:
    import pixeloe_bridge as _poe13e
    _POE13E = _poe13e.available()
except Exception:                                  # noqa: BLE001
    _POE13E = False
if not _POE13E:
    print("  skip  PixelOE checkout not available")
else:
    _t13e = tempfile.mkdtemp(prefix="ss_poe_")
    try:
        _cells13e = []
        for _s in range(4):
            _c = _np13.zeros((64, 64, 4), _np13.uint8)
            _yy, _xx = _np13.mgrid[0:64, 0:64]
            _c[..., 0] = ((_xx * 255) // 63).astype(_np13.uint8)
            _c[..., 1] = ((_yy * 255) // 63).astype(_np13.uint8)
            _c[..., 2] = 96
            _c[..., 3] = 255
            _c[(_yy - 32) ** 2 + (_xx - 24) ** 2 < 10 ** 2] = (250, 20, 20, 255)
            _cells13e.append(_c)
        _id13e = req("/api/editor/open", {"sheet": _sheet_raw(
            _t13e, "poe", _cells13e)})["doc"]["id"]

        _spec13e = [o for o in req("/api/editor/ops")["ops"]
                    if o["name"] == "pixelize_oe"]
        ok(_spec13e, "the route serves pixelize (outline) in the registry")
        _sp13e = _spec13e[0] if _spec13e else {"args": [], "group": ""}
        ok(_sp13e.get("group") == "Pixel art",
           "and it files under the Pixel art group, so the page groups it with "
           "the other pixel ops", _sp13e.get("group"))
        ok([a["k"] for a in _sp13e["args"]][0] == "pixel_size"
           and _sp13e["args"][0]["t"] == "int",
           "with the pixel size as its first control",
           [(a["k"], a["t"]) for a in _sp13e["args"]])

        _untouched13e = _cell_arr(_id13e, 3).copy()
        # The arguments below are stated explicitly rather than left to the
        # declared defaults: the defaults are asserted in the op-layer suite
        # (§32), and what this route test is for is that a *stated* argument set
        # survives JSON, the registry and the HTML round trip. Leaving them out
        # here would silently re-point every "8x8 block" assertion below at
        # whatever the default pixel size becomes.
        _r13e = req("/api/editor/%s/op" % _id13e,
                    {"name": "pixelize_oe", "sel": [0, 1],
                     "args": {"pixel_size": 8, "thickness": 0,
                              "mode": "contrast", "colors": 0,
                              "dither": "none", "sharpen": "none",
                              "sharpen_factor": 0, "color_match": False}})
        ok(_r13e["ok"], "the op runs over the route with the stated arguments",
           _r13e.get("error"))
        _arr13e = _cell_arr(_id13e, 0)
        ok(_arr13e.shape == (64, 64, 4),
           "and the served cell is still 64x64, so the cell size survived the "
           "round trip", _arr13e.shape)
        ok((_arr13e[..., 3] == 255).all(),
           "with alpha carried through")
        # Blockiness, measured on the pixels the route actually served.
        _corners13e = _arr13e[::8, ::8, :3]
        ok(_np13.array_equal(
            _arr13e[..., :3],
            _np13.repeat(_np13.repeat(_corners13e, 8, axis=0), 8,
                         axis=1)[:64, :64]),
           "and every 8x8 block is one flat colour, so what the page receives "
           "is a true block downscale")
        ok(_np13.array_equal(_cell_arr(_id13e, 3), _untouched13e),
           "the unselected frame is byte-identical")

        # The refusal must come back as a message, not a 500 or a silent no-op.
        _undo13e = _r13e["doc"]["undo"]
        _bad13e = req("/api/editor/%s/op" % _id13e,
                      {"name": "pixelize_oe", "sel": [0],
                       "args": {"pixel_size": 33}})
        ok(_bad13e["ok"] is False and "does not divide" in _bad13e["error"],
           "a pixel size the cell cannot take is refused at the route with the "
           "divisors listed", _bad13e.get("error"))
        ok(_bad13e["doc"]["undo"] == _undo13e,
           "and the refusal pushed no undo entry", _bad13e["doc"]["undo"])
        _shown13e = req("/api/editor/%s/cell/0.png" % _id13e, raw=True)
        ok(_shown13e[:8] == b"\x89PNG\r\n\x1a\n",
           "and the cell still serves as a PNG after both calls")

        # The palette picker: the same shared control the other two pixel ops
        # have, so the wire format is the same (a `text` arg named `palette`).
        _hexes13e = ["#101820", "#e8f0f8", "#4c9ad0", "#dc5050"]
        _allowed13e = {tuple(int(v) for v in row)
                       for row in _E13.hex_palette(",".join(_hexes13e))}
        _pal13e = req("/api/editor/%s/op" % _id13e,
                      {"name": "pixelize_oe", "sel": [0],
                       "args": {"pixel_size": 8, "palette": ",".join(_hexes13e)}})
        ok(_pal13e["ok"], "the op takes a palette from the picker over the route",
           _pal13e.get("error"))
        _arr13e2 = _cell_arr(_id13e, 0)
        _cols13e = {tuple(int(v) for v in px)
                    for px in _arr13e2[..., :3][_arr13e2[..., 3] > 0]
                    .reshape(-1, 3)}
        ok(_cols13e and _cols13e <= _allowed13e,
           "and the served pixels use the palette the picker sent, and nothing "
           "else", sorted(_cols13e - _allowed13e)[:4])
        ok("palette from the picker" in _pal13e["msg"],
           "with the status line saying where the palette came from")
        ok(_np13.array_equal(_cell_arr(_id13e, 3), _untouched13e),
           "and the unselected frame is still byte-identical, palette or not")
        _badpal13e = req("/api/editor/%s/op" % _id13e,
                         {"name": "pixelize_oe", "sel": [0],
                          "args": {"pixel_size": 8, "palette": "#ff0000,oops"}})
        ok(_badpal13e["ok"] is False and "hex colour" in _badpal13e["error"],
           "an unreadable palette entry is refused with a reason",
           _badpal13e.get("error"))
        req("/api/editor/%s/close" % _id13e, {})
    finally:
        shutil.rmtree(_t13e, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n" + "=" * 66)
print("%d checks, %d failures" % (N[0], len(FAILS)))
if FAILS:
    for f in FAILS:
        print("  FAILED:", f)
    sys.exit(1)
print("ALL PASS")
