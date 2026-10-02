"""Sprite Studio server.

    python app.py [--port 8765] [--host 127.0.0.1]

Then open http://127.0.0.1:8765

Design notes:

* The VRMBG-3.0 model is loaded **once** and cached in this process, guarded by a
  lock. Loading costs 2-3.5 s and cudnn autotuning makes the first ~25 frames
  5-10x slower, so keeping it warm makes every run after the first start fast.

* Jobs run one at a time in a worker thread. The model is a single shared GPU
  resource, so concurrency here would only cause OOM.

* Progress is polled (`GET /api/job/<id>`), not streamed. For a local single-user
  tool that is one fewer moving part, and it survives the browser tab sleeping.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

from fastapi import FastAPI, Query, Request
from fastapi.responses import (HTMLResponse, JSONResponse, FileResponse, Response,
                               StreamingResponse)
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402
import editor as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
UPLOADS = os.path.join(HERE, "uploads")
UI = os.path.join(HERE, "ui.html")
EDITOR_UI = os.path.join(HERE, "editor.html")
FAVICON = os.path.join(HERE, "favicon.ico")
WORKSPACE = os.path.dirname(HERE)

os.makedirs(RUNS, exist_ok=True)
os.makedirs(UPLOADS, exist_ok=True)

app = FastAPI(title="Sprite Studio")


@app.exception_handler(Exception)
async def _unhandled(request, exc):
    """Answer with JSON, always.

    The UI calls fetch() and then `r.json()`. An unhandled exception makes
    Starlette return a bare text/plain "Internal Server Error", so the browser
    reported `SyntaxError: Unexpected token 'I', "Internal S"... is not valid
    JSON` -- and the actual cause (a PermissionError on one file) never reached
    the user at all. Turning every unhandled error into JSON means the dialog can
    always show the reason.
    """
    traceback.print_exc()
    return JSONResponse({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)},
                        status_code=500)

# When this process started, and what the two modules looked like then. The page
# is re-read from disk on every request while this process holds its Python in
# memory, so the page can be newer than the server serving it -- and then a fix
# that *is* on disk is simply absent from the running server. Nothing 404s that
# used to exist, nothing errors, and the missing fix reads exactly like a broken
# feature. That has been reported as a bug three times ("overwriting not
# working", "when i click play button it says can not play and volume setting
# says http 404"). `/api/editor/version` compares these stamps against the files
# now, so a stale process announces itself instead of being mistaken for broken
# code -- no version number for anyone to remember to bump.
_BOOT_STAMPS = {}
for _mod in (os.path.abspath(__file__), os.path.abspath(E.__file__)):
    try:
        _BOOT_STAMPS[_mod] = os.path.getmtime(_mod)
    except OSError:                      # a frozen build: nothing to compare
        pass

# The process's own start time, which is what the footer badge prints and what
# `started` has always claimed to be. It used to report `min(_BOOT_STAMPS)`, the
# mtime of the *oldest watched file* -- a number that only looks like a start
# time when the code was just written. It reads wrong the moment the two files
# were not written together, and it read wrong in the worst way: a server started
# minutes ago announced a start two days earlier, on the one badge a user consults
# to decide whether their server is current.
_BOOT_TIME = time.time()

# --------------------------------------------------------------------------- #
# job state
# --------------------------------------------------------------------------- #

JOBS = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
MODEL_CACHE = {}
_CANCEL = set()

# open editor documents, held in this process so the browser only ships op
# commands and small per-cell PNGs
DOCS = {}
DOC_LOCK = threading.RLock()

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif")
SHEET_ROOTS = (os.path.join(WORKSPACE, "walk"),
               os.path.join(WORKSPACE, "sprites"),
               RUNS)
MAX_DOCS = 6
# The cap alone does not bound memory in a long session. A document is two full
# RGBA copies of the sheet -- `cells` plus the loaded `orig` reference, ~325 MB
# for a 124-frame 512x640 sheet -- and its undo stack is allowed 192 MB more.
# Six of those is over 2 GB, and nothing here ever gave one back: the cap was
# only tested when a *new* document was opened, and every page reload opens the
# same sheet under a fresh id, so the previous copies were only ever evicted by
# opening even more sheets. Idle, unmodified documents are now released. Ones
# with unsaved edits are kept -- those cannot be reopened from disk, and losing
# them is a worse outcome than the memory.
DOC_IDLE_SECONDS = int(os.environ.get("SPRITE_DOC_IDLE", "1800"))


def _new_job(cfg):
    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid,
        "status": "queued",
        "stage": "queued",
        "frac": 0.0,
        "log": [],
        "started": time.time(),
        "ended": None,
        "error": None,
        "result": None,
        "cfg": cfg,
        "run": None,
    }
    with JOBS_LOCK:
        JOBS[jid] = job
    return job


def _log(job, msg):
    job["log"].append("%s  %s" % (time.strftime("%H:%M:%S"), msg))
    if len(job["log"]) > 400:
        del job["log"][:100]


def _worker(job):
    cfg = job["cfg"]
    prefix = cfg.get("prefix") or os.path.splitext(os.path.basename(cfg.get("video", "")))[0]
    prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", prefix)[:60] or "sprite"
    run_dir = os.path.join(RUNS, "%s_%s" % (time.strftime("%m%d-%H%M%S"), prefix))
    job["run"] = os.path.basename(run_dir)
    cfg["prefix"] = prefix

    def emit(stage, frac, msg=""):
        job["stage"] = stage
        job["frac"] = float(frac)
        if msg:
            _log(job, "[%s] %s" % (stage, msg))

    with RUN_LOCK:
        if job["id"] in _CANCEL:
            job["status"] = "cancelled"
            job["ended"] = time.time()
            return
        job["status"] = "running"
        _log(job, "run dir: %s" % run_dir)
        try:
            res = P.run_pipeline(
                cfg, run_dir, emit=emit,
                cancel=lambda: job["id"] in _CANCEL,
                model_cache=MODEL_CACHE,
            )
            art = {k: os.path.relpath(v, RUNS).replace("\\", "/")
                   for k, v in res["artifacts"].items()}
            job["result"] = {
                "run": os.path.basename(run_dir),
                "prefix": res["prefix"],
                "frames": len(res["frames"]),
                "layout": res["layout"],
                "info": {k: v for k, v in res["info"].items() if k != "first_frame"},
                "bbox": res["bbox"],
                "artifacts": art,
                "verify": res.get("verify"),
                "sidecar": res.get("sidecar"),
            }
            job["status"] = "done"
            _log(job, "done")
        except Exception as ex:                      # noqa: BLE001
            job["status"] = "error"
            job["error"] = "%s: %s" % (type(ex).__name__, ex)
            _log(job, "ERROR " + job["error"])
            _log(job, traceback.format_exc()[-1500:])
        finally:
            job["ended"] = time.time()
            job["frac"] = 1.0
            _CANCEL.discard(job["id"])


# --------------------------------------------------------------------------- #
# api
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def index():
    with open(UI, "r", encoding="utf-8") as fh:
        html = fh.read()
    # Served from disk on every request; a cached copy hides fixes behind a
    # stale page while the server looks up to date.
    return HTMLResponse(html, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "Clear-Site-Data": '"cache"',
    })


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """The tab mark. Served from our own origin rather than linked to
    raw.githubusercontent.com: a cross-origin icon is a third-party request on
    every load, and it 404s into an empty tab when the network is down -- which
    is exactly the case this image exists for. 404 when the file has not been
    dropped in, so the browser falls back to its default instead of getting a
    broken icon."""
    if not os.path.isfile(FAVICON):
        return Response(status_code=404)
    # A day of caching is safe: the icon is not part of the edit loop, and the
    # pages themselves stay no-store.
    return FileResponse(FAVICON, media_type="image/x-icon",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/defaults")
def defaults():
    import platform
    cfg = dict(P.DEFAULT_CFG)
    # sensible model location: the env override first (that is what a container
    # sets), then alongside the app's parent, then the known Desktop path
    cands = [
        os.environ.get("SPRITE_MODEL_DIR", ""),
        os.path.join(os.path.dirname(HERE), "VRMBG-3.0"),
        r"C:\Users\PC\Desktop\VRMBG-3.0",
    ]
    for c in cands:
        # `c` is tested for truth first: an unset env var is "", and
        # os.path.join("", "config.json") is a *relative* path, so without this
        # the probe would match a config.json in whatever the cwd happens to be
        # and hand the pipeline a model_dir of "".
        if c and os.path.isfile(os.path.join(c, "config.json")):
            cfg["model_dir"] = c
            break
    return {
        "defaults": cfg,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cuda": _cuda(),
            "runs_dir": RUNS,
        },
    }


def _cuda():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return None
    except Exception:
        return None


@app.post("/api/probe")
async def probe(req: Request):
    body = await req.json()
    path = (body or {}).get("video", "")
    if not path:
        return JSONResponse({"ok": False, "error": "no path given"}, status_code=400)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "not found: %s" % path}, status_code=404)
    try:
        info = P.probe(path)
        import cv2
        ok, _ = True, None
        h, w = info["first_frame"].shape[:2]
        # a small preview of the first frame as a data URL
        import base64
        small = cv2.resize(info["first_frame"], (min(320, w), int(min(320, w) * h / w)))
        ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok2 else None
        return {
            "ok": True,
            "info": {k: v for k, v in info.items() if k != "first_frame"},
            # The first frame, as a still: the player's poster, and what is left
            # of the preview when the browser cannot decode the codec.
            "preview": ("data:image/jpeg;base64," + b64) if b64 else None,
            # Where the player should point. Built here rather than in the page,
            # so the URL shape lives with the route that answers it.
            "video": "/api/video?path=" + quote(os.path.abspath(path)),
            "divisors": P.divisors(info["frames"]),
            # The inference size this clip's own resolution calls for, so a 640px
            # sprite clip is not matted at the trained 1024 by default. `w`/`h`
            # come from the decoded frame above rather than from the container
            # metadata, which some formats report as 0x0.
            "infer_size": P.suggest_infer_size(w, h),
            # The clip's own border colour, as a suggestion for the colour key:
            # the page seeds the picker with it, because nobody can name their
            # green screen's RGB by eye (the walk clip's is #13ff38, not #00ff00).
            "backdrop": P.backdrop_color(info["first_frame"]),
        }
    except Exception as ex:                          # noqa: BLE001
        return JSONResponse({"ok": False, "error": "%s: %s" % (type(ex).__name__, ex)},
                            status_code=400)


# What a browser will actually decode, by extension. The matting stage reads far
# more than this (any codec ffmpeg has), but the point of this route is the
# player, so the list is what <video> can play rather than what the tool accepts.
VIDEO_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".webm": "video/webm", ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo", ".gif": "image/gif",
}

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


@app.get("/api/video")
def video(path: str = "", request: Request = None):
    """Stream a video file so the browser can play and seek it.

    The player in the info panel is a <video>, and a media element is a picky
    client: it opens with `Range: bytes=0-` and answers a seek by asking for the
    bytes it needs. A 200 with the whole body looks fine for the first play and
    then makes every scrub re-download from the start, so this speaks 206 with a
    correct Content-Range instead -- and streams, because the file can be
    hundreds of megabytes and this process is also holding the model.

    Only known video extensions are served. The path is the user's own, typed or
    dropped into the video box, and the tool already reads it to decode frames,
    so this is not a new capability -- but there is no reason to hand a browser
    every .json and .py on the disk through the same door.
    """
    p = os.path.abspath(path or "")
    if not p or not os.path.isfile(p):
        return JSONResponse({"ok": False, "error": "not found: %s" % path},
                            status_code=404)
    ext = os.path.splitext(p)[1].lower()
    if ext not in VIDEO_TYPES:
        return JSONResponse(
            {"ok": False, "error": "not a playable video: %s (%s)"
                                    % (os.path.basename(p), ext or "no extension")},
            status_code=415)
    size = os.path.getsize(p)
    start, end = 0, size - 1
    status = 200
    headers = {"Accept-Ranges": "bytes", "Content-Type": VIDEO_TYPES[ext],
               "Cache-Control": "no-store"}
    rng = request.headers.get("range") if request is not None else None
    if rng:
        m = RANGE_RE.match(rng.strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                # `bytes=-N`: the LAST N bytes, not the first
                start, end = max(0, size - int(m.group(2))), size - 1
            if start >= size or start > end:
                return Response(status_code=416,
                                headers={"Content-Range": "bytes */%d" % size})
            end = min(end, size - 1)
            status = 206
            headers["Content-Range"] = "bytes %d-%d/%d" % (start, end, size)
    headers["Content-Length"] = str(end - start + 1)

    def body():
        left = end - start + 1
        with open(p, "rb") as fh:
            fh.seek(start)
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    return StreamingResponse(body(), status_code=status, headers=headers)


@app.post("/api/upload")
async def upload(req: Request, name: str = "input.mp4", sub: str = ""):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(name)) or "input.mp4"
    # A mask clip is very often exported beside its source under a matching name
    # (`clip.mp4` and `clip_mask.mp4`, or sometimes just `clip.mp4` twice), and
    # both used to land in uploads/ -- so dropping the mask overwrote the source
    # it belongs to, and the run then read a mask as its own video. The mask gets
    # its own folder. `sub` arrives in a query string, so it is whitelisted
    # rather than joined into a path.
    folder = os.path.join(UPLOADS, "masks") if sub == "masks" else UPLOADS
    data = await req.body()
    if not data:
        return JSONResponse({"ok": False, "error": "empty upload"}, status_code=400)
    os.makedirs(folder, exist_ok=True)
    dst = os.path.join(folder, safe)
    with open(dst, "wb") as fh:
        fh.write(data)
    return {"ok": True, "path": dst, "bytes": len(data)}


@app.get("/api/browse")
def browse(dir: str = ""):
    """Minimal file browser so a raw video can be picked without typing a path."""
    d = dir or os.path.expanduser("~")
    d = os.path.abspath(d)
    if not os.path.isdir(d):
        return JSONResponse({"ok": False, "error": "not a directory: %s" % d},
                            status_code=400)
    dirs, vids = [], []
    try:
        for n in sorted(os.listdir(d), key=str.lower):
            p = os.path.join(d, n)
            try:
                if os.path.isdir(p):
                    dirs.append(n)
                elif n.lower().endswith(VIDEO_EXT):
                    vids.append({"name": n, "size": os.path.getsize(p), "path": p})
            except OSError:
                continue
    except PermissionError:
        return JSONResponse({"ok": False, "error": "permission denied"}, status_code=403)
    parent = os.path.dirname(d)
    return {"ok": True, "dir": d,
            "parent": parent if parent != d else None,
            "dirs": dirs[:300], "videos": vids[:300]}


@app.post("/api/run")
async def run(req: Request):
    cfg = await req.json()
    if not cfg.get("video"):
        return JSONResponse({"ok": False, "error": "no video selected"}, status_code=400)
    job = _new_job(cfg)
    threading.Thread(target=_worker, args=(job,), daemon=True).start()
    return {"ok": True, "id": job["id"]}


@app.get("/api/job/{jid}")
def job_status(jid: str):
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if not job:
        return JSONResponse({"ok": False, "error": "unknown job"}, status_code=404)
    return {"ok": True, "job": job}


@app.post("/api/job/{jid}/cancel")
def job_cancel(jid: str):
    _CANCEL.add(jid)
    return {"ok": True}


@app.get("/api/runs")
def runs():
    out = []
    for n in sorted(os.listdir(RUNS), reverse=True):
        d = os.path.join(RUNS, n)
        if not os.path.isdir(d):
            continue
        files = []
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                files.append({"name": f, "size": os.path.getsize(p)})
        out.append({"run": n, "files": files})
        if len(out) >= 40:
            break
    return {"runs": out}


@app.get("/files/{path:path}")
def files(path: str):
    p = os.path.abspath(os.path.join(RUNS, path))
    if not p.startswith(os.path.abspath(RUNS)):
        return JSONResponse({"ok": False, "error": "outside the runs dir"}, status_code=403)
    if not os.path.isfile(p):
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return FileResponse(p)


# --------------------------------------------------------------------------- #
# editor
# --------------------------------------------------------------------------- #

def _doc(doc_id):
    with DOC_LOCK:
        d = DOCS.get(doc_id)
        # Stamped on use, not on open: which document is abandoned is a question
        # about when it was last asked for, and every route for an open sheet
        # comes through here.
        if d is not None:
            d.last_used = time.time()
    if d is None:
        raise KeyError(doc_id)
    return d


def _evict_docs(need=1):
    """Make room for `need` more documents, releasing what is abandoned.

    Idle documents with no unsaved edits go first: they are reopenable from
    disk and are the common case in a long session -- each reload of the page
    opens a new id for the same sheet. Only if that is not enough does this
    fall back to plain least-recently-used order, so MAX_DOCS stays a hard cap.
    """
    now = time.time()
    with DOC_LOCK:
        for did, d in list(DOCS.items()):
            if (now - getattr(d, "last_used", now) > DOC_IDLE_SECONDS
                    and not d.dirty):
                DOCS.pop(did, None)
        while len(DOCS) + need > MAX_DOCS:
            lru = min(DOCS, key=lambda k: getattr(DOCS[k], "last_used", 0.0))
            DOCS.pop(lru, None)


def _payload(d):
    return {
        "id": d.id, "n": d.n, "rev": d.rev,
        "layout": d.layout, "meta": d.meta,
        "cell_rev": list(d.cell_rev),
        "dirty": sorted(d.dirty),
        "undo": len(d.undo), "redo": len(d.redo),
        "journal": d.journal[-40:],
        "source": d.src_sheet,
        # The sidecar the open actually resolved, so a path box that was given a
        # folder can show which JSON it found.
        "sidecar": d.src_sidecar,
        "audio": d.audio_payload(),
    }


@app.get("/editor", response_class=HTMLResponse)
def editor_page():
    with open(EDITOR_UI, "r", encoding="utf-8") as fh:
        html = fh.read()
    return HTMLResponse(html, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "Clear-Site-Data": '"cache"',
    })


@app.get("/api/editor/ops")
def editor_ops():
    """The op registry, so the UI builds its own controls."""
    return {"ops": [
        {"name": s["name"], "group": s["group"], "label": s["label"],
         "args": s["args"], "help": s["help"]}
        for s in E.OPS.values()]}


@app.get("/api/editor/version")
def editor_version():
    """Whether this process is older than the code on disk.

    `stale` names the modules whose file has been written since this process
    loaded it. A stale server serves the *old body of a route that exists*, so
    the page has no way to tell it from a broken feature -- which is how the same
    report arrived three times. The page asks this at boot and says so.
    """
    stale = []
    for path, boot in _BOOT_STAMPS.items():
        try:
            if os.path.getmtime(path) > boot + 0.001:
                stale.append(os.path.basename(path))
        except OSError:
            pass
    return {"ok": True, "stale": stale, "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                     time.localtime(_BOOT_TIME))}


@app.get("/api/sheets")
def sheet_library(limit: int = 600):
    """Find loadable sheet+sidecar pairs in walk/, sprites/ and runs/."""
    out, seen = [], set()
    for root in SHEET_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if x != "_cells"]
            for fn in filenames:
                if not fn.lower().endswith(".json"):
                    continue
                p = os.path.join(dirpath, fn)
                if p in seen:
                    continue
                seen.add(p)
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        sc = json.load(fh)
                except Exception:                    # noqa: BLE001
                    continue
                if not all(k in sc for k in ("columns", "rows",
                                             "frame_width", "frame_height")):
                    continue
                cands = []
                if sc.get("sheet"):
                    cands.append(os.path.join(dirpath, sc["sheet"]))
                cands.append(os.path.splitext(p)[0] + "_sheet.png")
                cands.append(os.path.splitext(p)[0] + ".png")
                sheet = next((c for c in cands if os.path.isfile(c)), None)
                if not sheet:
                    continue
                out.append({
                    "name": sc.get("name") or os.path.splitext(fn)[0],
                    "sheet": sheet, "sidecar": p,
                    "columns": sc.get("columns"), "rows": sc.get("rows"),
                    "frame_width": sc.get("frame_width"),
                    "frame_height": sc.get("frame_height"),
                    "frame_count": sc.get("frame_count"),
                    "fps": sc.get("fps"),
                    "group": os.path.basename(root),
                    "size": os.path.getsize(sheet),
                })
        if len(out) >= limit:
            break
    out.sort(key=lambda r: (r["group"] != "walk", r["name"].lower()))
    return {"sheets": out[:limit]}


@app.post("/api/editor/open")
async def editor_open(req: Request):
    body = await req.json() or {}
    sheet = body.get("sheet") or ""
    # exists, not isfile: a folder is a valid thing to point at, and load_doc
    # resolves the sheet sidecar in it.
    if not sheet or not os.path.exists(sheet):
        return JSONResponse({"ok": False, "error": "no such sheet: %s" % sheet},
                            status_code=404)
    did = uuid.uuid4().hex[:10]
    try:
        # `or None`: the UI sends 0 for "not given", and load_doc treats a
        # falsy column count as "discover the sidecar".
        d = E.load_doc(did, sheet, body.get("sidecar"),
                       body.get("columns") or None, body.get("rows") or None)
    except Exception as ex:                          # noqa: BLE001
        return JSONResponse({"ok": False, "error": "%s: %s" % (type(ex).__name__, ex)},
                            status_code=400)
    _evict_docs(1)
    with DOC_LOCK:
        DOCS[did] = d
        # Stamped under the lock: an entry with no `last_used` reads as the
        # oldest thing in the map, so a concurrent open could evict this one
        # before the caller had even seen its id.
        d.last_used = time.time()
    return {"ok": True, "doc": _payload(d)}


@app.get("/api/editor/{did}/cell/{i}.png")
def editor_cell(did: str, i: int):
    try:
        d = _doc(did)
        png = d.cell_png(i)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"}, status_code=404)
    except IndexError as ex:
        return JSONResponse({"ok": False, "error": str(ex)}, status_code=404)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/editor/{did}/snap_preview")
async def editor_snap_preview(did: str, req: Request):
    """Snap one frame so the Snap Pixels panel can show the result live.

    The op itself lives server-side with every other op; this is the same
    `pixel_snapper` call on a single cell, returned as a data URL and the
    dimensions it produced (which is what the panel zooms to).
    """
    body = await req.json() or {}
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"},
                            status_code=404)
    args = body.get("args") or {}
    n = d.n
    sel = sorted({int(i) for i in (body.get("sel") or [])
                  if 0 <= int(i) < n})
    i = sel[0] if sel else (int(body.get("cell") or 0) % max(1, n))
    try:
        import pixel_snapper
        outs = pixel_snapper.snap_pngs(
            [d.cell_png(i)],
            int(args.get("colors") or 16),
            int(args.get("pixel_size") or 0) or None,
            (args.get("palette") or "").strip() or None)
    except Exception as ex:                          # noqa: BLE001
        return JSONResponse({"ok": False, "error": "%s" % ex}, status_code=400)
    if not outs:
        return JSONResponse({"ok": False, "error": "snapper returned nothing"},
                            status_code=400)
    import base64
    o = outs[0]
    bw, bh = d.cell_w, d.cell_h
    return {"ok": True, "cell": i, "w": o["w"], "h": o["h"],
            "src_w": bw, "src_h": bh,
            "png": "data:image/png;base64," + base64.b64encode(o["png"]).decode("ascii")}


@app.get("/api/editor/{did}/sheet.png")
def editor_sheet_png(did: str):
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"}, status_code=404)
    import io as _io
    buf = _io.BytesIO()
    d.compose().save(buf, "PNG", compress_level=1)
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


# --- per-frame audio ------------------------------------------------------- #
# Clips are staged under uploads/ (gitignored, private to this server) while the
# document is open. save_doc copies them into the output's audio/ folder and the
# sidecar names that copy, which is the file an engine ships and loads.

def _audio_dir(doc_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id)[:80] or "doc"
    d = os.path.join(UPLOADS, "editor_audio", safe)
    os.makedirs(d, exist_ok=True)
    return d


# --- a folder of clips, for the editor's audio library --------------------- #
# The panel points at a folder and lists what is in it; a clip dragged onto a
# frame is attached **by path**, so its bytes never travel through the browser.
# That is the whole reason to browse a folder rather than upload files one at a
# time, and it is why this is a listing endpoint and not a file picker.

AUDIO_EXT = (".wav", ".mp3", ".ogg", ".oga", ".opus", ".flac", ".m4a", ".aac",
             ".aif", ".aiff", ".weba", ".webm")
AUDIO_LIB_LIMIT = 400                    # clips returned in one listing
AUDIO_LIB_DIRS = 2000                    # folders visited before a scan stops
MAX_AUDIO_BYTES = 64 * 1024 * 1024


def _audio_scan(folder, limit, max_dirs):
    """Audio files under `folder`, breadth-first, bounded on both axes.

    Breadth-first so a clip sitting in the folder itself is listed before one
    three levels down. Bounded because the path is whatever the user typed: a
    folder pointed at a drive root has to return promptly rather than walk it.
    Returns (entries, truncated).
    """
    out, queue, at, dirs = [], [folder], 0, 0
    while at < len(queue):
        cur = queue[at]
        at += 1
        if not os.path.isdir(cur):
            continue
        dirs += 1
        if dirs > max_dirs:
            return out, True
        try:
            names = sorted(os.listdir(cur))
        except OSError:                  # unreadable subfolder: skip, keep going
            continue
        for fn in names:
            p = os.path.join(cur, fn)
            if os.path.isdir(p):
                if not fn.startswith("."):
                    queue.append(p)
                continue
            if fn.startswith(".") or not fn.lower().endswith(AUDIO_EXT):
                continue
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            out.append({"name": fn, "path": p,
                        "rel": os.path.relpath(p, folder),
                        "ext": os.path.splitext(fn)[1].lower().lstrip("."),
                        "bytes": size})
            if len(out) >= limit:
                return out, True
    return out, False


def _clean_path(raw):
    """A path from a form field: trimmed, unquoted, ~ and %VAR% expanded."""
    p = str(raw or "").strip().strip('"').strip("'")
    if not p:
        return ""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))


@app.get("/api/editor/audio_lib")
def audio_library(folder: str = Query("", alias="dir"),
                  limit: int = AUDIO_LIB_LIMIT):
    """List the audio files in a folder, for the editor's Audio library panel."""
    want = _clean_path(folder)
    if not want:
        return JSONResponse({"ok": False, "error": "give a folder to list"},
                            status_code=400)
    if not os.path.isdir(want):
        return JSONResponse({"ok": False, "error": "not a folder: " + want},
                            status_code=400)
    limit = max(1, min(4000, int(limit)))
    items, truncated = _audio_scan(want, limit, AUDIO_LIB_DIRS)
    return {"ok": True, "dir": want, "audio": items, "truncated": truncated}


@app.get("/api/editor/audio_lib/file")
def audio_library_file(path: str = Query("")):
    """Serve one listed clip's bytes, so the panel can audition it before a drag.

    The listing hands the page an absolute path and nothing else -- the point of
    browsing a folder rather than uploading is that a clip's bytes never travel
    through the browser until it is actually attached. Auditioning is the one
    exception: a page cannot play a path it cannot fetch. So this serves the file
    the listing named, under the same rules the attach route applies and no
    others -- a real file, an audio extension, non-empty, under the size cap --
    and nothing outside that set can be reached through it.
    """
    src = _clean_path(path)
    if not src or not os.path.isfile(src):
        return JSONResponse({"ok": False, "error": "no such file: " + src},
                            status_code=400)
    if not src.lower().endswith(AUDIO_EXT):
        return JSONResponse({"ok": False, "error": "not an audio file: "
                            + os.path.basename(src)}, status_code=400)
    size = os.path.getsize(src)
    if size <= 0:
        return JSONResponse({"ok": False, "error": "that clip is empty"},
                            status_code=400)
    if size > MAX_AUDIO_BYTES:
        return JSONResponse({"ok": False, "error": "%s is over the %d MB cap"
                            % (os.path.basename(src),
                               MAX_AUDIO_BYTES // 1048576)}, status_code=400)
    mt = mimetypes.guess_type(src)[0] or "application/octet-stream"
    return FileResponse(src, media_type=mt, filename=os.path.basename(src),
                        headers={"Cache-Control": "no-store"})


@app.post("/api/editor/{did}/audio/from_path")
async def editor_audio_from_path(did: str, req: Request):
    """Attach a clip that is already on disk, without shipping it to the page.

    The clip is still *staged* into the document's own uploads/ folder rather
    than referenced where it sits, so save_doc keeps one rule about where a
    clip's bytes live: a library file that is later moved, edited or deleted
    cannot change what the saved sheet ships.
    """
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"},
                            status_code=404)
    body = await req.json() or {}
    src = _clean_path(body.get("path"))
    if not src or not os.path.isfile(src):
        return JSONResponse({"ok": False, "error": "no such file: " + src},
                            status_code=400)
    if not src.lower().endswith(AUDIO_EXT):
        return JSONResponse({"ok": False, "error": "not an audio file: "
                            + os.path.basename(src)}, status_code=400)
    size = os.path.getsize(src)
    if size <= 0:
        return JSONResponse({"ok": False, "error": "that clip is empty"},
                            status_code=400)
    if size > MAX_AUDIO_BYTES:
        return JSONResponse({"ok": False, "error": "%s is %.1f MB; the cap is %d MB"
                            % (os.path.basename(src), size / 1048576.0,
                               MAX_AUDIO_BYTES // 1048576)}, status_code=400)
    try:
        frame = int(body.get("frame") or 0)
        vol = 1.0 if body.get("volume") is None else float(body["volume"])
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "frame and volume must be numbers"},
                            status_code=400)
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(src)) or "clip"
    dest = os.path.join(_audio_dir(did), "%d_%s" % (time.time_ns(), base))
    shutil.copy2(src, dest)
    with DOC_LOCK:
        e = d.add_audio(max(0, min(d.n - 1, frame)), dest, name=base, volume=vol)
        payload = d.audio_payload()
    return {"ok": True, "id": e["id"], "frame": e["frame"], "bytes": size,
            "audio": payload}


@app.post("/api/editor/{did}/audio")
async def editor_audio_add(did: str, req: Request, frame: int = 0,
                           name: str = "clip"):
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"},
                            status_code=404)
    data = await req.body()
    if not data:
        return JSONResponse({"ok": False, "error": "empty audio upload"},
                            status_code=400)
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(name)) or "clip"
    dest = os.path.join(_audio_dir(did), "%d_%s" % (time.time_ns(), base))
    with open(dest, "wb") as fh:
        fh.write(data)
    with DOC_LOCK:
        e = d.add_audio(max(0, min(d.n - 1, int(frame))), dest, name=base)
        payload = d.audio_payload()
    return {"ok": True, "id": e["id"], "frame": e["frame"], "bytes": len(data),
            "audio": payload}


@app.post("/api/editor/{did}/audio/{aid}/volume")
async def editor_audio_volume(did: str, aid: int, req: Request):
    """Set an attached clip's volume.

    Not a preview-only knob: the value is what save_doc writes into the sidecar,
    so this is the number an engine plays the clip at. Refused rather than
    clamped for a value outside 0..1, so a wrong number cannot be mistaken for a
    quiet clip.
    """
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"},
                            status_code=404)
    try:
        body = await req.json() or {}
    except ValueError:
        return JSONResponse({"ok": False, "error": "a JSON body is required"},
                            status_code=400)
    try:
        vol = float(body.get("volume"))
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "volume must be a number"},
                            status_code=400)
    if not 0.0 <= vol <= 1.0:
        return JSONResponse({"ok": False,
                             "error": "volume must be between 0 and 1"},
                            status_code=400)
    with DOC_LOCK:
        e = d.set_audio_volume(aid, vol)
        if e is None:
            return JSONResponse({"ok": False, "error": "no such clip"},
                                status_code=404)
        payload = d.audio_payload()
    return {"ok": True, "id": e["id"], "volume": e["volume"], "audio": payload}


@app.get("/api/editor/{did}/audio/{aid}")
def editor_audio_get(did: str, aid: int):
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"},
                            status_code=404)
    e = next((x for x in d.audio if x["id"] == int(aid)), None)
    if e is None:
        return JSONResponse({"ok": False, "error": "no such clip"},
                            status_code=404)
    if not os.path.isfile(e["path"]):
        return JSONResponse({"ok": False, "error": "clip missing on disk"},
                            status_code=404)
    mt = mimetypes.guess_type(e["name"])[0] or "application/octet-stream"
    return FileResponse(e["path"], media_type=mt, filename=e["name"],
                        headers={"Cache-Control": "no-store"})


@app.delete("/api/editor/{did}/audio/{aid}")
def editor_audio_drop(did: str, aid: int):
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"},
                            status_code=404)
    with DOC_LOCK:
        gone = d.drop_audio(aid)
        payload = d.audio_payload()
    return {"ok": True, "removed": gone, "audio": payload}


@app.post("/api/editor/{did}/op")
async def editor_op(did: str, req: Request):
    body = await req.json() or {}
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"}, status_code=404)
    with DOC_LOCK:
        try:
            changed, msg, depth = E.run_op(d, body.get("name"),
                                           body.get("sel"), body.get("args"))
        except Exception as ex:                      # noqa: BLE001
            return {"ok": False, "error": "%s: %s" % (type(ex).__name__, ex),
                    "doc": _payload(d)}
    return {"ok": True, "changed": changed, "msg": msg, "undo_depth": depth,
            "doc": _payload(d)}


@app.post("/api/editor/{did}/undo")
def editor_undo(did: str):
    with DOC_LOCK:
        try:
            d = _doc(did)
        except KeyError:
            return JSONResponse({"ok": False, "error": "unknown document"}, status_code=404)
        e = d.undo_once()
    return {"ok": True, "restored": bool(e),
            "changed": list(range(d.n)) if (e and e["full"]) else (
                sorted(e["cells"]) if e else []),
            "doc": _payload(d)}


@app.post("/api/editor/{did}/redo")
def editor_redo(did: str):
    with DOC_LOCK:
        try:
            d = _doc(did)
        except KeyError:
            return JSONResponse({"ok": False, "error": "unknown document"}, status_code=404)
        e = d.redo_once()
    return {"ok": True, "restored": bool(e),
            "changed": list(range(d.n)) if (e and e["full"]) else (
                sorted(e["cells"]) if e else []),
            "doc": _payload(d)}


@app.get("/api/editor/{did}/verify")
def editor_verify(did: str):
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"}, status_code=404)
    ok, lines = E.verify_doc(d)
    return {"ok": True, "passed": ok, "lines": lines}


@app.post("/api/editor/{did}/save")
async def editor_save(did: str, req: Request):
    body = await req.json() or {}
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"}, status_code=404)
    name = body.get("name") or d.meta.get("name") or "sprite"
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:60] or "sprite"
    out_dir = body.get("out_dir") or ""
    # Whether the user actually chose a destination. A blank box is the default
    # and it means "the app's runs folder", so it is not a choice -- and that
    # difference is the whole of "overwriting not working": with a blank box the
    # ticked save went to a brand-new runs/editor_<stamp>/ folder instead of the
    # files it was loaded from.
    named_out = bool(out_dir)
    if out_dir:
        out_dir = os.path.abspath(out_dir)
    else:
        out_dir = os.path.join(RUNS, "editor_%s_%s"
                               % (time.strftime("%m%d-%H%M%S"), safe))
    log = []

    def emit(stage, frac, msg=""):
        log.append("%s %s" % (stage, msg))

    # "only the selected frames". An empty/absent list means the whole document,
    # so an old client that never sends `frames` behaves exactly as before.
    src = d
    n_out = d.n
    frames = body.get("frames")
    if frames:
        try:
            src, idx = E.export_subset(d, frames)
            n_out = len(idx)
        except Exception as ex:                      # noqa: BLE001
            return JSONResponse({"ok": False,
                                 "error": "%s: %s" % (type(ex).__name__, ex)},
                                status_code=400)

    try:
        arts = E.save_doc(src, out_dir, name=safe,
                          want_gif=bool(body.get("want_gif", True)),
                          want_preview=bool(body.get("want_preview", True)),
                          gif_scale=float(body.get("gif_scale", 0.5) or 0.5),
                          gif_bg=body.get("gif_bg", "checker"),
                          # Only ever true because the caller asked for it: the
                          # default is to refuse writing over the loaded sheet.
                          overwrite=bool(body.get("overwrite", False)),
                          # ...and only redirected to the loaded files when the
                          # caller named no destination of their own.
                          out_dir_named=named_out,
                          emit=emit)
    except Exception as ex:                          # noqa: BLE001
        return JSONResponse({"ok": False,
                             "error": "%s: %s" % (type(ex).__name__, ex),
                             "log": log}, status_code=400)
    # Where the sheet actually went, not where the box said to put it. Ticking
    # "allow overwriting the sheet this document was loaded from" with a blank
    # output folder writes everything back beside the loaded files -- sheet,
    # sidecar, preview and gif -- so reporting out_dir here would name a folder
    # the sheet is not in. Each artifact is also served on its own merits rather
    # than off one flag for the whole save, since an overwrite of files outside
    # the app's runs folder leaves all of them unservable together.
    sheet_dir = os.path.dirname(arts["sheet"])
    runs = os.path.abspath(RUNS)
    links = {}
    for k, v in arts.items():
        av = os.path.abspath(v)
        links[k] = (os.path.relpath(av, RUNS).replace("\\", "/")
                    if av.startswith(runs) else av)
    served = all(os.path.abspath(v).startswith(runs) for v in arts.values())
    return {"ok": True, "dir": sheet_dir, "served": served,
            "frames": n_out, "of": d.n,
            # The absolute path of the sheet that was just written, so the page
            # can reopen it and carry on editing the frames it just exported.
            # `artifacts.sheet` is relative to RUNS when it is servable, which is
            # a URL path, not something openSheet() can use.
            "sheet_path": os.path.abspath(arts["sheet"]),
            "artifacts": links, "log": log}


@app.post("/api/editor/{did}/save_meta")
async def editor_save_meta(did: str):
    """Write this document's metadata back into the sidecar it was loaded from.

    Deliberately not folded into the op route: an op is a pure change to the
    in-memory document, and one that quietly wrote a file would be a different
    kind of thing wearing the same name. The page calls this straight after
    set_meta, which is what makes Apply metadata persist.

    Only the JSON is touched -- the sheet PNG is not rewritten, which is why
    E.save_meta refuses when the grid has moved on from what the sidecar says.
    """
    try:
        d = _doc(did)
    except KeyError:
        return JSONResponse({"ok": False, "error": "unknown document"}, status_code=404)
    try:
        path = E.save_meta(d)
    except Exception as ex:                          # noqa: BLE001
        return JSONResponse({"ok": False,
                             "error": "%s: %s" % (type(ex).__name__, ex)},
                            status_code=400)
    return {"ok": True, "sidecar": path}


@app.post("/api/editor/{did}/close")
def editor_close(did: str):
    with DOC_LOCK:
        DOCS.pop(did, None)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# self-update -- pull the latest source from this app's own repo, then restart
# --------------------------------------------------------------------------- #
#
# The repo is public, so the update needs no token and asks for none: plain
# unauthenticated GitHub API (60 req/h, and an update is one request).
#
# Why the files are copied over the live tree rather than the container being
# rebuilt: this is a dev convenience -- click, get latest, keep working. It is
# explicitly NOT durable. The installed source lives in the container and dies
# with it; a rebuild replaces it. The durable path is still a rebuild.
#
# Why the restart is a re-exec: `ui.html` is re-read per request, but `app.py`
# and `editor.py` are Python held in memory, so new code on disk is invisible
# until the process restarts. `os.execv` replaces the process image in place,
# which keeps the same PID 1 and the same container -- a plain `sys.exit`
# would stop the container instead (the default restart policy is `no`).

APP_REPO = os.environ.get("SPRITE_APP_REPO", "camenduru/TostAI-Sprite-Sheet-Studio")
APP_REV_FILE = os.path.join(HERE, ".sprite_rev")
UPDATE_BACKUP = os.path.join(HERE, ".update_backup")

# Never overwritten by an update. `runs/` and `uploads/` are this container's
# own state, are gitignored upstream, and so should never appear in the tarball
# at all -- this is belt and braces, because the cost of being wrong is somebody's
# output. `.update_backup` is excluded so a backup never contains itself.
UPDATE_KEEP = ("runs", "uploads", ".update_backup", ".git")

_UA = {"User-Agent": "sprite-studio-update",
       "Accept": "application/vnd.github+json"}


def _gh_json(url):
    """GET a GitHub API URL. Returns (data, None) or (None, human_error)."""
    req = urllib.request.Request(url, headers=dict(_UA))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, ("GitHub returned 404 for %s. The repository name "
                          "is wrong or the repo is gone." % url)
        if e.code == 403:
            return None, ("GitHub returned 403 (rate limit or blocked). "
                          "Wait a while and try again.")
        return None, "GitHub returned HTTP %d." % e.code
    except Exception as ex:                                     # noqa: BLE001
        return None, "%s: %s" % (type(ex).__name__, ex)


def _current_rev():
    try:
        with open(APP_REV_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _restart_soon(delay=1.5):
    """Replace this process with a fresh one, once the response has flushed.

    The delay is load-bearing: both paths below tear the server down, so doing it
    before the response is written turns a successful update into a network error
    on the user's screen.
    """
    def _go():
        time.sleep(delay)
        argv = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
        try:
            if os.name == "nt":
                # Windows: os.execv is NOT usable from here. Measured on this
                # machine, and reproducible in isolation -- a uvicorn server that
                # execv's itself from a background thread dies outright: the
                # in-flight response is cut off mid-body (200 headers, empty
                # body), no replacement process is ever started, nothing is
                # printed, and the port goes dead. The identical execv from a
                # plain script's MAIN thread works, so it is the uvicorn +
                # background-thread combination, not execv itself.
                #
                # So spawn a detached replacement and leave. Exiting is safe on
                # Windows because there is no PID 1 to keep alive.
                flags = (subprocess.DETACHED_PROCESS
                         | subprocess.CREATE_NEW_PROCESS_GROUP)
                subprocess.Popen(argv, cwd=HERE, close_fds=True,
                                 creationflags=flags)
                os._exit(0)
            else:
                # POSIX, and specifically a container: execve(2) swaps the image
                # inside the SAME pid, so PID 1 stays PID 1 and the container is
                # not stopped. That is the entire reason a re-exec is used here
                # instead of exiting -- exiting would end the container, and the
                # default restart policy is `no`.
                os.execv(sys.executable, argv)
        except Exception:                                       # noqa: BLE001
            # Restart failed, so this process is still serving the OLD code.
            # Say so rather than exiting: a stopped container is worse than a
            # stale one that still works.
            traceback.print_exc()
    threading.Thread(target=_go, daemon=True, name="sprite-restart").start()


# The commit subject the running process started from. Set on a successful
# POST and read by the GET below, so the dialog can name what is installed
# rather than only its hash.
_rev_subject = ""


@app.get("/api/update")
def update_status():
    """What is installed, so the dialog can say so before anything is typed."""
    rev = _current_rev()
    return {"ok": True, "repo": APP_REPO, "rev": rev, "short": rev[:10],
            "subject": _rev_subject}


@app.post("/api/update")
async def update_apply():
    global _rev_subject
    # No body needed: the repo is public, so there is nothing to send.

    # 1. What is upstream right now?
    commits, err = _gh_json(
        "https://api.github.com/repos/%s/commits?per_page=1" % APP_REPO)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if not commits:
        return JSONResponse({"ok": False, "error": "the repository has no commits"},
                            status_code=400)
    latest = commits[0]["sha"]
    subject = (commits[0].get("commit", {}).get("message") or "").splitlines()[0]
    current = _current_rev()

    if latest == current:
        _rev_subject = subject
        return {"ok": True, "updated": False, "rev": latest, "subject": subject,
                "message": "already up to date at %s" % latest[:10]}

    tmp = tempfile.mkdtemp(prefix="sprite_update_")
    try:
        # 2. Fetch that exact tree. The API redirects to codeload with a signed
        #    token in the URL, so the redirect is followed without the header.
        tarball = os.path.join(tmp, "src.tar.gz")
        dl = urllib.request.Request(
            "https://api.github.com/repos/%s/tarball/%s" % (APP_REPO, latest),
            headers=dict(_UA))
        try:
            with urllib.request.urlopen(dl, timeout=180) as r, \
                    open(tarball, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as ex:                                 # noqa: BLE001
            return JSONResponse({"ok": False, "error": "download failed: %s: %s"
                                 % (type(ex).__name__, ex)}, status_code=400)

        root = os.path.join(tmp, "x")
        os.makedirs(root)
        with tarfile.open(tarball, "r:gz") as tf:
            try:
                tf.extractall(root, filter="data")              # py3.12+
            except TypeError:
                tf.extractall(root)                             # py3.10/3.11

        # GitHub wraps the tree in a single <owner>-<repo>-<shortsha> directory.
        entries = sorted(e for e in os.listdir(root) if not e.startswith("."))
        if len(entries) != 1 or not os.path.isdir(os.path.join(root, entries[0])):
            return JSONResponse({"ok": False, "error":
                                 "unexpected tarball layout: %r" % entries},
                                status_code=400)
        src = os.path.join(root, entries[0])

        if not os.path.isfile(os.path.join(src, "app.py")):
            return JSONResponse({"ok": False, "error":
                                 "refusing to install: the new tree has no app.py"},
                                status_code=400)

        # 3. Parse every new .py BEFORE anything is swapped in. This is the guard
        #    that stops a typo upstream from bricking the studio: a file that
        #    fails to parse would be re-exec'd into a container that never comes
        #    back, and the default restart policy is `no`.
        #
        #    `compile()` on the source, NOT `py_compile.compile(..., cfile=...)`:
        #    py_compile refuses /dev/null on Linux -- "is a non-regular file and
        #    will be changed into a regular one" -- and there is no reason to
        #    write bytecode at all just to check syntax. Reading bytes lets
        #    compile() honour any PEP 263 coding declaration in the file.
        bad = []
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in UPDATE_KEEP]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "rb") as fh:
                        compile(fh.read(), p, "exec")
                except SyntaxError as ex:
                    bad.append("%s: line %s: %s" % (os.path.relpath(p, src),
                                                    ex.lineno, ex.msg))
                except (OSError, ValueError) as ex:
                    bad.append("%s: %s" % (os.path.relpath(p, src), ex))
        if bad:
            return JSONResponse({"ok": False, "error":
                                 "refusing to install -- the new source does not "
                                 "compile:\n" + "\n".join(bad[:10])},
                                status_code=400)

        # 4. Back up the current tree, then copy the new one over it. Copy, not
        #    replace, so files deleted upstream linger harmlessly rather than
        #    `runs/` and `uploads/` being swept away with them.
        if os.path.isdir(UPDATE_BACKUP):
            shutil.rmtree(UPDATE_BACKUP, ignore_errors=True)
        try:
            shutil.copytree(HERE, UPDATE_BACKUP,
                            ignore=shutil.ignore_patterns(*UPDATE_KEEP))
        except Exception as ex:                                 # noqa: BLE001
            return JSONResponse({"ok": False, "error":
                                 "could not write a backup, so nothing was "
                                 "changed: %s" % ex}, status_code=400)

        copied = 0
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in UPDATE_KEEP]
            rel = os.path.relpath(dirpath, src)
            dst_dir = HERE if rel == "." else os.path.join(HERE, rel)
            os.makedirs(dst_dir, exist_ok=True)
            for fn in filenames:
                shutil.copy2(os.path.join(dirpath, fn),
                             os.path.join(dst_dir, fn))
                copied += 1

        with open(APP_REV_FILE, "w", encoding="utf-8") as f:
            f.write(latest + "\n")
        _rev_subject = subject

        _restart_soon()
        return {"ok": True, "updated": True, "rev": latest, "subject": subject,
                "files": copied,
                "message": "updated to %s, restarting" % latest[:10]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Sprite Studio server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    import uvicorn
    import sys
    # Headless-proof: under pythonw with no console (Startup folder, Task
    # Scheduler) sys.stdout/sys.stderr are None, and the prints below -- plus
    # uvicorn's own logging -- would crash startup. Point them at a log file
    # instead, line-buffered so a running server's output is actually readable.
    if sys.stdout is None or sys.stderr is None:
        _log = open(os.path.join(RUNS, "studio-console.log"), "a",
                    encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = _log
        if sys.stderr is None:
            sys.stderr = _log
    print("Sprite Studio -> http://%s:%d" % (args.host, args.port))
    print("runs dir      -> %s" % RUNS)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
