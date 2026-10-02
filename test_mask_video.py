"""Regression test: the optional mask video, the alpha it produces, and the
wiring that carries it from the page to the pipeline.

The feature is one sentence long -- "white is the subject, black is the
background, apply it to the source" -- and every interesting case is an edge of
it:

  * a mask that runs out before the source. `zip` stops at the shorter clip, so
    the run would come back with fewer frames than were asked for and the grid
    would be built from the wrong count. Section 4 pins the error.
  * a mask at another resolution. Resized, and said out loud once. Section 5.
  * a mask an encoder muddied. Section 6.
  * the start/end/step range applying to BOTH clips, or the two drift a frame
    apart and the alpha belongs to the wrong pose. Section 3.
  * the model NOT running. A mask that quietly ran VRMBG anyway would be slow,
    and on a clip the model gets wrong it would be wrong too. Section 3 proves
    it by handing the pipeline an empty model cache and checking it stays empty.

The last section is the wiring guard. Three files have to agree about four new
config keys and eight new element ids and nothing links them: `pipeline.py`
declares and coerces the keys, `ui.html` owns the controls and the FIELDS list
that collect() walks, `app.py` owns the upload folder. A field added to FIELDS
with no element is silently skipped by collect() -- the run simply behaves as if
the control did not exist, with nothing in any log saying so. That is the same
class of silent wrong answer the infer_size guard in test_config_types.py
exists for, so it gets the same treatment.

    python test_mask_video.py
"""
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402

import cv2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
fails = []


def check(name, cond, detail=""):
    print("  %-58s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(detail)))
    if not cond:
        fails.append(name)


def _err(fn):
    """Run `fn`, return its exception as a string -- "" if it did not raise."""
    try:
        fn()
        return ""
    except Exception as ex:                                   # noqa: BLE001
        return "%s: %s" % (type(ex).__name__, ex)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
W, H, N = 48, 40, 6
BACKDROP_BGR = (10, 20, 30)
SUBJECT_BGR = (200, 60, 40)
SUBJ_Y = (10, 30)                      # rows the block occupies
FPS = 12.0


def block_x(i):
    return 6 + i * 4                   # the block moves, so a misaligned mask shows


def write_video(path, frames):
    """FFV1 in an AVI: lossless, so a pixel asserted here is the pixel read back.

    The lossy default (mp4v) would leave the mask's black at 2-5 and its white
    at 250-253, and every exact-alpha assertion below would need a tolerance --
    which is the same thing as not testing the mapping at all.
    """
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (w, h))
    if not vw.isOpened():
        raise RuntimeError("no FFV1 encoder available -- cannot build fixtures")
    for f in frames:
        vw.write(f)
    vw.release()
    return path


def source_frames():
    out = []
    for i in range(N):
        f = np.zeros((H, W, 3), np.uint8)
        f[:, :] = BACKDROP_BGR
        f[SUBJ_Y[0]:SUBJ_Y[1], block_x(i):block_x(i) + 14] = SUBJECT_BGR
        out.append(f)
    return out


def mask_frames(scale=1, invert=False, lo=0, hi=255):
    """The mask: a white block exactly where the subject is, black elsewhere.

    `scale` writes it at 1/scale resolution (the resized-mask case), `invert`
    flips it (black subject on white), and `lo`/`hi` set the levels -- the
    encoder-mud case is lo=3, hi=250.
    """
    out = []
    for i in range(N):
        m = np.zeros((H, W, 3), np.uint8)
        m[SUBJ_Y[0]:SUBJ_Y[1], block_x(i):block_x(i) + 14] = 255
        if scale != 1:
            m = cv2.resize(m, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
        if lo or hi != 255:
            m = (m.astype(np.int16) // 255 * (hi - lo) + lo).astype(np.uint8)
        if invert:
            m = 255 - m
        out.append(m)
    return out


def read_rgba(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)     # BGRA
    return im[..., [2, 1, 0, 3]]                    # -> RGBA


print("--- 1. mask_to_alpha: the mapping, and both switches ---")
white = np.full((8, 8, 3), 255, np.uint8)
black = np.zeros((8, 8, 3), np.uint8)
grey = np.full((8, 8, 3), 128, np.uint8)
check("white -> opaque", int(P.mask_to_alpha(white).max()) == 255)
check("black -> transparent", int(P.mask_to_alpha(black).max()) == 0)
check("grey survives as grey (a soft edge stays soft)",
      int(P.mask_to_alpha(grey).min()) == 128 and int(P.mask_to_alpha(grey).max()) == 128)
check("invert: white -> transparent", int(P.mask_to_alpha(white, invert=True).max()) == 0)
check("invert: black -> opaque", int(P.mask_to_alpha(black, invert=True).min()) == 255)
check("invert is its own inverse",
      np.array_equal(P.mask_to_alpha(P.mask_to_alpha(grey, True), True), P.mask_to_alpha(grey)))
check("binary: grey >= threshold -> opaque",
      int(P.mask_to_alpha(grey, binary=True, threshold=128).min()) == 255)
check("binary: grey < threshold -> transparent",
      int(P.mask_to_alpha(grey, binary=True, threshold=129).max()) == 0)
check("binary is applied AFTER invert (invert then threshold)",
      int(P.mask_to_alpha(grey, invert=True, binary=True, threshold=128).max()) == 0
      and int(P.mask_to_alpha(grey, invert=True, binary=True, threshold=100).min()) == 255)
check("a single-channel frame is passed through, not reinterpreted",
      int(P.mask_to_alpha(np.full((4, 4), 77, np.uint8)).max()) == 77)
four = np.zeros((4, 4, 4), np.uint8)
four[..., :3] = 255
four[..., 3] = 77
check("a frame with a real alpha band uses that band",
      int(P.mask_to_alpha(four).max()) == 77)
coloured = np.zeros((4, 4, 3), np.uint8)
coloured[:, :] = (255, 255, 255)                 # white on a green mask
check("a colour-coded white mask still reads as opaque",
      int(P.mask_to_alpha(coloured).min()) == 255)
check("the default is white-is-subject (the user's convention)",
      P.DEFAULT_CFG["mask_invert"] is False and P.DEFAULT_CFG["mask_binary"] is False
      and P.DEFAULT_CFG["mask_video"] == "")

print("\n--- 2. merge_cfg coerces the mask fields, and never trusts the caller ---")
c = P.merge_cfg({"mask_video": "C:/x/m.avi", "mask_invert": "true",
                 "mask_binary": "1", "mask_threshold": "96"})
check("mask_invert 'true' -> True", c["mask_invert"] is True, repr(c["mask_invert"]))
check("mask_binary '1' -> True", c["mask_binary"] is True, repr(c["mask_binary"]))
check("mask_threshold '96' -> 96 (int)",
      c["mask_threshold"] == 96 and isinstance(c["mask_threshold"], int),
      repr(c["mask_threshold"]))
check("a mask path survives", c["mask_video"] == "C:/x/m.avi", c["mask_video"])
c = P.merge_cfg({"mask_video": None})
check("a null path becomes '', not None", c["mask_video"] == "", repr(c["mask_video"]))
c = P.merge_cfg({"mask_video": "  C:/x/m.avi  "})
check("a path copied out of Explorer loses its trailing space",
      c["mask_video"] == "C:/x/m.avi", repr(c["mask_video"]))
c = P.merge_cfg({"mask_threshold": "abc"})
check("garbage threshold falls back to the default",
      c["mask_threshold"] == P.DEFAULT_CFG["mask_threshold"], c["mask_threshold"])
check("no mask_video in the payload -> the empty default",
      P.merge_cfg({})["mask_video"] == "")

tmp = tempfile.mkdtemp(prefix="ss_masktest_")
try:
    src = write_video(os.path.join(tmp, "clip.avi"), source_frames())
    msk = write_video(os.path.join(tmp, "mask.avi"), mask_frames())

    print("\n--- 3. end to end: the mask is the alpha, and the model is not run ---")
    log = []
    cache = {}
    res = P.run_pipeline(
        {"video": src, "mask_video": msk, "want_gif": False, "do_verify": True,
         "cell_mode": "subject", "pad": 4},
        os.path.join(tmp, "run1"),
        emit=lambda s, f, m="": log.append((s, m)), model_cache=cache)
    frames = res["frames"]
    check("every frame came out", len(frames) == N, len(frames))
    check("the model was never loaded (the cache is still empty)",
          cache == {}, cache)
    a0 = read_rgba(frames[0])
    check("alpha is opaque over the subject",
          int(a0[SUBJ_Y[0] + 2:SUBJ_Y[1] - 2, block_x(0) + 2:block_x(0) + 12, 3].min()) == 255,
          int(a0[..., 3].max()))
    check("alpha is transparent off it",
          int(a0[0, 0, 3]) == 0 and int(a0[H - 1, W - 1, 3]) == 0,
          (int(a0[0, 0, 3]), int(a0[H - 1, W - 1, 3])))
    check("nothing is left semi-transparent by a hard mask",
          set(np.unique(a0[..., 3]).tolist()) == {0, 255},
          np.unique(a0[..., 3]).tolist()[:6])
    check("the source's own RGB is kept where the mask is opaque",
          tuple(int(v) for v in a0[20, block_x(0) + 5, :3]) == SUBJECT_BGR[::-1],
          (a0[20, block_x(0) + 5, :3].tolist(), SUBJECT_BGR[::-1]))
    check("and kept where it is transparent -- the mask sets alpha, not colour",
          tuple(int(v) for v in a0[0, 0, :3]) == BACKDROP_BGR[::-1],
          (a0[0, 0, :3].tolist(), BACKDROP_BGR[::-1]))
    # the moving block means a one-frame slip shows up as the wrong column
    check("frame 3's alpha follows frame 3's block, not frame 1's",
          int(read_rgba(frames[2])[20, block_x(2) + 5, 3]) == 255
          and int(read_rgba(frames[2])[20, block_x(0) + 5, 3]) == 0,
          int(read_rgba(frames[2])[20, block_x(0) + 5, 3]))
    check("the log names the mask and says the model was skipped",
          any("VRMBG-3.0 not run" in m for _, m in log),
          [m for _, m in log if "mask" in m.lower()][:2])
    side = res["sidecar"]
    check("the sidecar records the matte as the mask video",
          side.get("matte") == "mask video", side.get("matte"))
    check("the sidecar's note names the mask file",
          "mask video mask.avi" in side.get("matte_note", ""), side.get("matte_note"))
    check("verify ran and passed on a mask-built sheet",
          (res.get("verify") or {}).get("ok") is True,
          (res.get("verify") or {}).get("lines"))
    check("the sheet was built", os.path.isfile(res["artifacts"]["sheet"]))
    sheet = read_rgba(res["artifacts"]["sheet"])
    check("the sheet carries the mask's alpha through",
          set(np.unique(sheet[..., 3]).tolist()) == {0, 255},
          np.unique(sheet[..., 3]).tolist()[:6])

    print("\n--- 4. a mask that runs out is an error, not a short sheet ---")
    short = write_video(os.path.join(tmp, "short.avi"), mask_frames()[:4])
    try:
        P.run_pipeline({"video": src, "mask_video": short, "want_sidecar": False,
                        "want_preview": False},
                       os.path.join(tmp, "run2"))
        check("a short mask raises", False, "it returned normally")
    except ValueError as ex:
        check("a short mask raises ValueError", True)
        check("and says which frame it died on",
              "ran out at source frame 5 of 6" in str(ex), str(ex)[:160])
        check("and names the End that would line the two up",
              "End=4 lines the two up" in str(ex), str(ex)[:200])
    # A mask that exists but is entirely before the range: the source yields
    # frames from 4, the 2-frame mask yields none, so the very first `next()`
    # fails and there is no "frame it died on" to name.
    early = write_video(os.path.join(tmp, "early.avi"), mask_frames()[:2])
    err0 = _err(lambda: P.run_pipeline(
        {"video": src, "mask_video": early, "start": 4, "want_sidecar": False,
         "want_preview": False}, os.path.join(tmp, "run3b")))
    check("a mask that never overlaps the range is its own error",
          "produced no frames at all" in err0, err0[:160])
    err = _err(lambda: P.run_pipeline(
        {"video": src, "mask_video": os.path.join(tmp, "nope.avi")},
        os.path.join(tmp, "run3")))
    check("a missing mask file is refused before anything is loaded",
          "mask video not found" in err, err[:120])

    print("\n--- 5. a mask at another resolution is resized, and said so ---")
    half = write_video(os.path.join(tmp, "half.avi"), mask_frames(scale=2))
    log2 = []
    res2 = P.run_pipeline({"video": src, "mask_video": half, "want_sidecar": False,
                           "want_preview": False, "cell_mode": "none"},
                          os.path.join(tmp, "run4"),
                          emit=lambda s, f, m="": log2.append(m))
    check("the resize is reported once",
          sum(1 for m in log2 if "resized to the source's" in m) == 1,
          [m for m in log2 if "resized" in m])
    check("the message names both sizes",
          any("24x20" in m and "48x40" in m for m in log2),
          [m for m in log2 if "resized" in m])
    a = read_rgba(res2["frames"][0])
    check("the upscaled mask is still opaque over the subject",
          int(a[SUBJ_Y[0] + 2:SUBJ_Y[1] - 2, block_x(0) + 2:block_x(0) + 12, 3].min()) == 255)
    check("and still transparent in the corner", int(a[0, 0, 3]) == 0)

    print("\n--- 6. the range applies to both clips, and the two switches work ---")
    log3 = []
    res3 = P.run_pipeline({"video": src, "mask_video": msk, "want_sidecar": False,
                           "want_preview": False, "cell_mode": "none",
                           "start": 2, "end": 5, "frame_step": 2},
                          os.path.join(tmp, "run5"),
                          emit=lambda s, f, m="": log3.append(m))
    check("start/end/step select the same frames from the mask (2 of 6)",
          len(res3["frames"]) == 2, len(res3["frames"]))
    # column block_x(0)=6 is inside block 0 and outside every later block, so a
    # one-frame slip reads as opaque where it must be transparent.
    check("frame 1 of the range is source frame 2, with frame 2's alpha",
          int(read_rgba(res3["frames"][0])[20, block_x(1) + 5, 3]) == 255
          and int(read_rgba(res3["frames"][0])[20, block_x(0), 3]) == 0,
          int(read_rgba(res3["frames"][0])[20, block_x(0), 3]))
    check("frame 2 of the range is source frame 4, with frame 4's alpha",
          int(read_rgba(res3["frames"][1])[20, block_x(3) + 5, 3]) == 255
          and int(read_rgba(res3["frames"][1])[20, block_x(1), 3]) == 0,
          int(read_rgba(res3["frames"][1])[20, block_x(1), 3]))

    inv = write_video(os.path.join(tmp, "inv.avi"), mask_frames(invert=True))
    res4 = P.run_pipeline({"video": src, "mask_video": inv, "want_sidecar": True,
                           "want_preview": False, "cell_mode": "none",
                           "mask_invert": True},
                          os.path.join(tmp, "run6"))
    ai = read_rgba(res4["frames"][0])
    check("invert recovers a black-subject mask",
          int(ai[20, block_x(0) + 5, 3]) == 255 and int(ai[0, 0, 3]) == 0,
          (int(ai[20, block_x(0) + 5, 3]), int(ai[0, 0, 3])))
    check("and the sidecar says it was inverted",
          ", inverted" in res4["sidecar"].get("matte_note", ""),
          res4["sidecar"].get("matte_note"))

    mud = write_video(os.path.join(tmp, "mud.avi"), mask_frames(lo=3, hi=250))
    soft = P.run_pipeline({"video": src, "mask_video": mud, "want_sidecar": False,
                           "want_preview": False, "cell_mode": "none"},
                          os.path.join(tmp, "run7"))
    check("a muddied mask leaves the background at alpha 3, not 0",
          int(read_rgba(soft["frames"][0])[0, 0, 3]) == 3,
          int(read_rgba(soft["frames"][0])[0, 0, 3]))
    hard = P.run_pipeline({"video": src, "mask_video": mud, "want_sidecar": True,
                           "want_preview": False, "cell_mode": "none",
                           "mask_binary": True, "mask_threshold": 128},
                          os.path.join(tmp, "run8"))
    check("binarising at the threshold cleans it to 0",
          int(read_rgba(hard["frames"][0])[0, 0, 3]) == 0,
          int(read_rgba(hard["frames"][0])[0, 0, 3]))
    check("and keeps the subject opaque",
          int(read_rgba(hard["frames"][0])[20, block_x(0) + 5, 3]) == 255)
    check("the sidecar records the binarisation",
          ", binarised at 128" in hard["sidecar"].get("matte_note", ""),
          hard["sidecar"].get("matte_note"))

    print("\n--- 7. no mask still means no mask ---")
    res5 = P.run_pipeline({"video": src, "mask_video": "", "do_matte": False,
                           "want_sidecar": True, "want_preview": False,
                           "cell_mode": "none"},
                          os.path.join(tmp, "run9"))
    check("an empty mask path runs the plain extract path",
          res5["sidecar"].get("matte") == "none", res5["sidecar"].get("matte"))
    check("and the frames are fully opaque",
          int(read_rgba(res5["frames"][0])[..., 3].min()) == 255)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n--- 8. the wiring: three files, no linker ---")
with open(os.path.join(HERE, "ui.html"), encoding="utf-8") as fh:
    ui = fh.read()
with open(os.path.join(HERE, "app.py"), encoding="utf-8") as fh:
    appsrc = fh.read()

_m = re.search(r"const FIELDS = \[(.*?)\];", ui, re.S)
fields = re.findall(r'"([a-z_0-9]+)"', _m.group(1)) if _m else []
check("the page's FIELDS list was found", bool(fields), _m)
for k in ("mask_video", "mask_invert", "mask_binary", "mask_threshold"):
    check("%-16s is in FIELDS (collect() will read it)" % k, k in fields)
    check("%-16s is a key the pipeline knows" % k, k in P.DEFAULT_CFG)
    check("%-16s has a control on the page" % k, ('id="%s"' % k) in ui)
# The generic half of the same guard: a name in FIELDS with no element is
# skipped by collect() in silence, so every entry is checked, not just the new
# ones. This is what caught nothing today and will catch the next control.
missing = [f for f in fields if ('id="%s"' % f) not in ui]
check("every FIELDS entry has an element (none are silently skipped)",
      not missing, missing)
check("the mask path is gated on its own checkbox, not on being non-empty",
      'cfg.mask_video = $("use_mask").checked' in ui)
check("dropping a mask switches the path on",
      '$("use_mask").checked = true' in ui)
check("the mask uploads to its own folder",
      'sub=masks' in ui and 'os.path.join(UPLOADS, "masks")' in appsrc)
check("the page renders a mask preview",
      'id="maskprobe"' in ui and '$("maskprobe").onclick' in ui)
# The click handler that passes `which` must be a closure: a bare function
# reference would be handed the MouseEvent as its first argument.
check("the drop zone's click handler does not leak the event into `which`",
      'zone.addEventListener("click", () => pickNative(which))' in ui)

print("\n%s (%d failure%s)" % ("ALL PASS" if not fails else "FAILURES", len(fails),
                               "" if len(fails) == 1 else "s"))
sys.exit(1 if fails else 0)
