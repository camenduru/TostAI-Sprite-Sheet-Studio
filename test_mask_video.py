"""Regression test: the optional mask video, the alpha it produces, and the
wiring that carries it from the page to the pipeline.

The feature is one sentence long -- "white is the subject, black is the
background, apply it to the source" -- and every interesting case is an edge of
it:

  * a mask that runs out before the source. This was a hard error first, and the
    user hit it (121-frame mask, 124-frame clip) -- the guard was protecting the
    grid from a frame count taken off the wrong clip, but the count comes from
    `len(paths)`, so a clamp is already a correct grid, just a shorter one.
    Section 4 pins the clamp and the three places it is reported.
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
c = P.merge_cfg({"mask_start_skip": "3", "mask_end_skip": "1"})
check("mask_start_skip '3' -> 3 (int)",
      c["mask_start_skip"] == 3 and isinstance(c["mask_start_skip"], int),
      repr(c["mask_start_skip"]))
check("mask_end_skip '1' -> 1 (int)",
      c["mask_end_skip"] == 1 and isinstance(c["mask_end_skip"], int),
      repr(c["mask_end_skip"]))
c = P.merge_cfg({"mask_start_skip": "-4", "mask_end_skip": "-1"})
check("a negative skip floors at 0 rather than walking the mask backwards",
      c["mask_start_skip"] == 0 and c["mask_end_skip"] == 0,
      (c["mask_start_skip"], c["mask_end_skip"]))
c = P.merge_cfg({"mask_start_skip": "abc"})
check("garbage skip falls back to the default", c["mask_start_skip"] == 0,
      c["mask_start_skip"])
check("the skips default to 0/0 -- nothing moves unless asked",
      P.DEFAULT_CFG["mask_start_skip"] == 0 and P.DEFAULT_CFG["mask_end_skip"] == 0)
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

    print("\n--- 4. a short mask clamps the sheet, and says so in three places ---")
    # This was a hard error first, and the user hit it: a 121-frame mask against
    # a 124-frame clip refused to run and made them work out End=121. The error
    # existed to protect the grid from a frame count taken off the wrong clip --
    # but the count comes from len(paths), so clamping is already a *correct*
    # grid, just a shorter one. Refusing the work was the wrong guard.
    short = write_video(os.path.join(tmp, "short.avi"), mask_frames()[:4])
    res_s = P.run_pipeline({"video": src, "mask_video": short, "want_sidecar": True,
                            "want_preview": False, "cell_mode": "none"},
                           os.path.join(tmp, "run2"))
    check("the sheet is as long as the mask, not the source",
          len(res_s["frames"]) == 4, len(res_s["frames"]))
    check("and the grid is still full -- it was built from the sheet's own count",
          res_s["layout"]["columns"] * res_s["layout"]["rows"] == 4,
          (res_s["layout"]["columns"], res_s["layout"]["rows"]))
    # column 20 is inside block 3 (18..31) and outside block 0 (6..19), so a
    # one-frame slip reads as transparent where it must be opaque.
    check("the four frames are the source's first four, with the mask's alpha",
          int(read_rgba(res_s["frames"][3])[20, 20, 3]) == 255
          and int(read_rgba(res_s["frames"][3])[20, 6, 3]) == 0,
          int(read_rgba(res_s["frames"][3])[20, 20, 3]))
    warn = " ".join(res_s["layout"]["warnings"])
    check("the clamp is a layout warning, not only a log line",
          "covers only 4 of the 6" in warn, res_s["layout"]["warnings"])
    check("and it points at the skips, which are the tool for this",
          "Skip at start / Skip at end" in warn, warn)
    check("the sidecar records the clamp",
          "clamped to 4 frames" in res_s["sidecar"].get("matte_note", ""),
          res_s["sidecar"].get("matte_note"))
    check("no mask, no clamp -- the plain path still yields all six",
          len(P.run_pipeline({"video": src, "do_matte": False, "want_sidecar": False,
                              "want_preview": False, "cell_mode": "none"},
                             os.path.join(tmp, "run2b"))["frames"]) == 6)
    # The other side of the same coin: a longer mask is read only as far as the
    # source goes, and nothing is claimed about a clamp.
    long_mask = write_video(os.path.join(tmp, "long.avi"),
                            mask_frames() + mask_frames()[:2])
    res_l = P.run_pipeline({"video": src, "mask_video": long_mask, "want_sidecar": False,
                            "want_preview": False, "cell_mode": "none"},
                           os.path.join(tmp, "run2c"))
    check("a mask longer than the source still yields the source's length",
          len(res_l["frames"]) == 6, len(res_l["frames"]))
    check("and no clamp is claimed for it",
          not res_l["layout"]["warnings"], res_l["layout"]["warnings"])

    # A mask that exists but is entirely before the range: the source yields
    # frames from 4, the 2-frame mask yields none, so the very first `next()`
    # fails and there is no "frame it died on" to name.
    early = write_video(os.path.join(tmp, "early.avi"), mask_frames()[:2])
    err0 = _err(lambda: P.run_pipeline(
        {"video": src, "mask_video": early, "start": 4, "want_sidecar": False,
         "want_preview": False}, os.path.join(tmp, "run3b")))
    check("a mask that never overlaps the range is still an error",
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

    print("\n--- 7. the skips place a shorter mask on the clip ---")
    # The user's own arithmetic, in miniature: 124 frames of video against 121 of
    # mask, and the missing three are at one end or the other. Here 6 against 4.
    # The skip cuts the SOURCE and consumes no mask frame -- so source frame k
    # pairs with mask frame k - skip_a, and a wrong skip shows up as the alpha of
    # a different pose, which is exactly what these assertions look for.
    four = write_video(os.path.join(tmp, "four.avi"), mask_frames()[:4])

    # skip 2 at the start: the range is 6, the mask covers 4, exact fit.
    r = P.run_pipeline({"video": src, "mask_video": four, "want_sidecar": True,
                        "want_preview": False, "cell_mode": "none",
                        "mask_start_skip": 2},
                       os.path.join(tmp, "run10"))
    check("skip 2 + a 4-frame mask is an exact fit (4 frames, no clamp)",
          len(r["frames"]) == 4 and not r["layout"]["warnings"],
          (len(r["frames"]), r["layout"]["warnings"]))
    f0 = read_rgba(r["frames"][0])
    # source frame 3's block is 14..27; mask frame 1's block is 6..19
    check("output frame 1 is source frame 3 (its own subject is there)",
          tuple(int(v) for v in f0[20, 18, :3]) == SUBJECT_BGR[::-1],
          (f0[20, 18, :3].tolist(), SUBJECT_BGR[::-1]))
    check("...paired with mask frame 1, not mask frame 3",
          int(f0[20, 18, 3]) == 255 and int(f0[20, 25, 3]) == 0,
          (int(f0[20, 18, 3]), int(f0[20, 25, 3])))
    check("the frames are numbered by output position, so the first is 001",
          os.path.basename(r["frames"][0]).endswith("_001.png"),
          os.path.basename(r["frames"][0]))
    check("the sidecar records the skips",
          "skip 2/0 (start/end)" in r["sidecar"].get("matte_note", ""),
          r["sidecar"].get("matte_note"))

    # skip 2 at the end: the same four frames of mask, dropped off the back of
    # the range instead, so source frame 4 meets mask frame 4 and they agree.
    r2 = P.run_pipeline({"video": src, "mask_video": four, "want_sidecar": False,
                         "want_preview": False, "cell_mode": "none",
                         "mask_end_skip": 2},
                        os.path.join(tmp, "run11"))
    check("skip 2 at the end is also an exact fit", len(r2["frames"]) == 4,
          len(r2["frames"]))
    f3 = read_rgba(r2["frames"][3])
    check("output frame 4 is source frame 4 paired with mask frame 4",
          tuple(int(v) for v in f3[20, 30, :3]) == SUBJECT_BGR[::-1]
          and int(f3[20, 30, 3]) == 255,
          (f3[20, 30, :3].tolist(), int(f3[20, 30, 3])))

    # both ends: the mask slides one frame the other way
    r3 = P.run_pipeline({"video": src, "mask_video": four, "want_sidecar": False,
                         "want_preview": False, "cell_mode": "none",
                         "mask_start_skip": 1, "mask_end_skip": 1},
                        os.path.join(tmp, "run12"))
    check("skip 1 + 1 leaves 4 of 6", len(r3["frames"]) == 4, len(r3["frames"]))
    g0 = read_rgba(r3["frames"][0])
    check("output frame 1 is source frame 2 with mask frame 1",
          tuple(int(v) for v in g0[20, 22, :3]) == SUBJECT_BGR[::-1]
          and int(g0[20, 8, 3]) == 255 and int(g0[20, 22, 3]) == 0,
          (int(g0[20, 8, 3]), int(g0[20, 22, 3])))

    # a mask still short AFTER the skips is still clamped, and the note counts
    # against the trimmed range rather than the whole one
    r4 = P.run_pipeline({"video": src, "mask_video": four, "want_sidecar": False,
                         "want_preview": False, "cell_mode": "none",
                         "mask_start_skip": 1},
                        os.path.join(tmp, "run13"))
    check("4 frames of mask against 5 wanted clamps to 4",
          len(r4["frames"]) == 4, len(r4["frames"]))
    check("and the note counts the trimmed range, not the source's",
          "covers only 4 of the 5 frames the range asks for"
          in " ".join(r4["layout"]["warnings"]),
          r4["layout"]["warnings"])
    check("skips that leave nothing are refused",
          "leave none of the 6" in _err(lambda: P.run_pipeline(
              {"video": src, "mask_video": four, "mask_start_skip": 4,
               "mask_end_skip": 3, "want_sidecar": False, "want_preview": False},
              os.path.join(tmp, "run14"))))

    print("\n--- 8. no mask still means no mask ---")
    res5 = P.run_pipeline({"video": src, "mask_video": "", "do_matte": False,
                           "want_sidecar": True, "want_preview": False,
                           "cell_mode": "none"},
                          os.path.join(tmp, "run9"))
    check("an empty mask path runs the plain extract path",
          res5["sidecar"].get("matte") == "none", res5["sidecar"].get("matte"))
    check("and the frames are fully opaque",
          int(read_rgba(res5["frames"][0])[..., 3].min()) == 255)

    # ----------------------------------------------------------------------- #
    # 9. auto: the run finds the offset instead of being told it
    # ----------------------------------------------------------------------- #
    # The fixture's block moves 4 px per frame (`block_x`), which is what makes
    # an offset identifiable at all -- a subject that never moves leaves every
    # lag overlapping every other one, and auto must say so rather than guess.
    print("\n--- 9. auto finds the offset, and admits when it cannot ---")
    frames = source_frames()                       # N = 6 frames, block walks
    write_video(os.path.join(tmp, "autosrc.avi"), frames)

    # a mask drawn on frames 3..6 -- so it belongs 2 frames into the range
    late = write_video(os.path.join(tmp, "autolate.avi"),
                       mask_frames()[2:])
    cfg_auto = P.merge_cfg({"video": os.path.join(tmp, "autosrc.avi"),
                            "mask_video": late, "mask_auto": True})
    log = []
    a, b, short = P.find_mask_offset(cfg_auto["video"], late, cfg_auto,
                                     P.count_frames(cfg_auto["video"]),
                                     P.count_frames(late),
                                     emit=lambda s, f, m: log.append(m))
    check("a mask drawn on frames 3..6 is placed at skip 2/0", (a, b) == (2, 0), (a, b))
    check("and the sidecar line names it", short == "auto fit 2/0 (start/end)", short)
    check("the log says which source frame the mask's frame 1 pairs with",
          "pairs with source frame 3" in (log[-1] if log else ""), log)
    check("and quotes the overlap that won",
          "overlap 1.00 there" in (log[-1] if log else ""), log)

    # a mask drawn on frames 1..4 -- start-aligned, and auto must not move it
    early = write_video(os.path.join(tmp, "autoearly.avi"), mask_frames()[:4])
    log = []
    a, b, short = P.find_mask_offset(cfg_auto["video"], early, cfg_auto,
                                     P.count_frames(cfg_auto["video"]),
                                     P.count_frames(early),
                                     emit=lambda s, f, m: log.append(m))
    check("a mask drawn on frames 1..4 stays start-aligned", (a, b) == (0, 2), (a, b))
    check("and the log shows what it was kept against",
          "at the best other offset" in (log[-1] if log else ""), log)

    # a subject that does not move: the offset is not in the clips, and auto has
    # to say that rather than pick a lag out of the noise
    still = np.zeros((H, W, 3), np.uint8)
    still[:, :] = BACKDROP_BGR
    still[SUBJ_Y[0]:SUBJ_Y[1], block_x(0):block_x(0) + 14] = SUBJECT_BGR
    stillsrc = write_video(os.path.join(tmp, "stillsrc.avi"), [still] * N)
    stillmsk = write_video(os.path.join(tmp, "stillmsk.avi"), mask_frames()[:4])
    log = []
    a, b, short = P.find_mask_offset(stillsrc, stillmsk, cfg_auto,
                                     P.count_frames(stillsrc),
                                     P.count_frames(stillmsk),
                                     emit=lambda s, f, m: log.append(m))
    check("a subject that never moves falls back to start-aligned", (a, b) == (0, 2), (a, b))
    check("and the log says the clips do not determine the fit",
          "do not determine the fit" in (log[-1] if log else ""), log)

    # a mask that covers the whole range needs no fit, and auto must not invent one
    full = write_video(os.path.join(tmp, "autofull.avi"), mask_frames())
    a, b, short = P.find_mask_offset(cfg_auto["video"], full, cfg_auto,
                                     P.count_frames(cfg_auto["video"]),
                                     P.count_frames(full))
    check("a mask that covers the range is left alone", (a, b) == (0, 0), (a, b))

    # end to end: auto OVERRIDES the manual boxes rather than adding to them
    ra = P.run_pipeline({"video": os.path.join(tmp, "autosrc.avi"),
                         "mask_video": late, "mask_auto": True,
                         "mask_start_skip": 99, "mask_end_skip": 99,
                         "want_sidecar": True, "want_preview": False,
                         "cell_mode": "none"},
                        os.path.join(tmp, "run15"))
    check("auto runs with the manual skips set to nonsense", len(ra["frames"]) == 4,
          len(ra["frames"]))
    ga = read_rgba(ra["frames"][0])
    check("output frame 1 is source frame 3, which is what auto chose",
          tuple(int(v) for v in ga[20, 22, :3]) == SUBJECT_BGR[::-1]
          and int(ga[20, 22, 3]) == 255 and int(ga[20, 8, 3]) == 0,
          (ga[20, 22, :3].tolist(), int(ga[20, 22, 3]), int(ga[20, 8, 3])))
    check("the sidecar records the fit auto chose, not the boxes",
          "auto fit 2/0 (start/end)" in ra["sidecar"].get("matte_note", ""),
          ra["sidecar"].get("matte_note"))
    check("and claims no clamp",
          "clamped" not in ra["sidecar"].get("matte_note", ""),
          ra["sidecar"].get("matte_note"))

    # the manual path is untouched by any of this
    rm = P.run_pipeline({"video": os.path.join(tmp, "autosrc.avi"),
                         "mask_video": late, "mask_auto": False,
                         "mask_start_skip": 2, "want_sidecar": True,
                         "want_preview": False, "cell_mode": "none"},
                        os.path.join(tmp, "run16"))
    check("with auto off the manual skip still decides, and reaches the same frame",
          len(rm["frames"]) == 4
          and "skip 2/0 (start/end)" in rm["sidecar"].get("matte_note", ""),
          rm["sidecar"].get("matte_note"))
    check("and the sidecar does not claim auto ran",
          "auto fit" not in rm["sidecar"].get("matte_note", ""),
          rm["sidecar"].get("matte_note"))

    # Both boxes ticked is NOT a third behaviour. The mask replaces the model --
    # `run_pipeline` branches on the mask first -- so this combination has to be
    # byte-identical to the mask-only run and must never construct the model. It
    # is the state a user lands on by ticking the two boxes in the panel, so it
    # is worth pinning rather than leaving to the order of an if/elif.
    cache = {}
    rboth = P.run_pipeline({"video": os.path.join(tmp, "autosrc.avi"),
                            "mask_video": late, "do_matte": True,
                            "mask_start_skip": 2, "want_sidecar": True,
                            "want_preview": False, "cell_mode": "none"},
                           os.path.join(tmp, "run17"), model_cache=cache)
    check("ticking VRMBG as well never loads the model", cache == {}, cache)
    check("and the frames are byte-identical to the mask-only run",
          len(rboth["frames"]) == len(rm["frames"])
          and all(open(a, "rb").read() == open(b, "rb").read()
                  for a, b in zip(rboth["frames"], rm["frames"])),
          (len(rboth["frames"]), len(rm["frames"])))
    check("and the sidecar still credits the mask",
          rboth["sidecar"].get("matte") == "mask video",
          rboth["sidecar"].get("matte"))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n--- 10. the wiring: three files, no linker ---")
with open(os.path.join(HERE, "ui.html"), encoding="utf-8") as fh:
    ui = fh.read()
with open(os.path.join(HERE, "app.py"), encoding="utf-8") as fh:
    appsrc = fh.read()

_m = re.search(r"const FIELDS = \[(.*?)\];", ui, re.S)
fields = re.findall(r'"([a-z_0-9]+)"', _m.group(1)) if _m else []
check("the page's FIELDS list was found", bool(fields), _m)
for k in ("mask_video", "mask_invert", "mask_binary", "mask_threshold",
          "mask_start_skip", "mask_end_skip", "mask_auto"):
    check("%-16s is in FIELDS (collect() will read it)" % k, k in fields)
    check("%-16s is a key the pipeline knows" % k, k in P.DEFAULT_CFG)
    check("%-16s has a control on the page" % k, ('id="%s"' % k) in ui)
# The fit readout: the two skip boxes are useless without the arithmetic, and the
# arithmetic is useless if it is not re-run when any of its five inputs moves.
check("the page has a mask-fit readout", 'id="maskfit"' in ui)
check("the readout is recomputed when its inputs move",
      'for (const id of ["mask_start_skip","mask_end_skip","start","end","frame_step"])'
      in ui)
check("the readout is fed the clip's own length by the probe",
      "SRC_FRAMES = info.frames" in ui and "MASK_FRAMES = d.info.frames" in ui)
check("the readout names the leftover/short case, not just the numbers",
      "left over, which are not read" in ui and "short, so the sheet stops at" in ui)
# Auto owns the offset when it is ticked, so the two boxes have to stop taking
# input -- and the readout has to stop printing arithmetic the run will not use.
# A disabled box is the visible half of "auto overrides these"; without it the
# numbers would sit there looking authoritative while being ignored.
check("auto disables the two manual skip boxes",
      'el.disabled = !on || auto' in ui)
check("auto's own tick re-runs the gate", '$("mask_auto").onchange = syncMask' in ui)
check("the readout switches to the auto wording when it is ticked",
      'if ($("mask_auto").checked){' in ui and "The boxes above are ignored" in ui)
check("auto is gated on the mask path being on, like the path itself",
      'cfg.mask_auto = $("use_mask").checked && $("mask_auto").checked' in ui)
# The mask REPLACES the model, so the two are ALTERNATIVES and the page has to
# make that unreachable-as-a-combination rather than leaving two ticks that can
# both be on. One radio group, and the two keys the pipeline knows are derived
# from the single selection.
check("the two paths are one radio group, not two checkboxes",
      ui.count('name="matte_src"') == 3
      and 'type="radio" name="matte_src" id="do_matte"' in ui
      and 'type="radio" name="matte_src" id="use_mask"' in ui
      and '<input type="checkbox" id="use_mask"' not in ui)
check("and there is a third choice for no background removal at all",
      'type="radio" name="matte_src" id="matte_none"' in ui)
check("collect() resolves the choice to do_matte, not to a radio's value",
      'cfg.do_matte = $("do_matte").checked' in ui)
check("do_matte is out of FIELDS, so the generic loop cannot misread the radio",
      "do_matte" not in fields)
check("the starting radio comes from DEFAULT_CFG, not from the markup alone",
      '$("do_matte").checked = !!d.defaults.do_matte' in ui
      and '$("matte_none").checked = !d.defaults.do_matte' in ui)
check("the model's own settings grey when VRMBG is not the chosen path",
      'el.disabled = !vrmbg' in ui)
check("but the range settings do not, because every path reads them",
      '"model_dir","infer_size"' in ui)
# The generic half of the same guard: a name in FIELDS with no element is
# skipped by collect() in silence, so every entry is checked, not just the new
# ones. This is what caught nothing today and will catch the next control.
missing = [f for f in fields if ('id="%s"' % f) not in ui]
check("every FIELDS entry has an element (none are silently skipped)",
      not missing, missing)
check("the mask path is gated on its own radio, not on being non-empty",
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
