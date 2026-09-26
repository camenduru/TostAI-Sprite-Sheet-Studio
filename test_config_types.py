"""Regression test: config values arriving as strings must not crash the pipeline.

This is the bug the browser driver caught. An <input type="range"> yields a
string, so `black_max` reached the server as "4" and
`max(RGB) < "4"` died inside numpy with a UFuncTypeError that pointed nowhere
near the real cause. Every range input was affected.

Two layers are tested, because the fix is in both:
  1. merge_cfg coerces numeric fields regardless of what the client sends.
  2. a real (tiny) end-to-end run succeeds with an all-string config.

    python test_config_types.py
"""
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402

VIDEO = r"C:\Users\PC\Desktop\monsters\sprite\walk\032137_px_00002_front_walk.mp4"
fails = []


def check(name, cond, detail=""):
    print("  %-52s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(detail)))
    if not cond:
        fails.append(name)


print("--- 1. merge_cfg coerces strings (what the browser actually sends) ---")
# exactly the shape ui.html's collect() produces: every range/number field as a
# string, checkboxes as real booleans
stringy = {
    "video": VIDEO, "infer_size": "640", "start": "1", "end": "4", "frame_step": "1",
    "black_max": "4", "columns": "0", "cell_w": "0", "cell_h": "0", "pad": "6",
    "max_texture": "32768", "fps": "24", "trim_to": "0", "leak_alpha": "200",
    "bright": "64", "erode_max": "24", "gif_scale": "0.5",
    "key_tol": "48", "key_color": "13FF38",
    "do_matte": True, "do_repair": True, "do_key": True,
    "want_sheet": True, "want_sidecar": True,
    "want_preview": False, "want_gif": False, "want_frames": False, "do_verify": True,
    "square_cell": False, "cell_mode": "subject", "play_mode": "pingpong",
}
cfg = P.merge_cfg(stringy)
for k in P._INT_FIELDS:
    check("int  %-14s -> %r" % (k, cfg[k]), isinstance(cfg[k], int) and not isinstance(cfg[k], bool),
          "got %s %r" % (type(cfg[k]).__name__, cfg[k]))
for k in P._FLOAT_FIELDS:
    check("float %-13s -> %r" % (k, cfg[k]), isinstance(cfg[k], float))
for k in P._BOOL_FIELDS:
    check("bool %-14s -> %r" % (k, cfg[k]), isinstance(cfg[k], bool))

print("\n--- 2. garbage and edge inputs must not crash ---")
for bad in ({"black_max": "abc"}, {"pad": None}, {"infer_size": ""}, {"erode_max": []}):
    try:
        c = P.merge_cfg(dict(stringy, **bad))
        k = list(bad)[0]
        check("recovered from %-24s -> %r" % (json.dumps(bad), c[k]),
              c[k] == P.DEFAULT_CFG[k])
    except Exception as ex:                                   # noqa: BLE001
        check("recovered from %s" % json.dumps(bad), False, ex)

print("\n--- 3. the inference size ladder ---")
# suggest_infer_size is what /api/probe hands the page, so the picker is seeded
# from the clip's own resolution instead of always sitting on the trained 1024.
# The rule: the largest ladder step that does not exceed the source's LONGER
# edge, floored at 640. Every boundary is pinned here, both sides of each step.
for (w, h), want in (
    ((1920, 1080), 1024),   # bigger than the trained size -- stay at 1024
    ((1024, 1024), 1024),   # exactly the trained size
    ((1023, 768), 768),     # just under 1024 -> the next step down
    ((768, 768), 768),      # exactly 768
    ((767, 640), 640),      # just under 768 -> 640
    ((640, 480), 640),      # exactly 640
    ((512, 512), 640),      # below the ladder -> the 640 floor, never lower
    ((100, 100), 640),
    ((0, 0), 1024),         # a container that will not report its own size
    ((-5, -5), 1024),       # nonsense must not pick the fastest size
    (("x", "y"), 1024),
    ((None, None), 1024),
):
    got = P.suggest_infer_size(w, h)
    check("suggest(%-9s, %-9s) -> %s" % (repr(w), repr(h), want), got == want,
          "got %r" % got)

# The longer edge decides, because the model takes a square input -- a 1000x700
# clip is asked for 768, and the same clip transposed gives the same answer.
check("the longer edge decides, either orientation",
      P.suggest_infer_size(1000, 700) == P.suggest_infer_size(700, 1000) == 768)

# The ladder and the <select> must agree. They are two files with no link
# between them, and a suggestion the picker cannot hold fails silently: the
# value is assigned, no option matches, and the control keeps its old value
# while the panel cheerfully reports the new one.
_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
with open(_html, encoding="utf-8") as fh:
    _ui = fh.read()
_m = re.search(r'<select id="infer_size">(.*?)</select>', _ui, re.S)
_offered = [int(v) for v in re.findall(r'value="(\d+)"', _m.group(1))] if _m else []
check("the page has an infer_size select with options", bool(_offered), _offered)
check("the page offers exactly the ladder",
      sorted(_offered, reverse=True) == list(P.INFER_LADDER),
      "select=%s ladder=%s" % (_offered, list(P.INFER_LADDER)))

# Three files with no link between them: pipeline computes the suggestion,
# app.py names it in the response, ui.html destructures it. Rename it on either
# side and the page keeps the trained 1024 while the panel reports the new
# value -- a silent wrong answer, which is the failure mode worth a guard.
# No suite drives the generator's HTTP routes, so this is the only thing
# standing between a rename and that bug.
_appsrc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
with open(_appsrc_path, encoding="utf-8") as fh:
    _appsrc = fh.read()
check("the probe route computes the suggestion",
      '"infer_size": P.suggest_infer_size(' in _appsrc)
check("the page reads the key the route emits", "d.infer_size" in _ui)

print("\n--- 4. end-to-end run with an all-string config ---")
tmp = tempfile.mkdtemp(prefix="ss_cfgtest_")
try:
    res = P.run_pipeline(dict(stringy, want_preview=False, want_sidecar=True),
                         tmp)
    check("pipeline completed", True)
    check("frames produced", len(res["frames"]) == 4, len(res["frames"]))
    L = res["layout"]
    check("grid is full", L["columns"] * L["rows"] == L["frame_count"],
          "%dx%d vs %d" % (L["columns"], L["rows"], L["frame_count"]))
    check("sheet written", os.path.isfile(res["artifacts"]["sheet"]))
    v = res.get("verify") or {}
    check("verify ran and passed", v.get("ok") is True, v.get("lines"))
    for line in (v.get("lines") or []):
        print("        " + line)
    # the colour key is on in this config, so all three layers must have seen it:
    # the sidecar names it, the verifier reports its rim count, and the normalised
    # colour is what both of them say.
    side = json.load(open(res["artifacts"]["sidecar"]))
    check("sidecar records the colour key",
          "colour key #13ff38 (tol=48" in side.get("matte_repair", ""), side.get("matte_repair"))
    check("sidecar records the flood fill too (both ran)",
          "near-black anywhere" in side.get("matte_repair", ""), side.get("matte_repair"))
    check("the key's sidecar wording shows the extension default (off)",
          "band=6)" in side.get("matte_repair", "")
          and "+ border-connected" not in side.get("matte_repair", ""),
          side.get("matte_repair"))
    check("verifier reports the key spill on the rim",
          any(l.startswith("I5 key spill") for l in v.get("lines") or []), v.get("lines"))
except Exception as ex:                                       # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("end-to-end run", False, "%s: %s" % (type(ex).__name__, ex))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%s (%d failure%s)" % ("ALL PASS" if not fails else "FAILURES", len(fails),
                               "" if len(fails) == 1 else "s"))
sys.exit(1 if fails else 0)
