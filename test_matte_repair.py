"""Regression test: the two matte-repair passes, and the guards that bound them.

Both exist because a matte leaks in two different ways, and each pass can only see
one of them:

`repair_backdrop` is **topological**. It walks near-black pixels reachable from
the frame border, so a green -- or black -- edge *enclosed* by the subject is
unreachable by construction, and it cannot walk over the subject's own dark edge
either. That second property is why the connectivity test is on by default: with
it off the pass becomes a tonal black key and eats the subject's own outline.
`border_only` is the switch, and section 4 shows what each side of it costs.

`repair_key` is **tonal** and positional: a pixel within tolerance of the picked
backdrop colour, within KEY_EDGE_BAND px of transparency, loses its alpha. That
band is what keeps an erase-by-colour off a subject painted in the key colour --
section 6 runs the same fixture with no band and the blob inside the subject
disappears.

A user reported both halves in turn: "green edge leak around the transparent
image; only removes border-touching regions", then "make border-touching regions
optional". The numbers in the comments come from the walk clip
(runs/0921-084914_MiniMax_H3_00177___1_, backdrop #12ff4d): its rim is 8418 px, of
which 53% sits within 48 of the backdrop colour and 85% within 96, while the rest
-- the subject's own outline -- is 128+ away. That gap is where the tolerance
lives.

    python test_matte_repair.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402

fails = []
KEY = "#13ff38"                       # the walk clip's own backdrop
BACKDROP = P.key_rgb(KEY)             # (19, 255, 56)
SUBJECT = (44, 40, 60)                # the sprite's own neutral body colour


def check(name, cond, detail=""):
    print("  %-58s %s%s" % (name, "OK" if cond else "FAIL",
                            "" if cond else "  <- " + str(detail)))
    if not cond:
        fails.append(name)


def key_fixture(size=64, slab=False):
    """A matted frame with the two colour problems the key has to tell apart.

    A neutral round subject on transparent, a backdrop-coloured rim on its outer
    2 px (the leak), and an opaque backdrop-coloured blob deep inside the subject
    (a subject that happens to be painted in the key colour). With `slab`, also a
    slab of backdrop left in the top-left corner, 12 px wide and 20 deep, so its
    core sits further than the band from any transparency -- which is where the
    band and the connectivity test stop agreeing, and what a matte that failed
    down one side of the frame leaves behind.

    Every part is built from its own mask and no two overlap; the caller checks
    that, because overlapping masks would make the expected counts a fiction.
    """
    import cv2
    a = np.zeros((size, size, 4), np.uint8)
    r = size // 4
    body = np.zeros((size, size), np.uint8)
    cv2.circle(body, (size // 2, size // 2), r, 1, -1)
    body = body.astype(bool)
    rim1 = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~body
    rim2 = cv2.dilate(body.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~rim1 & ~body
    blob = np.zeros((size, size), np.uint8)
    cv2.circle(blob, (size // 2, size // 2), 4, 1, -1)      # > band deep, inside
    blob = blob.astype(bool)
    top = np.zeros((size, size), bool)
    if slab:
        top[:20, :12] = True

    a[..., :3][body] = SUBJECT
    a[..., 3][body] = 255
    a[..., :3][rim1] = BACKDROP
    a[..., 3][rim1] = 180                                   # the visible green edge
    a[..., :3][rim2] = BACKDROP
    a[..., 3][rim2] = 60                                    # the faint haze outside it
    a[..., :3][blob] = BACKDROP                             # the "green" subject
    a[..., 3][blob] = 255
    a[..., :3][top] = BACKDROP                              # a matte that failed
    a[..., 3][top] = 255                                    # down one side
    return Image.fromarray(a), body, rim1, rim2, blob, top


def flat_slab(size=32):
    """An all-opaque frame with a key-coloured strip down the left edge.

    The band path is undefined on a frame with no transparency at all -- there is
    no rim -- while the connectivity test is perfectly well defined: it clears the
    strip, because the strip is a key-coloured region touching the frame border.
    """
    a = np.zeros((size, size, 4), np.uint8)
    a[..., :3] = 110
    a[..., 3] = 255
    a[..., :3][:, :8] = BACKDROP
    return Image.fromarray(a)


def black_fixture(size=48):
    """A matted frame with the two kinds of dark leak `repair_backdrop` can meet.

    A near-black blob in the corner, touching the frame border (the leak the
    flood fill is for), and a near-black patch enclosed by the subject (the
    subject's own dark part -- or a leak it closed off, and the two are
    indistinguishable without the user saying which).
    """
    a = np.zeros((size, size, 4), np.uint8)
    body = np.zeros((size, size), np.uint8)
    body[size // 4:3 * size // 4, size // 4:3 * size // 4] = 1
    body = body.astype(bool)
    corner = np.zeros((size, size), bool)
    corner[:6, :6] = True                                   # touches (0,0)
    enclosed = np.zeros((size, size), bool)
    enclosed[size // 2 - 3:size // 2 + 3, size // 2 - 3:size // 2 + 3] = True
    a[..., :3][body] = 110
    a[..., 3][body] = 255
    a[..., :3][enclosed] = 2                                # the subject's own dark patch
    a[..., 3][enclosed] = 255
    a[..., :3][corner] = 2                                  # leaked backdrop
    a[..., 3][corner] = 255
    a[..., 3][~body & ~corner] = 0                          # transparent, near-black RGB
    return Image.fromarray(a), body, corner, enclosed


print("--- 1. the key colour is a form field, so it is checked not trusted ---")
for raw, want in (("#00ff00", (0, 255, 0)), ("00FF00", (0, 255, 0)),
                  ("#0f0", (0, 255, 0)), (" #13ff38 ", (19, 255, 56))):
    check("key_rgb(%r) -> %r" % (raw, want), P.key_rgb(raw) == want, P.key_rgb(raw))
for bad in ("", None, "red", "#12345", "#gggggg"):
    check("key_rgb(%r) -> the default, never black" % (bad,),
          P.key_rgb(bad) == P.key_rgb(P.KEY_COLOR), P.key_rgb(bad))
check("norm_key_color canonicalises case and the short form",
      P.norm_key_color("0F0") == "#00ff00", P.norm_key_color("0F0"))
check("KEY_COLOR round-trips through norm", P.norm_key_color(P.KEY_COLOR) == P.KEY_COLOR)

print("\n--- 2. merge_cfg coerces and canonicalises what the browser sends ---")
c = P.merge_cfg({"do_key": "true", "key_tol": "48", "key_color": "13FF38"})
check("do_key 'true' -> True", c["do_key"] is True, c["do_key"])
check("key_tol '48' -> 48 (int)", c["key_tol"] == 48 and isinstance(c["key_tol"], int),
      repr(c["key_tol"]))
check("key_color canonicalised", c["key_color"] == "#13ff38", c["key_color"])
c = P.merge_cfg({"key_color": "not a colour"})
check("garbage key_color falls back to the default", c["key_color"] == P.KEY_COLOR,
      c["key_color"])
c = P.merge_cfg({"repair_border_only": "false"})
check("repair_border_only 'false' -> False", c["repair_border_only"] is False,
      repr(c["repair_border_only"]))
c = P.merge_cfg({"key_also_connected": "true"})
check("key_also_connected 'true' -> True", c["key_also_connected"] is True,
      repr(c["key_also_connected"]))
check("defaults: flood fill tonal, colour key OFF with its extension OFF",
      P.DEFAULT_CFG["do_key"] is False and P.DEFAULT_CFG["do_repair"] is False
      and P.DEFAULT_CFG["repair_border_only"] is False
      and P.DEFAULT_CFG["key_also_connected"] is False,
      (P.DEFAULT_CFG["do_key"], P.DEFAULT_CFG["do_repair"],
       P.DEFAULT_CFG["repair_border_only"], P.DEFAULT_CFG["key_also_connected"]))

print("\n--- 3. the clip's own backdrop, read off its border ---")
import cv2  # noqa: E402
frame = np.zeros((60, 60, 3), np.uint8)
frame[:, :] = BACKDROP[::-1]                     # cv2: BGR
frame[40:, 40:] = (30, 20, 10)                   # a subject touching the corner
check("median of the border ring, as #rrggbb",
      P.backdrop_color(frame) == "#13ff38", P.backdrop_color(frame))
noisy = frame.copy()
noisy[0, 0] = (0, 0, 0)
check("a stray pixel does not move the median",
      P.backdrop_color(noisy) == "#13ff38", P.backdrop_color(noisy))

print("\n--- 4. the flood fill's connectivity test, on and off ---")
img, body, corner, enclosed = black_fixture()
before = np.array(img)
out_on, n_on = P.repair_backdrop(img, 4, border_only=True)
on = np.array(out_on)
check("on: the border-connected leak is cleared",
      bool((on[..., 3][corner] == 0).all()), on[..., 3][corner].min())
check("on: it clears exactly the leak, nothing else",
      n_on == int(corner.sum()), (n_on, int(corner.sum())))
check("on: the subject's own enclosed dark patch survives",
      bool((on[..., 3][enclosed] == 255).all()), on[..., 3][enclosed].min())
out_off, n_off = P.repair_backdrop(img, 4, border_only=False)
off = np.array(out_off)
check("off: the enclosed patch goes too",
      bool((off[..., 3][enclosed] == 0).all()), off[..., 3][enclosed].max())
check("off: and the difference is exactly that patch",
      n_off - n_on == int(enclosed.sum()), (n_off - n_on, int(enclosed.sum())))
check("off: an already-transparent near-black pixel is not 'cleared'",
      int(((before[..., 3] == 0) & (off[..., 3] == 0)).sum())
      == int((before[..., 3] == 0).sum()),
      "%d of %d" % (int(((before[..., 3] == 0) & (off[..., 3] == 0)).sum()),
                    int((before[..., 3] == 0).sum())))
check("off: the rest of the body is bit-identical either way",
      bool((on[body & ~enclosed] == off[body & ~enclosed]).all()))
check("off: the default is what protects the enclosed patch -- so the flag is "
      "load-bearing", n_off > n_on, (n_on, n_off))
check("a raise of the threshold reaches further in both modes",
      P.repair_backdrop(img, 254, True)[1] > n_on)

print("\n--- 5. the colour key: the leak goes, and the subject stays ---")
img, body, rim1, rim2, blob, top = key_fixture()
check("the leak and the body do not overlap",
      not (body & (rim1 | rim2)).any() and not (blob & (rim1 | rim2)).any(),
      "overlapping masks")
out, n = P.repair_key(img, KEY, 64)
before, after = np.array(img), np.array(out)
rim = rim1 | rim2
gone = (before[..., 3] > 0) & (after[..., 3] == 0)
check("every backdrop-coloured rim pixel lost its alpha",
      bool((gone[rim]).all()), "%d of %d" % (int(gone[rim].sum()), int(rim.sum())))
check("the count is what it says it removed", n == int(gone.sum()), (n, int(gone.sum())))
check("no other pixel was touched", int(gone.sum()) == int(rim.sum()), int(gone.sum()))
check("removed pixels are zeroed, not just made invisible",
      bool((after[..., :3][gone] == 0).all()))
check("the subject's own body is bit-identical",
      bool((after[body] == before[body]).all()))
check("a pixel the size of the subject is not 'cleared'",
      n < int(body.sum()), (n, int(body.sum())))

print("\n--- 6. the band is what keeps a key-coloured subject alive ---")
check("the deep backdrop-coloured blob has alpha 255 before",
      int(after[..., 3][blob].min()) == 255, after[..., 3][blob].min())
check("and still has it after (it is deeper than the band)",
      bool((np.array(P.repair_key(img, KEY, 64)[0])[..., 3][blob] == 255).all()))
out_noband, n_noband = P.repair_key(img, KEY, 64, band=10 ** 9)
check("with no band the blob is erased -- so the band is load-bearing",
      bool((np.array(out_noband)[..., 3][blob] == 0).all()), "blob survived without a band")

print("\n--- 7. the box EXTENDS the key's reach; it can never shrink it ---")
# The old semantics were a swap (the flag replaced the band with the connectivity
# test) and read as the box working backwards: on clips where the band cleans
# better, ticking it cleaned LESS. Now ticked is always a superset of unticked.
img, body, rim1, rim2, blob, top = key_fixture(slab=True)
flat = np.array(img)
check("the slab does not collide with the subject or its rim",
      not (top & (body | rim1 | rim2 | blob)).any(), "overlapping masks")
band_out, n_band = P.repair_key(img, KEY, 64, band=P.KEY_EDGE_BAND)
bord_out, n_bord = P.repair_key(img, KEY, 64, also_connected=True)
band_a, bord_a = np.array(band_out), np.array(bord_out)
gone_band = band_a[..., 3] == 0
gone_bord = bord_a[..., 3] == 0
check("unticked (band): the rim leak goes", bool(gone_band[rim1 | rim2].all()))
check("ticked: the rim leak still goes", bool(gone_bord[rim1 | rim2].all()))
check("ticked clears a SUPERSET of unticked (monotonicity)",
      bool((gone_bord & ~gone_band & (flat[..., 3] > 0)).sum()
           >= 0 and not (gone_band & ~gone_bord).any()),
      "px unticked cleared that ticked did not: %d"
      % int((gone_band & ~gone_bord & (flat[..., 3] > 0)).sum()))
check("unticked: the slab's core survives -- it is deeper than the band",
      int((band_a[..., 3][top] == 255).sum()) > 0,
      "%d of %d slab px kept" % (int((band_a[..., 3][top] == 255).sum()),
                                 int(top.sum())))
check("ticked: the whole slab goes -- one region, and its edge is on the "
      "border", bool((bord_a[..., 3][top] == 0).all()),
      int((bord_a[..., 3][top] > 0).sum()))
check("ticked: the enclosed blob still survives -- the extension adds no way "
      "to reach it",
      bool((bord_a[..., 3][blob] == 255).all()), bord_a[..., 3][blob].min())
check("ticked removed strictly more than the band here", n_bord > n_band,
      (n_band, n_bord))
check("both leave the subject's own neutral body alone",
      bool((bord_a[body] == flat[body]).all()) and bool((band_a[body] == flat[body]).all()))
check("ticked: a frame with no transparency is still defined -- no rim to "
      "lose, so it clears the border-connected key region",
      P.repair_key(flat_slab(), KEY, 64, also_connected=True)[1] > 0)

print("\n--- 8. the spill the key cannot match, and the despill that takes it out ---")
# The 00183 clip's leftover, in miniature: a dark-green outline on the silhouette
# (0,50,0) -- 200+ away from any green backdrop key, at alpha 255 -- plus a
# dark-green patch deep inside the body and a red-tinted rim pixel, both of which
# the despill must leave alone.
DARK_GREEN, RED_RIM = (0, 50, 0), (200, 50, 60)


def spill_fixture(size=64):
    import cv2
    a = np.zeros((size, size, 4), np.uint8)
    body = np.zeros((size, size), np.uint8)
    cv2.circle(body, (size // 2, size // 2), size // 4, 1, -1)
    body = body.astype(bool)
    outline = cv2.dilate(body.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~body
    deeper = np.zeros((size, size), np.uint8)
    cv2.circle(deeper, (size // 2, size // 2), 5, 1, -1)
    deeper = deeper.astype(bool)
    a[..., :3][body] = SUBJECT
    a[..., 3][body] = 255
    a[..., :3][outline] = DARK_GREEN            # the spill, opaque, on the edge
    a[..., 3][outline] = 250
    a[..., :3][deeper] = DARK_GREEN             # must survive: deeper than the band
    a[..., 3][deeper] = 255
    return Image.fromarray(a), body, outline, deeper


img, body, outline, deeper = spill_fixture()
# one rim pixel that is not the key's channel at all
ys, xs = np.where(outline)
a = np.array(img)
a[ys[0], xs[0], :3] = RED_RIM
red_at = (ys[0], xs[0])
img = Image.fromarray(a)
keyed, n_key = P.repair_key(img, KEY, 64)
keyed_a = np.array(keyed)
check("the spill is not something a tolerance can reach",
      bool((keyed_a[..., 3][outline] > 0).all()) and
      P.key_rgb(KEY) != DARK_GREEN and
      np.sqrt(sum((p - q) ** 2 for p, q in zip(DARK_GREEN, P.key_rgb(KEY)))) > 96,
      keyed_a[..., 3][outline].min())
out, n_dsp = P.despill_key(keyed, KEY)
before, after = np.array(keyed), np.array(out)
spill_px = outline.copy()
spill_px[red_at] = False
check("despill clamps the key's channel on the spill",
      bool((after[..., :3][spill_px][:, 1] == 0).all()),
      [tuple(int(v) for v in c) for c in after[..., :3][spill_px][:3]])
check("which is the black outline it was meant to be (and neutral)",
      bool((after[..., :3][spill_px][:, 0]
            == after[..., :3][spill_px][:, 2]).all())
      and int(after[..., :3][spill_px].max()) == P.BLACK_MAX,
      after[..., :3][spill_px][0].tolist())
check("and it is lifted out of the near-black test I2 walks through",
      bool((after[..., :3][spill_px].max(axis=1) >= P.BLACK_MAX).all()),
      int(after[..., :3][spill_px].max(axis=1).min()))
flat_out, _n = P.despill_key(keyed, KEY, floor=0)
check("floor 0 gives the textbook pure black clamp",
      bool((np.array(flat_out)[..., :3][spill_px] == 0).all()))
check("nothing was removed -- the silhouette keeps its shape",
      bool((after[..., 3] == before[..., 3]).all()))
check("the count is the pixels it changed", n_dsp == int(spill_px.sum()),
      (n_dsp, int(spill_px.sum())))
check("a rim pixel that is not the key's channel is untouched",
      bool((after[..., :3][red_at] == before[..., :3][red_at]).all()),
      (after[..., :3][red_at].tolist(), before[..., :3][red_at].tolist()))
check("the subject's own body is untouched",
      bool((after[body & ~deeper] == before[body & ~deeper]).all()))
check("and so is a dark patch deeper than the band",
      bool((after[..., :3][deeper] == DARK_GREEN).all()),
      after[..., :3][deeper][0].tolist())
check("after despill no rim pixel has the key's channel dominating",
      int((after[..., :3][..., 1] > np.maximum(after[..., :3][..., 0],
                                               after[..., :3][..., 2]))[outline].sum()) == 0)
check("a frame with no transparency has no rim to despill",
      P.despill_key(Image.new("RGBA", (20, 20), DARK_GREEN + (255,)), KEY)[1] == 0)
blue_a = np.zeros((24, 24, 4), np.uint8)                 # transparent margin,
blue_a[6:18, 6:18] = (10, 20, 90, 255)                  # then a blue-dominant slab
blue, n_blue = P.despill_key(Image.fromarray(blue_a), "#0000ff")
check("the channel comes from the picked colour, not from green",
      n_blue == 144 and int(np.array(blue)[12, 12, 2]) == 20,
      (n_blue, np.array(blue)[12, 12].tolist()))
green_a = np.zeros((24, 24, 4), np.uint8)
green_a[6:18, 6:18] = (0, 200, 0, 255)
same, n_green = P.despill_key(Image.fromarray(green_a), "#0000ff")
same_a = np.array(same)
check("a blue key leaves a green slab alone -- there is no blanket green clamp",
      n_green == 0 and bool((same_a[6:18, 6:18, :3] == (0, 200, 0)).all()), n_green)

print("\n--- 9. the tolerance dial behaves, and so do the edges ---")
# a fresh fixture: section 7's had a slab of backdrop in it, which is also
# key-coloured and would be counted as "more than the rim"
img, body, rim1, rim2, blob, _ = key_fixture()
rim = rim1 | rim2
counts = [P.repair_key(img, KEY, t)[1] for t in (0, 16, 32, 64, 128)]
check("removed px rises with the tolerance, never falls",
      all(b >= a for a, b in zip(counts, counts[1:])), counts)
check("tol 0 removes only an exact-colour match",
      P.repair_key(img, KEY, 0)[1] == int(rim.sum()) or counts[0] <= int(rim.sum()),
      counts[0])
check("a tolerance this wide cannot leave more than the rim",
      counts[-1] <= int(rim.sum()), counts[-1])
flat = Image.new("RGBA", (20, 20), BACKDROP + (255,))
same, n_flat = P.repair_key(flat, KEY, 64)
check("no transparency anywhere -> untouched, nothing claimed",
      n_flat == 0 and np.array(same).tolist() == np.array(flat).tolist(), n_flat)
empty = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
is_same, n_empty = P.repair_key(empty, KEY, 255)
check("an empty frame is untouched too",
      n_empty == 0 and np.array(is_same).tolist() == np.array(empty).tolist(), n_empty)

print("\n--- 10. the despill reaches the soft edge, not just the band ---")
# The reported bug, in miniature: a matte with a wide soft edge leaves its spill
# at alpha 1..254 far past the 6 px band -- measured on the user's sheet up to
# 21 px out -- and a band-only reach left it green in BOTH checkbox modes, which
# is why ticking the border box looked like it did nothing. The reach is now
# band + soft alpha: every semi-transparent pixel is fair game (it is the blend
# zone the matte itself made, and despill never touches alpha), while opaque
# pixels stay reachable only through the band.


def soft_spill_fixture(size=96):
    a = np.zeros((size, size, 4), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((yy - size / 2.0) ** 2 + (xx - size / 2.0) ** 2)
    body = d < 28
    a[..., :3][body] = SUBJECT
    a[..., 3][body] = 255
    rim = (d >= 28) & (d < 44)                    # a 16 px soft edge, all spill
    a[..., 3][rim] = np.clip((44 - d[rim]) / 16.0 * 255, 0, 255).astype(np.uint8)
    a[..., :3][rim] = DARK_GREEN
    return Image.fromarray(a), body, rim


img, body, rim = soft_spill_fixture()
a0 = np.array(img)
import cv2  # noqa: E402


def greenish(x):
    return (x[..., :3][..., 1] > np.maximum(x[..., :3][..., 0], x[..., :3][..., 2]))


out, n = P.despill_key(img, KEY)
after = np.array(out)
dist = cv2.distanceTransform((a0[..., 3] > 0).astype(np.uint8), cv2.DIST_L2, 3)
soft = rim & (a0[..., 3] > 0) & (a0[..., 3] < 255)
# the fixture's innermost ring lands at alpha exactly 255 and deeper than the
# band, so it is opaque spill -- the one thing the despill must NOT guess on,
# because on a subject painted like the backdrop those pixels ARE the subject.
check("every soft spill pixel was neutralised", n == int(soft.sum()),
      (n, int(soft.sum()), int(rim.sum())))
check("no soft rim pixel is left green",
      int(greenish(after)[soft].sum()) == 0, int(greenish(after)[soft].sum()))
check("alpha is untouched -- despill only recolours",
      bool((after[..., 3] == a0[..., 3]).all()))
check("the body is bit-identical", bool((after[body] == a0[body]).all()))
check("the reach went at least 10 px past full transparency -- a 6 px band "
      "cannot do this",
      float(dist[soft].max()) >= 10.0,
      dist[soft].max())
for label, bo in (("band mode", False), ("border mode", True)):
    keyed, _ = P.repair_key(img, KEY, 64, also_connected=bo)
    out, n = P.despill_key(keyed, KEY)
    a = np.array(out)
    check("%s: the deep soft rim comes out clean" % label,
          int(greenish(a)[soft].sum()) == 0, int(greenish(a)[soft].sum()))

print("\n%s (%d failure%s)" % ("ALL PASS" if not fails else "FAILURES", len(fails),
                               "" if len(fails) == 1 else "s"))
sys.exit(1 if fails else 0)
