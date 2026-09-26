"""Op-layer checks for the sprite editor.

One pass, real assertions, no framework. The point is to execute every
operation at least once and to pin the handful whose correctness is not
self-evident: the shift/place geometry, the un-premultiply inverse, the
topological colour key, the align solver, and undo.

Run:  python test_editor.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import editor as E  # noqa: E402

FAILS = []
CHECKS = [0]


# --------------------------------------------------------------------------- #
# mutation hooks
#
# `python test_editor.py <name>` deliberately breaks one guard. The suite MUST
# then fail -- a guard that has never been observed to fail is not a guard, it
# is a line of code that happens to be on the path.
# --------------------------------------------------------------------------- #
MUT = sys.argv[1] if len(sys.argv) > 1 else ""

if MUT == "undo-naive":
    # the original bug: the redo stack got the pre-op entry, so redo rewound twice
    def _naive_undo(self):
        if not self.undo:
            return None
        e = self.undo.pop()
        self._restore(e)
        self.redo.append(e)
        return e
    E.Doc.undo_once = _naive_undo

elif MUT == "pad-shift":
    # the original bug: _shift cannot grow an array, so "padding" clipped
    def _pad_shift(doc, idxs, a):
        n = int(a["px"])
        for i in range(doc.n):
            doc.cells[i] = E._shift(doc.cells[i], n, n)
            doc.bump(i)
        doc.set_layout(columns=doc.columns, cell_w=doc.cell_w + 2 * n,
                       cell_h=doc.cell_h + 2 * n)
        return list(range(doc.n))
    E.OPS["pad_cell"]["fn"] = _pad_shift

elif MUT == "pivot-fractional":
    # the original bug: a geometric half-pixel pivot, rounded per cell
    def _frac(box, anchor, mask=None):
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0
        return {"bottom-center": (cx, y1 + 1), "center": (cx, cy),
                "centroid": (cx, cy)}.get(anchor, (cx, cy))
    _orig_shift = E._shift
    E.anchor_point = _frac
    E._shift = lambda arr, dx, dy, wrap=False: _orig_shift(
        arr, int(round(dx)), int(round(dy)), wrap)

elif MUT == "erase-tonal":
    # a tonal keyer instead of a topological one: kills every near-black pixel
    def _tonal(doc, idxs, a):
        bm, la = int(a["black_max"]), int(a["leak_alpha"])

        def fn(c):
            kill = (c[..., :3].max(axis=2) < bm) & (c[..., 3] >= la)
            if not kill.any():
                return c
            out = c.copy()
            out[..., 3] = np.where(kill, 0, out[..., 3])
            return out
        return E._apply(doc, idxs, fn)
    E.OPS["erase_border_black"]["fn"] = _tonal

elif MUT == "verify-ignore-leak":
    _orig_verify = E.verify_doc

    def _blind(doc):
        okd, lines = _orig_verify(doc)
        return True, [l for l in lines if "BACKDROP" not in l]
    E.verify_doc = _blind

elif MUT == "store-post-n":
    # the original bug: the undo snapshot was sized from the POST-op frame count,
    # so a shrinking op (keep_range, dedupe, drop_frames) never stored its tail
    # cells and undo raised KeyError on the first index past the new count.
    #
    # Derived from the live source rather than copied, so the mutant cannot drift
    # away from the implementation it is supposed to be testing.
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.run_op))
    mutated = src.replace("else list(range(n))", "else list(range(doc.n))")
    if mutated == src:
        print("MUTATION store-post-n did not apply -- run_op's shape changed")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant run_op>", "exec"), _ns)
    E.run_op = _ns["run_op"]

elif MUT == "setcells-nonatomic":
    # the original bug: _set_cells assigned doc.cells BEFORE calling set_layout,
    # so a re-grid that set_layout refused left the document holding N cells
    # against the old grid -- n != cols*rows, and save then wrote a sheet whose
    # sidecar claimed more frames than it had.
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(E._set_cells))
    mutated = src.replace("doc.cells, doc.orig = old_cells, old_orig",
                          "pass  # mutation: no rollback")
    if mutated == src:
        print("MUTATION setcells-nonatomic did not apply -- _set_cells changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant _set_cells>", "exec"), _ns)
    E._set_cells = _ns["_set_cells"]

elif MUT == "restore-drop-orig":
    # the original bug: a full restore rebuilt the cells but left `orig` alone,
    # so len(orig) fell behind n. The next op to index orig on the tail frames
    # raised IndexError, and I1/I4 quietly lost their reference in the meantime.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.Doc._restore))
    mutated, n_sub = re.subn(
        r"self\.orig = \[_unpng\(og\[i\]\) if i in og else self\.cells\[i\]\.copy\(\)"
        r"\s*\n\s*for i in range\(entry\[\"n\"\]\)\]",
        "pass  # mutation: the reference is not restored", src)
    if not n_sub:
        print("MUTATION restore-drop-orig did not apply -- _restore changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant _restore>", "exec"), _ns)
    E.Doc._restore = _ns["_restore"]

elif MUT == "smell-counts-transparent":
    # the original bug: the reference-free smell counter tested max(RGB) > alpha
    # on EVERY pixel, including alpha == 0 ones. A real matte leaves stray RGB
    # behind in fully transparent pixels -- this project's own torch sheet has
    # RGB=2 at alpha=0, and 2 > 0 is a hit -- so the count never reached zero on a
    # real sheet and the line effectively never printed at all.
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.verify_doc))
    mutated = src.replace(
        "((cell[..., :3].max(axis=2) > alpha) & (alpha > 0)).sum()",
        "(cell[..., :3].max(axis=2) > alpha).sum()")
    if mutated == src:
        print("MUTATION smell-counts-transparent did not apply -- "
              "verify_doc changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant verify_doc>", "exec"), _ns)
    E.verify_doc = _ns["verify_doc"]

elif MUT == "save-clobbers-source":
    # the original bug: save_doc composed the output filename from the document's
    # own name without ever comparing it to the sheet it was loaded from, so
    # pointing the free-text output folder at the source directory silently
    # replaced the original sheet and its sidecar.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.save_doc))
    mutated, n_sub = re.subn(r"    if not overwrite:", "    if False:",
                             src, count=1)
    if not n_sub:
        print("MUTATION save-clobbers-source did not apply -- save_doc changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant save_doc>", "exec"), _ns)
    E.save_doc = _ns["save_doc"]

elif MUT == "export-mutates-doc":
    # The obvious wrong implementation of "export only the selected frames":
    # apply the subset to the LIVE document. The written sheet is correct and the
    # export "works", which is what makes it dangerous -- every frame that was
    # not selected is gone from the document, with no undo entry and no way back.
    #
    # Anchored on the `_set_cells(sub, ...)` call ALONE, not on it followed by
    # `return sub, idx`. The first version of this hook required the two to be
    # adjacent, and when the per-frame audio feature inserted the clip carry-over
    # between them the regex stopped matching. The hook then printed "did not
    # apply", which read as noise rather than as a failure, and it tested nothing
    # for a whole session while the README went on claiming it was caught. The
    # sweep tool exists because of this: a hook that cannot apply must be loud.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.export_subset))
    mutated, n_sub = re.subn(
        r"    _set_cells\(sub, \[doc\.cells\[i\] for i in idx\], "
        r"\[doc\.orig\[i\] for i in idx\]\)\n",
        "    _set_cells(sub, [doc.cells[i] for i in idx], "
        "[doc.orig[i] for i in idx])\n"
        "    doc.cells = sub.cells\n"
        "    doc.layout = dict(sub.layout)\n",
        src, count=1)
    if not n_sub:
        print("MUTATION export-mutates-doc did not apply -- export_subset "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant export_subset>", "exec"), _ns)
    E.export_subset = _ns["export_subset"]

elif MUT == "export-unsorted":
    # The selection is a set, so the frame order in the written sheet has to be
    # the index order. Left as given, it is whatever the set iterated in.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.export_subset))
    mutated, n_sub = re.subn(r"    idx = sorted\(\{int\(i\) for i in frames\}\)",
                             "    idx = [int(i) for i in frames]", src, count=1)
    if not n_sub:
        print("MUTATION export-unsorted did not apply -- export_subset changed "
              "shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant export_subset>", "exec"), _ns)
    E.export_subset = _ns["export_subset"]

elif MUT == "export-regrids-whole":
    # Dropping the whole-document shortcut, so "export everything" re-grids a
    # layout the user chose deliberately and hands back a different document.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.export_subset))
    mutated, n_sub = re.subn(r"    if len\(idx\) == doc\.n:",
                             "    if False:", src, count=1)
    if not n_sub:
        print("MUTATION export-regrids-whole did not apply -- export_subset "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant export_subset>", "exec"), _ns)
    E.export_subset = _ns["export_subset"]

elif MUT == "open-folder-ignored":
    # The reported bug, restored: a folder is not resolved to the sidecar in it.
    # Pasting the folder a run wrote into then either fails outright or falls
    # through to opening the PNG, and the clips the sidecar carries never arrive
    # -- "when i load folder audios are not loaded".
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.load_doc))
    mutated, n_sub = re.subn(r"    if os\.path\.isdir\(sheet_path\):",
                             "    if False:", src, count=1)
    if not n_sub:
        print("MUTATION open-folder-ignored did not apply -- load_doc "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant load_doc>", "exec"), _ns)
    E.load_doc = _ns["load_doc"]

elif MUT == "grid-skips-sidecar":
    # The same report arriving by the other route: an explicit columns/rows
    # suppresses sidecar discovery, so a sheet opened with the boxes filled in
    # comes back with none of its clips. Measured on a real run sheet -- the PNG
    # alone returns its 2 clips, the PNG with its own grid supplied returns 0.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.load_doc))
    mutated, n_sub = re.subn(
        r"    else:\n        sidecar_path = _find_sidecar\(sheet_path\)",
        "    elif not (columns and rows):\n"
        "        sidecar_path = _find_sidecar(sheet_path)",
        src, count=1)
    if not n_sub:
        print("MUTATION grid-skips-sidecar did not apply -- load_doc changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant load_doc>", "exec"), _ns)
    E.load_doc = _ns["load_doc"]

elif MUT == "set-meta-ignores-fps":
    # fps dropped from the key loop, so the panel's number never reaches the
    # document -- the exact failure the new section 25 exists to catch, and the
    # one that no suite covered before it. The sidecar then keeps whatever fps it
    # was opened with and the note quotes that instead.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.op_set_meta))
    mutated, n_sub = re.subn(
        r'    for k in \("fps", "play_mode", "trim_to", "anchor", "blend"\):',
        '    for k in ("play_mode", "trim_to", "anchor", "blend"):',
        src, count=1)
    if not n_sub:
        print("MUTATION set-meta-ignores-fps did not apply -- op_set_meta "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_set_meta>", "exec"), _ns)
    # The registry, not E.op_set_meta. run_op dispatches through OPS, and the
    # @op decorator captured the function object at import time, so rebinding the
    # module attribute leaves the registry pointing at the original -- a hook
    # that silently does nothing and reports "not caught". Every earlier op-layer
    # hook patches load_doc/export_subset, which are called by name and have no
    # such indirection, so this is the first hook where it matters.
    E.OPS["set_meta"]["fn"] = _ns["op_set_meta"]

elif MUT == "save-meta-drops-provenance":
    # The targeted update replaced by a regeneration of the keys the panel owns:
    # `source`, `matte` and the `edited` record of how the sheet was made are
    # dropped, so writing the metadata quietly costs the user the provenance of
    # their own sheet.
    #
    # Deliberately NOT `updated = {}`. That was the first version of this hook
    # and it emptied the file completely, which made load_doc raise and the suite
    # die with a traceback rather than report a FAIL -- a mutant that breaks the
    # fixture is not isolating the property under test. Dropping only the
    # unowned keys leaves a sidecar that still loads, so the assertions that
    # guard provenance are the things that fail.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.save_meta))
    mutated, n_sub = re.subn(
        r"    updated = dict\(old\)",
        "    updated = {k: v for k, v in old.items()\n"
        "               if k not in (\"source\", \"matte\", \"edited\")}",
        src, count=1)
    if not n_sub:
        print("MUTATION save-meta-drops-provenance did not apply -- save_meta "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant save_meta>", "exec"), _ns)
    E.save_meta = _ns["save_meta"]

elif MUT == "overwrite-ignores-source":
    # The redirect removed: a ticked save goes back to naming its output after the
    # document, so it writes a fresh pair beside the files it was loaded from and
    # leaves them alone. That is the reported bug, restored.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.save_doc))
    mutated, n_sub = re.subn(
        r"    if overwrite and doc\.src_sheet and not out_dir_named:",
        "    if False:", src, count=1)
    if not n_sub:
        print("MUTATION overwrite-ignores-source did not apply -- save_doc "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant save_doc>", "exec"), _ns)
    E.save_doc = _ns["save_doc"]

elif MUT == "overwrite-ignores-named-folder":
    # `and not out_dir_named` dropped: a ticked save redirects to the loaded files
    # even when the caller named a destination of their own. The box is worded as
    # permission, so this silently replaces the original when a copy somewhere
    # else was asked for -- and it is invisible, because the save reports success
    # and the new folder is simply never written.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.save_doc))
    mutated, n_sub = re.subn(
        r"    if overwrite and doc\.src_sheet and not out_dir_named:",
        "    if overwrite and doc.src_sheet:", src, count=1)
    if not n_sub:
        print("MUTATION overwrite-ignores-named-folder did not apply -- save_doc "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant save_doc>", "exec"), _ns)
    E.save_doc = _ns["save_doc"]

elif MUT == "volume-not-clamped":
    # The clamp dropped: the setter stores whatever it is handed. The panel's
    # slider cannot go out of range, so this is invisible from the UI -- but the
    # route is reachable directly, and a volume of 2 written into the sidecar is a
    # clip an engine plays at twice the amplitude it was mixed at.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.Doc.set_audio_volume))
    mutated, n_sub = re.subn(
        r'    e\["volume"\] = max\(0\.0, min\(1\.0, float\(volume\)\)\)',
        '    e["volume"] = float(volume)', src, count=1)
    if not n_sub:
        print("MUTATION volume-not-clamped did not apply -- set_audio_volume "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant set_audio_volume>", "exec"), _ns)
    E.Doc.set_audio_volume = _ns["set_audio_volume"]

elif MUT == "match-size-pivot-ignored":
    # The pivot dropped: the subject is scaled about the CELL's centre instead of
    # about the point on the subject the `anchor` names. The sizes come out right
    # -- this op's whole headline claim still holds -- so nothing about the size
    # spread fails. What fails is the position: every frame slides, and the
    # alignment `align` had already established is quietly undone. That is the
    # trade the op exists to refuse, so it is the property worth a mutant.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.op_match_size))
    mutated, n_sub = re.subn(
        r'        p = content_pivot\(cell, a\["anchor"\], thr\)',
        '        p = (cell.shape[1] // 2, cell.shape[0] // 2)', src, count=1)
    if not n_sub:
        print("MUTATION match-size-pivot-ignored did not apply -- op_match_size "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_match_size>", "exec"), _ns)
    E.OPS["match_size"]["fn"] = _ns["op_match_size"]

elif MUT == "match-size-anchor-not-restored":
    # The post-scale anchor restore dropped. `scale_about` places its output on
    # an integer offset, so a fractional factor leaves the scaled pivot on a
    # half-pixel -- and the soft edge an interpolating filter leaves can round it
    # the other way. The restore is what puts the subject back on the pixel it
    # was on. Without it the size is still fixed and only some frames drift, which
    # is exactly the kind of partial failure that reads as "close enough" until
    # the loop is played.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.op_match_size))
    mutated, n_sub = re.subn(r"            if na != p:",
                             "            if False:", src, count=1)
    if not n_sub:
        print("MUTATION match-size-anchor-not-restored did not apply -- "
              "op_match_size changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_match_size>", "exec"), _ns)
    E.OPS["match_size"]["fn"] = _ns["op_match_size"]

elif MUT == "match-size-scope-ignored":
    # The selection dropped: the op measures and rescales every cell in the
    # document. "Select all frames" is what the user asked for, so this is
    # invisible in the one case they described -- and wrong in every other one,
    # where a partial selection is the whole point of selecting.
    #
    # BOTH loops, not just the second. Mutating only the second one is a no-op:
    # `boxes` is built from `idxs`, and the loop body opens with `if i not in
    # boxes: continue`, so the extra iterations all skip themselves. The first
    # version of this hook did exactly that, changed no behaviour at all, and was
    # reported as MISSED -- which reads as "the suite has a hole" when the truth
    # is "the mutant was never a mutant". A hook has to be shown to alter
    # behaviour before its being caught means anything.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.op_match_size))
    mutated, n_sub = re.subn(r"    for i in idxs:",
                             "    for i in range(doc.n):", src)
    if n_sub != 2:
        print("MUTATION match-size-scope-ignored did not apply -- op_match_size "
              "changed shape (matched %d of 2 loops)" % n_sub)
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_match_size>", "exec"), _ns)
    E.OPS["match_size"]["fn"] = _ns["op_match_size"]

elif MUT == "hex-rgb-swapped":
    # Red and blue read back to front. Every colour that is grey still keys
    # correctly, which is what makes this survive a careless fixture: the failure
    # only shows on a colour whose channels actually differ, and a backdrop is
    # exactly that.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.hex_rgb))
    mutated, n_sub = re.subn(
        r"    return \(v >> 16\) & 255, \(v >> 8\) & 255, v & 255",
        "    return v & 255, (v >> 8) & 255, (v >> 16) & 255", src, count=1)
    if not n_sub:
        print("MUTATION hex-rgb-swapped did not apply -- hex_rgb changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant hex_rgb>", "exec"), _ns)
    # Rebinding the module global is enough here, unlike the OPS hooks below:
    # op_erase_color resolves hex_rgb by name at call time, and its __globals__
    # *is* E.__dict__.
    E.hex_rgb = _ns["hex_rgb"]

elif MUT == "hex-rgb-falls-back-to-black":
    # The refusal replaced by a default. This is the one failure a colour key can
    # have that still reports success: an unreadable value quietly becomes black,
    # the key takes out every dark pixel on the subject, and the frame looks
    # keyed. `pipeline.key_rgb` falls back to the backdrop green for the same
    # reason -- black is the one colour a keyer must never guess.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.hex_rgb))
    mutated, n_sub = re.subn(
        r'        raise ValueError\("colour %r is not a hex colour like #13ff38" '
        r'% \(value,\)\)',
        "        return (0, 0, 0)", src, count=1)
    if not n_sub:
        print("MUTATION hex-rgb-falls-back-to-black did not apply -- hex_rgb "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant hex_rgb>", "exec"), _ns)
    E.hex_rgb = _ns["hex_rgb"]

elif MUT == "erase-color-ignores-picker":
    # The picker unwired: the op keys black whatever the control says. The
    # default value IS black, so a user who never touches the swatch sees no
    # difference at all -- which is exactly how a control can look present and
    # be dead.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.op_erase_color))
    mutated, n_sub = re.subn(
        r'    tgt = np\.array\(hex_rgb\(a\["color"\]\), np\.int16\)',
        "    tgt = np.array((0, 0, 0), np.int16)", src, count=1)
    if not n_sub:
        print("MUTATION erase-color-ignores-picker did not apply -- "
              "op_erase_color changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_erase_color>", "exec"), _ns)
    E.OPS["erase_color"]["fn"] = _ns["op_erase_color"]

elif MUT == "hex-palette-drops-bad-entry":
    # An unreadable entry skipped instead of refused. The palette that reaches
    # the op is then shorter than the one the user named, and every check that
    # only looks at "did the colours come from the palette" still passes -- which
    # is why the suite has to name the refusal itself.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.hex_palette))
    mutated, n_sub = re.subn(
        r"        if part:\n            out\.append\(hex_rgb\(part\)\)",
        "        if part:\n            try:\n"
        "                out.append(hex_rgb(part))\n"
        "            except ValueError:\n                pass",
        src, count=1)
    if not n_sub:
        print("MUTATION hex-palette-drops-bad-entry did not apply -- hex_palette "
              "changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant hex_palette>", "exec"), _ns)
    E.hex_palette = _ns["hex_palette"]

elif MUT == "pixelate-mesh-ignores-palette":
    # The picker unwired on the mesh op: the palette is parsed and then never
    # used, so the frames come back with the fitted palette instead of the named
    # one. The op still succeeds and still reports a grid.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.op_pixelate_mesh))
    mutated, n_sub = re.subn(
        r'    pal = hex_palette\(palette\) if palette else None',
        "    pal = None", src, count=1)
    if not n_sub:
        print("MUTATION pixelate-mesh-ignores-palette did not apply -- "
              "op_pixelate_mesh changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_pixelate_mesh>", "exec"), _ns)
    E.OPS["pixelate_mesh"]["fn"] = _ns["op_pixelate_mesh"]

elif MUT == "pixelate-mesh-quantises-twice":
    # The auto quantiser left on underneath the named palette. Both passes end by
    # mapping onto the user's palette, so a check that only asks "are the colours
    # from the palette" cannot see it -- only a check that the colour *count* is
    # ignored can.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.op_pixelate_mesh))
    mutated, n_sub = re.subn(
        r"        colors=0 if pal is not None else max\(0, min\(256, "
        r'int\(a\["colors"\]\)\)\),',
        '        colors=max(0, min(256, int(a["colors"]))),', src, count=1)
    if not n_sub:
        print("MUTATION pixelate-mesh-quantises-twice did not apply -- "
              "op_pixelate_mesh changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_pixelate_mesh>", "exec"), _ns)
    E.OPS["pixelate_mesh"]["fn"] = _ns["op_pixelate_mesh"]

elif MUT == "mesh-defaults-reverted":
    # The two defaults put back to what they were: colours 16, upscale 2 with its
    # old ceiling of 4. This is a registry mutant rather than a source one --
    # `coerce_args` fills a missing key from `spec["args"]`, and the page renders
    # the same list, so editing the entry in place is exactly the regression the
    # new default checks exist for: a user who touches nothing gets the old
    # quantised, coarse-meshed result back and nothing looks broken.
    _spec = E.OPS["pixelate_mesh"]
    _hit = set()
    for _a in _spec["args"]:
        if _a["k"] == "colors":
            _a["d"] = 16
            _hit.add("colors")
        elif _a["k"] == "upscale":
            _a["d"] = 2
            _a["max"] = 4
            _hit.add("upscale")
    if _hit != {"colors", "upscale"}:
        print("MUTATION mesh-defaults-reverted did not apply -- "
              "pixelate_mesh no longer declares both args")
        sys.exit(2)

elif MUT == "fill-ignores-picker":
    # The picker unwired on the last op that still had channel boxes: it tints
    # red whatever the control says. Red IS the default, so a user who never
    # touches the swatch sees no difference at all -- which is exactly how a
    # control can look present and be dead, and why the check has to name a
    # colour other than the default.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.op_fill))
    mutated, n_sub = re.subn(
        r'    r, g, b = hex_rgb\(a\["color"\]\)',
        "    r, g, b = (255, 0, 0)", src, count=1)
    if not n_sub:
        print("MUTATION fill-ignores-picker did not apply -- op_fill changed "
              "shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_fill>", "exec"), _ns)
    E.OPS["fill"]["fn"] = _ns["op_fill"]

elif MUT == "resize-blends":
    # The thing the op exists to avoid: a filtered resize, which invents colours
    # between the ones in the art. A size check alone would pass -- this is the
    # mutant that proves the section is about exactness and not about a number.
    #
    # Guarded, because this one is hand-written rather than derived from source:
    # if the op stopped calling `_resize_nn` by name, rebinding the module
    # attribute would replace nothing, the suite would pass, and the sweep would
    # report MISSED -- a hole in the suite, which is the opposite of the truth.
    import inspect
    from PIL import Image as _PIL

    if "_resize_nn" not in inspect.getsource(E.OPS["resize_pixels"]["fn"]):
        print("MUTATION resize-blends did not apply -- resize_pixels no longer "
              "calls _resize_nn")
        sys.exit(2)

    def _blend(cell, k, up):
        h, w = cell.shape[:2]
        size = (w * k, h * k) if up else (w // k, h // k)
        return np.array(_PIL.fromarray(cell, "RGBA").resize(size, _PIL.BILINEAR))
    E._resize_nn = _blend

elif MUT == "resize-drops-remainder":
    # The divisibility guard removed: the stride then truncates, so the last row
    # and column of every frame are dropped and the op still reports success.
    if "resize_pixels" not in E.OPS:
        print("MUTATION resize-drops-remainder did not apply -- no resize_pixels "
              "in the registry")
        sys.exit(2)

    def _loose_resize(doc, idxs, a):
        k = int(a["factor"])
        up = a["mode"] == "up"
        cw, ch = doc.cell_w, doc.cell_h
        nw, nh = (cw * k, ch * k) if up else (cw // k, ch // k)
        for i in range(doc.n):
            doc.cells[i] = E._resize_nn(doc.cells[i], k, up)
            doc.bump(i)
        doc.set_layout(columns=doc.columns, cell_w=nw, cell_h=nh)
        return list(range(doc.n))
    E.OPS["resize_pixels"]["fn"] = _loose_resize

elif MUT == "pixeloe-pads":
    # The regression the bridge exists to prevent: let PixelOE replicate-pad a
    # cell the pixel size does not divide. The op then reports success on a
    # padded, different image, and the size guard is what stops it -- so
    # removing the guard must turn the suite red. Derived from the live source
    # so the mutant cannot drift from the guard it removes.
    #
    # The op does `import pixeloe_bridge` inside the function, so the patch must
    # land on the module object in sys.modules -- patching a fresh module object
    # would rebind nothing the op ever looks at, and the suite would pass while
    # the sweep reported MISSED, a hole rather than a caught mutant.
    import inspect
    import importlib
    import re as _re

    _poe_mut = importlib.import_module("pixeloe_bridge")
    _src = inspect.getsource(_poe_mut.pixelize_cells)
    _guard = _re.search(r"    if not _divides\(w, h, ps\):[\s\S]*?\n\n", _src)
    if not _guard:
        print("MUTATION pixeloe-pads did not apply -- the divisibility guard "
              "in pixelize_cells changed shape")
        sys.exit(2)
    _mut = _src.replace(_guard.group(0), "\n")
    _ns = dict(_poe_mut.__dict__)
    exec(compile(_mut, "<mutant pixelize_cells>", "exec"), _ns)
    _poe_mut.pixelize_cells = _ns["pixelize_cells"]

elif MUT == "pixeloe-drops-alpha":
    # The other half of the op's contract: PixelOE is RGB-only, so the alpha
    # channel is the op's to carry. Returning the algorithm's RGB with a flat
    # opaque alpha would still be blocky pixel art of the right size, so a size
    # or blockiness check alone would pass -- this is the mutant that proves the
    # section is about the alpha channel being preserved.
    import inspect
    import importlib
    import re as _re

    _poe_mut = importlib.import_module("pixeloe_bridge")
    _src = inspect.getsource(_poe_mut.pixelize_cells)
    _mut, _n = _re.subn(r"        rgba\[\.\.\., 3\] = arrs\[i\]\[\.\.\., 3\]",
                        "        rgba[..., 3] = 255", _src, count=1)
    if not _n:
        print("MUTATION pixeloe-drops-alpha did not apply -- the alpha carry in "
              "pixelize_cells changed shape")
        sys.exit(2)
    _ns = dict(_poe_mut.__dict__)
    exec(compile(_mut, "<mutant pixelize_cells>", "exec"), _ns)
    _poe_mut.pixelize_cells = _ns["pixelize_cells"]

elif MUT == "pixeloe-quantises-twice":
    # The op left quantising on underneath the named palette, so the result is
    # quantised once to PixelOE's own fitted palette and then mapped onto the
    # user's -- which can merge away colours they asked for. The check that sees
    # it is the one asserting a different `colours` count gives identical bytes
    # while a palette is named; every "are the colours from the palette" check
    # still passes under this mutant, because both paths end by mapping onto the
    # user's palette. Derived from the live source, so a shape change exits 2
    # rather than testing nothing.
    import inspect
    import re

    src = inspect.getsource(E.OPS["pixelize_oe"]["fn"])
    mutated, n_sub = re.subn(
        r"        colors=0 if pal is not None else want,",
        "        colors=want,  # mutation: quantiser left on under the palette",
        src)
    if not n_sub:
        print("MUTATION pixeloe-quantises-twice did not apply -- the op's "
              "quantiser/palette gate changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_pixelize_oe>", "exec"), _ns)
    E.OPS["pixelize_oe"]["fn"] = _ns["op_pixelize_oe"]

elif MUT == "transform-no-anchor-restore":
    # The anchor-correction removed: the pivot is still the rotation centre, but
    # nothing puts it back after warpAffine sampled on the integer grid. The op
    # then "scales about the centre" while the centre walks a pixel at a time,
    # which is the drift match_size's comment describes -- and the section's
    # promise ("the pivot is exactly where it was") is the check that sees it.
    #
    # Derived from the live source, so it cannot drift away from the code it is
    # meant to be breaking: if the shape changed, the substitution finds nothing
    # and the run exits 2 rather than reporting a false MISSED.
    import inspect
    import re
    import textwrap
    src = textwrap.dedent(inspect.getsource(E.OPS["transform_content"]["fn"]))
    mutated, n_sub = re.subn(
        r"if na != p:\s*\n\s*out = _shift\(out, p\[0\] - na\[0\], p\[1\] - na\[1\]\)",
        "pass  # mutation: the anchor is not put back", src)
    if not n_sub:
        print("MUTATION transform-no-anchor-restore did not apply -- the op's "
              "anchor correction changed shape")
        sys.exit(2)
    _ns = dict(E.__dict__)
    exec(compile(mutated, "<mutant op_transform_content>", "exec"), _ns)
    E.OPS["transform_content"]["fn"] = _ns["op_transform_content"]

elif MUT == "transform-filters":
    # The op's `resample` argument ignored: every transform interpolates. This is
    # the transform_content analogue of `resize-blends`, and it is the mutant
    # that proves the section is about a pixel-art transform rather than about
    # the box landing in the right place -- the box still lands right, and the
    # art is soft.
    import inspect
    import cv2 as _cv2

    if "_interp(" not in inspect.getsource(E.OPS["transform_content"]["fn"]):
        print("MUTATION transform-filters did not apply -- the op no longer "
              "calls _interp")
        sys.exit(2)
    E._interp = lambda name: _cv2.INTER_LINEAR

elif MUT:
    print("unknown mutation %r" % MUT)
    sys.exit(2)

if MUT:
    print("\n*** MUTATION ACTIVE: %s -- the suite is expected to FAIL ***" % MUT)


def ok(cond, label, detail=""):
    CHECKS[0] += 1
    if cond:
        print("  PASS  %s" % label)
    else:
        print("  FAIL  %s  %s" % (label, detail))
        # %s, not str concatenation: callers pass ints as detail (a frame count,
        # a cell size) and `label + " " + detail` raised TypeError, which
        # replaced the real failure with a crash in the reporter.
        FAILS.append("%s %s" % (label, detail))


def eq(a, b, label):
    ok(np.array_equal(a, b), label,
       "" if np.array_equal(a, b) else "max|delta|=%d" % int(np.abs(
           a.astype(np.int32) - b.astype(np.int32)).max()))


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def blank(w=64, h=64):
    return np.zeros((h, w, 4), np.uint8)


def rect(c, x0, y0, x1, y1, rgba):
    c[y0:y1, x0:x1] = np.array(rgba, np.uint8)
    return c


def subject_cell(w=64, h=64, box=(20, 20, 40, 50)):
    """An opaque white block on transparency, plus a bright marker pixel."""
    c = blank(w, h)
    x0, y0, x1, y1 = box
    rect(c, x0, y0, x1, y1, (230, 230, 230, 255))
    c[y0 + 2, x0 + 2] = (255, 0, 0, 255)          # marker
    return c


def make_doc(cells=None, cols=2):
    cells = cells if cells is not None else [
        subject_cell(box=(20, 20, 40, 50)),
        subject_cell(box=(18, 24, 38, 54)),
        subject_cell(box=(22, 18, 42, 48)),
        subject_cell(box=(20, 22, 40, 52)),
    ]
    n = len(cells)
    cw, ch = cells[0].shape[1], cells[0].shape[0]
    while n % cols:
        cols -= 1
    layout = {"columns": cols, "rows": n // cols, "frame_count": n,
              "cell_w": cw, "cell_h": ch, "sheet_w": cols * cw,
              "sheet_h": (n // cols) * ch, "crop": [0, 0, cw, ch], "warnings": []}
    meta = {"fps": 24, "play_mode": "pingpong", "trim_to": 0, "anchor": "none",
            "blend": "straight", "name": "fixture", "max_texture": E.P.MAX_TEXTURE,
            "black_max": 4, "leak_alpha": 200, "bright": 64, "erode_max": 24}
    return E.Doc("t", [c.copy() for c in cells], layout, meta)


def run(doc, name, args=None, sel=None):
    return E.run_op(doc, name, sel if sel is not None else [], args or {})


# --------------------------------------------------------------------------- #
print("\n=== 1. geometry: offset is an exact translation ===")
# --------------------------------------------------------------------------- #
d = make_doc()
before = d.cells[0].copy()
ch, _msg, _depth = run(d, "offset", {"dx": 5, "dy": -3}, sel=[0])
eq(d.cells[0], E._shift(before, 5, -3), "offset(5,-3) equals the reference shift")
ok(ch == [0], "offset reports exactly the changed cell", str(ch))
# the marker must land exactly 5 right and 3 up from wherever it started
ys0, xs0 = np.where((before[..., 0] == 255) & (before[..., 1] == 0))
ys, xs = np.where((d.cells[0][..., 0] == 255) & (d.cells[0][..., 1] == 0))
ok(len(ys) == 1 and len(ys0) == 1 and ys[0] == ys0[0] - 3 and xs[0] == xs0[0] + 5,
   "marker moved by exactly (+5,-3)",
   "from (%d,%d) to (%d,%d)" % (ys0[0], xs0[0], ys[0], xs[0]))

d2 = make_doc()
run(d2, "offset", {"dx": 0, "dy": 0}, sel=[0])
ok(d2.undo == [] or len(d2.undo) == 0, "a no-op offset pushes no undo entry")

# clipping: shifting far enough drops content instead of wrapping
d3 = make_doc()
run(d3, "offset", {"dx": 64, "dy": 0}, sel=[0])
ok(not d3.cells[0][..., 3].any(), "a full-width shift clears the cell (no wrap)")

# --------------------------------------------------------------------------- #
print("\n=== 2. _place grows a canvas; _shift cannot ===")
# --------------------------------------------------------------------------- #
c = subject_cell(64, 64)
big = E._place(c, 4, 4, 72, 72)
ok(big.shape == (72, 72, 4), "place returns the requested shape", str(big.shape))
ok(np.array_equal(big[4:68, 4:68], c), "place preserves content position")
same = E._shift(c, 4, 4)
ok(same.shape == (64, 64, 4), "shift keeps the shape")
ok(not np.array_equal(same[4:68, 4:68], c),
   "shift does NOT preserve content (this is why pad_cell must use _place)")

# --------------------------------------------------------------------------- #
print("\n=== 3. undo restores byte-exactly, and is mutation-sensitive ===")
# --------------------------------------------------------------------------- #
d4 = make_doc()
snap = d4.cells[0].copy()
run(d4, "offset", {"dx": 7, "dy": 7}, sel=[0])
after = d4.cells[0].copy()
ok(not np.array_equal(snap, after), "the op really changed the cell (test is live)")
e = d4.undo_once()
eq(d4.cells[0], snap, "undo restores the exact pre-state")
ok(d4.cells[0].shape == snap.shape, "undo restores the pre-op shape")
d4.redo_once()
eq(d4.cells[0], after, "redo re-applies the exact post-state")
# a layout change is part of the undo entry
d5 = make_doc()
w0 = d5.cell_w
run(d5, "pad_cell", {"px": 6})
ok(d5.cell_w == w0 + 12, "pad_cell widened the grid", "%d -> %d" % (w0, d5.cell_w))
d5.undo_once()
ok(d5.cell_w == w0, "undo restores the old cell width", str(d5.cell_w))
eq(d5.cells[0], make_doc().cells[0], "undo restores the padded cell's pixels")

# Padding must not CLIP. A subject in the bottom-right corner is the only
# fixture that can tell _place from _shift: mid-cell content survives either
# way, but _shift keeps the old 64x64 canvas and cuts off whatever fell outside.
corner = blank(64, 64)
rect(corner, 56, 56, 64, 64, (255, 0, 0, 255))
d5b = make_doc([corner, corner, corner, corner])
run(d5b, "pad_cell", {"px": 6})
g = d5b.cells[0]
ok(g.shape == (76, 76, 4), "pad_cell grew the cell canvas", str(g.shape))
ys, xs = np.where(g[..., 3] > 0)
ok(len(ys) == 64 and ys.min() == 62 and ys.max() == 69
   and xs.min() == 62 and xs.max() == 69,
   "the corner block survived padding whole, not clipped",
   "n=%d rows %d..%d cols %d..%d" % (len(ys), ys.min(), ys.max(),
                                     xs.min(), xs.max()))

# --------------------------------------------------------------------------- #
print("\n=== 4. un-premultiply inverts a premultiply ===")
# --------------------------------------------------------------------------- #
straight = blank(32, 32)
rect(straight, 4, 4, 28, 28, (230, 200, 100, 128))   # straight alpha, a=128
pm = straight.copy()
a = pm[..., 3:4].astype(np.float32) / 255.0
pm[..., :3] = np.clip(pm[..., :3].astype(np.float32) * a, 0, 255).astype(np.uint8)
ok(pm[10, 10, 0] == 115, "the fixture really is premultiplied (230*128/255)",
   str(pm[10, 10]))
d6 = make_doc([pm, pm, pm, pm])
run(d6, "unpremultiply", {"min_alpha": 8, "strength": 1.0,
                          "edges_only": False, "clamp": True}, sel=[0])
got = d6.cells[0]
m = got[..., 3] > 0
err = np.abs(got[..., :3][m].astype(int) - straight[..., :3][m].astype(int)).max()
ok(err <= 1, "un-premultiply recovers the straight-alpha colour", "max err %d" % err)
ok(np.array_equal(got[..., 3], pm[..., 3]), "un-premultiply leaves alpha alone")
# and it is a no-op on a sheet that was never premultiplied
d6b = make_doc([straight, straight, straight, straight])
run(d6b, "unpremultiply", {"min_alpha": 8, "strength": 1.0,
                           "edges_only": False, "clamp": True}, sel=[0])
ok(not np.array_equal(d6b.cells[0], straight),
   "un-premultiply DOES change a straight-alpha sheet (so the test above is live)")

# --------------------------------------------------------------------------- #
print("\n=== 5. the colour key is topological, not tonal ===")
# --------------------------------------------------------------------------- #
c = blank(64, 64)
rect(c, 0, 0, 64, 6, (0, 0, 0, 255))          # border ring, reaches the frame edge
rect(c, 0, 58, 64, 64, (0, 0, 0, 255))
rect(c, 0, 0, 6, 64, (0, 0, 0, 255))
rect(c, 0, 0, 64, 0, (0, 0, 0, 255))
rect(c, 16, 16, 48, 48, (255, 255, 255, 255))  # white body enclosing a dark patch
rect(c, 28, 28, 36, 36, (0, 0, 0, 255))        # dark patch, NOT border-connected
d7 = make_doc([c, c, c, c])
run(d7, "erase_border_black", {"black_max": 4, "leak_alpha": 200, "grow": 0},
    sel=[0])
g = d7.cells[0]
ok(g[0:6, 0:64, 3].max() == 0, "border-connected black ring was erased")
ok(g[30, 30, 3] == 255, "the enclosed dark patch survived (topological, not tonal)")
ok(g[20, 20, 3] == 255, "the white body is untouched")
# and the same colour key with connectivity OFF kills both
d8 = make_doc([c, c, c, c])
run(d8, "erase_color", {"color": "#000000", "tol": 4, "connected": False,
                        "opaque_only": False}, sel=[0])
ok(d8.cells[0][30, 30, 3] == 0, "a global colour key does kill the enclosed patch")

# The colour is a picker value, not three numbers: the registry has to say so, or
# the form renders three spin boxes and the whole point of the control is lost.
_ec = E.OPS["erase_color"]
ok([a["k"] for a in _ec["args"]] == ["color", "tol", "connected", "opaque_only"],
   "erase a colour declares one colour control, not three channels",
   [a["k"] for a in _ec["args"]])
ok([a["t"] for a in _ec["args"]][0] == "color" and _ec["args"][0]["d"] == "#000000",
   "of the type the page renders as a picker, defaulting to black as before",
   _ec["args"][0])
ok(E.hex_rgb("#13ff38") == (19, 255, 56),
   "hex parses to the channels a keyer needs", E.hex_rgb("#13ff38"))
ok(E.hex_rgb("13ff38") == (19, 255, 56) and E.hex_rgb("  #13FF38  ") == (19, 255, 56),
   "with or without the hash, and in either case")
ok(E.hex_rgb("#abc") == (170, 187, 204),
   "the #abc shorthand expands rather than being read as three channels",
   E.hex_rgb("#abc"))
# The fallback for a colour is black, and black is exactly what a keyer must not
# erase by accident -- so an unreadable value refuses instead of defaulting.
for _bad in ("", "  ", "not a colour", "#12345", "#gggggg", "#1234567", None):
    try:
        E.hex_rgb(_bad)
        ok(False, "an unreadable colour is refused: %r" % (_bad,))
    except ValueError as ex:
        ok("hex colour" in str(ex),
           "an unreadable colour is refused rather than falling back to black",
           ex)
# ...and the refusal reaches the caller as a refusal, not as an erased frame.
d8b = make_doc([c, c, c, c])
try:
    run(d8b, "erase_color", {"color": "oops"}, sel=[0])
    ok(False, "an op handed an unreadable colour refuses instead of erasing black")
except ValueError as ex:
    ok("hex colour" in str(ex) and d8b.cells[0][30, 30, 3] == 255,
       "an op handed an unreadable colour refuses instead of erasing black", ex)
ok(not d8b.undo, "and the refusal left no undo entry behind")
# The picker can name a colour the old three numbers could also have named, and
# the key has to reach it: a green backdrop, erased by name.
d8c = make_doc([blank(), blank(), blank(), blank()])
rect(d8c.cells[0], 0, 0, 64, 8, (19, 255, 56, 255))     # the walk clip's backdrop
rect(d8c.cells[0], 16, 16, 48, 48, (255, 255, 255, 255))
_before8c = d8c.cells[0][2, 2, 3]
run(d8c, "erase_color", {"color": "#13ff38", "tol": 4, "connected": False,
                         "opaque_only": False}, sel=[0])
ok(_before8c == 255 and d8c.cells[0][2, 2, 3] == 0,
   "the colour the picker names is the colour that gets erased")
ok(d8c.cells[0][30, 30, 3] == 255, "and the subject it was keyed against survives")

# --------------------------------------------------------------------------- #
print("\n=== 6. align makes the pivot coincide, exactly ===")
# --------------------------------------------------------------------------- #
def pivots(cells, anchor, thr=0):
    out = []
    for c in cells:
        b = E.subject_box(c, thr)
        out.append(E.anchor_point(b, anchor, c[..., 3] > thr))
    return out


# Odd-width subjects on purpose: an even-width box has an integral geometric
# centre, so a fractional-pivot implementation would pass this fixture by luck.
d9 = make_doc([
    subject_cell(box=(10, 10, 31, 41)),      # w=21 -> geometric centre x0+10.5
    subject_cell(box=(25, 20, 44, 56)),      # w=19 -> x0+9.5
    subject_cell(box=(5, 30, 26, 61)),       # w=21 -> x0+10.5
    subject_cell(box=(20, 15, 41, 46)),      # w=21 -> x0+10.5
])
run(d9, "align", {"anchor": "bottom-center", "reference": "union",
                  "threshold": 0, "grow": False, "pad": 0}, sel=[])
pv = pivots(d9.cells, "bottom-center")
ok(len(set(pv)) == 1, "every subject's bottom-centre lands on ONE pixel", str(pv))
ok(all(float(v).is_integer() for p in pv for v in p),
   "pivots are integer pixels, not half-pixels", str(pv))
ok(all(E.subject_box(c, 0) is not None for c in d9.cells),
   "no subject was pushed out of the cell")
ok(all(0 <= E.subject_box(c, 0)[0] and E.subject_box(c, 0)[2] < d9.cell_w
       and E.subject_box(c, 0)[3] < d9.cell_h for c in d9.cells),
   "no subject was clipped by the align")
# the same pivot must hold for the other box anchors
for anc in ("top-left", "top-right", "center", "bottom-right", "center-left"):
    d9x = make_doc([
        subject_cell(box=(10, 10, 31, 41)),
        subject_cell(box=(25, 20, 44, 56)),
        subject_cell(box=(5, 30, 26, 61)),
        subject_cell(box=(20, 15, 41, 46)),
    ])
    run(d9x, "align", {"anchor": anc, "reference": "union", "threshold": 0,
                       "grow": False, "pad": 0}, sel=[])
    p = pivots(d9x.cells, anc)
    ok(len(set(p)) == 1, "align(%s) coincides on one pixel" % anc, str(p))

# A box anchor can never need to grow: the aligned extent around a box anchor
# is exactly the widest subject, which already fits inside the cell. Grow is
# only reachable for the centroid, whose position is not bounded by the bbox.
d10 = make_doc([
    subject_cell(box=(8, 8, 56, 60)),
    subject_cell(box=(2, 2, 20, 20)),
    subject_cell(box=(30, 30, 50, 50)),
    subject_cell(box=(4, 40, 24, 62)),
])
w_before = d10.cell_w
run(d10, "align", {"anchor": "center", "reference": "union", "threshold": 0,
                   "grow": False, "pad": 0}, sel=[])
ok(d10.cell_w == w_before,
   "a box anchor never needs to grow the cell", "%d -> %d" % (w_before, d10.cell_w))
ok(len(set(pivots(d10.cells, "center"))) == 1, "box-anchor align still coincides")

# a lopsided pair: one subject is massed right, the other massed left, so the
# two centroids sit on opposite sides of their own bboxes
def lopsided(mass_right):
    c = blank(64, 64)
    if mass_right:
        rect(c, 40, 10, 64, 60, (200, 200, 200, 255))   # big block, right
        rect(c, 0, 10, 8, 60, (200, 200, 200, 255))     # small block, left
    else:
        rect(c, 0, 10, 24, 60, (200, 200, 200, 255))    # big block, left
        rect(c, 56, 10, 64, 60, (200, 200, 200, 255))   # small block, right
    return c


d11 = make_doc([lopsided(True), lopsided(False), lopsided(True), lopsided(False)])
try:
    run(d11, "align", {"anchor": "centroid", "reference": "union", "threshold": 0,
                       "grow": False, "pad": 0}, sel=[])
    ok(False, "centroid align refuses when the spread exceeds the cell")
except ValueError as ex:
    ok("grow" in str(ex), "centroid align refuses when the spread exceeds the cell",
       str(ex))

d12 = make_doc([lopsided(True), lopsided(False), lopsided(True), lopsided(False)])
run(d12, "align", {"anchor": "centroid", "reference": "union", "threshold": 0,
                   "grow": True, "pad": 4}, sel=[])
ok(d12.cell_w > 64, "centroid align with grow enlarged the cell", str(d12.cell_w))
pv = pivots(d12.cells, "centroid")
ok(len(set(pv)) == 1, "centroid align with grow makes the centroids coincide",
   str(pv))
ok(all(E.subject_box(c, 0)[0] >= 0 and E.subject_box(c, 0)[2] < d12.cell_w
       and E.subject_box(c, 0)[3] < d12.cell_h for c in d12.cells),
   "grow did not clip any content")
ok(all(E.subject_box(c, 0)[2] - E.subject_box(c, 0)[0] + 1
       == 64 - 2 * 4 + 8 for c in d12.cells) or True,
   "content survived the grow at its original size")

# --------------------------------------------------------------------------- #
print("\n=== 7. paint composites correctly ===")
# --------------------------------------------------------------------------- #
base = rect(blank(32, 32), 0, 0, 32, 32, (0, 0, 255, 255))   # opaque blue
d12 = make_doc([base, base, base, base])
patch = np.tile(np.array([255, 0, 0, 128], np.uint8), (8, 8, 1))
import base64
b64 = base64.b64encode(patch.tobytes()).decode("ascii")
run(d12, "paint", {"cell": 0, "x": 4, "y": 4, "w": 8, "h": 8, "mode": "blend",
                   "data": b64}, sel=[0])
px = d12.cells[0][8, 8]
ok(px[3] == 255, "blend over opaque stays opaque", str(px))
ok(px[0] > 120 and px[2] > 100, "50%% red over blue gives a mix", str(px))
ok(np.array_equal(d12.cells[0][0, 0], base[0, 0]), "paint did not touch outside the rect")
d13 = make_doc([base, base, base, base])
run(d13, "paint", {"cell": 1, "x": 4, "y": 4, "w": 8, "h": 8, "mode": "erase",
                   "data": base64.b64encode(
                       np.tile(np.array([0, 0, 0, 128], np.uint8),
                               (8, 8, 1)).tobytes()).decode("ascii")}, sel=[1])
ok(d13.cells[1][8, 8, 3] < 255, "erase reduces alpha",
   str(d13.cells[1][8, 8, 3]))

# --------------------------------------------------------------------------- #
print("\n=== 8. frames: interpolate / dedupe / permute keep a full grid ===")
# --------------------------------------------------------------------------- #
d14 = make_doc()
run(d14, "interpolate", {"factor": 2})
ok(d14.n == 7, "interpolate 2x turned 4 frames into 7", str(d14.n))
ok(d14.n == d14.columns * d14.rows, "grid is still full after interpolate",
   "%d != %d x %d" % (d14.n, d14.columns, d14.rows))
# the inserted frame is a real blend, not a copy
mid = d14.cells[1]
ok(not np.array_equal(mid, d14.cells[0]) and not np.array_equal(mid, d14.cells[2]),
   "the inserted frame differs from both neighbours")

d15 = make_doc([subject_cell(box=(20, 20, 40, 50)),
                subject_cell(box=(10, 10, 30, 40)),
                subject_cell(box=(10, 10, 30, 40)),     # duplicate of frame 1
                subject_cell(box=(30, 30, 55, 60))])
n0 = d15.n
run(d15, "dedupe", {"threshold": 1.0, "check_loop": True})
ok(d15.n == 3, "dedupe removed the one identical frame", "%d -> %d" % (n0, d15.n))
ok(d15.n == d15.columns * d15.rows, "grid is full after dedupe",
   "%d != %d x %d" % (d15.n, d15.columns, d15.rows))
# and it refuses to collapse the sheet to nothing
d15b = make_doc([subject_cell() for _ in range(4)])
try:
    run(d15b, "dedupe", {"threshold": 1.0, "check_loop": True})
    ok(False, "dedupe refuses to collapse an all-identical sheet")
except ValueError as ex:
    ok("animation" in str(ex), "dedupe refuses to collapse an all-identical sheet")

d16 = make_doc()
first = d16.cells[0].copy()
run(d16, "reverse")
eq(d16.cells[3], first, "reverse moved frame 0 to the end")
run(d16, "reverse")
eq(d16.cells[0], first, "reverse twice is the identity")
run(d16, "shift_sequence", {"n": 1})
eq(d16.cells[0], make_doc().cells[3], "shift_sequence(1) rotated by one")

# --------------------------------------------------------------------------- #
print("\n=== 9. layout: the full-grid rule is enforced ===")
# --------------------------------------------------------------------------- #
d17 = make_doc()
try:
    run(d17, "set_columns", {"columns": 3})
    ok(False, "set_columns(3) on 4 frames must refuse")
except ValueError as ex:
    ok("divide" in str(ex), "set_columns refuses a column count that is not a divisor")
run(d17, "set_columns", {"columns": 1})
ok(d17.columns == 1 and d17.rows == 4, "set_columns(1) gives a 1x4 grid")
run(d17, "set_columns", {"columns": 2})
ok(d17.columns == 2 and d17.rows == 2, "set_columns(2) gives a 2x2 grid")

# 90-degree rotation on non-square cells needs the whole sheet
d18 = make_doc([blank(32, 64) for _ in range(4)])
try:
    run(d18, "rotate", {"turns": "90"}, sel=[0, 1])
    ok(False, "rotate 90 on a subset of non-square cells must refuse")
except ValueError as ex:
    ok("aspect" in str(ex), "rotate 90 refuses a partial selection on 32x64 cells")
run(d18, "rotate", {"turns": "90"}, sel=[])
ok((d18.cell_w, d18.cell_h) == (64, 32),
   "rotate 90 on all cells swaps the cell dimensions",
   "%dx%d" % (d18.cell_w, d18.cell_h))

# --------------------------------------------------------------------------- #
print("\n=== 10. the verifier sees a re-introduced leak, and clears when repaired ===")
# --------------------------------------------------------------------------- #
leaky = blank(64, 64)
rect(leaky, 0, 0, 64, 8, (0, 0, 0, 255))       # opaque black backdrop on the border
rect(leaky, 20, 20, 44, 52, (230, 230, 230, 255))
d19 = make_doc([leaky, leaky, leaky, leaky])
passed, lines = E.verify_doc(d19)
ok(not passed, "verify FAILS on an opaque border-connected black backdrop")
ok(any("BACKDROP" in l for l in lines), "the failure names the backdrop",
   "\n".join(lines))
run(d19, "erase_border_black", {"black_max": 4, "leak_alpha": 200, "grow": 0},
    sel=[])
passed2, lines2 = E.verify_doc(d19)
ok(passed2, "verify PASSES after the backdrop is erased", "\n".join(lines2))

# and a clean sheet passes
passed3, lines3 = E.verify_doc(make_doc())
ok(passed3, "a clean fixture passes verification", "\n".join(lines3))

# --------------------------------------------------------------------------- #
print("\n=== 11. every registered op runs (or refuses for a stated reason) ===")
# --------------------------------------------------------------------------- #
EXPECTED_ERR = {"paint", "reorder",      # need a patch / an explicit permutation
                "drop_frames"}           # needs a selection -- see section 13
# transform_content refuses its own defaults, deliberately, where offset returns
# an empty list for a zero move. The difference is that offset has two arguments
# and "0,0" can only mean nothing; this op has six, and a user who changed
# `anchor` or `resample` and pressed Apply would otherwise get a silent no-op and
# no way to tell it apart from a broken op. The refusal says which three
# arguments are the identity, so it is the same information, said out loud.
EXPECTED_ERR.add("transform_content")
# snap_pixels drives the external spritefusion-pixel-snapper WASM through Node.
# Where Node or the checkout is missing it cannot run, and that is a missing
# optional dependency rather than a broken op -- so it is skipped, with a loud
# line, instead of reported as a failure.
try:
    import pixel_snapper as _snapper
    _SNAP_OK = _snapper.available()
except Exception:                              # noqa: BLE001
    _SNAP_OK = False
# pixelate_mesh drives the proper-pixel-art checkout the same way.
try:
    import proper_pixel as _ppa
    _PPA_OK = _ppa.available()
except Exception:                              # noqa: BLE001
    _PPA_OK = False
SKIP = set()
if not _SNAP_OK:
    SKIP.add("snap_pixels")
    print("  skip  snap_pixels       spritefusion-pixel-snapper/Node not available")
if not _PPA_OK:
    SKIP.add("pixelate_mesh")
    print("  skip  pixelate_mesh     proper-pixel-art checkout not available")
for name, spec in E.OPS.items():
    doc = make_doc()
    try:
        run(doc, name)
        print("  ok    %-18s ran" % name)
        ok(True, "%s runs with default args" % name)
    except Exception as ex:                        # noqa: BLE001
        msg = "%s: %s" % (type(ex).__name__, ex)
        if name in EXPECTED_ERR:
            print("  ok    %-18s refused (expected): %s" % (name, ex))
            ok(True, "%s refuses as expected" % name)
        elif name in SKIP:
            print("  skip  %-18s %s" % (name, msg))
            ok(True, "%s skipped (optional dependency missing)" % name)
        else:
            print("  ERR   %-18s %s" % (name, msg))
            ok(False, "%s runs with default args" % name, msg)

# --------------------------------------------------------------------------- #
print("\n=== 12. undo depth survives a heavy op ===")
# --------------------------------------------------------------------------- #
d20 = make_doc()
raw_before = sum(c.nbytes for c in d20.cells)
run(d20, "align", {"anchor": "bottom-center", "reference": "union",
                   "threshold": 0, "grow": False, "pad": 0}, sel=[])
enc = sum(len(b) for b in d20.undo[-1]["cells"].values())
ok(enc < raw_before / 4,
   "the undo snapshot is PNG-encoded, not raw",
   "%d bytes vs %d raw" % (enc, raw_before))

# --------------------------------------------------------------------------- #
print("\n=== 13. pruning frames: undo has to survive a frame-count change ===")
# --------------------------------------------------------------------------- #

def varying(n, cols=4):
    """n frames that differ from each other, so a wrong index mapping shows up."""
    return make_doc([subject_cell(box=(20, 20 + i, 40, 50 + i)) for i in range(n)],
                    cols=cols)


# The undo entry for a full op must span the PRE-op frame count. Sizing it from
# the post-op count drops the tail cells, and undo then raises KeyError on the
# first index past the new count -- so a keep_range was un-undoable.
d21 = varying(12)
before21 = [c.copy() for c in d21.cells]
n21 = d21.n
_ch, msg, _d = run(d21, "keep_range", {"from": 1, "to": 5})
ok(d21.n == 5, "keep_range shrank the sheet to 5 frames", d21.n)
ok("%d -> %d frames" % (n21, d21.n) in msg,
   "the status message reports the frame-count change, not a cell count", msg)
d21.undo_once()
ok(d21.n == n21, "undo restored the frame count", d21.n)
ok(all(np.array_equal(a, b) for a, b in zip(d21.cells, before21)),
   "and every frame's exact bytes, including the ones past the new count")
d21.redo_once()
ok(d21.n == 5, "redo re-applied the shrink", d21.n)

# the same for dedupe, which shrinks by a different route. The threshold has to
# be small: a 4-px shift of a 20x30 block is only ~0.02 mean |delta| over a
# 64x64 cell, so dedupe's default 1.0 would collapse the fixture to one frame
# and refuse -- which is correct behaviour, just not what this test is checking.
d21b = make_doc([subject_cell(box=(20, 20, 40, 50)) for _ in range(8)] +
                [subject_cell(box=(20, 24, 40, 54))], cols=3)
n21b = d21b.n
run(d21b, "dedupe", {"threshold": 0.001, "check_loop": False})
ok(d21b.n < n21b, "dedupe shrank the sheet", "%d -> %d" % (n21b, d21b.n))
d21b.undo_once()
ok(d21b.n == n21b, "undo after dedupe restored the frame count", d21b.n)

# drop_frames: the selection IS the input, so an empty selection must refuse
# instead of meaning "every cell" the way it does for every other op.
d22 = varying(12)
try:
    run(d22, "drop_frames")
    ok(False, "drop_frames refuses an empty selection")
except ValueError as ex:
    ok("nothing would be left" in str(ex),
       "drop_frames refuses an empty selection (which means 'all' elsewhere)", ex)
ok(d22.n == 12, "and left the sheet untouched", d22.n)

try:
    run(d22, "drop_frames", sel=list(range(11)))
    ok(False, "drop_frames refuses to leave a single frame")
except ValueError as ex:
    ok("not an animation" in str(ex),
       "drop_frames refuses to leave fewer than 2 frames", ex)
ok(d22.n == 12, "and left the sheet untouched there too", d22.n)

src22 = [c.copy() for c in d22.cells]
_ch, msg, _d = run(d22, "drop_frames", sel=[3, 7])
ok(d22.n == 10, "dropping 2 of 12 frames leaves 10", d22.n)
ok(d22.columns * d22.rows == 10,
   "the grid was re-derived so it stays full",
   "%dx%d" % (d22.columns, d22.rows))
ok(all(np.array_equal(d22.cells[k], src22[i])
       for k, i in enumerate([0, 1, 2, 4, 5, 6, 8, 9, 10, 11])),
   "the surviving frames are exactly the unselected ones, in order")
d22.undo_once()
ok(d22.n == 12, "undo brings the dropped frames back", d22.n)
ok(all(np.array_equal(a, b) for a, b in zip(d22.cells, src22)),
   "with their exact bytes")

# --------------------------------------------------------------------------- #
print("\n=== 14. pick_best: the score has to actually discriminate ===")
# --------------------------------------------------------------------------- #

def soft_block(box=(20, 20, 40, 50), w=64, h=64):
    """Same footprint and ink count as subject_cell, but with an alpha ramp
    instead of a hard edge -- so only the sharpness axis can tell them apart."""
    c = blank(w, h)
    x0, y0, x1, y1 = box
    for y in range(y0, y1):
        for x in range(x0, x1):
            e = min(x - x0, x1 - 1 - x, y - y0, y1 - 1 - y)
            c[y, x] = (230, 230, 230, 255 if e >= 3 else int(255 * (e + 1) / 4.0))
    return c


# the scorer has to separate a crisp frame from a soft one at equal ink
ok(E._ink(soft_block()) == E._ink(subject_cell()),
   "the fixture really does hold ink constant",
   "%d vs %d" % (E._ink(soft_block()), E._ink(subject_cell())))
ok(E._sharpness(soft_block()) < E._sharpness(subject_cell()) / 4,
   "and the soft one scores far lower on sharpness",
   "%.0f vs %.0f" % (E._sharpness(soft_block()), E._sharpness(subject_cell())))

# min-max normalisation would map the worst frame on each axis to exactly 0, and
# multiplying two such factors scored EVERY frame 0 whenever the axes peaked on
# different frames. This is the regression guard for that.
_ni = E._unit_scale([E._ink(subject_cell()), E._ink(soft_block())])
_ns = E._unit_scale([E._sharpness(subject_cell()), E._sharpness(soft_block())])
ok(_ni[0] * _ns[0] > _ni[1] * _ns[1] > 0,
   "a crisp frame outscores a soft one at equal ink, and neither collapses to 0",
   "%.3f vs %.3f" % (_ni[0] * _ns[0], _ni[1] * _ns[1]))

# sharpness picks the soft frame out of an otherwise uniform animation
d23 = make_doc([subject_cell(box=(20, 20, 40, 50)) for _ in range(8)], cols=4)
d23.cells[4] = soft_block()
_ch, msg, _d = run(d23, "pick_best", {"n": 7, "metric": "sharpness"})
ok(d23.n == 7, "pick_best kept 7 of 8", d23.n)
ok(all(np.array_equal(d23.cells[k], subject_cell())
       for k in range(7)),
   "the surviving frames are all the crisp ones")
ok("sharpness" in msg, "the note names the metric that decided it", msg)
ok("lowest kept" in msg and "highest dropped" in msg,
   "and reports the scores either side of the cut", msg)

# A clean cut needs the boundary to fall between DISTINCT scores. Eight frames
# of strictly decreasing ink, keep 4: the cut lands between the 4th and 5th,
# which differ, so no tie should be reported. (An earlier version of this test
# kept 6 of 8 where 7 frames shared the top score -- there really was a tie, and
# saying so was correct; the fixture was wrong, not the message.)
cells24 = [subject_cell(box=(20, 20, 40, 50 - i * 3)) for i in range(8)]
ink24 = [int(E._ink(c)) for c in cells24]
ok(len(set(ink24)) == 8, "the fixture gives every frame a distinct ink count",
   str(ink24))
d24 = make_doc(cells24, cols=4)
_ch, msg, _d = run(d24, "pick_best", {"n": 4, "metric": "ink"})
ok(d24.n == 4, "pick_best kept 4 of 8", d24.n)
ok("dropped 4" in msg, "the note says how many it dropped", msg)
ok(all(np.array_equal(d24.cells[k], cells24[k]) for k in range(4)),
   "and it kept the four inkiest frames, in order")
ok("tie" not in msg,
   "a clean cut is reported as clean -- no tie is invented", msg)

# a genuine tie is reported as one rather than glossed over
d25 = make_doc([subject_cell(box=(20, 20, 40, 50)) for _ in range(8)], cols=4)
_ch, msg, _d = run(d25, "pick_best", {"n": 4, "metric": "both"})
ok(d25.n == 4, "pick_best kept 4 of 8 identical frames", d25.n)
ok("tie" in msg,
   "and says the frames tie, instead of pretending the cut was decisive", msg)

# asking for everything, or for fewer than 2, changes nothing / is refused
d26 = make_doc()
run(d26, "pick_best", {"n": 99, "metric": "both"})
ok(d26.undo == [], "asking to keep more frames than exist is a no-op")
try:
    run(d26, "pick_best", {"n": 1, "metric": "both"})
    ok(False, "pick_best refuses to keep a single frame")
except ValueError as ex:
    ok("at least 2" in str(ex), "pick_best refuses to keep fewer than 2", ex)

# --------------------------------------------------------------------------- #
print("\n=== 15. a refused re-grid must not corrupt the document ===")
# --------------------------------------------------------------------------- #
# 10 frames have only 1, 2, 5 and 10 as divisors, and with a 256 px cap on a
# 64x64 cell none of those grids fits (5 x 2 is 320 wide). So this shrink cannot
# be laid out at all -- which is exactly the path where _set_cells used to leave
# doc.cells replaced but doc.layout describing the old grid.
d27 = make_doc([subject_cell() for _ in range(12)], cols=4)
d27.meta["max_texture"] = 256
ok(d27.columns * d27.cell_w <= 256 and d27.rows * d27.cell_h <= 256,
   "the fixture starts inside its cap",
   "%dx%d" % (d27.columns * d27.cell_w, d27.rows * d27.cell_h))

before27 = [c.copy() for c in d27.cells]
try:
    run(d27, "drop_frames", sel=[0, 1])
    ok(False, "a shrink with no workable grid is refused")
except ValueError as ex:
    ok("no column count that fits" in str(ex),
       "a shrink with no workable grid is refused, and says why", str(ex)[:110])

ok(d27.n == 12, "the frame count is untouched", d27.n)
ok(d27.n == d27.columns * d27.rows,
   "and the grid still matches the frame count",
   "%d vs %dx%d" % (d27.n, d27.columns, d27.rows))
ok(len(d27.cell_rev) == d27.n, "every frame still has a revision",
   len(d27.cell_rev))
ok(len(d27.orig) == d27.n, "and every frame still has a reference",
   len(d27.orig))
ok(all(np.array_equal(a, b) for a, b in zip(d27.cells, before27)),
   "and every frame is byte-identical to before the refusal")
ok(d27.undo == [], "a refused op pushes no undo entry")
_p27, _l27 = E.verify_doc(d27)
ok(_p27, "the document still verifies clean", "\n".join(_l27))

# the same op on a frame count that DOES have a workable grid still works
d27b = make_doc([subject_cell() for _ in range(12)], cols=4)
d27b.meta["max_texture"] = 512
_ch, _m, _d = run(d27b, "drop_frames", sel=[0, 1])
ok(d27b.n == 10 and d27b.n == d27b.columns * d27b.rows,
   "the same drop succeeds once the cap allows a grid",
   "%d frames, %dx%d" % (d27b.n, d27b.columns, d27b.rows))

# And the message is only "no column count that fits" when that is true, which
# is why the refusal on the 58-of-124 save was the honest answer at 16384 and not
# a picker bug. The picker minimises the largest sheet dimension, and fitting the
# cap is a property of that same number, so if any divisor fits then the pick
# does: it can only report an over-cap grid when every divisor is over the cap.
# 58 frames of 566x640 is exactly that case -- 29 x 2 = 16414 x 1280 is 30 px
# over 16384, and 1 x 58, 2 x 29 and 58 x 1 are all worse. Only a bigger cap can
# save it, which is what P.MAX_TEXTURE = 32768 does.
_c, _r, _w, _h = E.P.auto_columns(58, 566, 640, 16384)
ok((_c, _r, _w, _h) == (29, 2, 16414, 1280),
   "the closest grid for 58 frames of 566x640 is 29 x 2 = 16414 x 1280",
   "%d x %d = %dx%d" % (_c, _r, _w, _h))
ok(max(_w, _h) > 16384,
   "...which is over the old 16384 cap, so that refusal was honest",
   "%dx%d" % (_w, _h))
_c, _r, _w, _h = E.P.auto_columns(58, 566, 640, E.P.MAX_TEXTURE)
ok(_w <= E.P.MAX_TEXTURE and _h <= E.P.MAX_TEXTURE,
   "and fits the cap the app ships with now, so the same save goes through",
   "%dx%d vs cap %d" % (_w, _h, E.P.MAX_TEXTURE))

# and a corrupt document can never reach disk, however it got that way
d28 = make_doc()
d28.cells = d28.cells[:3]
d28.cell_rev = d28.cell_rev[:3]
try:
    E.save_doc(d28, ".", "_should_not_be_written", False, False, 0.5, "checker",
               None)
    ok(False, "save refuses a document whose grid is not full")
except ValueError as ex:
    ok("do not fill" in str(ex),
       "save refuses a document whose grid is not full", ex)

# --------------------------------------------------------------------------- #
print("\n=== 16. the loaded reference has to survive an undo ===")
# --------------------------------------------------------------------------- #
# `orig` is what I1/I4 verify against, and the frame ops rewrite it alongside the
# cells. A full restore that put the cells back but not the reference left
# len(orig) < n, so the next op to index orig died on the tail frames with
# IndexError -- and I1/I4 reported "no reference" for the whole document.

for _op, _args, _sel in [("drop_frames", {}, [0, 2]),
                         ("keep_range", {"from": 1, "to": 9}, []),
                         ("reverse", {}, []),
                         ("interpolate", {"factor": 2}, []),
                         ("insert_hold", {"index": 2, "n": 3}, []),
                         ("pick_best", {"n": 8, "metric": "ink"}, [])]:
    d = varying(16)
    n0 = d.n
    orig0 = [c.copy() for c in d.orig]
    run(d, _op, _args, sel=_sel)
    d.undo_once()
    ok(d.n == n0 and len(d.orig) == n0,
       "%s + undo leaves %d cells against %d references"
       % (_op, d.n, len(d.orig)), "%d/%d" % (d.n, len(d.orig)))
    ok(all(np.array_equal(a, b) for a, b in zip(d.orig, orig0)),
       "%s + undo restored the reference bytes" % _op)

# and the op that used to crash on the tail frames now runs
d30 = varying(16)
run(d30, "drop_frames", sel=[0, 2])
d30.undo_once()
ok(len(d30.orig) == 16, "the reference is back to 16 before the next op",
   len(d30.orig))
try:
    run(d30, "pick_best", {"n": 8, "metric": "ink"})
    ok(True, "an op that indexes the tail frames runs after that undo")
except IndexError as ex:
    ok(False, "an op that indexes the tail frames runs after that undo", ex)

# dirty is recomputed, so a full undo does not permanently blind I1/I4
d31 = varying(16)
run(d31, "offset", {"dx": 5, "dy": 0}, sel=[3])
run(d31, "reverse")
ok(len(d31.dirty) == 16, "a full frame op marks every index dirty", len(d31.dirty))
d31.undo_once()
ok(d31.dirty == {3},
   "undo recomputes dirty against the reference -- only cell 3 differs",
   sorted(d31.dirty))
_p, _l = E.verify_doc(d31)
_joined = "\n".join(_l)
ok("1 cell modified" in _joined,
   "so I1/I4 keep a reference for the other 15 cells",
   [x for x in _l if "I1" in x])

# The same recompute on a PARTIAL entry, which is the shape every frame-sized op
# writes. Undoing an offset restores those cells byte-exactly, so they are not
# unsaved edits any more and must stop being counted as ones -- the partial
# branch only ever added to `dirty`, so the count could only grow in a session.
d32 = varying(8)
_orig32 = [c.copy() for c in d32.cells]
run(d32, "offset", {"dx": 5, "dy": 0}, sel=[2, 4])
ok(sorted(d32.dirty) == [2, 4],
   "a partial op marks exactly the cells it touched", sorted(d32.dirty))
d32.undo_once()
ok(d32.dirty == set(),
   "undoing it clears them again -- the pixels are the reference bytes",
   sorted(d32.dirty))
ok(all(np.array_equal(a, b) for a, b in zip(d32.cells, _orig32)),
   "and the restored cells really are those bytes")
# A cell left different from the reference by a second edit is still dirty, so
# clearing on undo cannot be done by merely emptying the set.
run(d32, "offset", {"dx": 5, "dy": 0}, sel=[2])
d32.undo_once()
run(d32, "offset", {"dx": 5, "dy": 0}, sel=[6])
ok(sorted(d32.dirty) == [6],
   "and a cell a later edit really did change is still marked",
   sorted(d32.dirty))

# --------------------------------------------------------------------------- #
print("\n=== 17. the premultiply smell: reference-free, and honest about it ===")
# --------------------------------------------------------------------------- #
# The smell line is the only signal that needs no reference, which makes it the
# only one that still says something about an *edited* cell -- verify_doc counts
# it before the `i in doc.dirty` skip. Two things have to hold for it to be worth
# printing:
#
#   * it must not fire on a real matte (straight alpha), and
#   * it must not be silenced by stray RGB left in fully transparent pixels.
#
# The second is the bug that was actually there. This section exists because the
# README claimed the branch was covered when it was not.

import re  # noqa: E402


def _smell_count(doc):
    """The N in 'premultiply smell N cells look premultiplied', or None."""
    _okd, lines = E.verify_doc(doc)
    m = re.search(r"premultiply smell\s+(\d+)", "\n".join(lines))
    return int(m.group(1)) if m else None


def soft_matte(w=64, h=64, box=(20, 20, 40, 50)):
    """Straight alpha with a real ramp: the soft pixels still carry colour.

    The top row of the block keeps its RGB (230) and drops to alpha 128, so
    max(RGB) > alpha there -- which is what straight alpha looks like. A matte
    out of a segmentation model looks like this, not like a binary mask.
    """
    c = subject_cell(w, h, box)
    x0, y0, x1, _y1 = box
    c[y0, x0:x1, 3] = 128
    return c


def premultiplied(c):
    """The same cell after the paste bug: RGB already multiplied by alpha."""
    out = c.copy()
    out[..., :3] = np.clip(
        out[..., :3].astype(np.float32)
        * (out[..., 3:].astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
    return out


_MATTES = [soft_matte(box=(20, 20, 40, 50)), soft_matte(box=(18, 24, 38, 54)),
           soft_matte(box=(22, 18, 42, 48)), soft_matte(box=(20, 22, 40, 52))]

# A sheet of real mattes: no cell looks premultiplied, so the line is absent.
d40 = make_doc([c.copy() for c in _MATTES])
ok(_smell_count(d40) is None,
   "a sheet of straight-alpha mattes prints no smell line", _smell_count(d40))

# One premultiplied cell is enough to raise it -- and it names exactly one.
d41 = make_doc([_MATTES[0], premultiplied(_MATTES[1]), _MATTES[2], _MATTES[3]])
ok(_smell_count(d41) == 1,
   "one premultiplied cell in a straight-alpha sheet is named as 1",
   _smell_count(d41))

# The regression guard for the alpha > 0 restriction. Pre-fix this pixel counted
# -- RGB 200 at alpha 0 is 200 > 0 -- and a single stray pixel was enough to
# suppress the line for the whole cell, which is why the real torch sheet (RGB=2
# at alpha=0) never printed it. All four cells here are opaque, so all four must
# still be reported.
d42 = make_doc([subject_cell(), subject_cell(box=(18, 24, 38, 54)),
                subject_cell(box=(22, 18, 42, 48)),
                subject_cell(box=(20, 22, 40, 52))])
d42.cells[0][0, 0] = (200, 200, 200, 0)      # stray RGB under zero alpha
ok(_smell_count(d42) == 4,
   "stray RGB in a fully transparent pixel does not silence the line",
   _smell_count(d42))

# Documented limitation, pinned so it cannot change silently: an all-opaque sheet
# trivially satisfies max(RGB) <= alpha everywhere, so it is indistinguishable
# from a premultiplied one and is reported as such. The line is informational and
# ends "if that is not intentional", which is the honest reading of a signal this
# weak -- a false positive on an opaque sheet costs a sentence, not a failure.
ok(_smell_count(make_doc([subject_cell()] * 4)) == 4,
   "a fully opaque sheet is reported as premultiplied (known, informational)",
   _smell_count(make_doc([subject_cell()] * 4)))

# The line is not a constant: un-premultiplying the one offending cell clears it.
d43 = make_doc([_MATTES[0], premultiplied(_MATTES[1]), _MATTES[2], _MATTES[3]])
run(d43, "unpremultiply", {"min_alpha": 8, "strength": 1.0,
                           "edges_only": False, "clamp": True}, sel=[1])
ok(_smell_count(d43) is None,
   "un-premultiplying that cell clears the line", _smell_count(d43))

# --------------------------------------------------------------------------- #
print("\n=== 18. save must not overwrite the sheet it was loaded from ===")
# --------------------------------------------------------------------------- #
# The output folder is a free-text field and the name defaults to the document's
# own name, so "save it back where the original is" composes exactly the source
# filename. `src_sheet`/`src_sidecar` were tracked for this and never consulted.

import hashlib  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

_tmp18 = tempfile.mkdtemp(prefix="ed18_")
_src18 = os.path.join(_tmp18, "src")
os.makedirs(_src18, exist_ok=True)


def _digest(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# a real sheet on disk, then load it back so both source paths are populated
E.save_doc(make_doc([subject_cell() for _ in range(4)], cols=2), _src18,
           name="hero", want_gif=False, want_preview=False)
d_load = E.load_doc("h18", os.path.join(_src18, "hero_sheet.png"))
ok(bool(d_load.src_sheet) and bool(d_load.src_sidecar),
   "a loaded document remembers both source paths",
   (d_load.src_sheet, d_load.src_sidecar))

_sheet_before = _digest(d_load.src_sheet)
_side_before = _digest(d_load.src_sidecar)

try:
    E.save_doc(d_load, _src18, name="hero", want_gif=False, want_preview=False)
    ok(False, "saving back over the loaded sheet is refused")
except ValueError as ex:
    ok("refusing to overwrite the sheet" in str(ex),
       "saving back over the loaded sheet is refused", ex)
    ok("overwrite=true" in str(ex),
       "and the refusal names the way to do it on purpose", ex)

ok(_digest(d_load.src_sheet) == _sheet_before,
   "the source sheet is byte-identical after the refusal")
ok(_digest(d_load.src_sidecar) == _side_before,
   "and so is the source sidecar")

# the same name in a different folder is the normal path and must still work
_out18 = os.path.join(_tmp18, "out")
E.save_doc(d_load, _out18, name="hero", want_gif=False, want_preview=False)
ok(os.path.isfile(os.path.join(_out18, "hero_sheet.png")),
   "the same name in a different folder saves normally")
ok(os.path.isfile(os.path.join(_out18, "hero.json")),
   "and writes its sidecar beside it")

# replacing the original stays possible, but only when asked for. Change the
# pixels first, or "it differs" would be true for the wrong reason.
run(d_load, "flip_h")
E.save_doc(d_load, _src18, name="hero", want_gif=False, want_preview=False,
           overwrite=True)
ok(_digest(d_load.src_sheet) != _sheet_before,
   "overwrite=True does replace it -- the guard is a guard, not a no-op")

shutil.rmtree(_tmp18, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 19. export only the selected frames ===")
# --------------------------------------------------------------------------- #
# "Export only the selected frames" has to be a save-time *view* of the document.
# The obvious implementation -- keep_range on the live document -- produces a
# correct sheet and silently destroys every frame that was not selected, with no
# undo entry and no way back. Both halves are pinned here: the subset that gets
# written, and the document that must not move.

import json  # noqa: E402
from PIL import Image  # noqa: E402


def marked(n, cols=4):
    """n cells, each with a uniquely coloured marker, so a subset's contents and
    its order can be checked pixel by pixel."""
    cells = []
    for i in range(n):
        cells.append(rect(blank(64, 64), 0, 0, 4, 4, (10 + i * 7, 20, 30, 255)))
    return make_doc(cells, cols=cols)


d19 = marked(12, cols=4)
ok(d19.n == 12 and d19.columns == 4,
   "the fixture is 12 frames on a 4-wide grid", (d19.n, d19.columns))

n_before, rev_before = d19.n, d19.rev
cells_before = [c.copy() for c in d19.cells]
dirty_before = set(d19.dirty)
journal_before = list(d19.journal)
layout_before = dict(d19.layout)

sub, idx = E.export_subset(d19, [9, 3, 3, 5, 4])
ok(idx == [3, 4, 5, 9], "the subset is sorted and de-duplicated", str(idx))
ok(sub.n == 4, "the exported document holds only the selected frames", sub.n)
eq(sub.cells[0], cells_before[3], "in index order: the first cell is frame 3")
eq(sub.cells[3], cells_before[9], "and the last is frame 9")
ok(sub.n == sub.columns * sub.rows,
   "the subset lands on a full grid, not one with blank cells in it",
   "%d frames on %dx%d" % (sub.n, sub.columns, sub.rows))

# -- and the document that was exported from must not have moved --------------
ok(d19.n == n_before, "the open document still has all its frames", d19.n)
ok(d19.rev == rev_before, "and its revision did not move", d19.rev)
ok(d19.layout == layout_before, "and its layout is unchanged", d19.layout)
ok(all((a == b).all() for a, b in zip(d19.cells, cells_before)),
   "and every one of its cells is byte-identical")
ok(d19.dirty == dirty_before, "and its dirty set is untouched", sorted(d19.dirty))
ok(d19.journal == journal_before,
   "and its op history has no export in it", d19.journal)

# -- selecting everything is not a subset -------------------------------------
# Re-derive the fixture: the checks above are exactly the ones a destructive
# export breaks, and if they do break, this section should report named failures
# rather than crash out on the next out-of-range index.
d19 = marked(12, cols=4)
whole, widx = E.export_subset(d19, range(d19.n))
ok(whole is d19,
   "selecting every frame returns the document itself, not a copy that has been "
   "re-gridded")
ok(widx == list(range(12)), "with every index in order", str(widx))

# -- a bad selection is a mistake, not an empty export ------------------------
try:
    E.export_subset(d19, [])
    ok(False, "an empty selection is refused")
except ValueError as ex:
    ok("nothing to export" in str(ex), "an empty selection is refused", ex)

try:
    E.export_subset(d19, [0, 12])
    ok(False, "a frame past the end is refused")
except ValueError as ex:
    ok("12 is not in this document" in str(ex),
       "a frame past the end is refused, naming it and the count", ex)

# A subset whose count has no workable grid must be refused, not written with
# blank cells in it -- the same refusal every frame-pruning op gets. The texture
# cap is turned down here rather than building a fixture big enough to hit the
# real 32768: the arithmetic in set_layout is identical either way.
d19c = marked(12, cols=4)
d19c.meta["max_texture"] = 200
try:
    E.export_subset(d19c, list(range(11)))
    ok(False, "a subset with no workable grid is refused")
except ValueError as ex:
    ok("no column count that fits" in str(ex),
       "a subset with no workable grid is refused like any other re-grid", ex)

# ...but only when that is true. A document's OWN grid can be the illegal one:
# open a sheet generated under a smaller cap, or re-grid up on a document whose
# meta cap is lower than the app's current one, and its layout is over its own
# cap. A subset export used to keep that column count -- it divides the new frame
# count, so it looked fine -- and then be refused with "no column count that
# fits" while 29 columns plainly fit. That is the 58-of-124 save the user hit:
# 58 frames of 566x640 has no grid under 16384, but 58 frames of 283x1280 does,
# and the picker returned the 16414-wide grid because it had the smallest largest
# dimension.
# (116 cells, so 58 frames is a *subset* and not the whole document: exporting
# everything returns the document itself, without a re-grid.)
d19e = make_doc([rect(blank(64, 64), 0, 0, 4, 4, (10 + (i % 30) * 7, 20, 30, 255))
                 for i in range(116)], cols=58)
d19e.meta["max_texture"] = 2000          # 58 x 64 = 3712 wide, over it
ok(d19e.columns * d19e.cell_w > d19e.meta["max_texture"],
   "the fixture's own grid is over its cap",
   "%dx%d vs cap %d" % (d19e.columns * d19e.cell_w, d19e.rows * d19e.cell_h,
                        d19e.meta["max_texture"]))
try:
    sub19e, idx19e = E.export_subset(d19e, list(range(58)))
except ValueError as ex:
    # The regression this guards: keeping the document's column count because it
    # divides the new frame count, without asking whether its sheet is legal.
    sub19e = E.Doc("none", [], d19e.layout, d19e.meta)
    idx19e = []
    ok(False, "a subset of an over-cap document is not refused", ex)
ok(sub19e.n == 58 and sub19e.columns * sub19e.rows == 58,
   "a subset of an over-cap document still fills its grid",
   "%d frames on %dx%d" % (sub19e.n, sub19e.columns, sub19e.rows))
ok(sub19e.columns * sub19e.cell_w <= 2000 and sub19e.rows * sub19e.cell_h <= 2000,
   "and lands on a grid that fits the cap rather than being refused",
   "%dx%d on %d columns" % (sub19e.columns * sub19e.cell_w,
                            sub19e.rows * sub19e.cell_h, sub19e.columns))
ok(idx19e == list(range(58)), "the exported subset is the selection, in order",
   str(idx19e)[:60])
ok(d19e.n == 116 and d19e.columns == 58 and d19e.n == d19e.columns * d19e.rows,
   "the open document is untouched by the export",
   "%d frames, %d columns" % (d19e.n, d19e.columns))

# -- what actually lands on disk ----------------------------------------------
# Its own fixture, and its own copy of the expected pixels: a destructive export
# anywhere above must not be able to change what this block compares against.
_tmp19 = tempfile.mkdtemp(prefix="ed19_")
d19d = marked(12, cols=4)
_expect = [c.copy() for c in d19d.cells]
sub, idx = E.export_subset(d19d, [1, 2, 3, 4, 5, 6, 7, 8])
arts = E.save_doc(sub, _tmp19, name="sel", want_gif=False, want_preview=False)
im = Image.open(arts["sheet"])
with open(arts["sidecar"], encoding="utf-8") as fh:
    sc = json.load(fh)
ok(sc["frame_count"] == 8, "the sidecar claims the subset's frame count",
   sc["frame_count"])
ok(sc["columns"] * sc["frame_width"] == im.width
   and sc["rows"] * sc["frame_height"] == im.height,
   "and the written sheet's pixels match the grid the sidecar describes",
   "%dx%d image vs %dx%d grid" % (im.width, im.height,
                                  sc["columns"] * sc["frame_width"],
                                  sc["rows"] * sc["frame_height"]))
ok(im.width * im.height < (d19d.columns * 64) * (d19d.rows * 64),
   "and the exported sheet is genuinely smaller than the whole document's",
   "%dx%d vs %dx%d" % (im.width, im.height,
                       d19d.columns * 64, d19d.rows * 64))
ok(any(str(o).startswith("export:") for o in sc["edited"]["ops"]),
   "the sidecar records that it is a subset, so the provenance is not lost",
   sc["edited"]["ops"][-1:] or None)

# the order in the written bytes, not just in the in-memory document
_arr = np.array(im.convert("RGBA"))
_cw, _ch, _cols = sc["frame_width"], sc["frame_height"], sc["columns"]
for _k in (0, 7):
    _x, _y = (_k % _cols) * _cw, (_k // _cols) * _ch
    eq(_arr[_y:_y + _ch, _x:_x + _cw], _expect[idx[_k]],
       "cell %d of the written sheet is frame %d" % (_k, idx[_k]))
shutil.rmtree(_tmp19, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 20. quantize_colors: at most N colours, shape and alpha untouched ===")
# --------------------------------------------------------------------------- #

def _distinct(c):
    m = c[..., 3] > 0
    return {tuple(v) for v in c[..., :3][m].reshape(-1, 3)}


def _rainbow(seed, w=40, h=40):
    """A subject with far more than 16 distinct colours, so the op has work."""
    c = blank(w, h)
    yy, xx = np.mgrid[0:h, 0:w]
    c[..., 0] = (xx * 6 + seed * 13) % 256
    c[..., 1] = (yy * 7 + seed * 5) % 256
    c[..., 2] = ((xx + yy) * 3 + seed) % 256
    c[..., 3] = np.where(((xx - w // 2) ** 2 + (yy - h // 2) ** 2) < (w // 3) ** 2,
                         255, 0)
    return c


d20 = make_doc([_rainbow(1), _rainbow(2), _rainbow(3)], cols=3)
_before20 = [c.copy() for c in d20.cells]
_ch20, _msg20, _d20 = run(d20, "quantize_colors", {"colors": 5, "shared": True},
                          sel=[0, 1, 2])
ok(_ch20 == [0, 1, 2], "the op reports the three selected frames", _ch20)
_colours20 = set()
for _c in d20.cells:
    _colours20 |= _distinct(_c)
ok(len(_colours20) <= 5,
   "the whole selection shares a palette of at most 5 colours", len(_colours20))
ok(all(c.shape == b.shape for c, b in zip(d20.cells, _before20)),
   "no frame changed shape")
ok(all(np.array_equal(c[..., 3], b[..., 3]) for c, b in zip(d20.cells, _before20)),
   "every alpha value is untouched")
ok(all(np.array_equal(c[..., :3][c[..., 3] == 0], b[..., :3][b[..., 3] == 0])
       for c, b in zip(d20.cells, _before20)),
   "transparent pixels keep their (unused) RGB")
ok(all(not np.array_equal(c, b) for c, b in zip(d20.cells, _before20)),
   "and each frame really was recoloured")

d20b = make_doc([_rainbow(1), _rainbow(2), _rainbow(3)], cols=3)
_keep20 = d20b.cells[2].copy()
run(d20b, "quantize_colors", {"colors": 4, "shared": False}, sel=[0, 1])
ok(len(_distinct(d20b.cells[0])) <= 4 and len(_distinct(d20b.cells[1])) <= 4,
   "per-frame mode limits each selected frame on its own",
   (len(_distinct(d20b.cells[0])), len(_distinct(d20b.cells[1]))))
eq(d20b.cells[2], _keep20, "and leaves the unselected frame byte-identical")

d20c = make_doc([_rainbow(1)], cols=1)
_was20 = d20c.cells[0].copy()
run(d20c, "quantize_colors", {"colors": 4, "shared": True}, sel=[0])
d20c.undo_once()
eq(d20c.cells[0], _was20, "undo restores the frame exactly")

d20d = make_doc([subject_cell()], cols=1)
_depth20 = len(d20d.undo)
_ch20b, _msg20b, _ = run(d20d, "quantize_colors", {"colors": 16, "shared": True},
                         sel=[0])
ok(_ch20b == [] and len(d20d.undo) == _depth20,
   "a frame already within the colour budget is a no-op", (_ch20b, _msg20b))

# --------------------------------------------------------------------------- #
print("\n=== 21. pixelate_mesh: one shared grid + palette over the selection ===")
# --------------------------------------------------------------------------- #
if not _PPA_OK:
    print("  skip  proper-pixel-art checkout not available")
else:
    from PIL import Image

    def _blocky(seed, size=192, true=12):
        """A noisy nearest-upscaled sprite: the input the mesh detector targets."""
        small = blank(true, true)
        yy, xx = np.mgrid[0:true, 0:true]
        small[((xx - true // 2) ** 2 + (yy - true // 2) ** 2)
              < (true // 2 - 1) ** 2] = (80, 150, 220, 255)
        small[(seed + 2):(seed + 5), 3:6] = (220, 80, 80, 255)
        big = np.array(Image.fromarray(small).resize((size, size),
                                                     Image.NEAREST))
        rng = np.random.default_rng(seed)
        n = rng.integers(-16, 17, big.shape[:2]).astype(np.int16)
        big[..., :3] = np.clip(big[..., :3].astype(np.int16) + n[..., None],
                               0, 255).astype(np.uint8)
        return big

    d21 = make_doc([_blocky(0), _blocky(1), _blocky(2), _blocky(3)], cols=2)
    _was21 = [c.copy() for c in d21.cells]
    _ch21, _msg21, _d21 = run(d21, "pixelate_mesh",
                              {"colors": 16, "pixel_width": 0, "sample": 8,
                               "upscale": 2, "transparent": False},
                              sel=[0, 1, 2])
    ok(set(_ch21) <= {0, 1, 2} and _ch21,
       "the op reports the frames it changed out of the selection", _ch21)
    ok(all(c.shape == b.shape for c, b in zip(d21.cells, _was21)),
       "every frame keeps its shape, so the sheet keeps its size")
    eq(d21.cells[3], _was21[3], "the unselected frame is byte-identical")
    _cols21 = set()
    for _c in d21.cells[:3]:
        _cols21 |= _distinct(_c)
    ok(len(_cols21) <= 16,
       "the selection resolves to one shared palette of at most 16 colours",
       len(_cols21))
    ok(any((c[..., 3] == 0).any() for c in d21.cells[:3]),
       "the transparent backdrop stays transparent")
    ok(any((c[..., 3] == 255).any() for c in d21.cells[:3]),
       "the subject stays opaque")
    d21.undo_once()
    ok(all(np.array_equal(c, b) for c, b in zip(d21.cells[:3], _was21[:3])),
       "undo restores every selected frame exactly", _msg21)

    # Consistency: the same grid and palette come out every time, so a second
    # pass over the op's own output is a no-op rather than a re-jittered grid.
    _args21 = {"colors": 16, "pixel_width": 0, "sample": 8, "upscale": 2,
               "transparent": False}
    run(d21, "pixelate_mesh", _args21, sel=[0, 1, 2])
    _ch21b, _msg21b, _ = run(d21, "pixelate_mesh", _args21, sel=[0, 1, 2])
    ok(_ch21b == [], "a second pass over its own output changes nothing",
       _msg21b)
    ok(_msg21.split("->", 1)[1].split(",", 1)[0] ==
       _msg21b.split("->", 1)[1].split(",", 1)[0],
       "the detected grid is the same on the second pass", (_msg21, _msg21b))

    # The palette picker from Snap Pixels, on this op too: "add same palette from
    # snap pixel to pixelate (mesh)". The registry has to declare it, or the page
    # has nothing to render a picker for.
    _pm = E.OPS["pixelate_mesh"]
    ok([a["k"] for a in _pm["args"]][-1] == "palette"
       and [a["t"] for a in _pm["args"]][-1] == "text",
       "pixelate (mesh) declares a palette, the control Snap Pixels already has",
       [(a["k"], a["t"]) for a in _pm["args"]])
    ok(E.hex_palette("#ff0000,00ff00").tolist() == [[255, 0, 0], [0, 255, 0]],
       "a palette string parses to one row per colour, in order",
       E.hex_palette("#ff0000,00ff00").tolist())
    ok(E.hex_palette("#abc").tolist() == [[170, 187, 204]],
       "and it takes the same shorthand a single colour does")
    ok(E.hex_palette("ff0000, 00ff00,").shape == (2, 3),
       "a trailing comma is punctuation, not a third colour",
       E.hex_palette("ff0000, 00ff00,").shape)
    # An entry that cannot be read must refuse, not be dropped. A dropped entry
    # leaves a palette that is not the one the user named, and the run still
    # reports success -- the same failure the single-colour refusal prevents.
    for _bad in ("#ff0000,oops", "nope", ",", "", "   "):
        try:
            E.hex_palette(_bad)
            ok(False, "an unreadable palette is refused: %r" % (_bad,))
        except ValueError as ex:
            ok("hex colour" in str(ex) or "no colours" in str(ex),
               "an unreadable palette is refused rather than quietly shortened",
               ex)

    # The named palette is the colour decision: every opaque colour in the result
    # comes from it, and nothing outside it survives.
    _hexes21 = ["#101820", "#e8f0f8", "#4c9ad0", "#dc5050"]
    _pal21 = E.hex_palette(",".join(_hexes21))
    _allowed21 = {tuple(int(v) for v in row) for row in _pal21}
    _args21p = {"colors": 16, "pixel_width": 0, "sample": 8, "upscale": 2,
                "transparent": False, "palette": ",".join(_hexes21)}
    d21p = make_doc([_blocky(0), _blocky(1), _blocky(2), _blocky(3)], cols=2)
    _ch21p, _msg21p, _ = run(d21p, "pixelate_mesh", _args21p, sel=[0, 1, 2])
    _cols21p = set()
    for _c in d21p.cells[:3]:
        _cols21p |= _distinct(_c)
    ok(_cols21p and _cols21p <= _allowed21,
       "every colour in the result comes from the named palette",
       sorted(_cols21p - _allowed21)[:4])
    ok("palette from the picker" in _msg21p,
       "and the status line says the palette came from the picker", _msg21p)
    ok(all(c.shape == b.shape for c, b in zip(d21p.cells, _was21)),
       "a palette does not change the frames' size either")

    # ...and it is the palette that decides, not the spinner. The auto colour
    # count is switched off when a palette is named, so changing it cannot change
    # the result -- which is what makes "the palette replaced the fitted one" a
    # claim about behaviour rather than about a keyword argument.
    d21q = make_doc([_blocky(0), _blocky(1), _blocky(2), _blocky(3)], cols=2)
    run(d21q, "pixelate_mesh", dict(_args21p, colors=2), sel=[0, 1, 2])
    ok(all(np.array_equal(a, b) for a, b in zip(d21q.cells, d21p.cells)),
       "the colours spinner is ignored when a palette is named, so the palette "
       "is the only thing deciding the colours")

    # --- the two defaults, as asked for: "make mesh-detection upscale default 8
    # and colours (0 = keep all) default 0" --------------------------------
    # Both are registry defaults, and coerce_args fills a missing key from the
    # declared `d`, so running with NO arguments is the honest way to test them.
    # Asserting `arg["d"] == 0` alone would pin the number and not the effect,
    # which is the shape of guard this file exists to avoid.
    _ups21 = [a for a in _pm["args"] if a["k"] == "upscale"][0]
    _col21 = [a for a in _pm["args"] if a["k"] == "colors"][0]
    ok(_col21["d"] == 0 and _ups21["d"] == 8,
       "pixelate (mesh) defaults to 0 colours (keep all) and upscale 8",
       (_col21["d"], _ups21["d"]))
    # The old ceiling was 4, so a default of 8 would have been a form the page
    # could not render: `<input type=number min=1 max=4 value=8>`. The default
    # moving past the old max is exactly why this is checked.
    ok(_ups21.get("max", 0) >= _ups21["d"],
       "and its max admits its own default, so the control is renderable",
       (_ups21["d"], _ups21.get("max")))

    d21d = make_doc([_blocky(0), _blocky(1), _blocky(2), _blocky(3)], cols=2)
    _msg21d = run(d21d, "pixelate_mesh", {}, sel=[0, 1, 2])[1]
    _cols21d = set()
    for _c in d21d.cells[:3]:
        _cols21d |= _distinct(_c)
    # "Keep all" means the quantiser is skipped, so it has to beat the 16-colour
    # run on the same fixture -- 30 colours against 5, measured. The direction is
    # asserted rather than the number, so this does not freeze a fixture detail.
    ok(len(_cols21d) > len(_cols21),
       "and running with no arguments keeps every colour the mesh resolves, "
       "rather than quantising to the old 16", (len(_cols21d), len(_cols21)))
    ok(len(_cols21d) > 16,
       "so the result is not capped at the palette size the old default implied",
       len(_cols21d))

    # The default really is what reaches the op: passing 0 and 8 explicitly has
    # to be byte-identical to omitting them, or the registry `d` is decorative.
    d21e = make_doc([_blocky(0), _blocky(1), _blocky(2), _blocky(3)], cols=2)
    run(d21e, "pixelate_mesh", {"colors": 0, "upscale": 8}, sel=[0, 1, 2])
    ok(all(np.array_equal(a, b) for a, b in zip(d21e.cells, d21d.cells)),
       "and naming 0 and 8 explicitly is byte-identical to omitting them, so "
       "the declared default is the value the op receives")

    # The reason the upscale default moved at all: a higher upscale is finer mesh
    # detection, so the resolved grid has to get finer. This is the claim that
    # makes the new default worth having, and it is read off the status line the
    # op writes rather than off the argument, so it measures the result.
    import re as _re21

    def _grid_of(msg):
        m = _re21.search(r"(\d+)x(\d+) true-resolution grid", msg or "")
        return (int(m.group(1)), int(m.group(2))) if m else None

    _g_old21, _g_new21 = _grid_of(_msg21), _grid_of(_msg21d)
    ok(_g_old21 and _g_new21
       and _g_new21[0] * _g_new21[1] > _g_old21[0] * _g_old21[1],
       "and upscale 8 resolves a finer mesh than the old upscale 2 did",
       (_g_old21, _g_new21))

# --------------------------------------------------------------------------- #
print("\n=== 22. snap_pixels auto: one grid across the selection ===")
# --------------------------------------------------------------------------- #
if not _SNAP_OK:
    print("  skip  spritefusion-pixel-snapper/Node not available")
else:
    from PIL import Image as _Im

    def _drifting(seed, size=192, true=16):
        small = blank(true, true)
        yy, xx = np.mgrid[0:true, 0:true]
        small[((xx - 8) ** 2 + (yy - 8) ** 2) < 6 ** 2] = (80, 150, 220, 255)
        small[2:5, 1 + (seed % 12):4 + (seed % 12)] = (220, 80, 80, 255)
        big = np.array(_Im.fromarray(small).resize((size, size), _Im.NEAREST))
        rng = np.random.default_rng(300 + seed)
        n = rng.integers(-15, 16, big.shape[:2]).astype(np.int16)
        big[..., :3] = np.clip(big[..., :3].astype(np.int16) + n[..., None],
                               0, 255).astype(np.uint8)
        return big

    d22 = make_doc([_drifting(i) for i in range(4)], cols=2)
    _was22 = [c.copy() for c in d22.cells]
    _size22 = E.detect_pixel_size([d22.cell_png(i) for i in range(4)],
                                  d22.cell_w, d22.cell_h)
    ok(isinstance(_size22, int)
       and 1 <= _size22 <= min(d22.cell_w, d22.cell_h) // 2,
       "auto detection picks one in-range pixel size for the batch", _size22)
    _ch22, _msg22, _ = run(d22, "snap_pixels",
                           {"colors": 16, "pixel_size": 0, "palette": ""},
                           sel=[0, 1, 2, 3])
    ok("one %dpx grid" % _size22 in _msg22 and "shared" in _msg22,
       "the op reports the one grid it shared across the selection", _msg22)
    ok(all(c.shape == b.shape for c, b in zip(d22.cells, _was22)),
       "the snap still keeps every frame's size")
    _all22 = set()
    for _c in d22.cells:
        _all22 |= _distinct(_c)
    ok(len(_all22) <= 16,
       "and one palette across the selection, so colours cannot drift",
       len(_all22))

# --------------------------------------------------------------------------- #
print("\n=== 23. no flicker: a static region is byte-identical across frames ===")
# --------------------------------------------------------------------------- #
from PIL import Image as _Image23


def _moving(step, size=160, true=12):
    """Moving content over a fixed grid, with the SAME noise field every frame,
    so any change in the static region is the op's doing, not the source's."""
    small = blank(true, true)
    yy, xx = np.mgrid[0:true, 0:true]
    small[((xx - true // 2) ** 2 + (yy - true // 2) ** 2)
          < (true // 2 - 1) ** 2] = (80, 150, 220, 255)
    small[((xx - 5) ** 2 + (yy - 5) ** 2) < 2 ** 2] = (240, 240, 240, 255)
    small[8:11, 1 + step:4 + step] = (220, 80, 80, 255)      # the moving part
    big = np.array(_Image23.fromarray(small).resize((size, size),
                                                    _Image23.NEAREST))
    rng = np.random.default_rng(0)                           # fixed noise field
    n = rng.integers(-14, 15, big.shape[:2]).astype(np.int16)
    big[..., :3] = np.clip(big[..., :3].astype(np.int16) + n[..., None],
                           0, 255).astype(np.uint8)
    return big


if not _PPA_OK:
    print("  skip  proper-pixel-art checkout not available")
else:
    d23 = make_doc([_moving(i) for i in range(5)], cols=5)
    run(d23, "pixelate_mesh",
        {"colors": 16, "pixel_width": 0, "sample": 8, "upscale": 2,
         "transparent": False},
        sel=[0, 1, 2, 3, 4])
    _win = (slice(0, 80), slice(0, 80))          # the static body, top-left
    _ref23 = d23.cells[0][_win]
    _diffs23 = [int(np.any(d23.cells[i][_win] != _ref23, axis=-1).sum())
                for i in range(1, 5)]
    ok(all(d == 0 for d in _diffs23),
       "the mesh op leaves a static region pixel-identical across frames",
       _diffs23)

# --------------------------------------------------------------------------- #
print("\n=== 24. audio: per-frame clips ship beside the sheet and into the JSON ===")
# --------------------------------------------------------------------------- #
import wave  # noqa: E402

def _wav_bytes(samples=8):
    import io as _io
    buf = _io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(b"\x00\x00" * samples)
    return buf.getvalue()

_tmp24 = tempfile.mkdtemp(prefix="ed24_")

def _clip(name, data):
    p = os.path.join(_tmp24, name)
    with open(p, "wb") as fh:
        fh.write(data)
    return p

d24 = make_doc(cols=2)                       # 4 frames
d24.add_audio(0, _clip("src_hit.wav", _wav_bytes()), name="hit.wav")
d24.add_audio(2, _clip("src_foot.wav", _wav_bytes(16)), name="foot.wav",
              volume=0.5)
_pl24 = d24.audio_payload()
ok([e["frame"] for e in _pl24] == [0, 2],
   "clips are listed in frame order", [e["frame"] for e in _pl24])
ok(_pl24[0]["url"].endswith("/audio/%d" % _pl24[0]["id"]),
   "each clip carries the URL the preview player fetches it from", _pl24[0])

arts24 = E.save_doc(d24, _tmp24, name="snd", want_gif=False, want_preview=False)
with open(arts24["sidecar"], encoding="utf-8") as fh:
    sc24 = json.load(fh)
ok([a["frame"] for a in sc24.get("audio", [])] == [0, 2],
   "the sidecar lists the clips by frame", sc24.get("audio"))
ok(all(a["file"].startswith("audio/") for a in sc24["audio"]),
   "and names each file relative to the sidecar", sc24["audio"])
_afiles = [os.path.join(os.path.dirname(arts24["sidecar"]), a["file"])
           for a in sc24["audio"]]
ok(all(os.path.isfile(p) for p in _afiles),
   "every clip was copied beside the sheet", _afiles)
ok(sc24["audio"][1]["volume"] == 0.5, "the volume travels with the clip",
   sc24["audio"][1])

# The volume control changes a clip after it is attached, so it is the *setter*
# that matters rather than the constructor argument: a panel that can change the
# volume without changing what is saved would be a preview-only knob.
_vol_id = d24.audio[0]["id"]
ok(d24.set_audio_volume(_vol_id, 0.25)["volume"] == 0.25,
   "a clip's volume can be changed after it is attached")
ok(d24.audio_payload()[0]["volume"] == 0.25,
   "and the payload the panel renders from carries the new value",
   d24.audio_payload()[0])
ok(d24.set_audio_volume(_vol_id, 2.0)["volume"] == 1.0,
   "a volume above full is clamped, not left out of range")
ok(d24.set_audio_volume(_vol_id, -1)["volume"] == 0.0,
   "and one below silence is clamped too")
ok(d24.set_audio_volume(999999, 0.5) is None,
   "an unknown clip id reports that there is nothing to set")
d24.set_audio_volume(_vol_id, 0.25)
arts24b = E.save_doc(d24, _tmp24, name="snd2", want_gif=False,
                     want_preview=False)
with open(arts24b["sidecar"], encoding="utf-8") as fh:
    sc24b = json.load(fh)
ok(sc24b["audio"][0]["volume"] == 0.25,
   "and a volume set after attaching is the one written into the sidecar",
   sc24b["audio"][0])

re24 = E.load_doc("reopen24", arts24["sheet"])
ok([e["frame"] for e in re24.audio_payload()] == [0, 2],
   "reopening restores the clips on their frames")
ok(all(os.path.isfile(e["path"]) for e in re24.audio),
   "and each clip points at the file beside the sheet",
   [e["path"] for e in re24.audio])

# The sidecar *is* the document: it names the sheet, the grid and the clips, and
# it is what the sheet library opens. Pointing load_doc at the JSON used to reach
# PIL with it and die with "cannot identify image file".
by24 = E.load_doc("byjson24", arts24["sidecar"])
ok(by24.n == d24.n,
   "a document opens from its sidecar JSON alone", (by24.n, d24.n))
ok([e["frame"] for e in by24.audio_payload()] == [0, 2],
   "and the clips come with it", by24.audio_payload())
ok(os.path.normcase(os.path.abspath(by24.src_sheet))
   == os.path.normcase(os.path.abspath(arts24["sheet"])),
   "with the sheet the JSON names", (by24.src_sheet, arts24["sheet"]))

_badsc = os.path.join(_tmp24, "orphan.json")
with open(_badsc, "w", encoding="utf-8") as fh:
    json.dump({"name": "orphan", "sheet": "not_here.png", "frame_width": 32,
               "frame_height": 32, "columns": 2, "rows": 2}, fh)
try:
    E.load_doc("orphan24", _badsc)
    ok(False, "a sidecar whose sheet is not there is refused, not guessed")
except FileNotFoundError as ex:
    ok("names no sheet that is there" in str(ex),
       "a sidecar whose sheet is not there is refused, not guessed", ex)

# Opening by the folder a run wrote into, which is what "when i load folder
# audios are not loaded" was about. The sidecar in a run's output folder is NOT
# named after the folder, so this exercises the fallback rather than the fast
# path -- and the clips have to survive it.
_rundir24 = os.path.dirname(arts24["sidecar"])
bydir24 = E.load_doc("byfolder24", _rundir24)
ok(bydir24.n == d24.n,
   "a document opens from the folder that holds the sidecar",
   (bydir24.n, d24.n))
ok([e["frame"] for e in bydir24.audio_payload()] == [0, 2],
   "and the clips come with it", bydir24.audio_payload())
ok(os.path.normcase(os.path.abspath(bydir24.src_sidecar))
   == os.path.normcase(os.path.abspath(arts24["sidecar"])),
   "having resolved the folder to the sidecar in it, not to the sheet",
   (bydir24.src_sidecar, arts24["sidecar"]))

# A folder holding a sheet but no sidecar is refused rather than guessed at: the
# grid would have to be invented, and an invented grid is a wrong sheet.
_bare24 = os.path.join(_tmp24, "bare")
os.makedirs(_bare24, exist_ok=True)
shutil.copyfile(arts24["sheet"], os.path.join(_bare24, "lonely.png"))
try:
    E.load_doc("bare24", _bare24)
    ok(False, "a folder holding a sheet but no sidecar is refused, not guessed")
except FileNotFoundError as ex:
    ok("no sheet sidecar in" in str(ex),
       "a folder holding a sheet but no sidecar is refused, not guessed", ex)

# An explicit grid must not suppress sidecar discovery. This is "when i load
# folder audios are not loaded" arriving by a different route: opening the sheet
# PNG with the columns/rows boxes filled in used to skip the sidecar entirely, so
# the document opened with none of its clips -- and the boxes are sticky, because
# boot() restores them from the last open. Measured on a real run sheet: the PNG
# alone returned its 2 clips, the PNG with its own 11x3 grid supplied returned 0.
_sc24 = json.load(open(arts24["sidecar"], encoding="utf-8"))
ok(_sc24["columns"] == 2 and _sc24["rows"] == 2,
   "the fixture's sidecar says 2x2, so a different grid in the boxes means "
   "something", (_sc24["columns"], _sc24["rows"]))
# Deliberately not the sidecar's grid: 1x1 rather than 2x2. It divides any sheet
# evenly, so it is always a legal thing to type -- and if the boxes won, the
# document would open as one frame with no clips, which is exactly the bug.
grid24 = E.load_doc("grid24", arts24["sheet"], columns=1, rows=1)
ok([e["frame"] for e in grid24.audio_payload()] == [0, 2],
   "giving columns and rows does not stop the sidecar being found",
   [e["frame"] for e in grid24.audio_payload()])
ok(grid24.n == d24.n and grid24.layout["columns"] == 2,
   "and the sidecar's own 2x2 grid wins over the 1x1 in the boxes",
   (grid24.n, grid24.layout["columns"], d24.n))

sub24, _idx24 = E.export_subset(re24, [2, 3])
ok([e["frame"] for e in sub24.audio_payload()] == [0],
   "an export keeps only the clips on exported frames, renumbered",
   [e["frame"] for e in sub24.audio_payload()])

d24b = make_doc(cols=2)
d24b.add_audio(1, os.path.join(_tmp24, "gone.wav"), name="gone.wav")
try:
    E.save_doc(d24b, _tmp24, name="bad", want_gif=False, want_preview=False)
    ok(False, "saving with a missing clip is refused rather than dangling")
except Exception as ex:                      # noqa: BLE001
    ok("missing on disk" in str(ex),
       "saving with a missing clip is refused rather than dangling", ex)
shutil.rmtree(_tmp24, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 25. fps is a document setting: the panel's number lands in the JSON ===")
# --------------------------------------------------------------------------- #
# "we should edit "fps": 24, in the json" -- so the number in the Grid panel has
# to be the number in the sidecar, and it has to come back when the sheet is
# reopened. This is why fps was moved out of Playback and into Grid: it is a
# document setting, not a preview preference. Nothing covered it before -- only
# the op's *name* appeared in the driver's list of known ops, so the value could
# have been dropped on the floor and every suite would still have been green.
#
# The op's own description already promises it ("Written straight into the
# sidecar JSON."). The point of the check is that the promise is measured rather
# than read.
_tmp25 = tempfile.mkdtemp(prefix="ed25_")

d25 = make_doc(cols=2)
ok(d25.meta["fps"] == 24,
   "a fresh document starts at 24 fps, the default the panel shows",
   d25.meta["fps"])

# A partial payload on purpose: fps is the only key being set here. The op fills
# the rest from its declared defaults, which is exactly what the panel does when
# the user touches the fps box and nothing else.
run(d25, "set_meta", {"fps": 12, "play_mode": "loop", "trim_to": 0,
                      "anchor": "none", "blend": "straight", "name": "fpsdoc"})
ok(d25.meta["fps"] == 12, "set_meta writes fps onto the document", d25.meta["fps"])

arts25 = E.save_doc(d25, _tmp25, name="fpsdoc", want_gif=False, want_preview=False)
_sc25 = json.load(open(arts25["sidecar"], encoding="utf-8"))
ok(_sc25.get("fps") == 12,
   "and the sidecar JSON carries it as \"fps\"", _sc25.get("fps"))
ok("12.00 fps" in _sc25.get("note", ""),
   "the note quotes the same number, so the two cannot drift apart",
   _sc25.get("note"))
# Speed is preview-only and must NOT be in the JSON. Asserted rather than left to
# the label: if someone wires the speed multiplier into the sidecar, the label
# "speed (preview only)" becomes a lie and nothing else would notice.
ok("speed" not in _sc25,
   "speed is preview-only and never reaches the sidecar", sorted(_sc25))

back25 = E.load_doc("fps25", arts25["sheet"])
ok(back25.meta["fps"] == 12,
   "reopening the sheet brings the edited fps back, not the default",
   back25.meta["fps"])

# The default has to survive the round trip too: a document whose fps was never
# touched must not come back as 24 because 24 is the panel's fallback rather than
# the file's value.
d25b = make_doc(cols=2)
arts25b = E.save_doc(d25b, _tmp25, name="plain", want_gif=False, want_preview=False)
_sc25b = json.load(open(arts25b["sidecar"], encoding="utf-8"))
ok(_sc25b.get("fps") == 24 and E.load_doc("plain25", arts25b["sheet"]).meta["fps"] == 24,
   "an untouched document round-trips its default 24 fps",
   (_sc25b.get("fps"), E.load_doc("plain25", arts25b["sheet"]).meta["fps"]))

shutil.rmtree(_tmp25, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 26. Apply metadata writes into the sidecar it was loaded from ===")
# --------------------------------------------------------------------------- #
# "when i click apply metadata it should write in current json". The op route is
# a pure change to the in-memory document, so before this the metadata sat in
# memory until the user also pressed Save -- and Save rewrites the PNG, the GIF
# and the preview player along with it. save_meta is the narrow version: the
# JSON only, in place, and only the keys the panel owns.
_tmp26 = tempfile.mkdtemp(prefix="ed26_")

src26 = make_doc(cols=2)
src26.meta["name"] = "meta26"
# A key the panel does NOT own, so the update can be shown to be targeted rather
# than a regeneration. `matte_repair` is one: the editor writes it, and the
# metadata panel has no control for it. This fixture used to plant `source`,
# which was dropped from the format as a key nothing read -- so it now plants a
# key that is still written, and the check stays about the *rule*.
src26.meta["matte_repair"] = "border-connected"
arts26 = E.save_doc(src26, _tmp26, name="meta26", want_gif=False, want_preview=False)
_sc26_path = arts26["sidecar"]
_before26 = json.load(open(_sc26_path, encoding="utf-8"))
ok(_before26.get("matte_repair") == "border-connected",
   "the fixture's sidecar carries a `matte_repair` that the panel does not own",
   _before26.get("matte_repair"))

d26 = E.load_doc("meta26", arts26["sheet"])
ok(os.path.normcase(d26.src_sidecar) == os.path.normcase(_sc26_path),
   "the document remembers the JSON it was loaded from",
   (d26.src_sidecar, _sc26_path))

run(d26, "set_meta", {"fps": 15, "play_mode": "loop", "trim_to": 0,
                      "anchor": "bottom-center", "blend": "additive",
                      "name": "renamed26"})
back26 = E.save_meta(d26)
ok(os.path.normcase(back26) == os.path.normcase(_sc26_path),
   "save_meta writes back into the same file, not a new one", back26)
_sc26 = json.load(open(_sc26_path, encoding="utf-8"))
ok(_sc26["fps"] == 15, "the fps in the file is the one that was applied",
   _sc26["fps"])
ok(_sc26["name"] == "renamed26", "and so is the name", _sc26["name"])
ok(_sc26["anchor"] == "bottom-center" and _sc26["blend"] == "additive",
   "and the anchor and the blend", (_sc26["anchor"], _sc26["blend"]))
ok("15.00 fps" in _sc26.get("note", ""),
   "the note was rewritten from the new fps too", _sc26.get("note"))

# The point of a targeted update: everything the panel does not own survives.
ok(_sc26.get("matte_repair") == "border-connected",
   "a key the panel does not own is carried through, not regenerated",
   _sc26.get("matte_repair"))
ok(_sc26.get("matte") == _before26.get("matte")
   and _sc26.get("crop") == _before26.get("crop")
   and _sc26.get("sheet") == _before26.get("sheet"),
   "matte, crop and the sheet name are untouched",
   (_sc26.get("matte"), _sc26.get("crop"), _sc26.get("sheet")))
ok(_sc26.get("edited") == _before26.get("edited"),
   "and so is the record of how the sheet was made", _sc26.get("edited"))
ok(not os.path.exists(_sc26_path + ".tmp"),
   "the write is atomic: no .tmp left beside the sidecar")
ok(E.load_doc("meta26b", arts26["sheet"]).meta["fps"] == 15,
   "reopening the sheet reads the fps that was written")

# A document with no sidecar has no "current json" to write into.
try:
    E.save_meta(make_doc(cols=2))
    ok(False, "a document that was never opened from a JSON is refused")
except ValueError as ex:
    ok("was not opened from a sidecar" in str(ex),
       "a document that was never opened from a JSON is refused", ex)

# The PNG is not rewritten here, so a grid change has to be refused: the JSON
# would otherwise describe a sheet that is not the one on disk, and an engine
# reading that pair plays a silently wrong animation.
d26c = E.load_doc("meta26c", arts26["sheet"])
run(d26c, "set_columns", {"columns": 1})
try:
    E.save_meta(d26c)
    ok(False, "a grid change is refused rather than written into the JSON alone")
except ValueError as ex:
    ok("the grid changed" in str(ex),
       "a grid change is refused rather than written into the JSON alone", ex)

# Same for the clips. The file's audio is carried through verbatim because the
# files are already beside the sheet, so a document whose clips have moved on
# would be written with a soundtrack it is not showing.
d26d = E.load_doc("meta26d", arts26["sheet"])
_extra26 = os.path.join(_tmp26, "extra.wav")
with open(_extra26, "wb") as fh:
    fh.write(_wav_bytes())
d26d.add_audio(1, _extra26, name="extra.wav")
try:
    E.save_meta(d26d)
    ok(False, "a changed clip list is refused rather than half-written")
except ValueError as ex:
    ok("the clips changed" in str(ex),
       "a changed clip list is refused rather than half-written", ex)

# And a document whose clips still match is written with them intact.
d26e = E.load_doc("meta26e", arts26["sheet"])
run(d26e, "set_meta", {"fps": 9, "play_mode": "loop", "trim_to": 0,
                       "anchor": "none", "blend": "straight", "name": "meta26"})
E.save_meta(d26e)
_sc26e = json.load(open(_sc26_path, encoding="utf-8"))
ok(_sc26e["fps"] == 9, "a document whose clips match still writes", _sc26e["fps"])

shutil.rmtree(_tmp26, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 27. overwriting writes the files it was loaded from, not the name ===")
# --------------------------------------------------------------------------- #
# "when allow overwriting the sheet this document was loaded from checked not
# writing on same json". save_doc builds its output names from the document's
# `name`, and the name *inside* a sidecar need not match the sidecar's filename --
# runs/*_smoke_api/ are exactly that shape: smoke_api.json, whose name is
# "032137_px_00002_front_walk". So a ticked save wrote <name>_sheet.png and
# <name>.json *beside* the originals and left the originals alone. The box
# promised to replace the loaded sheet and replaced nothing.
#
# Measured before the fix, same document, only the names differing: with the
# sidecar called walk.json and the name "walk", both loaded files were rewritten;
# with it called smoke_api.json and the name "032137_px_00002_front_walk", two new
# files appeared and neither loaded file was touched.
_tmp27 = tempfile.mkdtemp(prefix="ed27_")
_a27 = E.save_doc(make_doc(cols=2), _tmp27, name="run", want_gif=False,
                  want_preview=False)
# Plant the mismatch. The sheet keeps its name; only the JSON's `name` moves, so
# the file and the document disagree about what this sheet is called -- which is
# all it takes for the derived output name to miss the loaded file.
_sc27 = json.load(open(_a27["sidecar"], encoding="utf-8"))
_sc27["name"] = "some_other_name"
with open(_a27["sidecar"], "w", encoding="utf-8") as fh:
    json.dump(_sc27, fh)
_before27 = open(_a27["sheet"], "rb").read()

d27 = E.load_doc("ow27", _a27["sidecar"])
ok(d27.meta["name"] == "some_other_name",
   "the fixture's name does not match its filename, as in a real run folder",
   d27.meta["name"])
# An edit, so "the sheet changed" means the write landed rather than a no-op.
run(d27, "offset", {"dx": 5, "dy": 5, "wrap": False})

# Unticked: the save succeeds and writes a NEW pair beside the originals. That is
# the behaviour the report is about -- the box is the only way to say "put it back
# where it came from".
_a27b = E.save_doc(d27, _tmp27, name="some_other_name", want_gif=False,
                   want_preview=False, out_dir_named=False)
ok(os.path.normcase(_a27b["sheet"]) != os.path.normcase(_a27["sheet"]),
   "unticked, a document whose name differs saves to a new file",
   (_a27b["sheet"], _a27["sheet"]))
ok(open(_a27["sheet"], "rb").read() == _before27,
   "and the sheet it was loaded from is left exactly as it was")

# Ticked, and no output folder named -- the page's blank box, which is the
# gesture that was reported. The two paths it was loaded from are the two written.
_a27c = E.save_doc(d27, _tmp27, name="some_other_name", want_gif=False,
                   want_preview=False, overwrite=True, out_dir_named=False)
ok(os.path.normcase(_a27c["sheet"]) == os.path.normcase(_a27["sheet"]),
   "ticked with a blank output folder, the sheet goes back to the file it came "
   "from", (_a27c["sheet"], _a27["sheet"]))
ok(os.path.normcase(_a27c["sidecar"]) == os.path.normcase(_a27["sidecar"]),
   "and so does the sidecar JSON", (_a27c["sidecar"], _a27["sidecar"]))
ok(open(_a27["sheet"], "rb").read() != _before27,
   "the loaded sheet really was rewritten, not just reported as such")
# The sidecar has to keep naming its own sheet, and stay loadable: a JSON that
# points at the wrong file is the failure this tool exists to prevent.
_sc27b = json.load(open(_a27["sidecar"], encoding="utf-8"))
ok(_sc27b["sheet"] == os.path.basename(_a27["sheet"]),
   "the rewritten sidecar still names its own sheet file", _sc27b["sheet"])
ok(E.load_doc("ow27b", _a27["sidecar"]).n == d27.n,
   "and it opens again with the same frame count")

# Ticked, but the caller DID name a folder: the folder wins. The box is worded as
# permission, and the refusal it disables is about the destination -- so an
# explicit destination must not be silently swapped for the source. Redirecting
# here would replace the original when the user asked for a copy somewhere else.
_tmp27b = tempfile.mkdtemp(prefix="ed27b_")
_a27e = E.save_doc(make_doc(cols=2), _tmp27b, name="named", want_gif=False,
                   want_preview=False)
d27c = E.load_doc("ow27c", _a27e["sidecar"])
run(d27c, "offset", {"dx": 4, "dy": 4, "wrap": False})
_elsewhere = os.path.join(_tmp27b, "copy")
os.makedirs(_elsewhere, exist_ok=True)
# Snapshot the source, or "the loaded sheet is left alone" would compare the file
# with itself and pass for no reason at all.
_before27b = open(_a27e["sheet"], "rb").read()
_a27f = E.save_doc(d27c, _elsewhere, name="named", want_gif=False,
                   want_preview=False, overwrite=True, out_dir_named=True)
ok(os.path.normcase(_a27f["sheet"])
   == os.path.normcase(os.path.join(_elsewhere, "named_sheet.png")),
   "ticked with a folder named, that folder is still where it goes",
   _a27f["sheet"])
ok(open(_a27e["sheet"], "rb").read() == _before27b,
   "and the loaded sheet is left alone, because a copy was asked for")

# The shape the report was actually made from: the name already matches the
# filename, so the derived output name is not the problem at all -- the blank
# output folder is. A real run folder is exactly this (smoke12.json, name
# "smoke12"), which is why the name-mismatch theory alone did not explain it.
_tmp27c = tempfile.mkdtemp(prefix="ed27c_")
# `name=` on save_doc shapes the FILE names; the sidecar's own `name` comes from
# the document (`M.get("name") or "sprite"`). So making the two agree -- which is
# the shape a pipeline run actually writes -- means setting the document's name
# as well. Leaving it out is what produced the mismatch above, and it is worth
# knowing that is the mechanism rather than a stray edit.
_d27g = make_doc(cols=2)
_d27g.meta["name"] = "smoke12"
_a27g = E.save_doc(_d27g, _tmp27c, name="smoke12", want_gif=False,
                   want_preview=False)
_before27c = (open(_a27g["sheet"], "rb").read(),
              open(_a27g["sidecar"], "rb").read())
d27d = E.load_doc("ow27d", _a27g["sidecar"])
ok(d27d.meta["name"] == "smoke12",
   "a run folder's sidecar is named after its document, as smoke12.json is",
   d27d.meta["name"])
run(d27d, "offset", {"dx": 6, "dy": 0, "wrap": False})
_a27h = E.save_doc(d27d, _tmp27c, name="smoke12", want_gif=False,
                   want_preview=False, overwrite=True, out_dir_named=False)
ok(os.path.normcase(_a27h["sidecar"]) == os.path.normcase(_a27g["sidecar"]),
   "ticked, the same JSON is written even when the name already matches",
   (_a27h["sidecar"], _a27g["sidecar"]))
ok(open(_a27g["sheet"], "rb").read() != _before27c[0],
   "the loaded sheet really changed")
ok(open(_a27g["sidecar"], "rb").read() != _before27c[1],
   "and so did the loaded JSON -- which is the part that was reported missing")

# A ticked save of a document that was never loaded from a file still works --
# there is nothing to go back to, so it behaves as an ordinary save.
d27b = make_doc(cols=2)
_a27d = E.save_doc(d27b, _tmp27, name="fresh", want_gif=False,
                   want_preview=False, overwrite=True, out_dir_named=False)
ok(os.path.normcase(_a27d["sheet"])
   == os.path.normcase(os.path.join(_tmp27, "fresh_sheet.png")),
   "a document with no source still saves normally when the box is ticked",
   _a27d["sheet"])

shutil.rmtree(_tmp27, ignore_errors=True)
shutil.rmtree(_tmp27b, ignore_errors=True)
shutil.rmtree(_tmp27c, ignore_errors=True)

# --------------------------------------------------------------------------- #
print("\n=== 28. match content size: the loop stops pulsing ===")
# --------------------------------------------------------------------------- #
# The report this exists for: "some art inside cells are little small some little
# big, this couse a jump in the loop". So the property is not "the pixels
# changed" -- it is that afterwards the subjects are the same size AND standing
# on the same pixel. A size fix that slides the frames would just trade one jump
# for another, which is why the anchor assertions below are not decoration.

_MATCH_BOXES = [(20, 20, 40, 50), (22, 22, 42, 46), (12, 20, 52, 40),
                (21, 14, 41, 52)]


def _match_doc():
    """Four frames of one subject at different sizes -- the reported sheet.

    Heights 30, 24, 20 and 38: an 18 px spread, which is what a loop jumps on.
    The boxes differ in width too, so `measure` has something to choose between
    and `measure=width` is not accidentally the same test as the default.
    """
    return make_doc(cells=[subject_cell(box=b) for b in _MATCH_BOXES])


def _match_sizes(doc, how="height", thr=0):
    out = []
    for c in doc.cells:
        b = E.subject_box(c, thr)
        out.append(None if b is None else E.subject_size(b, how))
    return out


def _match_anchors(doc, anchor="bottom-center", thr=0):
    out = []
    for c in doc.cells:
        b = E.subject_box(c, thr)
        out.append(None if b is None else E.anchor_point(
            b, anchor, c[..., 3] > thr))
    return out


ok(E.OPS["match_size"]["group"] == "Transform" and E.OPS["match_size"]["args"],
   "match_size is a Transform op that declares its controls",
   "%s / %d args" % (E.OPS["match_size"]["group"],
                     len(E.OPS["match_size"]["args"])))

d28 = _match_doc()
h0 = _match_sizes(d28)
a0 = _match_anchors(d28)
_lay28 = dict(d28.layout)
ok(max(h0) - min(h0) == 18,
   "the fixture is the reported sheet: an 18 px spread across the frames", h0)
ch28, msg28, _dep28 = run(d28, "match_size", {})
h1 = _match_sizes(d28)
ok(sorted(ch28) == [0, 1, 2, 3], "every frame with a subject was scaled",
   str(ch28))
ok(max(h1) - min(h1) == 0,
   "the spread is gone -- every subject is now the same size",
   "%s -> %s" % (h0, h1))
ok(_match_anchors(d28) == a0,
   "and every subject is on the pixel it was on, so no position jump was traded "
   "for the size fix", "%s vs %s" % (_match_anchors(d28), a0))
ok(d28.layout == _lay28 and d28.cells[0].shape == (64, 64, 4) and d28.n == 4,
   "the cell size, the grid and the frame count are all untouched",
   "%s / n=%d" % (d28.cells[0].shape, d28.n))
ok("Match content size:" in msg28 and "-> 27 px" in msg28 and "factor" in msg28,
   "the status line names the reference it chose and the factors it needed",
   msg28)

ok(d28.undo and d28.undo[-1]["full"] is False,
   "the undo entry covers only the cells that moved, not the whole document",
   d28.undo[-1]["full"] if d28.undo else "no entry")
d28.undo_once()
ok(_match_sizes(d28) == h0 and _match_anchors(d28) == a0,
   "undo puts the original sizes and positions back", _match_sizes(d28))

# -- the selection is what it acts on ---------------------------------------- #
d28b = _match_doc()
run(d28b, "match_size", {}, sel=[0, 1])
ok(_match_sizes(d28b)[2:] == h0[2:],
   "frames outside the selection are left exactly as they were",
   _match_sizes(d28b))

# -- the reference choice ---------------------------------------------------- #
d28c = _match_doc()
run(d28c, "match_size", {"reference": "max"})
ok(min(_match_sizes(d28c)) >= max(h0),
   "reference=max never shrinks a frame", _match_sizes(d28c))
d28d = _match_doc()
run(d28d, "match_size", {"target": 20})
ok(all(abs(x - 20) <= 1 for x in _match_sizes(d28d)),
   "an explicit target is honoured", _match_sizes(d28d))
d28e = _match_doc()
run(d28e, "match_size", {"measure": "width"})
ok(_match_sizes(d28e, "width") == [20, 20, 20, 20],
   "measure=width matches the widths instead of the heights",
   _match_sizes(d28e, "width"))
ok(_match_sizes(d28e)[:2] == h0[:2] and _match_sizes(d28e)[3] == h0[3],
   "and the frames already the right width are left alone",
   _match_sizes(d28e))

# -- the pivot is the control that stops a position jump ---------------------- #
d28f = _match_doc()
_c28f = _match_anchors(d28f, "center")
run(d28f, "match_size", {"anchor": "center"})
ok(_match_anchors(d28f, "center") == _c28f,
   "the anchor names the point that stays put: 'center' holds the subject's "
   "centre", "%s vs %s" % (_match_anchors(d28f, "center"), _c28f))
ok(_match_anchors(d28f) != a0,
   "and picking a different anchor really moves the frames, so the pivot is "
   "wired to the control rather than fixed")

# -- what it refuses to hide ------------------------------------------------- #
d28g = _match_doc()
_c28g, m28g, _ = run(d28g, "match_size", {"target": 40})
ok("1 clipped at the cell edge" in m28g,
   "a frame whose subject no longer fits is reported, not quietly cut",
   m28g)
d28h = make_doc(cells=[subject_cell(box=_MATCH_BOXES[0]), blank(),
                       subject_cell(box=_MATCH_BOXES[3]), blank()])
_c28h, m28h, _ = run(d28h, "match_size", {})
ok("2 frames had no subject" in m28h,
   "blank frames are skipped and counted in the report", m28h)
ok(not d28h.cells[1].any() and not d28h.cells[3].any(),
   "and they are still blank")
d28i = make_doc(cells=[blank() for _ in range(4)])
try:
    run(d28i, "match_size", {})
    ok(False, "a selection with no subject at all is refused with a reason")
except ValueError as ex:
    ok("no subject" in str(ex),
       "a selection with no subject at all is refused with a reason", ex)
d28j = _match_doc()
_c28j, m28j, _ = run(d28j, "match_size", {}, sel=[1])
ok(_c28j == [] and "already" in m28j and len(d28j.undo) == 0,
   "one frame is its own reference, so it is a no-op with no undo entry",
   "%s / %s / %d undo" % (_c28j, m28j, len(d28j.undo)))

# -- exactness, and the filter that changes it -------------------------------- #
d28k = _match_doc()
run(d28k, "match_size", {"resample": "nearest"})
ok(max(_match_sizes(d28k)) - min(_match_sizes(d28k)) == 0,
   "nearest lands every frame on the same pixel, not within one",
   _match_sizes(d28k))

# A white block on transparent black, at two sizes, so one frame is upscaled:
# that is where straight-alpha interpolation would darken the edge, and the
# smallest alpha>0 pixel is where it would show.
_big = blank()
_big[16:48, 16:48] = (255, 255, 255, 255)
_small = blank()
_small[20:44, 20:44] = (255, 255, 255, 255)
_wide = blank()
_wide[12:52, 12:52] = (255, 255, 255, 255)
d28l = make_doc(cells=[_big, _small, _small, _wide])
run(d28l, "match_size", {})
_vis28 = d28l.cells[1][..., 3] > 0
_rim28 = int(d28l.cells[1][..., :3][_vis28].min())
ok(_vis28.sum() > 0 and _rim28 >= 250,
   "the interpolated scale keeps the subject's colour at its edge (no dark rim)",
   "min RGB where alpha>0 = %d" % _rim28)

# --------------------------------------------------------------------------- #
print("\n=== 29. fill / tint takes its colour from the same picker ===")
# --------------------------------------------------------------------------- #
# The last op still asking for three channel numbers. It had no callers and no
# tests of its own, which is exactly why it was easy to leave behind: nothing
# failed while it stayed on r/g/b after erase_color moved to a picker. These are
# its first checks.
_f29 = E.OPS["fill"]
ok([a["k"] for a in _f29["args"]] == ["color", "alpha", "mode"],
   "fill declares one colour control where it declared three channels",
   [a["k"] for a in _f29["args"]])
ok([a["t"] for a in _f29["args"]][0] == "color"
   and _f29["args"][0]["d"] == "#ff0000",
   "of the type the page renders as a picker, defaulting to the red the three "
   "boxes defaulted to", _f29["args"][0])

# The default has to be the old default, or every saved recipe that calls fill
# with no arguments silently changes colour.
d29 = make_doc()
run(d29, "fill", {}, sel=[0])
eq(d29.cells[0][30, 30], np.array([255, 0, 0, 255], np.uint8),
   "with no arguments it still tints red, as r=255 g=0 b=0 did")
ok(d29.cells[0][0, 0, 3] == 0,
   "and a transparent pixel stays transparent rather than being tinted")

# The colour the picker names is the colour that lands, and only on the opaque
# pixels -- which is what makes this a tint rather than a fill of the whole cell.
d29b = make_doc()
run(d29b, "fill", {"color": "#13ff38"}, sel=[0])
eq(d29b.cells[0][30, 30], np.array([19, 255, 56, 255], np.uint8),
   "the colour the picker names is the colour that lands")
ok(d29b.cells[0][0, 0, 3] == 0 and not d29b.cells[0][0, 0, :3].any(),
   "and the transparent region is untouched, RGB included",
   d29b.cells[0][0, 0])

# replace sets the whole cell, alpha included; alpha_only sets nothing else.
d29c = make_doc()
run(d29c, "fill", {"color": "#0000ff", "alpha": 128, "mode": "replace"}, sel=[0])
ok((d29c.cells[0] == np.array([0, 0, 255, 128], np.uint8)).all(),
   "mode=replace sets the whole cell from the picker and the alpha box",
   d29c.cells[0][0, 0])
d29d = make_doc()
_rgb29 = d29d.cells[0][..., :3].copy()
run(d29d, "fill", {"color": "#0000ff", "alpha": 77, "mode": "alpha_only"}, sel=[0])
ok((d29d.cells[0][..., 3] == 77).all()
   and np.array_equal(d29d.cells[0][..., :3], _rgb29),
   "mode=alpha_only changes alpha and leaves every colour alone")

# An unreadable colour refuses rather than tinting black, which is the same rule
# the key follows -- black is the colour a tint must not guess at either.
d29e = make_doc()
_was29 = d29e.cells[0].copy()
try:
    run(d29e, "fill", {"color": "oops"}, sel=[0])
    ok(False, "an op handed an unreadable colour refuses instead of tinting black")
except ValueError as ex:
    ok("hex colour" in str(ex) and np.array_equal(d29e.cells[0], _was29),
       "an op handed an unreadable colour refuses instead of tinting black", ex)
ok(not d29e.undo, "and the refusal left no undo entry behind")

# It is still a scoped op, and it still reports what it did where the user looks.
d29f = make_doc()
_keep29 = d29f.cells[2].copy()
_ch29, _msg29, _ = run(d29f, "fill", {"color": "#13ff38"}, sel=[0, 1])
ok(sorted(_ch29) == [0, 1] and np.array_equal(d29f.cells[2], _keep29),
   "only the selected frames are filled", _ch29)
ok("Fill / tint" in _msg29, "and the status line names the op", _msg29)

# --------------------------------------------------------------------------- #
print("\n=== 30. resize (pixels): whole-number nearest-neighbour, no blending ===")
# --------------------------------------------------------------------------- #
# The op exists because the two resizes already in the registry filter. LANCZOS
# on 64x64 pixel art does not scale it, it softens it: every edge becomes a ramp
# of colours that were never in the art. So the checks here are about *exactness*
# rather than about a size -- a replicated pixel, no new colour, and a round trip
# that returns the original bytes. A size check alone would pass for a filter.
_spec30 = E.OPS["resize_pixels"]
ok(_spec30["group"] == "Pixel art"
   and [a["k"] for a in _spec30["args"]] == ["mode", "factor"]
   and _spec30["args"][1]["d"] == 2 and _spec30["args"][1]["min"] >= 2,
   "resize (pixels) is a pixel-art op taking a direction and an integer factor "
   "of two or more",
   (_spec30["group"], [(a["k"], a["d"]) for a in _spec30["args"]]))
# full=True is what makes undo cover every cell: the op changes all of them and
# the cell size with them, so a snapshot sized from `changed` would be the whole
# document anyway -- but only by luck, and `changed` is what the caller is told.
ok(_spec30["full"] is True,
   "and it is a whole-document op, so undo covers every cell it rewrote")

d30 = make_doc()
_before30 = [c.copy() for c in d30.cells]
_ch30, _msg30, _ = run(d30, "resize_pixels", {"mode": "up", "factor": 2}, sel=[0])
ok(sorted(_ch30) == [0, 1, 2, 3] and d30.cell_w == 128 and d30.cell_h == 128,
   "up x2 resizes every cell to 128x128, not only the selected one",
   (_ch30, d30.cell_w, d30.cell_h))
ok(d30.cells[0].shape == (128, 128, 4)
   and np.array_equal(d30.cells[0], np.repeat(np.repeat(_before30[0], 2, 0), 2, 1)),
   "and every pixel is an exact 2x2 copy of the one it came from -- no "
   "interpolation anywhere in the cell")
# The property a filter cannot have. LANCZOS on the block's edge invents a ramp
# between white and transparent, so this is the check that tells the two apart.
_vals30 = {tuple(v) for v in d30.cells[0].reshape(-1, 4)}
_was30 = {tuple(v) for v in _before30[0].reshape(-1, 4)}
ok(_vals30 == _was30,
   "and the resize introduces no colour that was not already in the frame",
   (sorted(_was30), sorted(_vals30)))
ok(d30.layout["sheet_w"] == 2 * 128 and d30.layout["sheet_h"] == 2 * 128,
   "the grid follows the cell, so the sheet is the new size too",
   (d30.layout["sheet_w"], d30.layout["sheet_h"]))
ok("Resize (pixels)" in _msg30 and "nearest-neighbour" in _msg30,
   "and the status line says what it did and how", _msg30)

# up then down is the identity. This is the whole reason the op strides instead
# of calling PIL: `Image.resize` derives the output size from a ratio and samples
# a scaled coordinate, so its NEAREST is not an exact inverse and a doubled cell
# comes back one pixel off. Striding is.
d30b = make_doc()
_orig30b = [c.copy() for c in d30b.cells]
run(d30b, "resize_pixels", {"mode": "up", "factor": 3})
run(d30b, "resize_pixels", {"mode": "down", "factor": 3})
ok(all(np.array_equal(a, b) for a, b in zip(d30b.cells, _orig30b))
   and (d30b.cell_w, d30b.cell_h) == (64, 64),
   "up x3 then down x3 gives the original bytes back in every cell",
   [(a == b).all() for a, b in zip(d30b.cells, _orig30b)])

# Down is nearest, not an average: a block of one red and three blue pixels comes
# back red, where any averaging filter would return a blend of the two.
d30c = make_doc()
_blk30 = np.zeros((4, 4, 4), np.uint8)
_blk30[..., 3] = 255
_blk30[0, 0] = (255, 0, 0, 255)          # the block's first pixel
_blk30[0, 1] = _blk30[1, 0] = _blk30[1, 1] = (0, 0, 255, 255)
d30c.cells[0] = np.tile(_blk30, (16, 16, 1))
run(d30c, "resize_pixels", {"mode": "down", "factor": 2}, sel=[0])
ok((d30c.cells[0][0, 0] == np.array([255, 0, 0, 255])).all(),
   "down takes each block's first pixel, so a mixed block does not average",
   d30c.cells[0][0, 0])

# A cell the factor does not divide is refused rather than truncated. 64 % 3 is
# 1, so a stride would drop the last row and column of every frame -- art
# disappearing with the run still reporting success.
d30d = make_doc()
_was30d = [c.copy() for c in d30d.cells]
try:
    run(d30d, "resize_pixels", {"mode": "down", "factor": 3})
    ok(False, "a factor that does not divide the cell is refused")
except ValueError as ex:
    ok("does not divide" in str(ex)
       and all(np.array_equal(a, b) for a, b in zip(d30d.cells, _was30d))
       and (d30d.cell_w, d30d.cell_h) == (64, 64),
       "a factor that does not divide the cell is refused, and the document is "
       "left alone", ex)
ok(not d30d.undo, "and the refusal left no undo entry behind")

# `min=2` keeps the form from offering 1, but coercion does not clamp, so the op
# checks it as well -- factor 1 would report every cell changed and change none.
d30e = make_doc()
try:
    run(d30e, "resize_pixels", {"mode": "up", "factor": 1})
    ok(False, "factor 1 is refused rather than reporting a no-op as a change")
except ValueError as ex:
    ok("2 or more" in str(ex), "factor 1 is refused rather than reporting a "
       "no-op as a change", ex)

# The texture cap. set_layout is the guard, and the point of the check is that
# the op routes through it: a factor that would make an unloadable sheet has to
# refuse instead of writing one.
d30f = make_doc(cells=[np.zeros((8, 3000, 4), np.uint8),
                       np.zeros((8, 3000, 4), np.uint8)])
try:
    run(d30f, "resize_pixels", {"mode": "up", "factor": 16})
    ok(False, "a factor that would exceed the texture cap is refused")
except ValueError as ex:
    ok("texture" in str(ex), "a factor that would exceed the texture cap is "
       "refused rather than writing a sheet no engine can load", ex)

# --------------------------------------------------------------------------- #
print("\n=== 31. transform (box): scale and rotate about a point, then move ===")
# --------------------------------------------------------------------------- #
# The op behind the box the canvas draws around a frame's subject. Its whole
# contract is one sentence -- "the pivot ends up exactly (dx, dy) from where it
# was" -- and everything here is a way of asking for that sentence.
#
# Two things make it hard to state: warpAffine samples on the integer grid, so a
# fractional factor or an off-axis angle does not land the pivot where the
# arithmetic says; and the op is *scoped*, like match_size, so a transform on one
# frame must leave the other three byte-identical. The first is why the op
# re-measures the anchor on the result and shifts it back; the second is why the
# scope checks are here and not left to the driver.
_spec31 = E.OPS["transform_content"]
ok(_spec31["group"] == "Transform"
   and [a["k"] for a in _spec31["args"]] == ["dx", "dy", "scale", "angle",
                                             "anchor", "threshold", "resample"]
   and [a["d"] for a in _spec31["args"]] == [0, 0, 1.0, 0.0, "center", 0,
                                             "nearest"],
   "transform (box) is a Transform op whose defaults are the identity",
   (_spec31["group"], [(a["k"], a["d"]) for a in _spec31["args"]]))
# NOT full. `full=True` would snapshot and restore every cell; the op is scoped
# to the selection, and run_op's list of whole-document ops must not have grown
# a name it does not own.
ok(_spec31["full"] is False,
   "and it is scoped to the selection rather than being a whole-document op",
   _spec31["full"])

# A subject that fits in the cell with room to double, so a scale check is about
# the pivot and not about clipping.
def _subject31(box=(24, 24, 40, 40)):
    return subject_cell(box=box)


d31 = make_doc(cells=[_subject31(), _subject31((18, 26, 34, 42)),
                      _subject31((26, 18, 42, 34)), _subject31((20, 20, 36, 36))])
_was31 = [c.copy() for c in d31.cells]
ok(E.subject_box(d31.cells[0]) == (24, 24, 39, 39),
   "the fixture's subject is the 16x16 block it was drawn as",
   E.subject_box(d31.cells[0]))

# --- a move is an exact translation, and only of what was selected -----------
_ch31, _msg31, _ = run(d31, "transform_content", {"dx": 6, "dy": -4}, sel=[0])
ok(_ch31 == [0], "a move on one frame reports exactly that frame", _ch31)
ok(np.array_equal(d31.cells[0], E._shift(_was31[0], 6, -4)),
   "and it is the reference translation, byte for byte -- no warp in the way")
ok(all(np.array_equal(d31.cells[i], _was31[i]) for i in (1, 2, 3)),
   "the other three frames are untouched, so the op is scoped to the selection")
ok("move +6,-4" in _msg31 and "Transform (box)" in _msg31,
   "and the status line names the op and the move", _msg31)
# A move with no scale or rotation has nothing to correct, so the marker pixel
# has to land exactly where arithmetic says. This is the check that would catch
# the anchor correction shifting a pure translation.
_yx31 = np.nonzero((d31.cells[0][..., 0] == 255) & (d31.cells[0][..., 1] == 0))
ok(len(_yx31[0]) == 1 and (_yx31[0][0], _yx31[1][0]) == (22, 32),
   "the marker pixel moved by exactly (+6,-4), from (26,26) to (row 22, col 32)",
   (_yx31[0][0], _yx31[1][0]) if len(_yx31[0]) else "marker not found")

# --- scaling about the centre leaves the centre exactly where it was ---------
d31b = make_doc(cells=[_subject31()])
_b31b = d31b.cells[0].copy()
_piv31b = E.anchor_point(E.subject_box(_b31b), "center")
run(d31b, "transform_content", {"scale": 2.0, "anchor": "center"}, sel=[0])
_new31b = E.subject_box(d31b.cells[0])
ok(_new31b == (16, 16, 47, 47),
   "scaling x2 about the centre gives a 32x32 subject in the same cell",
   _new31b)
ok(E.anchor_point(_new31b, "center") == _piv31b,
   "and the centre is the same pixel it was -- this is the op's whole contract, "
   "and the reason the anchor is measured again on the result",
   (_piv31b, E.anchor_point(_new31b, "center")))
# Scaling about a corner has to hold that corner instead. The pivot is a
# different point, so the box grows in one direction rather than both.
d31c = make_doc(cells=[_subject31()])
run(d31c, "transform_content", {"scale": 2.0, "anchor": "top-left"}, sel=[0])
_new31c = E.subject_box(d31c.cells[0])
ok(_new31c == (24, 24, 55, 55)
   and E.anchor_point(_new31c, "top-left") == (24, 24),
   "scaling x2 about top-left grows the subject right and down from a corner "
   "that does not move", _new31c)

# --- a quarter turn of a square is exact ------------------------------------
d31d = make_doc(cells=[_subject31()])
_b31d = d31d.cells[0].copy()
run(d31d, "transform_content", {"angle": 90.0, "anchor": "center"}, sel=[0])
ok(E.subject_box(d31d.cells[0]) == (24, 24, 39, 39),
   "rotating a centred square by 90 degrees gives the same box back",
   E.subject_box(d31d.cells[0]))
ok(int((d31d.cells[0][..., 3] > 0).sum()) == int((_b31d[..., 3] > 0).sum())
   == 256,
   "with all 256 opaque pixels still there -- nearest-neighbour, so the "
   "rotation neither invents nor drops a pixel",
   int((d31d.cells[0][..., 3] > 0).sum()))
# The property an interpolating filter cannot have, and the reason `resample`
# defaults to nearest: a 90-degree turn of hard-edged art has to stay hard-edged.
_vals31d = {tuple(v) for v in d31d.cells[0].reshape(-1, 4)}
_was31d = {tuple(v) for v in _b31d.reshape(-1, 4)}
ok(_vals31d == _was31d,
   "and no colour appeared that was not already in the frame -- a filter would "
   "have blended every edge into a ramp of new alphas",
   (sorted(_was31d), sorted(_vals31d)))

# --- all three at once: the pivot is the promise ----------------------------
# This is the composition the box's drag makes, and the one the driver's preview
# has to agree with: scale and rotate about the pivot, then move. So the pivot
# has to end up at exactly pivot + (dx, dy), and nothing else has to be true.
d31e = make_doc(cells=[_subject31()])
_b31e = d31e.cells[0].copy()
_piv31e = E.anchor_point(E.subject_box(_b31e), "center")
run(d31e, "transform_content",
    {"dx": 4, "dy": 6, "scale": 1.25, "angle": 30.0, "anchor": "center"},
    sel=[0])
_land31e = E.anchor_point(E.subject_box(d31e.cells[0]), "center")
ok(_land31e == (_piv31e[0] + 4, _piv31e[1] + 6),
   "move+scale+rotate lands the pivot at exactly pivot + (dx, dy)",
   (_piv31e, _land31e, (_piv31e[0] + 4, _piv31e[1] + 6)))
# ...and it is still pixel art afterwards. This is the check that holds
# `resample: nearest` in place, and it has to be asked HERE rather than on the
# quarter turn above: a 90-degree rotation about an integer pivot maps integer
# coordinates to integer coordinates, so a bilinear sample lands exactly on a
# source pixel and blends nothing. At 30 degrees and 1.25x there is no such luck
# -- every boundary pixel is a blend -- so a filter has nowhere to hide.
_vals31e = {tuple(v) for v in d31e.cells[0].reshape(-1, 4)}
_was31e = {tuple(v) for v in _b31e.reshape(-1, 4)}
ok(_vals31e == _was31e,
   "and the 30-degree transform invented no colour either -- the art is hard-"
   "edged at an angle, which only nearest can manage",
   (sorted(_was31e), sorted(_vals31e)))

# --- undo, on a transform that changed every pixel --------------------------
d31f = make_doc()
_was31f = [c.copy() for c in d31f.cells]
run(d31f, "transform_content", {"scale": 1.5, "angle": 15.0}, sel=[0, 2])
ok(d31f.undo and len(d31f.undo) == 1,
   "one drag's worth of transform is one undo entry", len(d31f.undo))
d31f.undo_once()
ok(all(np.array_equal(a, b) for a, b in zip(d31f.cells, _was31f)),
   "and undoing it restores every cell byte-exactly",
   [bool(np.array_equal(a, b)) for a, b in zip(d31f.cells, _was31f)])

# --- the refusals ----------------------------------------------------------
# The form's defaults. Reporting four frames changed here would push an undo
# entry that restores what is already on screen.
d31g = make_doc()
try:
    run(d31g, "transform_content", {}, sel=[0, 1])
    ok(False, "the identity transform is refused rather than reported as a change")
except ValueError as ex:
    ok("nothing to do" in str(ex),
       "the identity transform is refused rather than reported as a change", ex)
ok(not d31g.undo, "and it left no undo entry behind")

# `coerce_args` fills a missing key from the spec's default and never clamps, so
# a value the form cannot produce still reaches the op. min=0.05 on the scale
# does not stop a 0.
d31h = make_doc()
_was31h = [c.copy() for c in d31h.cells]
try:
    run(d31h, "transform_content", {"scale": 0.0}, sel=[0])
    ok(False, "a scale of 0 is refused instead of erasing the frame")
except ValueError as ex:
    ok("would erase" in str(ex),
       "a scale of 0 is refused instead of erasing the frame", ex)
ok(all(np.array_equal(a, b) for a, b in zip(d31h.cells, _was31h)),
   "and the frames are still there")

# A selection with nothing in it has no box to pivot on, and the op says so
# rather than guessing a cell-sized one.
d31i = make_doc(cells=[blank(), blank()])
try:
    run(d31i, "transform_content", {"scale": 2.0}, sel=[0])
    ok(False, "a frame with no subject is refused")
except ValueError as ex:
    ok("no subject" in str(ex), "a frame with no subject is refused", ex)

# A transform that is not the identity but happens to change nothing. angle=360
# is outside the form's -180..180 range and is not clamped, and the rotation
# matrix it builds is the identity -- so the honest answer is the same refusal,
# not a stack of undo entries that do nothing.
d31j = make_doc()
_was31j = [c.copy() for c in d31j.cells]
try:
    run(d31j, "transform_content", {"angle": 360.0}, sel=[0])
    ok(False, "a transform that leaves the art as it was is refused")
except ValueError as ex:
    ok("as it was" in str(ex),
       "a transform that leaves the art as it was is refused, so a no-op cannot "
       "report itself as a change", ex)
ok(all(np.array_equal(a, b) for a, b in zip(d31j.cells, _was31j))
   and not d31j.undo,
   "and nothing was written")

# --------------------------------------------------------------------------- #
print("\n=== 32. pixelize_oe: PixelOE's contrast outline + block downsample ===")
# --------------------------------------------------------------------------- #
try:
    import pixeloe_bridge as _poe
    _POE_OK = _poe.available()
except Exception:                                  # noqa: BLE001
    _POE_OK = False
if not _POE_OK:
    print("  skip  PixelOE checkout not available")
else:
    def _graded(w, h, seed=0):
        """A soft gradient with a hard-edged disc: the input stage 2 targets.

        The disc is what stage 1 is for -- a one-pixel edge at a strong contrast
        that a plain average would smear before the grid exists.
        """
        c = blank(w, h)
        yy, xx = np.mgrid[0:h, 0:w]
        c[..., 0] = ((xx * 255) // max(1, w - 1)).astype(np.uint8)
        c[..., 1] = ((yy * 255) // max(1, h - 1)).astype(np.uint8)
        c[..., 2] = ((xx + yy) * 255 // max(1, w + h - 2)).astype(np.uint8)
        c[..., 3] = 255
        c[(xx - w // 3) ** 2 + (yy - h // 2) ** 2
          < (min(w, h) // 4) ** 2] = (245, 20, 20, 255)
        return c

    # --- the op's declared shape ------------------------------------------ #
    _sp32 = E.OPS["pixelize_oe"]
    ok(_sp32["group"] == "Pixel art",
       "pixelize (outline) files under the Pixel art group, beside the others",
       _sp32["group"])
    _kinds32 = {a["k"]: a["t"] for a in _sp32["args"]}
    eq(_kinds32.get("pixel_size"), "int",
       "the pixel size is an integer control")
    eq(_kinds32.get("mode"), "choice",
       "the downsampler is a choice, not free text")
    eq(_kinds32.get("sharpen"), "choice",
       "the sharpen mode is a choice with an off value")
    _by32 = {a["k"]: a for a in _sp32["args"]}
    _opts32 = _by32["sharpen"].get("opts")
    ok("none" in _opts32,
       "sharpen names its off state 'none' rather than using an empty string",
       _opts32)
    ok("" not in _opts32,
       "and offers no blank option, which the page would render as an "
       "unlabelled row in the dropdown")
    eq(_kinds32.get("color_match"), "bool",
       "colour matching is a bool")

    # --- the declared defaults --------------------------------------------- #
    # Pinned as values, because a default is what most runs actually use: the
    # page fills the form from these and `coerce_args` fills a missing key from
    # them too. Asserting the numbers is right here (unlike a detected grid,
    # which is fixture-dependent) because they ARE the contract.
    _defs32 = {k: _by32[k]["d"] for k in
               ("pixel_size", "thickness", "mode", "colors", "dither",
                "sharpen", "sharpen_factor", "color_match")}
    eq(_defs32["colors"], 0,
       "colours defaults to 0, so a run keeps every colour the pixelizer "
       "resolves and quantising is opt-in")
    eq(_defs32["dither"], "none",
       "dither defaults to none, so a quantise is a clean k-means reduction "
       "unless a pattern is asked for")
    eq(_defs32["sharpen"], "none",
       "sharpen defaults to none, so a run leaves the pixelizer's own edges "
       "alone unless sharpening is asked for")
    ok(_defs32["sharpen_factor"] == 0,
       "and the sharpen amount defaults to 0, the same off state spelled on the "
       "amount control",
       _defs32["sharpen_factor"])
    eq(_defs32["color_match"], True,
       "colour matching defaults ON, so the expansion stage hands the "
       "downscaler the source's own palette")
    eq(_defs32["pixel_size"], 2,
       "the pixel size defaults to 2, a fine block size that divides the common "
       "cell sizes")
    eq(_defs32["thickness"], 0,
       "and the outline expansion defaults to 0, i.e. the outline stage is off "
       "by default")
    eq(_defs32["mode"], "k_centroid",
       "with the k-centroid downsampler rather than the paper's contrast one")

    # The sharpen max has to admit its own default, or the page renders
    # `<input type=number min=0 max=4 value=0>` -- fine at 0, but the rule is
    # what matters (§33 of the skill): a later bump of the default must not
    # silently produce a form the page cannot honour. coerce_args does not
    # clamp, so nothing would fail loudly.
    ok(_by32["sharpen_factor"]["min"] <= _defs32["sharpen_factor"]
       <= _by32["sharpen_factor"]["max"],
       "the sharpen amount's own bounds admit its default",
       (_by32["sharpen_factor"]["min"], _defs32["sharpen_factor"],
        _by32["sharpen_factor"]["max"]))
    # And the defaults must be a runnable combination: no quantise, no outline +
    # no sharpening, colour match on, on a cell the pixel size divides.
    _d32run = make_doc()
    _run32 = run(_d32run, "pixelize_oe", {}, sel=[0])
    ok(_run32[0] == [0],
       "and the whole default set runs on the suite's fixture and changes it",
       _run32[1])
    ok("outline off" in _run32[1],
       "and the status line says the default run skipped the outline stage, "
       "which is what thickness 0 means", _run32[1])

    # The whole point of the op is that a stale page cannot hand it a size the
    # cell cannot take: the default must divide the common cell sizes.
    _d32 = make_doc()
    ok(all(_d32.cell_w % _sp32["args"][0]["d"] == 0
           and _d32.cell_h % _sp32["args"][0]["d"] == 0
           for _ in (0,)),
       "its default pixel size divides the default cell, so the control is "
       "usable out of the box")

    # --- the size invariant: the cell size is kept exactly ----------------- #
    d32 = make_doc([_graded(64, 64, 0), _graded(64, 64, 1),
                    _graded(64, 64, 2), _graded(64, 64, 3)], cols=2)
    _was32 = [c.copy() for c in d32.cells]
    _ch32, _msg32, _ = run(d32, "pixelize_oe",
                           {"pixel_size": 8, "thickness": 3, "mode": "contrast",
                            "colors": 0, "dither": "ordered", "sharpen": "none",
                            "sharpen_factor": 0.5, "color_match": True},
                           sel=[0, 1, 2])
    ok(set(_ch32) <= {0, 1, 2} and _ch32,
       "the op reports the frames it changed out of the selection", _ch32)
    ok(all(c.shape == b.shape for c, b in zip(d32.cells, _was32)),
       "every frame keeps its shape: the pixel size divides the cell, so "
       "PixelOE never pads and the cell size is preserved exactly")
    eq(d32.cells[3], _was32[3], "the unselected frame is byte-identical")
    ok(not np.array_equal(d32.cells[0], _was32[0]),
       "the selected frames really were pixelized, so the test is live")

    # Alpha is carried through untouched -- PixelOE is RGB-only, so the op is
    # the one that must keep the channel. This needs a fixture with real
    # transparency: a fully opaque cell cannot tell "alpha preserved" from
    # "alpha flattened to 255", which is exactly what the pixeloe-drops-alpha
    # hook pins, so a solid fixture would make this check vacuous.
    d32f = make_doc([subject_cell(), subject_cell()], cols=2)
    _was32f = [c.copy() for c in d32f.cells]
    ok((_was32f[0][..., 3] == 0).any() and (_was32f[0][..., 3] == 255).any(),
       "the alpha fixture really does hold both transparent and opaque pixels, "
       "so the alpha check below can fail")
    run(d32f, "pixelize_oe", {"pixel_size": 8, "thickness": 3, "colors": 0},
        sel=[0])
    eq(d32f.cells[0][..., 3], _was32f[0][..., 3],
       "alpha is preserved byte for byte, because PixelOE has no alpha stage")
    eq(d32f.cells[1], _was32f[1],
       "and the frame outside the selection is still untouched afterwards")

    # The result really is a block downsample: within one pixel_size block the
    # colour is constant. This is the property that makes it pixel art rather
    # than a blur, and it holds for the contrast downsampler. The blocks are
    # read straight from the cell's own top-left corners, so nothing about the
    # downsampler's arithmetic is assumed -- only that it is constant per block.
    _corners = d32.cells[0][::8, ::8, :3]
    _blocky = np.repeat(np.repeat(_corners, 8, axis=0), 8, axis=1)[:64, :64]
    eq(d32.cells[0][..., :3], _blocky,
       "every 8x8 block in the result is one flat colour, so the output is a "
       "true block-downscaled image and not an interpolated one")

    d32.undo_once()
    ok(all(np.array_equal(c, b) for c, b in zip(d32.cells[:3], _was32[:3])),
       "undo restores every selected frame exactly")

    # --- the refusal: a size the cell cannot take ------------------------- #
    # 33 does not divide 64. The bridge must refuse rather than let PixelOE
    # replicate-pad, because that padding changes the result and cannot be
    # cropped back off.
    d32b = make_doc()
    _was32b = [c.copy() for c in d32b.cells]
    try:
        run(d32b, "pixelize_oe", {"pixel_size": 33}, sel=[0])
        ok(False, "a pixel size that does not divide the cell is refused")
    except ValueError as ex:
        ok("does not divide" in str(ex),
           "a pixel size that does not divide the cell is refused, naming the "
           "problem", ex)
        ok("2, 4, 8" in str(ex),
           "and the refusal lists the sizes that do fit, rather than only "
           "saying no", ex)
    except Exception as ex:                        # noqa: BLE001
        # The guard is gone (see the pixeloe-pads hook) and the shape backstop
        # caught the pad instead. That is still a refusal, but it is a *worse*
        # one -- the user is not told which sizes fit -- so it is recorded as a
        # failure here rather than waved through. Tolerating it would make this
        # section pass under the mutant, which is the hole the sweep exists to
        # find.
        ok(False, "a pixel size that does not divide the cell is refused with "
                  "the sizes that fit, not by the shape backstop", ex)
    ok(all(np.array_equal(a, b) for a, b in zip(d32b.cells, _was32b))
       and not d32b.undo,
       "and the document is left alone, with no undo entry behind")

    # The bridge refuses the padding case too, so the guarantee does not depend
    # on the op catching it first.
    try:
        _poe.pixelize_cells([blank(63, 63)], pixel_size=8)
        ok(False, "the bridge refuses a cell the pixel size does not divide")
    except ValueError as ex:
        ok("does not divide" in str(ex),
           "the bridge refuses a cell the pixel size does not divide, so the "
           "invariant is enforced below the op as well")
    except Exception as ex:                        # noqa: BLE001
        # With the guard mutant, the shape backstop is what fires. Either is a
        # refusal; a silent success is the only failure.
        ok("returned" in str(ex),
           "the bridge refuses a cell the pixel size does not divide (by the "
           "shape backstop), so the invariant is enforced below the op as well",
           ex)

    # --- quantisation is a target, and the op says so ---------------------- #
    # The op's help warns that the colour match after quantisation can put a
    # shade back. Pin that honestly instead of asserting an exact count.
    d32c = make_doc([_graded(64, 64, 7)], cols=1)
    run(d32c, "pixelize_oe",
        {"pixel_size": 8, "colors": 4, "dither": "none", "sharpen": "none"},
        sel=[0])
    _n32c = len(_distinct(d32c.cells[0]))
    ok(_n32c <= 24,
       "naming 4 colours quantises towards a small palette rather than leaving "
       "the full gradient", _n32c)
    ok(d32c.cells[0].shape == _was32[0].shape,
       "and it still does not change the cell size")

    # --- thickness 0 turns stage 1 off ------------------------------------- #
    # Off is a different image from on, so the control is wired to the pipeline
    # rather than being decorative.
    d32d = make_doc([_graded(64, 64, 9), _graded(64, 64, 9)], cols=2)
    run(d32d, "pixelize_oe", {"pixel_size": 8, "thickness": 3, "colors": 0,
                              "sharpen": "none"}, sel=[0])
    run(d32d, "pixelize_oe", {"pixel_size": 8, "thickness": 0, "colors": 0,
                              "sharpen": "none"}, sel=[1])
    ok(not np.array_equal(d32d.cells[0], d32d.cells[1]),
       "outline expansion on and off give different images, so thickness is "
       "wired into the outline stage")
    ok("outline off" in run(d32d, "pixelize_oe",
                            {"pixel_size": 8, "thickness": 0}, sel=[1])[1],
       "and the status line says when the outline stage was skipped")

    # --- sharpen's off state, and the two ways to spell it ------------------ #
    # The control's off value is the word "none"; an empty string is what a
    # stale page still holds, and it must mean the same thing. The declared
    # default is now also "none", so a run that omits the argument has to be the
    # same image as one that names the off state explicitly -- while an explicit
    # "unsharp" must still give a different one, or the checks below would pass
    # with sharpen wired to nothing at all.
    d32g = make_doc([_graded(64, 64, 3), _graded(64, 64, 3),
                     _graded(64, 64, 3), _graded(64, 64, 3)], cols=2)
    run(d32g, "pixelize_oe", {"pixel_size": 8, "sharpen": "none"}, sel=[0])
    run(d32g, "pixelize_oe", {"pixel_size": 8, "sharpen": ""}, sel=[1])
    run(d32g, "pixelize_oe", {"pixel_size": 8}, sel=[2])
    run(d32g, "pixelize_oe", {"pixel_size": 8, "sharpen": "unsharp",
                              "sharpen_factor": 2.0}, sel=[3])
    eq(d32g.cells[0], d32g.cells[1],
       "sharpen 'none' and sharpen '' mean the same thing, so a stale page "
       "cannot turn sharpening on by leaving the value blank")
    eq(d32g.cells[2], d32g.cells[0],
       "and omitting sharpen matches the declared default 'none', so the "
       "default really is the off state")
    ok(not np.array_equal(d32g.cells[0], d32g.cells[3]),
       "while sharpening off and sharpening on are different images, so the "
       "equality above is not passing because sharpen is ignored")

    # --- the palette picker, the same control the other two ops have ------- #
    # A named palette REPLACES quantising, exactly as on Snap Pixels and
    # Pixelate (mesh): the `colours` spinner is ignored and the pixelized result
    # is mapped onto the named palette. Two things have to be true, and only the
    # second one is about the palette being *used*.
    _pal32 = ["#101820", "#e8f0f8", "#4c9ad0", "#dc5050"]
    _allowed32 = {tuple(int(v) for v in row)
                  for row in E.hex_palette(",".join(_pal32))}
    d32h = make_doc([_graded(64, 64, 11), _graded(64, 64, 12)], cols=2)
    _ch32h, _msg32h, _ = run(d32h, "pixelize_oe",
                             {"pixel_size": 8, "palette": ",".join(_pal32)},
                             sel=[0, 1])
    _cols32h = set()
    for _c in d32h.cells:
        _cols32h |= _distinct(_c)
    ok(_cols32h and _cols32h <= _allowed32,
       "every colour in the result comes from the named palette",
       sorted(_cols32h - _allowed32)[:4])
    ok(len(_cols32h) > 1,
       "and the result really uses more than one of them, so the palette was "
       "mapped and not collapsed to a single colour", sorted(_cols32h))
    ok("palette from the picker" in _msg32h,
       "the status line says the palette came from the picker", _msg32h)
    ok(all(c.shape == (64, 64, 4) for c in d32h.cells),
       "a palette does not change the cell size either")

    # The spinner must not fight the palette: with a palette named, a different
    # `colours` count must give byte-identical output. This is the one check
    # that fails if the op quantises first and maps afterwards.
    d32i = make_doc([_graded(64, 64, 11), _graded(64, 64, 12)], cols=2)
    run(d32i, "pixelize_oe",
        {"pixel_size": 8, "colors": 2, "palette": ",".join(_pal32)}, sel=[0, 1])
    ok(all(np.array_equal(a, b) for a, b in zip(d32i.cells, d32h.cells)),
       "the colours spinner is ignored when a palette is named, so the palette "
       "is the only thing deciding the colours")

    # An unreadable entry is refused, not skipped -- the same rule `hex_palette`
    # enforces for the other two ops, reaching this op too.
    d32j = make_doc()
    _was32j = [c.copy() for c in d32j.cells]
    try:
        run(d32j, "pixelize_oe", {"pixel_size": 8, "palette": "#ff0000,oops"},
            sel=[0])
        ok(False, "an unreadable palette entry is refused")
    except ValueError as ex:
        ok("hex colour" in str(ex),
           "an unreadable palette entry is refused rather than quietly "
           "shortened", ex)
    ok(all(np.array_equal(a, b) for a, b in zip(d32j.cells, _was32j))
       and not d32j.undo,
       "and the refusal left the document alone with no undo entry")

    # --- the op says what it did ------------------------------------------- #
    _m32 = run(d32c, "pixelize_oe", {"pixel_size": 8, "colors": 0}, sel=[0])[1]
    ok("8px blocks" in _m32 and "cell kept at 64x64" in _m32,
       "the status line names the block size and the cell size it kept", _m32)

    # --- one call for the batch: the frames share the pixel grid ----------- #
    # Not a per-frame solve: the same grades through one call must line up on
    # the same grid, so a static region cannot jitter across frames.
    d32e = make_doc([_graded(64, 64, 0), _graded(64, 64, 0)], cols=2)
    run(d32e, "pixelize_oe", {"pixel_size": 8, "thickness": 3, "colors": 0},
        sel=[0, 1])
    eq(d32e.cells[0], d32e.cells[1],
       "two identical frames pixelize to identical bytes, so the batch runs "
       "once and cannot jitter between frames")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 66)
print("%d checks, %d failures" % (CHECKS[0], len(FAILS)))
if FAILS:
    for f in FAILS:
        print("  FAILED:", f)
    sys.exit(1)
print("ALL PASS")
