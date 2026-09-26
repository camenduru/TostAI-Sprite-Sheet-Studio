"""Mutation-test the served editor page: are the driver's checks sensitive?

Each mutant is derived from the live editor.html text, so a drift makes this
script exit 2 rather than silently mutate nothing. editor.html is restored in a
finally, so a crashed driver cannot leave a mutant on disk -- and a copy of the
pristine page is parked next to it while a sweep runs, so a run that is killed
outright (SIGKILL, a closed terminal) is put back by the next invocation instead
of being derived from.

    python mutate_editor_page.py            # all mutants
    python mutate_editor_page.py A C        # just those
"""
import os
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "editor.html")
PY = r"C:/Users/PC/AppData/Local/Programs/Python/Python313/python.exe"

MUTANTS = [
    # --- the restart recovery (section 16) ---------------------------------
    # The reopen still happens; only the success flag is forced. This is the
    # narrow mutant: it isolates "the status line claims a reopen that failed"
    # from "no reopen happened at all".
    ("A", "recovery claims a reopen that failed",
     "const back = await openSheet(path);",
     "const back = (await openSheet(path), true);"),
    ("B", "the page never notices the dead document",
     "    if (/unknown document/i.test(d.error)) recoverFromRestart();\n",
     ""),
    # --- shift-select (section 3) -----------------------------------------
    # The reported bug, restored: a grid rectangle on the canvas while the
    # timeline keeps the frame range.
    ("C", "the canvas shift-click is a rectangle again",
     "      selectRange(S.anchorCell, ci);",
     "      (() => { const cols = SHEET().cols, a = S.anchorCell;"
     " const r0 = Math.floor(a/cols), c0 = a%cols,"
     " r1 = Math.floor(ci/cols), c1 = ci%cols;"
     " for (let r = Math.min(r0,r1); r <= Math.max(r0,r1); r++)"
     " for (let c = Math.min(c0,c1); c <= Math.max(c0,c1); c++)"
     " S.sel.add(r*cols+c); })();"),
    # A stale anchor from the previous sheet is no longer reset on open.
    ("D", "the anchor is not reset when a new sheet opens",
     "if (!prev || prev.id !== d.id){",
     "if (false){"),
    # The range is no longer clamped to the document.
    ("E", "the range is not clamped to the document",
     "  const lo = Math.max(0, x), hi = Math.min(n - 1, y);",
     "  const lo = x, hi = y;"),
    # --- export only the selected frames (section 13b) ---------------------
    # The box is rendered and enabled, but the selection never reaches the
    # server: the export silently writes the whole document.
    ("F", "the export box is decorative and sends nothing",
     '      frames: $("onlySel").checked ? selArr() : [],',
     "      frames: [],"),
    # The subset is written, but there is no way back into it.
    ("G", "no edit action on the save result",
     "    if (d.sheet_path)",
     "    if (false)"),
    # The label states the count whether or not the option is on -- the wording
    # that was read as a promise about Save while the box was unticked.
    ("H", "the export label ignores whether the box is ticked",
     '  $("onlySelLabel").textContent = on\n'
     '    ? "only the selected " + k + " of " + n + " frames"\n'
     '    : (k ? "export only the selected frames (" + k + " selected)"\n'
     '         : "export only the selected frames (nothing selected)");',
     '  $("onlySelLabel").textContent = k\n'
     '    ? "only the selected " + k + " of " + n + " frames"\n'
     '    : "only the selected frames (nothing selected)";'),
    # The hint beside Save stops following the box.
    ("I", "the save hint ignores the export box",
     "  h.textContent = on\n",
     "  h.textContent = false\n"),
    # --- the side-panel layout (section 1) --------------------------------
    # The palette opens by default again, which is what pushes the Audio panel
    # (moved to the left column under it) off the bottom of the column.
    ("J", "the Operations palette is open again",
     '    <details class="panel">\n      <summary>Operations</summary>',
     '    <details class="panel" open>\n      <summary>Operations</summary>'),
    # --- the sound mark on the cell (section 16e) -------------------------
    # The clips are still attached and still listed; only the mark that says so
    # on the sheet itself is gone, which is what the canvas-pixel check reads.
    ("K", "a frame with a sound is no longer marked on the sheet",
     "  if (legible && sound.length){",
     "  if (false){"),
    # --- the player's own fps (section 8c) --------------------------------
    # The coupling restored: the playback loop reads the document's fps instead
    # of the player's, so setting the document's frame rate silently changes how
    # the preview plays. Both rate checks invert -- the 240/1 pair and the
    # 60-at-1x/4x pair -- so this is caught four ways.
    ("L", "the playback loop reads the document's fps again",
     '    const fps = Math.max(1, (Number($("playFps").value) || 24) * speed);',
     '    const fps = Math.max(1, (Number($("fps").value) || 24) * speed);'),
    # --- Apply metadata actually writing (section 16h) --------------------
    # The button stops calling the route, so the metadata stays in memory and the
    # JSON on disk keeps the old value. This is the wiring no other harness can
    # check: the op layer proves save_meta does the right thing, and only a real
    # click can prove the button reaches it.
    ("M", "Apply metadata never writes the JSON",
     '    const r = await api("/api/editor/" + S.doc.id + "/save_meta", {});\n'
     '    const line = "metadata written to " + r.sidecar;',
     '    const r = {sidecar: "(nothing written)"};\n'
     '    const line = "metadata written to " + r.sidecar;'),
    # --- a checkbox whose words are not part of it (section 16i) ----------
    # Back to a <div>, so the text beside the box is not clickable and the row
    # looks broken to anyone who clicks the words rather than the 13px square.
    ("N", "the overwrite checkbox's words stop being part of the control",
     '<label class="chk" style="margin-top:6px"><input type="checkbox" id="overwrite">',
     '<div class="chk" style="margin-top:6px"><input type="checkbox" id="overwrite">'),
    # --- the audio library drag (section 16j) -----------------------------
    # The drop attaches to whatever frame is current instead of the frame it was
    # dropped on. The gesture still appears to work -- a clip does get attached --
    # which is exactly why "it landed where I dropped it" needs its own check, and
    # why section 16j puts the current frame somewhere else first.
    ("O", "the drop lands on the current frame, not the one under the pointer",
     '  const cv = e.target.closest && e.target.closest("#strip canvas");\n'
     '  if (cv) attachFromLibrary(dragPath(e), Number(cv.dataset.i));',
     '  attachFromLibrary(dragPath(e), S.cur);'),
    # Any drag is treated as a library clip. This is the mutant for the design
    # decision rather than for a symptom: the removed panel's drop targets
    # consulted a module-level flag, so anything could satisfy them. Section 16j
    # drags a foreign payload that *would* attach if accepted, which is what makes
    # its "nothing was attached" a real statement about the page.
    ("P", "the drop target stops checking what kind of drag it is",
     '  return !!e.dataTransfer\n'
     '    && [...(e.dataTransfer.types || [])].includes(LIB_MIME);',
     '  return true;'),
    # --- the palette dropdown while playing (section 16q) ------------------
    # The reported bug, restored: the listener closes on *any* outside scroll
    # again. This is the mutant that matters for 16q, because "always close"
    # still satisfies the half of the section that asserts the panel goes when
    # its own column scrolls -- only the playback half can tell the two
    # listeners apart. The anchor spans a line break and contains backticks,
    # so it is written with \n like the others and run through as_page_endings.
    ("Q", "the dropdown closes on any outside scroll again",
     '    const t = e.target;\n'
     '    // Only a scroll that could have moved the *button* closes the panel, so\n'
     '    // `t.contains(btn)` is the whole test. There is deliberately no separate\n'
     '    // clause for the viewport: a viewport scroll reports `document` as its\n'
     '    // target, and `document.contains(btn)` is true, so it is already covered.\n'
     '    // That was measured rather than assumed -- a forced document scroll arrives\n'
     '    // with target `document` and never with `document.scrollingElement` (which\n'
     '    // is `<html>`), so a clause naming that element could not have fired.\n'
     '    if (t.contains && t.contains(btn)) panel.hidden = true;',
     '    panel.hidden = true;'),
    # --- a superseded audio-library listing (section 16j) ------------------
    # boot() starts restoreLibDir() without awaiting it, so the remembered
    # folder's listing can be in flight when the user asks for another one. With
    # the guard gone the late answer renders anyway, and the panel ends up
    # showing a folder the box does not name while LIB.items hands out its
    # paths. Section 16j holds the first response open so the late answer is
    # guaranteed to be the wrong one.
    ("R", "a superseded listing is rendered anyway",
     '    if (mine !== LIB_REQ) return null;\n'
     '    // A route that is not there answers a 404 carrying no `audio` list at all,\n',
     '    // A route that is not there answers a 404 carrying no `audio` list at all,\n'),
    # The same guard on the failure path. Without it a superseded request that
    # failed clears the rows the newer listing just filled and puts "not a
    # folder" in the note, under a box that no longer names that folder.
    ("S", "a superseded failure clears the panel anyway",
     '    // A failure that has been superseded must not clear the panel either: the\n'
     '    // note would blame the folder that is no longer in the box.\n'
     '    if (mine !== LIB_REQ) return null;\n',
     '    // A failure that has been superseded must not clear the panel either: the\n'
     '    // note would blame the folder that is no longer in the box.\n'),
    # --- a new op that changes the cell size (section 16r) ----------------
    # The op itself is fine; only its name is missing from the list of ops that
    # refit the view after they run. That is the wiring the op layer and the HTTP
    # smoke structurally cannot see, and it is invisible on a sheet that already
    # fits -- which is why 16r forces a zoom the sheet does not fit in first.
    ("T", "a cell-size op is left out of the refit list",
     '        || o.name === "rotate" || o.name === "resize_pixels") fitView();',
     '        || o.name === "rotate") fitView();'),
    # --- transform box (section 16s) ---------------------------------------
    # The box should only intercept presses under the select tool, so the pencil
    # can still paint inside the frame. Removing the tool guard makes the box
    # swallow a stroke -- the pencil test in 16s presses inside the frame and
    # asserts the drag kind is "paint", not "xform".
    ("U", "the transform box intercepts every tool's press",
     '  if (S.xform && S.tool === "select"){',
     '  if (S.xform){'),
]

only = [a.upper() for a in sys.argv[1:]]
if only:
    MUTANTS = [m for m in MUTANTS if m[0] in only]
    if not MUTANTS:
        print("no mutant matches %s" % only)
        sys.exit(2)

# Flags go to the driver; bare words are mutant tags and must NOT. Forwarding
# the tags as well made `python mutate_editor_page.py Q` hand "Q" to
# editor_drive.py, whose argparse rejected it and exited 2 before running a
# single check -- so every mutant reported CAUGHT for a reason that had nothing
# to do with the mutation, in about a second instead of a couple of minutes.
# A one-mutant run is exactly how a hook gets validated, so this mattered.
PASSTHRU = [a for a in sys.argv[1:] if a.startswith("-")]

original = open(HTML, "r", encoding="utf-8", newline="").read()

# A previous run that was killed mid-sweep leaves a mutant in the file this
# script derives its mutants from, so this runs before anything reads `original`
# again. It is after write_page() on purpose: that is the function it needs.
if restore_stale():
    print("!! a previous sweep was killed before it restored the page -- "
          "editor.html has been put back from its pre-run copy")
    print("   (that is the state this run re-reads below, not the mutant)")
    original = open(HTML, "r", encoding="utf-8", newline="").read()


def _restore_on_term(signum, frame):
    """Ctrl-C is handled by the finally below; a SIGTERM is not, so it is here."""
    write_page(original)
    drop_baseline()
    sys.exit(130)


if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _restore_on_term)

def write_page(text):
    """Replace editor.html in one step, byte for byte.

    Two reasons this is not a plain open(..., "w").write(). The server reads
    this file on every request, so a truncate-then-write leaves a window in
    which a page fetch gets half a mutant -- a page that behaves like nothing
    else and cannot be reproduced from either version. And the default text mode
    rewrites line endings on Windows, so the "restored" size printed below was
    not the size that was there before the run.
    """
    tmp = HTML + ".mutant"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, HTML)


# The pristine page is parked here for the duration of a sweep.
#
# A run killed with SIGKILL -- a closed terminal, a task manager kill -- never
# reaches the `finally` that restores the page, and what is left on disk is a
# mutant being served as if it were the editor. That is not hypothetical: mutant
# D ("the anchor is not reset when a new sheet opens") was found applied in the
# working tree, which made the served page carry a stale anchor, an AUDIO.cache
# that was never dropped between documents and a box cache keyed to the wrong
# sheet -- while every source-derived check agreed with it, because it derived
# its expectations from the mutated text.
#
# So finding this file at startup can only mean the previous run did not finish,
# and it is restored rather than used as the new source of truth.
BASELINE = HTML + ".baseline"


def restore_stale():
    """Put a killed sweep's mutant back. True if there was one to undo."""
    if not os.path.exists(BASELINE):
        return False
    with open(BASELINE, "r", encoding="utf-8", newline="") as fh:
        pristine = fh.read()
    write_page(pristine)
    os.remove(BASELINE)
    return True


def drop_baseline():
    try:
        os.remove(BASELINE)
    except OSError:
        pass


def as_page_endings(text):
    """Rewrite a mutant's line endings to the page's own.

    The anchors in MUTANTS are written with \\n. The served page is CRLF on
    Windows, and `original` is read with newline="" so it keeps those CR. Several
    anchors span a line break (B, H, I, J, O, P), so without this they never match
    -- and because the pre-flight below exits on the first miss, one of them took
    the entire sweep down with it while looking like a one-line drift problem.
    """
    if "\r\n" in original:
        return text.replace("\r\n", "\n").replace("\n", "\r\n")
    return text.replace("\r\n", "\n")


def plant(tag, label, old, new):
    """The mutated page text, or None when the anchor is not in the source."""
    old_c, new_c = as_page_endings(old), as_page_endings(new)
    if old_c not in original:
        return None
    mutated = original.replace(old_c, new_c, 1)
    # A replacement that changed nothing would run the pristine page and report
    # MISSED -- which reads like a weak mutant rather than a broken harness.
    if mutated == original:
        print("!! mutation %s applied but changed nothing: %s" % (tag, label))
        sys.exit(2)
    return mutated


# Fail loudly rather than silently testing nothing -- but say *which* anchors
# drifted, so one bad anchor cannot masquerade as a sweep that ran.
missing = [t for t, l, o, n in MUTANTS if plant(t, l, o, n) is None]
if missing:
    print("!! these mutations do not apply, source has drifted: %s"
          % ", ".join(missing))
    sys.exit(2)

results = []
try:
    # Parked before the first mutant, removed by the restore below: its presence
    # afterwards can only mean this run did not get to finish.
    with open(BASELINE, "w", encoding="utf-8", newline="") as fh:
        fh.write(original)
    for tag, label, old, new in MUTANTS:
        print("\n" + "=" * 66)
        print("MUTANT %s: %s" % (tag, label))
        print("=" * 66)
        write_page(plant(tag, label, old, new))
        try:
            # The URL and port are passed through so the sweep can be pointed at
            # a server started for the run. Without this it always drives the
            # default port, which may be a server someone else is using -- and
            # one started before the latest app.py changes, in which case a
            # section that needs a new route fails on every mutant and the sweep
            # reports "CAUGHT" for the wrong reason.
            r = subprocess.run([PY, "editor_drive.py"] + PASSTHRU, cwd=HERE,
                               capture_output=True, text=True, timeout=900)
        finally:
            write_page(original)
        out = r.stdout + r.stderr
        for ln in out.strip().splitlines():
            if ln.strip().startswith("FAIL"):
                print("   " + ln.strip()[:170])
        for ln in out.strip().splitlines()[-3:]:
            if "checks," in ln or "ALL PASS" in ln:
                print("   " + ln.strip())
        # A driver that never reported a result did not test the mutation. This
        # is the same trap as the tag-forwarding bug above: an argparse error, a
        # missing server or a crash all exit non-zero, and reading that as
        # CAUGHT makes a broken run look like a strong mutant. A mutant is only
        # caught when the suite actually ran and failed.
        #
        # "Actually ran" has to include the driver that STARTED and then died.
        # `wait_for` raises TimeoutError rather than recording a FAIL, so a
        # mutant that makes the page never reach a state the section waits for
        # kills the driver before its summary -- and that is a detection, not a
        # broken harness. Requiring the summary line reported those as BROKEN,
        # which reads as "the mutant tested nothing" when the truth is "the
        # mutant hung the page and the driver said so by dying". The PASS lines
        # are the evidence that it got as far as running checks.
        ran = (("checks," in out) or ("ALL PASS" in out)
               or ("Traceback" in out and "  PASS  " in out))
        if ran:
            state = ("CAUGHT" if (r.returncode != 0 or "FAIL" in out
                                  or "TimeoutError" in out or "Traceback" in out)
                     else "MISSED")
        else:
            state = "BROKEN"
            tail = [l for l in out.strip().splitlines() if l.strip()][-1:]
            print("   !! the driver reported no result (exit %d): %s"
                  % (r.returncode, (tail[0] if tail else "(no output)")[:150]))
        results.append((tag, label, state))
        print("   -> %s" % state)
finally:
    write_page(original)
    drop_baseline()
    # Now a real claim: byte-exact, so this number can be compared with the one
    # taken before the run instead of being a number that merely looks right.
    ok_restore = os.path.getsize(HTML) == len(original.encode("utf-8"))
    print("\neditor.html restored (%d bytes, %s)"
          % (os.path.getsize(HTML),
             "byte-exact" if ok_restore else "SIZE CHANGED"))

print("\n" + "=" * 66)
for tag, label, state in results:
    print("  %-4s %-8s %s" % (tag, state, label))
print("%d of %d mutants caught"
      % (sum(1 for _, _, s in results if s == "CAUGHT"), len(results)))
sys.exit(0 if all(s == "CAUGHT" for _, _, s in results) else 1)
