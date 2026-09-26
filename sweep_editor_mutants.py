"""Run every mutation hook in test_editor.py and classify what happened.

Each hook deliberately breaks one guard and re-runs the whole suite. The suite
must then FAIL. Three outcomes, and only one of them is a pass:

  0  MISSED   the mutant survived -- the guard the hook broke is not actually
              covered by an assertion, so the suite would not have noticed the
              real bug.
  1  CAUGHT   the suite failed, which is the expected outcome.
  2  BROKEN   the hook could not apply -- the source it anchors on has moved.
              This is NOT a pass. A hook that cannot apply tests nothing, and
              reports the same green as a hook that works.

`export-mutates-doc` is why this exists: its regex required `return sub, idx` to
follow `_set_cells(sub, ...)` in export_subset, and when the per-frame audio
feature inserted the clip carry-over between those two lines the regex stopped
matching. The hook then printed "did not apply", which read as noise rather than
a failure, and it stayed silently dead for a session. Classifying by exit code
makes a dead hook loud.

Run:  python sweep_editor_mutants.py
      python sweep_editor_mutants.py --timeout 900
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(HERE, "test_editor.py")

MISSED, CAUGHT, BROKEN = "MISSED", "CAUGHT", "BROKEN"
CAUGHT_BY_TRACEBACK = "CAUGHT*"
# Every verdict that means the suite refused to pass. The star is not a
# different outcome, just a coarser one: the suite died on an uncaught exception
# instead of failing a labelled check.
CAUGHT_ANY = (CAUGHT, CAUGHT_BY_TRACEBACK)


def hooks(path):
    """Every mutation name the suite knows, read from the suite itself.

    Read rather than listed, so a hook added to test_editor.py is swept without
    anyone remembering to add it here -- a hard-coded list would go stale the
    first time someone added a hook and is exactly the failure this tool exists
    to prevent, one level up.
    """
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    found = re.findall(r'MUT == "([^"]+)"', src)
    # `elif MUT:` and `if MUT:` are the fallthroughs, not hooks.
    return [h for h in dict.fromkeys(found) if h]


def classify(code, out):
    """Exit code -> verdict.

    Exit 1 is a catch whether or not the suite reached its summary line. Three
    hooks in this suite (`store-post-n`, `restore-drop-orig`,
    `open-folder-ignored`) are caught by an uncaught exception in the fixtures --
    `KeyError: 5`, `IndexError`, `PermissionError` -- so the suite dies with a
    traceback and never prints "N checks". That is a cruder failure than a named
    assertion, and the README says so, but it is still the suite refusing to
    pass, which is what a mutation hook is for. Counting them as errors would
    have this tool report three false alarms on every run, and a tool that cries
    wolf gets ignored -- which is the same failure mode it was written to fix.
    """
    if code == 2:
        return BROKEN
    if code == 0:
        return MISSED
    if code == 1:
        return CAUGHT if "checks," in out else CAUGHT_BY_TRACEBACK
    return "ERROR(%d)" % code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds per hook")
    args = ap.parse_args()

    names = hooks(SUITE)
    if not names:
        print("no mutation hooks found in %s -- the pattern has moved" % SUITE)
        return 2
    print("sweeping %d hooks from %s\n" % (len(names), os.path.basename(SUITE)))

    tally = {}
    bad = []
    t0 = time.time()
    for name in names:
        started = time.time()
        try:
            p = subprocess.run([sys.executable, SUITE, name], cwd=HERE,
                               capture_output=True, text=True,
                               timeout=args.timeout)
            code, out = p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            code, out = 3, "timed out after %ds" % args.timeout
        verdict = classify(code, out)
        tally[verdict] = tally.get(verdict, 0) + 1
        if verdict not in CAUGHT_ANY:
            bad.append((name, verdict, out))
        # The failing assertion is the interesting part of a CAUGHT hook: it says
        # which guard held. Worth one line even on a pass.
        first = ""
        if verdict == CAUGHT:
            m = re.search(r"^  FAIL  (.*)$", out, re.M)
            first = "  <- %s" % (m.group(1)[:80] if m else "?")
        elif verdict == CAUGHT_BY_TRACEBACK:
            m = re.search(r"^(\w*(?:Error|Exception)): (.*)$", out, re.M)
            first = "  <- %s" % (m.group(0)[:80] if m else "traceback")
        print("  %-8s %-34s %5.1fs%s"
              % (verdict, name, time.time() - started, first))

    print("\n" + "=" * 66)
    print("%d hooks: %s" % (len(names), ", ".join(
        "%d %s" % (n, k) for k, n in sorted(tally.items()))))
    print("total %.0fs" % (time.time() - t0))
    if bad:
        print("\nnot caught:")
        for name, verdict, out in bad:
            print("  %s  %s" % (verdict, name))
            for line in out.strip().splitlines()[:6]:
                print("      " + line[:160])
        return 1
    print("every hook is caught (a * means by a traceback rather than a "
          "labelled check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
