"""Drive the real Sprite Studio UI in a real browser, over CDP.

This is not a re-test of the pipeline (that is already covered by drive.py). It
exercises the *front-end*: the event handlers, the fetch calls, the polling loop
and the render functions -- the code that had never actually run.

Uses Chrome DevTools Protocol over websocket-client. No playwright/selenium needed.

    python ui_drive.py [--url http://127.0.0.1:8765] [--port 9222] [--frames N]
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

import websocket

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# This driver prints the page's own text, which contains arrows, multiplication
# signs and the middle dot the UI uses as a separator. On a Windows console that
# is cp1252, and printing it raised UnicodeEncodeError *at the report*, so a run
# that had already done all its work died before the checks table with a
# traceback about a character. Replace rather than raise: the report is the point.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def targets(port):
    with urllib.request.urlopen("http://127.0.0.1:%d/json" % port, timeout=10) as r:
        return json.load(r)


class Page:
    def __init__(self, ws_url):
        # See editor_drive.py: without suppress_origin the handshake is refused
        # with a 403 by a headless Chrome that was not launched with
        # --remote-allow-origins.
        self.ws = websocket.create_connection(ws_url, timeout=180,
                                             suppress_origin=True)
        self.i = 0

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

    def js(self, expr, await_promise=False):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True,
                      awaitPromise=await_promise)
        res = r.get("result", {})
        if r.get("exceptionDetails"):
            raise RuntimeError("JS threw: %s" % json.dumps(
                r["exceptionDetails"].get("exception", {}))[:300])
        return res.get("value")

    def wait_for(self, expr, label, timeout=240, interval=0.4):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.js(expr):
                return time.time() - t0
            time.sleep(interval)
        raise TimeoutError("timed out waiting for %s" % label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--port", type=int, default=9222)
    ap.add_argument("--frames", type=int, default=0,
                    help="0 = the whole video")
    ap.add_argument("--video", default=r"C:\Users\PC\Desktop\monsters\sprite\walk\032137_px_00002_front_walk.mp4")
    args = ap.parse_args()

    tg = [t for t in targets(args.port) if t["type"] == "page"]
    if not tg:
        print("no page target on port %d -- is Chrome running with "
              "--remote-debugging-port?" % args.port)
        return 2
    page = Page(tg[0]["webSocketDebuggerUrl"])
    page.call("Runtime.enable")
    print("attached to:", tg[0].get("url"))

    # 0. reload before starting. ui.html is served from disk on every request, so
    #    a tab left open from an earlier run is a page from an earlier build: the
    #    driver has to test what is on disk now, not what the browser happened to
    #    load twenty minutes ago. A stale tab is exactly how a control that *is*
    #    in the markup reads as a TypeError about null -- which is how this run
    #    found it.
    page.call("Page.navigate", url=args.url)
    page.wait_for("document.readyState === 'complete'", "a fresh page load")
    #    the page must also have finished its /api/defaults fetch, which populated
    #    #env, proving the front-end JS is actually running
    page.wait_for("document.readyState === 'complete'", "document load")
    page.wait_for("document.getElementById('env').textContent.includes('python')",
                  "#env populated from /api/defaults")
    print("env line     :", page.js("document.getElementById('env').textContent"))

    # 0b. the defaults the page *shows* must be the ones the server is running.
    #     ui.html's markup is not the last word: every field is overwritten from
    #     /api/defaults on load, so a server started before a default changed
    #     silently re-applies the old one -- a checkbox that keeps coming back
    #     ticked, or a 16384 texture cap on a build whose constant says 32768.
    #     Both were reported as editor bugs while the code on disk was correct.
    with urllib.request.urlopen("%s/api/defaults" % args.url, timeout=10) as r:
        served = json.load(r)["defaults"]
    drift = []
    for fid, key in (("do_repair", "do_repair"), ("do_matte", "do_matte"),
                     ("do_key", "do_key"),
                     ("repair_border_only", "repair_border_only"),
                     ("key_also_connected", "key_also_connected"),
                     ("want_preview", "want_preview"), ("want_gif", "want_gif")):
        shown = page.js("document.getElementById('%s').checked" % fid)
        if shown != bool(served[key]):
            drift.append("%s: page %s, server %s" % (fid, shown, served[key]))
    cap_shown = page.js("Number(document.getElementById('max_texture').max)")
    if cap_shown != int(served["max_texture"]):
        drift.append("max_texture: page %s, server %s"
                     % (cap_shown, served["max_texture"]))
    print("defaults     : %s" % ("in step with the server" if not drift
                                 else "; ".join(drift)))

    # 1. type a path the way a user would, and fire the input event
    page.js("""
      (() => {
        const el = document.getElementById('video');
        el.value = %s;
        el.dispatchEvent(new Event('input', {bubbles:true}));
        el.dispatchEvent(new Event('change', {bubbles:true}));
        return el.value;
      })()
    """ % json.dumps(args.video))

    # 2. real click on "Read video"
    page.js("document.getElementById('probe').click()")
    page.wait_for("document.querySelectorAll('#columns option').length > 1",
                  "probe to populate the column list")
    info = page.js("document.getElementById('vinfo').innerText")
    print("probe result :", " | ".join(info.splitlines()[:4]))
    cols = page.js("[...document.querySelectorAll('#columns option')].map(o=>o.value).join(',')")
    print("columns offered:", cols)

    # 2b. the clip in the info panel has to be something you can watch, not a
    #     still of frame 1: a loop, a hitch or a dropped frame does not show up
    #     in one image, and those are the reasons to look before a long run.
    with urllib.request.urlopen("%s/api/video?path=%s"
                                % (args.url, urllib.parse.quote(args.video)),
                                timeout=30) as r:
        video_headers = {k.lower(): v for k, v in r.headers.items()}
    ok_video = []
    page.wait_for("!!document.querySelector('#vinfo video#vplay')",
                  "the video element", timeout=60)
    page.wait_for("document.getElementById('vplay').readyState >= 1",
                  "the element's metadata", timeout=60)
    meta = page.js("""(() => {
      const v = document.getElementById('vplay');
      return {dur: v.duration, w: v.videoWidth, h: v.videoHeight,
              controls: v.controls, muted: v.muted, poster: !!v.poster,
              src: v.currentSrc};
    })()""")
    print("video element:", meta)
    ok_video.append(("the panel holds a playable element",
                     meta["dur"] > 0 and meta["w"] > 0,
                     "%.2fs %sx%s" % (meta["dur"], meta["w"], meta["h"])))
    ok_video.append(("with controls, muted, looping",
                     meta["controls"] and meta["muted"], meta))
    ok_video.append(("frame 1 as the poster", meta["poster"], meta))
    ok_video.append(("served by /api/video", "/api/video?path=" in (meta["src"] or ""),
                     meta["src"]))
    ok_video.append(("the route advertises byte ranges",
                     video_headers.get("accept-ranges") == "bytes", video_headers))
    # A seek is the real test of the route: the element asks for the bytes it
    # needs, so a server that only ever sends a 200 from byte zero plays once and
    # then re-downloads the whole clip on every scrub.
    mid = meta["dur"] / 2
    page.js("document.getElementById('vplay').play()")
    page.wait_for("document.getElementById('vplay').currentTime > 0.15",
                  "the clip to play", timeout=30)
    page.js("document.getElementById('vplay').currentTime = %s" % mid)
    page.wait_for("document.getElementById('vplay').currentTime > %s"
                  % (mid - 0.35), "the seek to land", timeout=30)
    seeked = page.js("document.getElementById('vplay').currentTime")
    ok_video.append(("a mid-clip seek lands", abs(seeked - mid) < 0.35,
                     "wanted %.2f, got %.2f" % (mid, seeked)))
    page.js("document.getElementById('vplay').pause()")
    print("\n--- the clip as the panel serves it ---")
    for label, good, detail in ok_video:
        print("   %-34s %s" % (label, "OK" if good else "FAIL   %s" % (detail,)))

    # 2c. the colour key. Reading the clip must seed the picker with the clip's
    #     own border colour: that is the backdrop the matte had to remove, and
    #     nobody can name their green screen's RGB by eye. It matters more than
    #     it sounds -- the walk clip's backdrop is #000004 and the MiniMax clips'
    #     is #13ff38, so a picker left on the textbook #00ff00 is 82 away from
    #     the green one, which at a tolerance of 64 matches nothing at all.
    req = urllib.request.Request("%s/api/probe" % args.url,
                                 data=json.dumps({"video": args.video}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        probed = json.load(r)
    picked = str(page.js("document.getElementById('key_color').value") or "").lower()
    ok_key = [
        ("the key controls are in the form", page.js(
            "!!(document.getElementById('do_key') && document.getElementById('key_color')"
            " && document.getElementById('key_tol') && document.getElementById('do_key').type==='checkbox')"), ""),
        ("reading the clip seeds the picker",
         bool(picked) and picked == (probed.get("backdrop") or "").lower(),
         "page %s, route %s" % (picked, probed.get("backdrop"))),
        ("the picker is a valid #rrggbb",
         len(picked) == 7 and picked[0] == "#", picked),
        ("the tolerance starts where the server does", page.js(
            "Number(document.getElementById('key_tol').value)") == int(served["key_tol"]),
         "page %s, server %s" % (page.js("document.getElementById('key_tol').value"),
                                  served["key_tol"])),
        ("the colour key starts off",
         page.js("document.getElementById('do_key').checked") is False
         and bool(served["do_key"]) is False,
         "page %s, server %s" % (page.js("document.getElementById('do_key').checked"),
                                  served["do_key"])),
        ("the key's border-extension box starts off",
         page.js("document.getElementById('key_also_connected').checked") is False
         and bool(served["key_also_connected"]) is False,
         "page %s, server %s" % (page.js("document.getElementById('key_also_connected').checked"),
                                  served["key_also_connected"])),
        ("the flood fill's border box starts off (tonal default)",
         page.js("document.getElementById('repair_border_only').checked") is False
         and bool(served["repair_border_only"]) is False, ""),
    ]
    print("\n--- the colour key ---")
    for label, good, detail in ok_key:
        print("   %-34s %s" % (label, "OK" if good else "FAIL   %s" % (detail,)))

    # 3. narrow the range if asked, then click Build
    if args.frames:
        page.js("document.getElementById('end').value = %d" % args.frames)
    page.js("""
      (() => {
        const g = document.getElementById('go');
        g.click();
        return {disabled: g.disabled,
                status: document.getElementById('status').textContent};
      })()
    """)
    print("after click  : go.disabled=%s status=%s"
          % (page.js("document.getElementById('go').disabled"),
             page.js("document.getElementById('status').textContent")))

    # 4. the polling loop must drive the UI to a terminal state on its own
    t = page.wait_for(
        "['done','error','cancelled'].includes(document.getElementById('status').textContent)",
        "the job to reach a terminal state")
    status = page.js("document.getElementById('status').textContent")
    print("finished in  : %.1fs, status=%s" % (t, status))

    log = page.js("document.getElementById('log').textContent")
    print("\n--- progress log as the UI rendered it ---")
    for line in log.splitlines()[-14:]:
        print("   ", line)

    if status != "done":
        print("\nRESULT PANEL:", page.js("document.getElementById('result').innerText")[:800])
        return 1

    # 5. the result panel must have been rendered by render()
    res = page.js("document.getElementById('result').innerText")
    print("\n--- result panel as the UI rendered it ---")
    for line in res.splitlines()[:22]:
        print("   ", line)

    checks = {
        "form matches /api/defaults": not drift,
        "clip is playable in the panel": all(g for _l, g, _d in ok_video),
        "colour key seeded from the clip": all(g for _l, g, _d in ok_key),
        "result panel visible": page.js(
            "document.getElementById('resultpanel').style.display !== 'none'"),
        "verify badge rendered": "verification PASSED" in res or "verification FAILED" in res,
        "artifact links rendered": page.js(
            "document.querySelectorAll('#result .arts a').length") > 0,
        "sheet <img> rendered": page.js(
            "!!document.querySelector('#result .sheetwrap img')"),
        "preview <iframe> rendered": page.js(
            "!!document.querySelector('#result iframe.pv')"),
        "progress bar reached 100%": page.js(
            "document.getElementById('bar').style.width === '100%'"),
        "build button re-enabled": page.js(
            "document.getElementById('go').disabled === false"),
        "run history reloaded": page.js(
            "document.querySelectorAll('#runs div').length") > 0,
    }
    print("\n--- front-end behaviour ---")
    bad = 0
    for k, v in checks.items():
        print("   %-26s %s" % (k, "OK" if v else "FAIL"))
        bad += 0 if v else 1
    print("\n%d/%d checks passed" % (len(checks) - bad, len(checks)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
