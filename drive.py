"""Drive a job through the running server and report the result.

    python drive.py <json-config-file-or-inline-json>
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8765"


def post(path, obj, raw=None):
    data = raw if raw is not None else json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=30))


def main():
    cfg = json.loads(open(sys.argv[1]).read() if sys.argv[1].endswith(".json")
                     else sys.argv[1])
    r = post("/api/run", cfg)
    if not r.get("ok"):
        print("run rejected:", r)
        return 1
    jid = r["id"]
    print("job", jid)
    last = None
    while True:
        j = get("/api/job/" + jid)["job"]
        key = (j["stage"], round(j["frac"], 2))
        if key != last:
            print("  %-8s %5.1f%%  %s" % (j["stage"], j["frac"] * 100,
                                          j["log"][-1][13:] if j["log"] else ""))
            last = key
        if j["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.5)
    print("\nstatus:", j["status"])
    if j["error"]:
        print("error :", j["error"])
        for line in j["log"][-12:]:
            print("   ", line)
        return 1
    res = j["result"]
    print("run    :", res["run"])
    print("frames :", res["frames"])
    L = res["layout"]
    print("grid   : %dx%d of %dx%d -> sheet %dx%d"
          % (L["columns"], L["rows"], L["cell_w"], L["cell_h"], L["sheet_w"], L["sheet_h"]))
    print("crop   :", L["crop"], " bbox:", res["bbox"])
    if L["warnings"]:
        print("warns  :", L["warnings"])
    print("artifacts:", res["artifacts"])
    v = res.get("verify") or {}
    print("verify :", "PASS" if v.get("ok") else ("FAIL" if v.get("ok") is False else "skipped"))
    for line in (v.get("lines") or []):
        print("   ", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
