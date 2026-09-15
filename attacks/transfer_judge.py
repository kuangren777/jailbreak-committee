"""Judge stored transfer generations with Llama Guard 3. Adds verdict, keeps no text."""
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import calibrate_vuln as cv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--guard-port", type=int, default=8006)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    cv.GUARD_PORT = a.guard_port
    gid = cv._served_id(a.guard_port)
    files = [p for pat in a.raw for p in sorted(glob.glob(pat))]
    print("[judge] guard=%s files=%d" % (gid, len(files)), flush=True)
    for path in files:
        rows = [r for r in (json.loads(l) for l in open(path)) if "text" in r]

        def work(r):
            o = {k: r[k] for k in ("bsha", "name", "orig_success")}
            try:
                o["verdict"] = cv.guard_verdict(gid, r["behavior"][:a.max_chars], r["text"][:a.max_chars])
            except Exception as e:
                o["verdict"] = "error"; o["error"] = str(e)[:100]
            return o
        out = os.path.join(a.outdir, "transfer_%s.jsonl" % rows[0]["name"])
        with open(out, "w") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
            for o in ex.map(work, rows):
                f.write(json.dumps(o) + "\n")
        print("[judge] wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
