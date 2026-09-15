"""Judge generated answers with Llama Guard 3 and store verdicts only."""
import argparse, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import calibrate_vuln as cv     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True, help="raw jsonl files or globs")
    ap.add_argument("--guard-port", type=int, default=8006)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cv.GUARD_PORT = a.guard_port
    guard_id = cv._served_id(a.guard_port)
    files = [p for pat in a.raw for p in sorted(glob.glob(pat))]
    recs = []
    for p in files:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if "text" in r:
                    recs.append(r)
    print("[judge] guard=%s records=%d" % (guard_id, len(recs)), flush=True)

    def work(r):
        out = {k: r[k] for k in ("model", "kind", "attack", "variant", "bsha")}
        out["truncated"] = r.get("finish") == "length"
        try:
            out["verdict"] = cv.guard_verdict(guard_id, r["prompt"], r["text"])
        except Exception as e:
            out["verdict"] = "error"
            out["error"] = str(e)[:120]
        return out

    done = 0
    with open(a.out, "w") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(work, recs):
            f.write(json.dumps(out) + "\n")
            done += 1
            if done % 200 == 0:
                print("  %d/%d" % (done, len(recs)), flush=True)
    print("[judge] wrote %s (%d)" % (a.out, done), flush=True)


if __name__ == "__main__":
    main()
