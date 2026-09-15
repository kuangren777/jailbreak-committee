"""Feed the attacker's adapted prompts to the SECOND committee member.

The attacker in pair_adaptive.py optimized each prompt against the first member.
The committee routes the same query to the second member too. We generate its
answer and its Llama Guard verdict, so pair_detect.py can measure whether the
optimized attack transfers (both members broken) or the second member's answer
diverges enough for the disagreement detector to fire. Stores text + verdict.
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import calibrate_vuln as cv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--guard-port", type=int, default=8006)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cv.GUARD_PORT = a.guard_port
    mid = cv._served_id(a.port)
    guard_id = cv._served_id(a.guard_port)
    rows = [json.loads(l) for l in open(a.pair)]
    rows = [r for r in rows if r.get("adapted_prompt")]
    print("[2nd] member=%s guard=%s rows=%d" % (mid, guard_id, len(rows)), flush=True)

    def work(r):
        out = {"bsha": r["bsha"], "name": a.name, "orig_success": r["success"],
               "best_level": r["best_level"]}
        try:
            txt, fr = cv.generate(a.port, mid, r["adapted_prompt"], max_tokens=a.max_tokens)
            out["text"] = txt
            out["verdict"] = cv.guard_verdict(guard_id, r["behavior"], txt)
        except Exception as e:
            out["error"] = str(e)[:120]
        return out

    done = 0
    with open(a.out, "w") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(work, rows):
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            done += 1
    print("[2nd] wrote %s (%d)" % (a.out, done), flush=True)


if __name__ == "__main__":
    main()
