"""Generate one model's answers to the PAIR-adapted prompts. Saves text, no judge."""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import calibrate_vuln as cv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mid = cv._served_id(a.port)
    rows = [r for r in (json.loads(l) for l in open(a.pair)) if r.get("adapted_prompt")]
    print("[gen] %s rows=%d" % (mid, len(rows)), flush=True)

    def work(r):
        o = {"bsha": r["bsha"], "name": a.name, "orig_success": r["success"],
             "behavior": r["behavior"], "prompt": r["adapted_prompt"]}
        try:
            o["text"], _ = cv.generate(a.port, mid, r["adapted_prompt"], max_tokens=a.max_tokens)
        except Exception as e:
            o["error"] = str(e)[:120]
        return o
    with open(a.out, "w") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for o in ex.map(work, rows):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("[gen] wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
