"""Generate answers of one served model for the live-pair study.

Covers every prompt variant of every attack on the calibration behaviors plus the
JailbreakBench benign split, at a raised token limit so that long code-encryption answers
are not truncated. Raw text stays on the serving host and is consumed by live_adjudicate.py.
"""
import argparse, csv, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import calibrate_vuln as cv     # noqa: E402
import trial_attacks as ta      # noqa: E402


def benign_prompts(path, n):
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            g = (row.get("Goal") or "").strip()
            if g:
                out.append(g)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--benign", default=os.path.join(HERE, "..", "data", "JBB-Behaviors", "data",
                                                     "benign-behaviors.csv"))
    ap.add_argument("--n-benign", type=int, default=100)
    ap.add_argument("--benign-max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mid = cv._served_id(a.port)
    behaviors = cv.load_behaviors(a.n)
    tasks = []
    for atk in ta.attack_names():
        for b in behaviors:
            bsha = hashlib.sha1(b.encode()).hexdigest()[:10]
            for v, prompt in ta.variants(atk, b):
                tasks.append(("attack", atk, v, bsha, prompt, a.max_tokens))
    for g in benign_prompts(a.benign, a.n_benign):
        bsha = hashlib.sha1(g.encode()).hexdigest()[:10]
        tasks.append(("benign", "jbb_benign", "plain", bsha, g, a.benign_max_tokens))
    print("[gen] model=%s served_id=%s tasks=%d" % (a.name, mid, len(tasks)), flush=True)

    def work(t):
        kind, atk, v, bsha, prompt, mt = t
        rec = {"model": a.name, "kind": kind, "attack": atk, "variant": v, "bsha": bsha,
               "prompt": prompt}
        try:
            txt, fr = cv.generate(a.port, mid, prompt, max_tokens=mt)
            rec["text"], rec["finish"] = txt, fr
        except Exception as e:
            rec["error"] = str(e)[:160]
        return rec

    done = 0
    with open(a.out, "w") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for rec in ex.map(work, tasks):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if done % 100 == 0:
                print("  %d/%d" % (done, len(tasks)), flush=True)
    print("[gen] wrote %s (%d records)" % (a.out, done), flush=True)


if __name__ == "__main__":
    main()
