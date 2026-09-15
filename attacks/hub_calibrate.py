"""Calibrate a 2026 model pool through the gateway with the same protocol as the local pool.

Same 20 prompt variants of the seven attacks and the same 25 AdvBench behaviors in the
same fixed hash order, so the 2026 matrix is directly comparable with the 2023 one.
Writes one jsonl per model with the response text; verdicts are added by a separate
judging pass, and only aggregates ever enter the repo.
"""
import argparse, hashlib, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import hub_client as hub          # noqa: E402
import trial_attacks as ta        # noqa: E402
import calibrate_vuln as cv       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gateway model id")
    ap.add_argument("--name", required=True, help="short name used in the matrix")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    behaviors = cv.load_behaviors(a.n)
    tasks = []
    for atk in ta.attack_names():
        for b in behaviors:
            bsha = hashlib.sha1(b.encode()).hexdigest()[:10]
            for v, prompt in ta.variants(atk, b):
                tasks.append((atk, v, bsha, prompt))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                if "text" in r:
                    done.add((r["attack"], r["variant"], r["bsha"]))
            except Exception:
                pass
    todo = [t for t in tasks if (t[0], t[1], t[2]) not in done]
    print("[hub] %s tasks=%d todo=%d" % (a.name, len(tasks), len(todo)), flush=True)

    lock = threading.Lock()
    f = open(a.out, "a")
    counter = {"n": 0, "err": 0}

    def work(t):
        atk, v, bsha, prompt = t
        rec = {"model": a.name, "gateway_id": a.model, "kind": "attack",
               "attack": atk, "variant": v, "bsha": bsha, "prompt": prompt}
        try:
            rec["text"], rec["finish"] = hub.chat(a.model, prompt, max_tokens=a.max_tokens)
        except Exception as e:
            rec["error"] = str(e)[:200]
        with lock:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            counter["n"] += 1
            counter["err"] += ("error" in rec)
            if counter["n"] % 50 == 0:
                print("  %d/%d err=%d" % (counter["n"], len(todo), counter["err"]), flush=True)
        return None

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    f.close()
    print("[hub] %s done, errors=%d" % (a.name, counter["err"]), flush=True)


if __name__ == "__main__":
    main()
