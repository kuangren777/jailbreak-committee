"""Recompute the selection and gate numbers of the paper from the released verdicts.

Runs tools/committee_select.py on the open pool (results/scale) and the hosted pool
(results/hub2026_scale), then prints each value the paper reports beside the value
recomputed here. Released worst-quarter ASR needs the answer embeddings, which are
withheld, so those two values are checked against tools/serving_replay_scale.json.
Needs numpy only.

Usage: python3 reproduce.py
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
f2 = lambda x: "%.2f" % x
BOUGHT, SINGLE, FILT = "Baichuan2-7b+Llama-2-7b", "Llama-2-7b", "Mistral-7B-v0.1+vicuna-7b"


def run(pool):
    out = os.path.join(tempfile.mkdtemp(), pool + ".json")
    subprocess.run([sys.executable, "tools/committee_select.py", pool, out], check=True, stdout=subprocess.DEVNULL)
    return json.load(open(out))


def main():
    o, h = run("scale"), run("hub2026_scale")
    unf = ("guard", "qwen3", "either", "both")
    rows = [
        ("pair bought under the union labels", BOUGHT, o["bought"]),
        ("W of the bought pair, union", "0.56", f2(o["cross"][BOUGHT]["either"])),
        ("W of the bought pair, Llama Guard", "0.26", f2(o["cross"][BOUGHT]["guard"])),
        ("W of the bought pair, Qwen3-14B", "0.43", f2(o["cross"][BOUGHT]["qwen3"])),
        ("best single model, union", SINGLE, o["judges"]["either"]["best_single"][0]),
        ("W of the best single model, union", "0.73", f2(o["cross_best_single"]["either"])),
        ("W of the best single model, Llama Guard", "0.48", f2(o["cross_best_single"]["guard"])),
        ("W of the best single model, Qwen3-14B", "0.66", f2(o["cross_best_single"]["qwen3"])),
        ("gate accepts the bought pair under all four label sets", "True",
         str(all(o["bought_gate"][j]["accept"] for j in unf))),
        ("largest margin delta at which the purchase holds", "0.15",
         "%.2f" % (min(o["bought_gate"][j]["gain_q"] for j in unf) - 0.005)),
        ("pair selected behind the filter", FILT, o["winners"]["filtered"]),
        ("W of that pair behind the filter", "0.08", f2(o["judges"]["filtered"]["top5"][0][1])),
        ("W of the best filtered single model", "0.26", f2(o["judges"]["filtered"]["best_single"][1])),
        ("margin below which the filtered purchase holds", "0.17", f2(o["judges"]["filtered"]["gate"]["gain_q"])),
        ("held-out W of JBS, Qwen3-14B", "0.42", f2(o["judges"]["qwen3"]["split_rules"]["joint_cvar"]["heldout_cvar"])),
        ("held-out W of the comonotone minimax, Qwen3-14B", "0.54",
         f2(o["judges"]["qwen3"]["split_rules"]["comonotone_minimax"]["heldout_cvar"])),
        ("held-out families where the pair beats the best single model, union", "4", str(o["judges"]["either"]["loao_wins"])),
        ("hosted pool, W of the best single model", "0.01", f2(h["judges"]["either"]["best_single"][1])),
        ("hosted pool, W of the best pair", "0.00",
         f2(min(c for _, c, _ in h["judges"]["either"]["top5"]))),
        ("hosted pool, draws in which the gain exceeds delta", "0",
         "%d" % round(max(j["gate"]["p_gain_gt_delta"] for j in h["judges"].values()) * 2000)),
    ]
    sr = json.load(open("tools/serving_replay_scale.json"))
    rows += [("released worst-quarter ASR of the bought pair at tau 0.8 (stored replay)", "0.23",
              f2(sr["committees"][BOUGHT]["either"]["0.8"]["released_cvar"])),
             ("released worst-quarter ASR of the best single model (stored replay)", "0.73",
              f2(sr["single"][SINGLE]["either"]["released_cvar"]))]
    ok = 0
    for name, paper, here in rows:
        match = paper == here
        ok += match
        print(f"{'ok ' if match else 'DIFF'}  {name:<72} paper {paper:<26} here {here}")
    print(f"\n{ok} of {len(rows)} checks reproduce")
    sys.exit(0 if ok == len(rows) else 1)


if __name__ == "__main__":
    main()
