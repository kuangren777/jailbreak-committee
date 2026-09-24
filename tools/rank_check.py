"""Does the calibration statistic W rank what the adjudicator actually releases?

For every pair, Spearman correlation between W (Eq. select, judge-level joint breach) and the
released worst-quarter ASR from tools/serving_replay.py, without a filter (union labels, tau
0.8) and behind the Llama Guard filter (filtered labels scored by Qwen3-14B, tau 0.5, the
threshold at matched service). Writes tools/rank_check.json.
"""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from committee_select import POOLS, tensor, load, consensus, joint, cvar  # noqa: E402

os.chdir(os.path.join(HERE, ".."))
sr = json.load(open("tools/serving_replay_scale.json"))
models, cols, behs, T = tensor({n: load(p, f) for p, f, n in POOLS["scale"]})
G, Q = T["guard"], T["qwen3"]
labels = {"either": consensus(G, Q, "either"), "filtered": np.where(np.isnan(Q) | np.isnan(G), np.nan, Q * (1 - G))}
w = np.ones(len(behs)) / len(behs)
rk = lambda x: np.argsort(np.argsort(x))
out = {}
for name, ck, judge, tau in (("either", "committees", "either", "0.8"), ("filtered", "committees_guard", "qwen3", "0.5")):
    W, R = [], []
    for S in itertools.combinations(range(len(models)), 2):
        W.append(cvar(joint(labels[name], S, w), 0.25))
        R.append(sr[ck]["+".join(models[i] for i in S)][judge][tau]["released_cvar"])
    W, R = np.array(W), np.array(R)
    out[name] = {"tau": tau, "spearman": float(np.corrcoef(rk(W), rk(R))[0, 1]),
                 "same_argmin": bool(np.argmin(W) == np.argmin(R))}
json.dump(out, open("tools/rank_check.json", "w"), indent=1)
print(out)
