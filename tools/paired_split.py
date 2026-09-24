"""Paired held-out difference between the joint-breach rule and the comonotone minimax rule.

Same random halves as split_eval (seed 0, 200 splits): on each split both rules choose on one
half and are scored on the other; report the mean and 90% interval of W(comonotone pick) -
W(joint pick) per label set. Also breaks down unparsed Qwen3-14B verdicts by model and attack.
Writes tools/paired_split.json.
"""
import collections, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from committee_select import POOLS, tensor, load, consensus, rule_choices, joint, cvar  # noqa: E402

os.chdir(os.path.join(HERE, ".."))
models, cols, behs, T = tensor({n: load(p, f) for p, f, n in POOLS["scale"]})
G, Q = T["guard"], T["qwen3"]
L = {"guard": G, "qwen3": Q, "either": consensus(G, Q, "either"),
     "filtered": np.where(np.isnan(Q) | np.isnan(G), np.nan, Q * (1 - G))}
nb = len(behs); out = {"diff": {}}
for name, X in L.items():
    rng = np.random.default_rng(0); d = []
    for _ in range(200):
        perm = rng.permutation(nb); tr, te = perm[: nb // 2], perm[nb // 2:]
        w_tr = np.zeros(nb); w_tr[tr] = 1 / len(tr); w_te = np.zeros(nb); w_te[te] = 1 / len(te)
        ch = rule_choices(X, w_tr, 0.25)
        d.append(cvar(joint(X, ch["comonotone_minimax"], w_te), .25) - cvar(joint(X, ch["joint_cvar"], w_te), .25))
        rng.integers(15)                       # keep the random stream aligned with split_eval
    d = np.array(d)
    out["diff"][name] = {"mean": float(d.mean()), "lo": float(np.quantile(d, .05)), "hi": float(np.quantile(d, .95)),
                         "p_pos": float((d > 0).mean()), "p_neg": float((d < 0).mean())}
miss = np.isnan(Q)
bym = {models[i]: float(miss[i].mean()) for i in range(len(models))}
bya = collections.defaultdict(list)
for j, (a, v) in enumerate(cols):
    bya[a].append(miss[:, j].mean())
bya = {a: float(np.mean(v)) for a, v in bya.items()}
cell = {f"{models[i]}|{cols[j][0]}/{cols[j][1]}": float(miss[i, j].mean()) for i in range(len(models)) for j in range(len(cols))}
top = max(cell.items(), key=lambda kv: kv[1])
out["missing"] = {"total": int(miss.sum()), "rate": float(miss.mean()), "by_model": bym, "by_attack": bya, "top_cell": top}
json.dump(out, open("tools/paired_split.json", "w"), indent=1)
print(json.dumps(out["diff"], indent=1)); print(json.dumps({k: out["missing"][k] for k in ("total", "rate", "by_model", "by_attack", "top_cell")}, indent=1))
