"""Checks raised in the second review, all on the 200-behavior open pool (results/scale).

1. ablation: 2x2 held-out split of {joint breach, comonotone bound min(ASR_a, ASR_b)} x
   {max over prompts, worst-quarter CVaR}, same 200 random halves as split_eval.
2. reselect: gate with the committee re-selected inside every bootstrap draw, behavior-only
   and behavior x prompt (two-way) resampling. Gain = W(best single) - W(re-selected pair),
   both scored on the draw.
3. rules_ci: paired held-out difference W(rule pick) - W(JBS pick) with 90% interval for every
   rule of Table 1.
4. tau_rank: Spearman between W of the 15 pairs and their released worst-quarter ASR, per tau.
Writes tools/review_checks.json.
"""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from committee_select import POOLS, tensor, load, consensus, rule_choices, joint, cvar, select, DELTA  # noqa: E402

os.chdir(os.path.join(HERE, ".."))
ALPHA = 0.25
models, cols, behs, T = tensor({n: load(p, f) for p, f, n in POOLS["scale"]})
G, Q = T["guard"], T["qwen3"]
L = {"guard": G, "qwen3": Q, "either": consensus(G, Q, "either"), "both": consensus(G, Q, "both"),
     "filtered": np.where(np.isnan(Q) | np.isnan(G), np.nan, Q * (1 - G))}
nb, nc, n = len(behs), len(cols), len(models)
pairs = list(itertools.combinations(range(n), 2))
singles = [(i,) for i in range(n)]


def marg(X, w):
    return np.nansum(X * w, axis=2) / np.maximum(np.sum(~np.isnan(X) * w, axis=2), 1e-12)


def agg(v, how):
    return float(v.max()) if how == "max" else cvar(v, ALPHA)


def pick(X, w, stat, how):
    if stat == "joint":
        return min(pairs, key=lambda p: (agg(joint(X, p, w), how), cvar(joint(X, p, w), ALPHA)))
    P = marg(X, w)
    return min(pairs, key=lambda p: (agg(np.minimum(P[p[0]], P[p[1]]), how), cvar(np.minimum(P[p[0]], P[p[1]]), ALPHA)))


def halves(reps=200, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        perm = rng.permutation(nb); tr, te = perm[: nb // 2], perm[nb // 2:]
        w_tr = np.zeros(nb); w_tr[tr] = 1 / len(tr); w_te = np.zeros(nb); w_te[te] = 1 / len(te)
        yield w_tr, w_te, rng


def ablation(X):
    acc = {f"{s}_{h}": [] for s in ("joint", "comono") for h in ("max", "cvar")}
    for w_tr, w_te, rng in halves():
        for s in ("joint", "comono"):
            for h in ("max", "cvar"):
                acc[f"{s}_{h}"].append(cvar(joint(X, pick(X, w_tr, s, h), w_te), ALPHA))
        rng.integers(15)
    base = np.array(acc["joint_cvar"])
    return {k: {"heldout_W": round(float(np.mean(v)), 3),
                "diff_vs_jbs": round(float(np.mean(np.array(v) - base)), 3),
                "lo": round(float(np.quantile(np.array(v) - base, .05)), 3),
                "hi": round(float(np.quantile(np.array(v) - base, .95)), 3)} for k, v in acc.items()}


def reselect(X, two_way, draws=1000, seed=0, rho=0.9):
    rng = np.random.default_rng(seed); g = []; same = 0
    win0 = select(X, pairs, np.ones(nb) / nb, ALPHA)[0][2]
    for _ in range(draws):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb) / nb
        Xd = X[:, rng.integers(0, nc, nc)] if two_way else X
        S = select(Xd, pairs, w, ALPHA)[0][2]; same += S == win0
        g.append(min(cvar(joint(Xd, s, w), ALPHA) for s in singles) - cvar(joint(Xd, S, w), ALPHA))
    g = np.array(g)
    return {"gain_mean": round(float(g.mean()), 3), "gain_q10": round(float(np.quantile(g, 1 - rho)), 3),
            "p_gain_gt_delta": round(float((g > DELTA).mean()), 3), "p_gain_pos": round(float((g > 0).mean()), 3),
            "accept": bool((g > DELTA).mean() >= rho), "same_winner": round(same / draws, 3)}


def rules_ci(X):
    d = {}
    for w_tr, w_te, rng in halves():
        ch = rule_choices(X, w_tr, ALPHA); ch["random_pair"] = pairs[rng.integers(len(pairs))]
        jb = cvar(joint(X, ch["joint_cvar"], w_te), ALPHA)
        for r in ("comonotone_minimax", "min_mean_asr", "max_disagreement", "random_pair"):
            d.setdefault(r, []).append(cvar(joint(X, ch[r], w_te), ALPHA) - jb)
    return {r: {"mean": round(float(np.mean(v)), 3), "lo": round(float(np.quantile(v, .05)), 3),
                "hi": round(float(np.quantile(v, .95)), 3)} for r, v in d.items()}


def tau_rank():
    from scipy.stats import spearmanr
    rep = json.load(open("tools/serving_replay_scale.json"))
    X = L["either"]; w = np.ones(nb) / nb
    short = {m: m for m in models}
    out = {}
    for tau in ("0.5", "0.6", "0.7", "0.75", "0.8", "0.85", "0.9"):
        Ws, Rs = [], []
        for p in pairs:
            key = "+".join(short[models[i]] for i in p)
            if key not in rep["committees"]:
                continue
            Ws.append(cvar(joint(X, p, w), ALPHA)); Rs.append(rep["committees"][key]["either"][tau]["released_cvar"])
        out[tau] = {"rho": round(float(spearmanr(Ws, Rs)[0]), 3), "n": len(Ws)}
    return out


res = {"ablation": {}, "reselect": {}, "rules_ci": {}}
for name, X in L.items():
    res["ablation"][name] = ablation(X)
    res["reselect"][name] = {"behavior": reselect(X, False), "two_way": reselect(X, True)}
    res["rules_ci"][name] = rules_ci(X)
    print(name, json.dumps({k: res[k][name] for k in res}), flush=True)
res["tau_rank"] = tau_rank(); print(res["tau_rank"])
json.dump(res, open("tools/review_checks.json", "w"), indent=1)
