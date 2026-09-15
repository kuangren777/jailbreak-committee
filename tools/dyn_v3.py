"""Scheduling study v3 on the per-variant matrix (one prompt per query).

- EXP3-IX without discounting. The learning rate is tuned on a separate random tuning stream.
- Neu (2015) rate as a reference, eta = sqrt(2 ln K / (K T)) and gamma = eta / 2.
- Static baselines chosen on the observed matrix M, a pure minimax committee and a mixed
  (randomized) minimax schedule found by multiplicative weights.
- Out-of-sample bootstrap. Each draw samples a true matrix from the Jeffreys posterior of every
  cell, keeps every static choice and the attack streams fixed at their values on the observed M,
  and runs the scheduler on the true matrix with common seeds.
Usage: python3 dyn_v3.py <per_variant.json> <out.json> [boot_seeds]
"""
import itertools, json, sys
from multiprocessing import Pool
import numpy as np

import dyn_pv as D   # reads sys.argv[1] (per_variant.json) at import
from dhr.exp3_se import Exp3SE
from dhr.baselines import RandomScheduler
from run_dynamic import build_emergence_stream, run_policy

ETAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
BOOT_SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
NE, T = len(D.E), D.PHASES * D.PLEN
MODELS6 = ["Llama-2-7b", "Mistral-7B-v0.1", "Qwen-7B", "internlm-7b", "vicuna-7b", "Baichuan2-7b"]
ATK7 = ["flipattack", "codechameleon", "renellm", "deepinception", "cipherchat", "pap", "pasttense"]


def comonotone(S, c, e):
    k = len(c) // 2 + 1
    return float(sorted((S[i, e] for i in c), reverse=True)[k - 1])


def table(S, m, model):
    cs = list(itertools.combinations(range(len(D.POOL)), m))
    f = D.committee_breach if model == "ind" else comonotone
    return cs, np.array([[f(S, c, e) for e in range(NE)] for c in cs])


def mixed_minimax(A, iters=100_000):
    """min_w max_e w.A[:, e] over rows by multiplicative weights; returns bounds and weights."""
    n, k = A.shape
    lw, la = np.zeros(n), np.zeros(k)
    w_avg, a_avg = np.zeros(n), np.zeros(k)
    lr = np.sqrt(8 * np.log(max(n, k)) / iters)
    for _ in range(iters):
        w = np.exp(lw - lw.max()); w /= w.sum()
        a = np.exp(la - la.max()); a /= a.sum()
        w_avg += w; a_avg += a
        lw -= lr * (A @ a)
        la += lr * (w @ A)
    w_avg /= iters; a_avg /= iters
    return {"upper": float((w_avg @ A).max()), "lower": float((A @ a_avg).min()), "weights": w_avg.tolist()}


def tuning_stream(seed=123):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < T:
        out.extend([int(rng.integers(NE))] * int(rng.integers(80, 400)))
    return out[:T]


def exp3ix(K, eta, gamma=None):
    return lambda s: Exp3SE(K, horizon=T, variant="exp3ix", seed=s, share=0.0, eta=eta, gamma=gamma)


def run(B, stream, make, seeds):
    return np.array([run_policy(make(s), B, stream, np.random.default_rng(1000 + s))[0] for s in range(seeds)])


def neu_rate(K):
    eta = float(np.sqrt(2 * np.log(K) / (K * T)))
    return eta, eta / 2


def tune(m):
    cs, B = table(D.S0, m, "ind")
    st = tuning_stream()
    scores = {eta: float(run(B, st, exp3ix(len(cs), eta), 30).mean()) for eta in ETAS}
    return min(scores, key=scores.get), scores


def evaluate(m, model, stream_name, eta):
    cs, B = table(D.S0, m, model)
    K = len(cs)
    stream = D.cycle_stream(D.S0) if stream_name == "switching" else \
        build_emergence_stream(D.S0, cs, B.tolist(), D.PHASES, D.PLEN)
    on = lambda row: float(np.mean(row[stream]))
    minimax = int(np.argmin(B.max(axis=1)))
    mixed = mixed_minimax(B)
    ex = run(B, stream, exp3ix(K, eta), 30)
    ne, ng = neu_rate(K)
    exn = run(B, stream, exp3ix(K, ne, ng), 30)
    return {"random": float(run(B, stream, lambda s: RandomScheduler(K, seed=s), 30).mean()),
            "first_phase_fixed": on(B[int(np.argmin(B[:, stream[:D.PLEN]].mean(axis=1)))]),
            "minimax": on(B[minimax]), "minimax_worst": float(B[minimax].max()),
            "minimax_committee": [D.POOL[i] for i in cs[minimax]],
            "mixed": float(np.mean(np.array(mixed["weights"]) @ B[:, stream])), "mixed_worst": mixed["upper"],
            "mixed_lower": mixed["lower"], "mixed_weights": dict(zip(["+".join(D.POOL[i] for i in c) for c in cs],
                                                                     np.round(mixed["weights"], 3).tolist())),
            "oracle": float(np.mean(B[:, stream].min(axis=0))),
            "exp3ix": float(ex.mean()), "exp3ix_std": float(ex.std()),
            "exp3ix_neu_rate": float(exn.mean()), "neu_eta": ne}


def heldout_on_stream(m, drop="codechameleon"):
    keep = [j for j, (a, _) in enumerate(D.E) if a != drop]
    cs, Bi = table(D.S0, m, "ind")
    _, Bc = table(D.S0, m, "com")
    stream = D.cycle_stream(D.S0)
    bound = Bc[:, keep].max(axis=1)
    tied = [c for c in range(len(cs)) if abs(bound[c] - bound.min()) < 1e-12]
    sel = min(tied, key=lambda c: Bi[c, keep].max())
    return {"tied": [[D.POOL[i] for i in cs[c]] for c in tied], "selected": [D.POOL[i] for i in cs[sel]],
            "selected_switching_ind": float(Bi[sel, stream].mean()),
            "tied_switching_ind_range": [float(min(Bi[c, stream].mean() for c in tied)),
                                         float(max(Bi[c, stream].mean() for c in tied))]}


def boot(args):
    i, m, eta = args
    rng = np.random.default_rng(40_000 + 100 * m + i)
    X = np.rint(D.S0 * D.NB)
    truth = rng.beta(X + 0.5, D.NB - X + 0.5)
    cs, B_obs = table(D.S0, m, "ind")
    _, B_true = table(truth, m, "ind")
    stream = D.cycle_stream(D.S0)
    minimax = int(np.argmin(B_obs.max(axis=1)))
    w = np.array(mixed_minimax(B_obs, iters=20_000)["weights"])
    sched = run(B_true, stream, exp3ix(len(cs), eta), BOOT_SEEDS).mean()
    return {"minus_minimax": float(sched - B_true[minimax, stream].mean()),
            "minus_mixed": float(sched - (w @ B_true[:, stream]).mean())}


def ci(vals):
    v = np.array(vals)
    return {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "mean": float(v.mean()),
            "frac_below_zero": float((v < 0).mean())}


if __name__ == "__main__":
    out = {"boot_seeds": BOOT_SEEDS}
    full = np.array([[D.PV[f"{a}|{mod}"][v] for a in ATK7 for v in D.PV[f"{a}|{mod}"]] for mod in MODELS6])
    out["mixed_single_full_pool"] = mixed_minimax(full)
    for m in (1, 2):
        eta, scores = tune(m)
        out[f"tune_m{m}"] = {"eta": eta, "scores": scores}
        for model in ("ind", "com"):
            for name in ("switching", "targeting"):
                out[f"{name}_m{m}_{model}"] = evaluate(m, model, name, eta)
        out[f"heldout_codechameleon_m{m}"] = heldout_on_stream(m)
        with Pool(8) as p:
            res = p.map(boot, [(i, m, eta) for i in range(200)])
        out[f"bootstrap_m{m}"] = {"minus_minimax": ci([r["minus_minimax"] for r in res]),
                                  "minus_mixed": ci([r["minus_mixed"] for r in res])}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out, indent=1))
