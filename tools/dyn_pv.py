"""Dynamic-scheduling study on the per-variant matrix (one prompt per query).

Each (attack, variant) pair is one attack of the stream, so all committee members face the
same prompt. Reuses committee_breach / stream builders / run_policy from
ICASSP-3/attacks/run_dynamic.py. Usage: python3 dyn_pv.py <per_variant.json> <out.json>
"""
import os
import itertools, json, sys
from multiprocessing import Pool
import numpy as np

REPO = os.environ.get("ICASSP3_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/attacks")
from dhr.exp3_se import Exp3SE                                                # noqa: E402
from dhr.baselines import RandomScheduler, RoundRobinScheduler, FixedScheduler  # noqa: E402
from run_dynamic import (committee_breach, build_emergence_stream, run_policy,  # noqa: E402
                         CURATED_POOL, CURATED_ATTACKS)

PV = json.load(open(sys.argv[1]))
POOL, ATK = CURATED_POOL, CURATED_ATTACKS
VARS = {a: list(PV[f"{a}|{POOL[0]}"].keys()) for a in ATK}
E = [(a, v) for a in ATK for v in VARS[a]]
S0 = np.array([[PV[f"{a}|{m}"][v] for (a, v) in E] for m in POOL])
PHASES, PLEN, SEEDS, NB = 8, 250, 30, 25


def breach_table(S, m):
    cs = list(itertools.combinations(range(len(POOL)), m))
    return cs, [[committee_breach(S, c, e) for e in range(len(E))] for c in cs]


def best_variant(a, S):
    idx = [j for j, (aa, _) in enumerate(E) if aa == a]
    return max(idx, key=lambda j: S[:, j].mean())


def cycle_stream(S):
    ids = [best_variant("codechameleon", S), best_variant("renellm", S)]
    return [ids[p % 2] for p in range(PHASES) for _ in range(PLEN)]


def evaluate(S, m, stream, seeds=SEEDS, configs=None, series=False):
    cs, B = breach_table(S, m)
    K, T = len(cs), len(stream)
    mean_on = lambda c: float(np.mean([B[c][f] for f in stream]))
    minimax = min(range(K), key=lambda c: max(B[c]))
    first = min(range(K), key=lambda c: np.mean([B[c][f] for f in stream[:PLEN]]))
    hind = min(range(K), key=mean_on)
    res = {"minimax": {"asr": mean_on(minimax), "committee": [POOL[i] for i in cs[minimax]]},
           "hindsight": {"asr": mean_on(hind), "committee": [POOL[i] for i in cs[hind]]},
           "oracle": float(np.mean([min(B[c][f] for c in range(K)) for f in stream]))}
    makers = {"random": lambda s: RandomScheduler(K, seed=s),
              "round_robin": lambda s: RoundRobinScheduler(K, seed=s),
              "first_phase_fixed": lambda s: FixedScheduler(K, arm=first, seed=s)}
    for name, (eta, share) in (configs or {}).items():
        makers[name] = (lambda eta, share: lambda s: Exp3SE(K, horizon=T, variant="exp3ix", seed=s,
                                                             share=share, eta=eta))(eta, share)
    for name, mk in makers.items():
        vals, ser0 = [], None
        for s in range(seeds):
            v, ser = run_policy(mk(s), B, stream, np.random.default_rng(1000 + s))
            vals.append(v)
            if s == 0:
                ser0 = ser
        res[name] = {"asr_mean": float(np.mean(vals)), "asr_std": float(np.std(vals))}
        if series:
            res[name]["series_seed0"] = ser0
    return res


CONFIGS = {"exp3ix_eta1_s0.02": (1.0, 0.02), "exp3ix_eta1_s0": (1.0, 0.0),
           "exp3ix_theory_s0": (None, 0.0), "exp3ix_eta0.3_s0.02": (0.3, 0.02),
           "exp3ix_eta3_s0.02": (3.0, 0.02)}


def boot(i):
    rng = np.random.default_rng(10_000 + i)
    S = rng.binomial(NB, S0) / NB
    r = evaluate(S, 1, cycle_stream(S), seeds=5, configs={"x": (1.0, 0.02)})
    return r["x"]["asr_mean"] - r["minimax"]["asr"]


if __name__ == "__main__":
    out = {"expanded_attacks": ["|".join(e) for e in E],
           "cycle_variants": ["|".join(E[j]) for j in sorted(set(cycle_stream(S0)))]}
    for m in (1, 2):
        out[f"cycle_m{m}"] = evaluate(S0, m, cycle_stream(S0), configs=CONFIGS, series=(m == 1))
        cs, B = breach_table(S0, m)
        emerge = build_emergence_stream(S0, cs, B, PHASES, PLEN)
        out[f"emerge_m{m}"] = evaluate(S0, m, emerge, configs={"exp3ix_eta1_s0.02": (1.0, 0.02)})
        out[f"emerge_m{m}"]["stream"] = ["|".join(E[emerge[p * PLEN]]) for p in range(PHASES)]
    with Pool(8) as p:
        gaps = p.map(boot, range(200))
    out["bootstrap_gap_m1"] = {"n": len(gaps), "mean": float(np.mean(gaps)),
                               "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
                               "frac_scheduler_better": float(np.mean(np.array(gaps) < 0))}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {kk: (vv if not isinstance(vv, dict) else {x: y for x, y in vv.items() if x != "series_seed0"})
                      for kk, vv in v.items()})
        else:
            print(k, v)
