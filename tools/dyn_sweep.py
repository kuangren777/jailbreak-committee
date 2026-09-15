"""Learning-rate sensitivity and m=2 bootstrap for the per-variant switching stream.

Usage: python3 dyn_sweep.py <per_variant.json> <out.json>
"""
import sys
from multiprocessing import Pool
import numpy as np

import dyn_pv as D   # dyn_pv reads sys.argv[1] at import, which is the same per_variant.json

ETAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def sweep_point(args):
    m, eta, s = args
    r = D.evaluate(D.S0, m, D.cycle_stream(D.S0), configs={"x": (eta, s)})
    return {"m": m, "eta": eta, "s": s, "asr_mean": r["x"]["asr_mean"], "asr_std": r["x"]["asr_std"],
            "minimax": r["minimax"]["asr"]}


def boot_m2(i):
    rng = np.random.default_rng(20_000 + i)
    S = rng.binomial(D.NB, D.S0) / D.NB
    r = D.evaluate(S, 2, D.cycle_stream(S), seeds=5, configs={"x": (1.0, 0.02)})
    return r["x"]["asr_mean"] - r["minimax"]["asr"]


if __name__ == "__main__":
    grid = [(m, eta, s) for m in (1, 2) for eta in ETAS + [None] for s in (0.0, 0.02)]
    with Pool(8) as p:
        pts = p.map(sweep_point, grid)
        gaps = p.map(boot_m2, range(200))
    series = D.evaluate(D.S0, 1, D.cycle_stream(D.S0),
                        configs={"eta1_s0.02": (1.0, 0.02), "theory_s0": (None, 0.0)}, series=True)
    out = {"sweep": pts,
           "bootstrap_gap_m2": {"n": len(gaps), "mean": float(np.mean(gaps)),
                                "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
                                "frac_scheduler_better": float(np.mean(np.array(gaps) < 0))},
           "series_m1": {k: series[k]["series_seed0"] for k in ("eta1_s0.02", "theory_s0", "random")},
           "series_ref_m1": {"minimax": series["minimax"]["asr"], "oracle": series["oracle"]}}
    import json
    json.dump(out, open(sys.argv[2], "w"))
    for q in pts:
        print(q)
    print("bootstrap_gap_m2", out["bootstrap_gap_m2"])
