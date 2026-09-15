"""Posterior bootstrap of dyn_v3 with 2000 draws and a Monte Carlo error estimate.
Usage: python3 boot2000.py per_variant.json boot2000_out.json"""
import json, sys
from multiprocessing import Pool
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "10"]
import dyn_v3 as V

out = {}
for m in (1, 2):
    eta = json.load(open("dyn_v3_out.json"))[f"tune_m{m}"]["eta"]
    with Pool(8) as p:
        res = p.map(V.boot, [(i, m, eta) for i in range(2000)])
    g = np.array([r["minus_minimax"] for r in res])
    rng = np.random.default_rng(0)
    lo = [np.percentile(rng.choice(g, len(g)), 2.5) for _ in range(500)]
    hi = [np.percentile(rng.choice(g, len(g)), 97.5) for _ in range(500)]
    out[f"m{m}"] = {"ci95": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
                    "mc_se_lower": float(np.std(lo)), "mc_se_upper": float(np.std(hi)),
                    "frac_scheduler_better": float((g < 0).mean()), "n": len(g)}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
