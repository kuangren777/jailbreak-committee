"""The selection gate.

Given a susceptibility matrix scored by the judge a deployer will actually run, decide
whether a two-model committee is worth its second inference. The rule of the paper picks
the pair minimizing max_f min(M[i,f], M[j,f]), but that argmin is only meaningful when one
pair separates from the field. The gate resamples every cell at its own behavior count
and reports how often the nominal winner stays the winner, then answers deploy-the-pair
or deploy-the-best-single.

Usage: build the matrix from verdict jsonl files, one field per judge, and run the gate
per judge. Verdict aggregates only.
"""
import argparse, collections, glob, itertools, json
import numpy as np


def build(files, field, restrict=None):
    """(model, attack, variant) -> list of 0/1. restrict limits to a key set."""
    acc = collections.defaultdict(list)
    for p in files:
        for line in open(p):
            r = json.loads(line)
            v = r.get(field)
            if v not in ("safe", "unsafe"):
                continue
            k = (r["model"], r["attack"], r["variant"], r["bsha"])
            if restrict is not None and k not in restrict:
                continue
            acc[(r["model"], r["attack"], r["variant"])].append(v == "unsafe")
    return acc


def gate(acc, draws=4000, seed=0, stability=0.9):
    models = sorted({k[0] for k in acc})
    cols = sorted({(k[1], k[2]) for k in acc})
    cols = [c for c in cols if all((m,) + c in acc for m in models)]
    P = np.array([[sum(acc[(m,) + c]) / len(acc[(m,) + c]) for c in cols] for m in models])
    N = np.array([[len(acc[(m,) + c]) for c in cols] for m in models])
    worst = P.max(axis=1)
    best_single_i = int(worst.argmin())

    idx = list(itertools.combinations(range(len(models)), 2))
    names = ["%s+%s" % (models[a], models[b]) for a, b in idx]
    vals = np.array([np.minimum(P[a], P[b]).max() for a, b in idx])
    order = np.argsort(vals)
    winner = names[order[0]]
    margin = float(vals[order[1]] - vals[order[0]])

    rng = np.random.default_rng(seed)
    wins = collections.Counter()
    beats_single = 0
    for _ in range(draws):
        S = rng.binomial(N, P) / N
        v = [np.minimum(S[a], S[b]).max() for a, b in idx]
        j = int(np.argmin(v))
        wins[names[j]] += 1
        if v[j] < S[best_single_i].max():
            beats_single += 1
    stab = wins[winner] / draws
    passed = bool(stab >= stability and beats_single / draws >= stability)
    return {
        "models": models, "n_variants": len(cols),
        "behaviors_per_cell": sorted(set(N.flatten().tolist())),
        "best_single": models[best_single_i], "best_single_worst": round(float(worst.min()), 3),
        "nominal_pair": winner, "nominal_pair_value": round(float(vals[order[0]]), 3),
        "runner_up": names[order[1]], "runner_up_value": round(float(vals[order[1]]), 3),
        "margin": round(margin, 3),
        "stability": round(stab, 3),
        "p_pair_beats_best_single": round(beats_single / draws, 3),
        "top_by_draws": [(k, round(c / draws, 3)) for k, c in wins.most_common(4)],
        "gate": "deploy pair" if passed else "deploy best single",
        "stability_threshold": stability,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True, help="one verdict field per judge")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--matched", action="store_true",
                    help="restrict every judge to the responses all judges parsed")
    ap.add_argument("--stability", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = [p for pat in a.verdicts for p in sorted(glob.glob(pat))]

    parsed = []
    for field in a.fields:
        s = set()
        for p in files:
            for line in open(p):
                r = json.loads(line)
                if r.get(field) in ("safe", "unsafe"):
                    s.add((r["model"], r["attack"], r["variant"], r["bsha"]))
        parsed.append(s)
    restrict = set.intersection(*parsed) if a.matched else None

    out = {"matched": bool(a.matched), "n_matched": len(restrict) if restrict else None}
    for field, label in zip(a.fields, a.labels):
        out[label] = gate(build(files, field, restrict), stability=a.stability)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
