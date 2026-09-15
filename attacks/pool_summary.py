"""Per-model worst-case ASR and judge agreement for one pool, from verdict aggregates.

Reads the same verdict jsonl files the gate reads, one verdict field per judge, and
reports the matrix summary a deployer sees before running the gate.
"""
import argparse, collections, glob, json
import numpy as np


def load(files, field):
    acc = collections.defaultdict(list)
    for p in files:
        for line in open(p):
            r = json.loads(line)
            v = r.get(field)
            if v in ("safe", "unsafe"):
                acc[(r["model"], r["attack"], r["variant"])].append(v == "unsafe")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", nargs="+", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = [p for pat in a.verdicts for p in sorted(glob.glob(pat))]

    out = {}
    per_key = {}
    for field, label in zip(a.fields, a.labels):
        acc = load(files, field)
        models = sorted({k[0] for k in acc})
        cols = sorted({(k[1], k[2]) for k in acc})
        cols = [c for c in cols if all((m,) + c in acc for m in models)]
        rows = {}
        for m in models:
            p = [sum(acc[(m,) + c]) / len(acc[(m,) + c]) for c in cols]
            j = int(np.argmax(p))
            rows[m] = {"worst": round(max(p), 3), "worst_col": list(cols[j]),
                       "mean": round(float(np.mean(p)), 3)}
        out[label] = {"n_variants": len(cols), "per_model": rows,
                      "pool_worst": round(max(r["worst"] for r in rows.values()), 3),
                      "best_single": min(rows, key=lambda m: rows[m]["worst"])}
        d = {}
        for p in files:
            for line in open(p):
                r = json.loads(line)
                if r.get(field) in ("safe", "unsafe"):
                    d[(r["model"], r["attack"], r["variant"], r["bsha"])] = r[field]
        per_key[label] = d

    if len(per_key) == 2:
        (la, da), (lb, db) = per_key.items()
        both = set(da) & set(db)
        agree = sum(da[k] == db[k] for k in both)
        out["agreement"] = {"n_both": len(both), "agree": round(agree / len(both), 3),
                            "judges": [la, lb]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
