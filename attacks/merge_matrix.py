"""Merge the rotation's partial result files into one 6-model x all-attack
best-of-variant susceptibility matrix and print it. Inputs are run_trial.py
outputs; later files override earlier cells on conflict.
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="run_trial.py json outputs, in order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    attacks, models = [], []
    cells = {}   # (attack, model) -> {asr, coverage, trunc}
    for path in args.inputs:
        d = json.load(open(path))
        for atk in d["attacks"]:
            if atk not in attacks:
                attacks.append(atk)
            for m in d["targets"]:
                if m not in models:
                    models.append(m)
                c = d["matrix"][atk][m]
                cells[(atk, m)] = {
                    "asr": c["asr_best_of_variant"], "coverage": c["coverage"],
                    "truncated": c.get("fails", {}).get("truncated", 0)}

    matrix = {atk: {m: cells.get((atk, m)) for m in models} for atk in attacks}
    doc = {"attacks": attacks, "models": models, "matrix": matrix,
           "note": "best-of-variant ASR, merged from rotation; verdict+sha1 only"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(doc, open(args.out, "w"), indent=1, ensure_ascii=False)

    print("=== 6-model x %d-attack best-of-variant ASR ===" % len(attacks))
    hdr = "%-15s " % "attack" + " ".join("%-13s" % m[:13] for m in models)
    print(hdr)
    for atk in attacks:
        row = "%-15s " % atk
        for m in models:
            c = matrix[atk][m]
            row += "%-13s" % ("--" if c is None else "%.2f" % c["asr"])
        print(row)
    # per-model mean ASR (how breakable each model is on average across attacks)
    print("\nmean ASR per model (across %d attacks):" % len(attacks))
    for m in models:
        vals = [matrix[a][m]["asr"] for a in attacks if matrix[a][m]]
        print("  %-15s %.3f" % (m, sum(vals) / len(vals) if vals else 0.0))
    print("\nwrote %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
