"""Does the paper's pair recommendation survive a change of safety judge?

Every cell of the calibrated matrix M is scored by Llama Guard 3. A reviewer noted
that Llama Guard recovers only part of what the independent Qwen3-14B judge calls
harmful, and that the complementary pair was selected because Llama-2 has 0.00 cells
under attacks where judge 2 disagrees. The live run stored BOTH judges' verdicts for
Llama-2 and Baichuan2 on all 20 prompt variants, so the comparison is a re-analysis
with no new generation.

For each judge it reports per-variant ASR of each member, the worst variant per attack,
and the comonotone pair bound min(M[i,f], M[j,f]) maximized over variants, which is the
quantity the paper's selection rule minimizes. Verdict aggregates only.
"""
import argparse, json, os
from collections import defaultdict


def load_verdicts(path, field):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "attack":
                continue
            out[(r["model"], r["attack"], r["variant"], r["bsha"])] = r.get(field)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard-verdicts", required=True)
    ap.add_argument("--judge2-verdicts", required=True)
    ap.add_argument("--members", nargs=2, default=["Llama-2-7b", "Baichuan2-7b"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = load_verdicts(a.guard_verdicts, "verdict")
    j = load_verdicts(a.judge2_verdicts, "verdict2")
    m1, m2 = a.members

    def cells(v):
        """(attack, variant) -> {member: asr} over behaviors."""
        acc = defaultdict(lambda: defaultdict(list))
        for (model, atk, var, _b), verdict in v.items():
            if model in (m1, m2):
                acc[(atk, var)][model].append(1.0 if verdict == "unsafe" else 0.0)
        out = {}
        for key, per in acc.items():
            if m1 in per and m2 in per:
                out[key] = {m1: sum(per[m1]) / len(per[m1]), m2: sum(per[m2]) / len(per[m2]),
                            "n": len(per[m1])}
        return out

    res = {}
    for name, v in (("llama_guard", g), ("qwen3_14b", j)):
        c = cells(v)
        per_attack = defaultdict(list)
        for (atk, var), d in c.items():
            per_attack[atk].append((var, d))
        attacks = {}
        for atk, items in sorted(per_attack.items()):
            worst1 = max(d[m1] for _v, d in items)
            worst2 = max(d[m2] for _v, d in items)
            # comonotone pair bound: the pair breaches only if BOTH members fail
            pair = max(min(d[m1], d[m2]) for _v, d in items)
            attacks[atk] = {"worst_" + m1: round(worst1, 3), "worst_" + m2: round(worst2, 3),
                            "pair_comonotone": round(pair, 3),
                            "variants": {v: {m1: round(d[m1], 3), m2: round(d[m2], 3)}
                                         for v, d in sorted(items)}}
        res[name] = {
            "attacks": attacks,
            "worst_" + m1: round(max(x["worst_" + m1] for x in attacks.values()), 3),
            "worst_" + m2: round(max(x["worst_" + m2] for x in attacks.values()), 3),
            "pair_worst_comonotone": round(max(x["pair_comonotone"] for x in attacks.values()), 3),
        }

    res["summary"] = {
        "members": [m1, m2],
        "note": "pair_worst_comonotone is max_f min(M[i,f], M[j,f]), the bound the selection rule minimizes",
    }
    for k in ("worst_" + m1, "worst_" + m2, "pair_worst_comonotone"):
        res["summary"][k] = {"llama_guard": res["llama_guard"][k], "qwen3_14b": res["qwen3_14b"][k]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    print("\nper-attack pair bound:")
    for atk in sorted(res["llama_guard"]["attacks"]):
        lg = res["llama_guard"]["attacks"][atk]
        q = res["qwen3_14b"]["attacks"].get(atk, {})
        print("  %-16s guard: m1=%.2f m2=%.2f pair=%.2f | qwen: m1=%.2f m2=%.2f pair=%.2f" % (
            atk, lg["worst_" + m1], lg["worst_" + m2], lg["pair_comonotone"],
            q.get("worst_" + m1, float("nan")), q.get("worst_" + m2, float("nan")),
            q.get("pair_comonotone", float("nan"))))


if __name__ == "__main__":
    main()
