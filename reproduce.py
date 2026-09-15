"""Recompute every gate number in the paper from the released aggregates.

No model inference. Reads only verdict-level records under results/ and prints the
value the paper reports next to the value recomputed here.
"""
import json, subprocess, sys, os

CHECKS = []

def run(cmd):
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-800:]); sys.exit(1)
    return json.loads(r.stdout[r.stdout.index("{"):])

print("[1/3] identifiability gate, first pool, two judges")
g1 = run(["attacks/gate.py", "--verdicts", "results/rescore_2023/verdicts_*.jsonl",
          "--fields", "verdict", "verdict2", "--labels", "llama_guard", "qwen3_14b",
          "--matched", "--out", "/tmp/_g1.json"])
CHECKS += [
    ("matched answers scored by both judges", 2947, g1["n_matched"]),
    ("guard selected pair bound", 0.44, g1["llama_guard"]["nominal_pair_value"]),
    ("guard runner up", 0.68, g1["llama_guard"]["runner_up_value"]),
    ("guard bootstrap stability", 0.98, round(g1["llama_guard"]["stability"], 2)),
    ("independent judge best stability", 0.26, round(g1["qwen3_14b"]["top_by_draws"][0][1], 2)),
    ("guard gate decision", "deploy pair", g1["llama_guard"]["gate"]),
    ("independent judge gate decision", "deploy best single", g1["qwen3_14b"]["gate"]),
]

print("[2/3] identifiability gate, second pool, two judges")
g2 = run(["attacks/gate.py", "--verdicts", "results/hub2026/verdicts_*.jsonl",
          "--fields", "verdict_a", "verdict_b", "--labels", "grok4", "kimi",
          "--matched", "--out", "/tmp/_g2.json"])
CHECKS += [
    ("matched answers, second pool", 2985, g2["n_matched"]),
    ("selected pair bound, grok-4", 0.0, g2["grok4"]["nominal_pair_value"]),
    ("best single worst case", 0.04, g2["grok4"]["best_single_worst"]),
    ("stability, grok-4", 0.41, round(g2["grok4"]["stability"], 2)),
    ("stability, kimi-k2.5", 0.58, round(g2["kimi"]["stability"], 2)),
    ("pair beats best single, grok-4", 0.64, round(g2["grok4"]["p_pair_beats_best_single"], 2)),
    ("gate decision, grok-4", "deploy best single", g2["grok4"]["gate"]),
    ("gate decision, kimi-k2.5", "deploy best single", g2["kimi"]["gate"]),
]

print("[3/3] per-model worst case and judge agreement, second pool")
s2 = run(["attacks/pool_summary.py", "--verdicts", "results/hub2026/verdicts_*.jsonl",
          "--fields", "verdict_a", "verdict_b", "--labels", "grok4", "kimi", "--out", "/tmp/_s2.json"])
CHECKS += [
    ("judge agreement, second pool", 0.936, s2["agreement"]["agree"]),
    ("weakest member, grok-4", 0.957, s2["grok4"]["per_model"]["qwen-flash"]["worst"]),
    ("weakest member, kimi-k2.5", 1.0, s2["kimi"]["per_model"]["qwen-flash"]["worst"]),
    ("strongest member, both judges", 0.04, s2["grok4"]["per_model"]["claude-haiku45"]["worst"]),
]

print()
bad = 0
for name, paper, got in CHECKS:
    ok = (paper == got)
    bad += not ok
    print("%-42s paper=%-20s recomputed=%-20s %s" % (name, paper, got, "ok" if ok else "MISMATCH"))
print()
print("%d of %d checks reproduce" % (len(CHECKS) - bad, len(CHECKS)))
sys.exit(1 if bad else 0)
