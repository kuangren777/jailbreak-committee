"""Fig. tradeoff: released worst-quarter ASR against benign service rate.

One panel per label set. Each pair of the pool is a threshold sweep (tau from 0.5 to 0.95);
the pair the gate accepts is blue, the other 14 are gray. Single models are gray dots,
single models behind a Llama Guard filter are orange rings (only in panels not scored by
the guard, since it cannot score its own filter). Reads tools/serving_replay_scale.json.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "serving_replay_scale.json")))
SEL = sys.argv[2] if len(sys.argv) > 2 else "Baichuan2-7b+Llama-2-7b"
OUT = os.path.join(HERE, "fig_tradeoff.pdf")
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#b4b4b0", "#3a3a38"
SHORT = {"Baichuan2-7b": "Baichuan2", "Llama-2-7b": "Llama-2", "Mistral-7B-v0.1": "Mistral",
         "Qwen-7B": "Qwen", "internlm-7b": "InternLM", "vicuna-7b": "Vicuna"}
TITLE = {"guard": "Llama Guard labels", "qwen3": "Qwen3-14B labels", "either": "Either-judge labels"}

# (a) no filter, union labels, the bought pair; (b) behind the Llama Guard filter, Qwen3-14B
# labels, the pair selected under filtered-breach labels
CS = json.load(open(os.path.join(HERE, "committee_select_scale.json")))
PANELS = [("committees", "either", CS["bought"], "single", "(a) No filter"),
          ("committees_guard", "qwen3", CS["winners"]["filtered"], "single_guard",
           "(b) Llama Guard filter")]
OFF = {("single", "internlm-7b"): (3, 1, "left"), ("single_guard", "Baichuan2-7b"): (0, -8, "center"), ("single_guard", "vicuna-7b"): (-3, -7, "right"),
       ("single_guard", "Llama-2-7b"): (-3, -2, "right"), ("single_guard", "Qwen-7B"): (3, 2, "left"),
       ("single_guard", "Mistral-7B-v0.1"): (0, 9, "center")}
# the three single models crowded at the top right of panel (a) are named in the caption
SKIP = {("single", "Baichuan2-7b"), ("single", "vicuna-7b"), ("single", "Mistral-7B-v0.1")}
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=True)
for ax, (ck, j, sel, sk, title) in zip(axes, PANELS):
    for name, v in R[ck].items():
        if name == sel:
            continue
        pts = sorted((x["benign_served"], x["released_cvar"]) for x in v[j].values())
        ax.plot(*zip(*pts), color="#d6d6d3", lw=0.45, zorder=1)
    pts = sorted((float(t), x["benign_served"], x["released_cvar"]) for t, x in R[ck][sel][j].items())
    ax.plot([p[1] for p in pts], [p[2] for p in pts], color=BLUE, lw=1.6, marker="o", ms=3.5, zorder=3)
    for t, x, y in (pts[0], pts[-1]):
        ax.annotate("$\\tau$=%g" % t, (x, y), xytext=(3, -6) if t < 0.6 else (3, 3),
                    textcoords="offset points", fontsize=8, color=BLUE)
    lab = "JBS (ours):\n" + " + ".join(SHORT[m] for m in sel.split("+"))
    ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontsize=9, color=BLUE, va="top")
    for m, v in R[sk].items():
        x, y = v[j]["benign_served"], v[j]["released_cvar"]
        if sk == "single":
            ax.scatter(x, y, s=22, color=INK, zorder=4, lw=0)
        else:
            ax.scatter(x, y, s=26, facecolors="none", edgecolors=ORANGE, lw=1.1, zorder=4)
        if (sk, m) in SKIP:
            continue
        dx, dy, ha = OFF.get((sk, m), (-2, 3, "right"))
        ax.annotate(SHORT.get(m, m), (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=INK, ha=ha)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlabel(r"Benign service rate ($\uparrow$)")
    ax.set_xlim(0, 1.03); ax.set_ylim(-0.02, 1.02)
    ax.grid(color="#e6e6e3", lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel(r"Released worst-quarter ASR ($\downarrow$)")
fig.tight_layout(pad=0.3, w_pad=0.5)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print("wrote", OUT)
