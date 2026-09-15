"""ICASSP figures (PDF) from the per-variant susceptibility matrix.

per_variant.json maps "attack|model" -> {variant: ASR over 25 AdvBench behaviors}. It is
extracted from ICASSP-3/results/attacks/{trial,part1_new,part2}.json, whose best-of-variant
cells match results/attacks/full_matrix.json. Usage: python3 make_icassp_figs.py <outdir>
"""
import itertools
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
PV = json.load(open(os.path.join(HERE, "per_variant.json")))
MODELS = ["Llama-2-7b", "Mistral-7B-v0.1", "Qwen-7B", "internlm-7b", "vicuna-7b", "Baichuan2-7b"]
LABELS = ["Llama-2", "Mistral", "Qwen", "InternLM", "Vicuna", "Baichuan2"]
ATTACKS = ["flipattack", "codechameleon", "renellm", "deepinception", "cipherchat", "pap", "pasttense"]
ATK_LABELS = ["FlipAttack", "CodeChameleon", "ReNeLLM", "DeepInception", "SelfCipher", "PAP", "Past tense"]
VARS = {a: list(PV[f"{a}|{MODELS[0]}"].keys()) for a in ATTACKS}

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6,
})
W, H = 3.35, 2.15


def p_ind(ps, k):
    pmf = np.array([1.0])
    for p in ps:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return float(pmf[k:].sum())


def p_com(ps, k):
    return sorted(ps, reverse=True)[k - 1]


def worst_case(S, f):
    k = len(S) // 2 + 1
    return max(f([PV[f"{a}|{m}"][v] for m in S], k) for a in ATTACKS for v in VARS[a])


def fig_matrix():
    M = np.array([[max(PV[f"{a}|{m}"].values()) for m in MODELS] for a in ATTACKS])
    fig, ax = plt.subplots(figsize=(W, H))
    im = ax.imshow(M, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(ATTACKS)))
    ax.set_yticklabels(ATK_LABELS)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] > 0.55 else "black", fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("ASR", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig_matrix.pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("matrix worst single-variant per model:", dict(zip(LABELS, M.max(axis=0).round(2))))


def fig_size():
    rows = {}
    for name, f in (("independent", p_ind), ("comonotone", p_com)):
        best, med = [], []
        for m in range(1, 7):
            wc = [worst_case(S, f) for S in itertools.combinations(MODELS, m)]
            best.append(min(wc))
            med.append(st.median(wc))
        rows[name] = (best, med)
    ms = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(W, H))
    style = {"independent": ("black", "o"), "comonotone": ("0.5", "s")}
    for name, (best, med) in rows.items():
        c, mk = style[name]
        ax.plot(ms, best, color=c, marker=mk, ms=4.5, ls="none", label=f"best, {name}")
        ax.plot(ms, med, color=c, marker=mk, ms=4.5, ls="none", mfc="white", label=f"median, {name}")
    ax.axhline(rows["independent"][0][0], color="0.6", lw=0.7, ls=":", zorder=0, label="Llama-2 alone")
    ax.set_xlabel("committee size $m$")
    ax.set_ylabel("worst-case ASR")
    ax.set_xticks(ms)
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2, handlelength=2.0, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig_size.pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    for name, (best, med) in rows.items():
        print(name, "best", [round(x, 3) for x in best], "median", [round(x, 3) for x in med])


def fig_dynamic():
    """ASR against learning rate on the switching stream (dyn_sweep_out.json, s=0) with the
    Neu (2015) rate and the tuned rate from dyn_v3_out.json."""
    d = json.load(open(os.path.join(HERE, "dyn_sweep_out.json")))
    v3 = json.load(open(os.path.join(HERE, "dyn_v3_out.json")))
    fig, ax = plt.subplots(figsize=(W, H))
    styles = {1: ("black", "o"), 2: ("0.5", "s")}
    for m in (1, 2):
        c, mk = styles[m]
        pts = sorted((p for p in d["sweep"] if p["m"] == m and p["s"] == 0.0 and p["eta"] is not None),
                     key=lambda p: p["eta"])
        ax.plot([p["eta"] for p in pts], [p["asr_mean"] for p in pts], color=c, marker=mk, ms=3.2, lw=1.1,
                label=f"EXP3-IX, $m={m}$")
        r = v3[f"switching_m{m}_ind"]
        ax.axhline(r["minimax"], color=c, lw=0.8, ls=":", label=f"minimax committee, $m={m}$")
        ax.plot([r["neu_eta"]], [r["exp3ix_neu_rate"]], marker="*", ms=8, color=c, ls="none")
        eta_t = v3[f"tune_m{m}"]["eta"]
        ax.plot([eta_t], [r["exp3ix"]], marker="D", ms=5, mfc="white", mec=c, ls="none")
    ax.set_xscale("log")
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel("ASR (mean of 30 seeds)")
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.42), framealpha=0.9, ncol=2, handlelength=2.0, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig_eta.pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


fig_matrix()
fig_size()
if os.path.exists(os.path.join(HERE, "dyn_sweep_out.json")):
    fig_dynamic()
