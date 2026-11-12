"""Build all project figures from SAVED experiment results (no recomputation).

Reads:  outputs/phase5/*.json, outputs/diagnostics/*.json
Writes: figures/*.png (150 dpi) and figures/*.pdf (vector)

Style per Scientific Studio guidance: every element named, one message per figure,
unified fonts/linewidths across figures.

Usage: python figures/make_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs"))
FIG = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
})

C_OPTIMISTIC = "#d62728"
C_STOCHASTIC = "#1f77b4"
C_ORACLE = "#2ca02c"
C_GREEDY = "#7f7f7f"


def _save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    print(f"wrote figures/{name}.png|.pdf")


def main_figure() -> None:
    """The project's one message: stochastic planning pays only when the state
    representation carries the dynamics."""
    env_s = json.load(open(OUT / "phase5" / "env_a_ablation.json"))["a3_ablation"]["aggregate"]
    env_c = json.load(open(OUT / "phase5" / "env_c_ablation.json"))["aggregate"]
    diff_s = json.load(open(OUT / "phase5" / "env_a_ablation.json"))["a3_ablation"]["stochastic_minus_optimistic"]
    diff_c = json.load(open(OUT / "phase5" / "env_c_ablation.json"))["stochastic_minus_optimistic"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)

    ax = axes[0]
    names = ["Greedy\n(myopic)", "Optimistic\n(paper T0)", "Stochastic\n(ours)", "Oracle\n(DP)"]
    vals = [env_s["greedy"]["mean"], env_s["A0_optimistic_depthH"]["mean"],
            env_s["A1_stochastic_depthH"]["mean"], env_s["oracle_dp"]["mean"]]
    errs = [env_s["greedy"]["std"], env_s["A0_optimistic_depthH"]["std"],
            env_s["A1_stochastic_depthH"]["std"], env_s["oracle_dp"]["std"]]
    colors = [C_GREEDY, C_OPTIMISTIC, C_STOCHASTIC, C_ORACLE]
    ax.bar(names, vals, yerr=errs, color=colors, alpha=0.85, capsize=4)
    ax.set_title("A. Faithful state (Env-S)")
    ax.set_ylabel("True-environment return J")
    ax.annotate(f"+{diff_s['mean']:.3f}\nCI95 [{diff_s['ci95'][0]:.3f}, {diff_s['ci95'][1]:.3f}]",
                xy=(2, vals[2]), xytext=(1.45, 0.60),
                arrowprops=dict(arrowstyle="->", lw=1.2), ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="#eef6ee", ec=C_ORACLE))

    ax = axes[1]
    names = ["Greedy MF\n(no planning)", "Optimistic\n2-step", "Stochastic\n2-step", "True-state\ngreedy (ref)"]
    vals = [env_c["greedy_mf"]["mean"], env_c["optimistic"]["mean"],
            env_c["stochastic"]["mean"], env_c["true_greedy"]["mean"]]
    errs = [env_c["greedy_mf"]["std"], env_c["optimistic"]["std"],
            env_c["stochastic"]["std"], env_c["true_greedy"]["std"]]
    ax.bar(names, vals, yerr=errs, color=colors, alpha=0.85, capsize=4)
    ax.set_title("B. Compressed MF posterior (Env-C)")
    ax.annotate(f"{diff_c['mean']:+.3f}\nCI95 [{diff_c['ci95'][0]:.3f}, {diff_c['ci95'][1]:.3f}]",
                xy=(2, vals[2]), xytext=(0.9, vals[2] + 0.35),
                arrowprops=dict(arrowstyle="->", lw=1.2), ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdeeee", ec=C_OPTIMISTIC))

    fig.suptitle("Stochastic feedback-conditioned planning pays only when the state carries the dynamics",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "main_figure")


def h3_figure() -> None:
    """H3: regret vs planning depth across response-noise levels (Env-S)."""
    rows = json.load(open(OUT / "phase5" / "env_a_noise.json"))["a4_noise"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for noise, marker in zip((0.5, 1.5, 3.0), ("o", "s", "^")):
        sub = sorted([r for r in rows if r["noise"] == noise and r["train_size"] == 3000],
                     key=lambda r: r["depth"])
        ax.plot([r["depth"] for r in sub], [r["regret_vs_oracle"] for r in sub],
                marker=marker, label=f"response noise σ={noise}", lw=1.8)
    ax.set_xlabel("planning depth (steps)")
    ax.set_ylabel("regret vs exact oracle")
    ax.set_title("Deeper planning reduces regret once the model is reasonable (Env-S)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "h3_error_depth")


def main() -> None:
    main_figure()
    h3_figure()


if __name__ == "__main__":
    main()
