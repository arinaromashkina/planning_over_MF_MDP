"""Build the one-slide project report (studio/slide_main.pdf).

Scientific Studio week-8 structure: figure/diagram + key message (1-2 phrases) +
basic notations + essential terms. Talk <= 1 minute; script in
studio/ONE_MINUTE_SCRIPT_RU.md. Regenerable: python studio/make_slide.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "main_figure.png"
OUT = Path(__file__).resolve().parent / "slide_main.pdf"

TITLE = "Когда планирование окупается: стохастические модели пользователя в рекомендациях"
GOAL = ("Цель: проверить, даёт ли моделирование стохастических реакций пользователя и обусловленных ими\n"
        "переходов состояния выигрыш в суммарной полезности J(π) по сравнению с оптимистичной моделью.")
MESSAGE = ("Ключевое сообщение: стохастическое планирование окупается только когда представление\n"
           "состояния несёт динамику — при честном состоянии оно достигает оракула (+0.122,\n"
           "CI95 [0.115, 0.129]), над сжатым MF-постериором — не отличается от оптимистичного (−0.004).")
NOTIONS = ("Обозначения: J(π) — ожидаемая суммарная полезность политики π в истинной среде; T0 — оптимистичный\n"
           "переход (принятие всегда); T1 — стохастическая модель отклика + точное взвешенное обновление;\n"
           "DP-оракул — точное решение методом обратной индукции.")
TERMS = ("Термины: оптимистичный переход; feedback-conditioned transition; контролируемая среда с известной\n"
         "динамикой; сжатое состояние (MF-постериор).")
FOOTER = "github.com/arinaromashkina/planning_over_MF_MDP · figures/main_figure.pdf"


def main() -> None:
    img = mpimg.imread(FIG)
    fig = plt.figure(figsize=(13.33, 7.5))  # 16:9
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.5, 0.978, TITLE, ha="center", va="top", fontsize=19, fontweight="bold")
    ax.text(0.5, 0.928, GOAL, ha="center", va="top", fontsize=10, color="#333333")

    # main figure (embedded PNG), centered
    ax_img = fig.add_axes([0.08, 0.315, 0.84, 0.52])
    ax_img.imshow(img)
    ax_img.axis("off")

    # message box
    box = FancyBboxPatch((0.04, 0.045), 0.92, 0.225, boxstyle="round,pad=0.012",
                         fc="#f4f8f4", ec="#2ca02c", lw=1.2, transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(0.5, 0.245, MESSAGE, ha="center", va="top", fontsize=11, fontweight="bold",
            color="#1a4d1a")
    ax.text(0.5, 0.14, NOTIONS, ha="center", va="top", fontsize=8.5, color="#333333")
    ax.text(0.5, 0.075, TERMS, ha="center", va="top", fontsize=8.5, color="#333333")

    ax.text(0.5, 0.012, FOOTER, ha="center", va="bottom", fontsize=9, color="#666666")

    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {OUT} and {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
