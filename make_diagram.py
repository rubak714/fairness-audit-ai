"""
make_diagram.py

Draws the project workflow and saves it to docs/workflow.png. Clean straight
boxes with colored outlines on a white background, so it reads like a tidy
whiteboard rather than a hand-drawn sketch.

Run:  python make_diagram.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, color, fontsize=11):
    """Draw one rounded, colored, hand-drawn box with centered text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=2.5, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#222222")


def arrow(ax, start, end, label="", color="#444444", style="-|>", dashed=False):
    """Draw a hand-drawn arrow between two points, with an optional label."""
    ap = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=18,
        linewidth=2, color=color,
        linestyle="--" if dashed else "-",
        shrinkA=4, shrinkB=4,
    )
    ax.add_patch(ap)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my, label, ha="center", va="center", fontsize=9,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # colors picked to feel like whiteboard markers
    magenta, orange, cyan = "#c026d3", "#ea580c", "#0891b2"
    blue, amber, green = "#2563eb", "#e0a000", "#16a34a"

    # left column: data then model
    box(ax, 0.6, 6.9, 3.3, 1.2, "Make loan data\n(bias hidden in answers)", magenta)
    box(ax, 0.6, 5.1, 3.3, 1.2, "Train the model\nWITHOUT gender", orange)

    # the audit container on the right, with three steps inside
    container = FancyBboxPatch(
        (5.2, 4.4), 6.2, 4.0,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        linewidth=2.5, edgecolor=cyan, facecolor="white",
    )
    ax.add_patch(container)
    ax.text(5.5, 8.05, "Fairness audit  (fairness_audit.py)", ha="left",
            va="center", fontsize=11, color=cyan)
    box(ax, 5.7, 7.0, 5.2, 0.9, "Measure 3 fairness numbers", cyan, fontsize=10)
    box(ax, 5.7, 5.8, 5.2, 0.9, "Give each group its own cutoff", cyan, fontsize=10)
    box(ax, 5.7, 4.6, 5.2, 0.9, "Draw a before / after chart", cyan, fontsize=10)

    # chatbot test on the lower left
    box(ax, 0.6, 3.0, 3.3, 1.2, "Same test on a chatbot\nswap 'she' and 'he'", blue)

    # human and final decision at the bottom
    box(ax, 4.4, 1.0, 2.6, 1.2, "Human\nreviewer", amber)
    box(ax, 8.2, 1.0, 3.0, 1.2, "Fairer decision\nfor a real person", green)

    # arrows
    arrow(ax, (2.25, 6.9), (2.25, 6.3), "applicants", magenta)
    arrow(ax, (3.9, 5.7), (5.7, 7.2), "scores 0 to 1", orange)
    arrow(ax, (2.25, 5.1), (2.25, 4.2), "same idea", blue, dashed=True)
    arrow(ax, (8.3, 4.6), (6.4, 2.2), "", cyan)          # audit -> human
    arrow(ax, (2.6, 3.0), (4.6, 2.2), "", blue)          # chatbot -> human
    arrow(ax, (7.0, 1.6), (8.2, 1.6), "final call", amber)

    fig.suptitle("FairnessAudit-AI  |  how it all works", fontsize=15,
                 fontweight="bold")
    os.makedirs("docs", exist_ok=True)
    out = os.path.join("docs", "workflow.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
