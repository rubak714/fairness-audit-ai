"""
make_diagram.py

Draws the project workflow and saves it to docs/how-it-works.png. Clean straight
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

    # left column: three boxes, all the same size and the same gap between them.
    # I compute the y positions from one gap value so the spacing stays equal.
    lx, lw, bh, gap = 0.6, 3.3, 1.2, 0.8
    cx = lx + lw / 2            # shared center line for the whole column
    y_data = 6.9
    y_model = y_data - bh - gap
    y_chat = y_model - bh - gap
    box(ax, lx, y_data, lw, bh, "Make loan data\n(bias hidden in answers)", magenta)
    box(ax, lx, y_model, lw, bh, "Train the model\nWITHOUT gender", orange)

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

    # chatbot test: third box in the left column, same size and gap as the others
    box(ax, lx, y_chat, lw, bh, "Same test on a chatbot\nswap 'she' and 'he'", blue)

    # human and final decision at the bottom
    box(ax, 4.4, 1.0, 2.6, 1.2, "Human\nreviewer", amber)
    box(ax, 8.2, 1.0, 3.0, 1.2, "Fairer decision\nfor a real person", green)

    # arrows (computed from the same positions so they line up with the boxes)
    arrow(ax, (cx, y_data), (cx, y_model + bh), "applicants", magenta)
    arrow(ax, (lx + lw, y_model + bh / 2), (5.7, 7.2), "scores 0 to 1", orange)
    arrow(ax, (cx, y_model), (cx, y_chat + bh), "same idea", blue, dashed=True)
    arrow(ax, (8.3, 4.6), (6.4, 2.2), "", cyan)          # audit -> human
    arrow(ax, (cx + 0.35, y_chat), (4.6, 2.2), "", blue)  # chatbot -> human
    arrow(ax, (7.0, 1.6), (8.2, 1.6), "final call", amber)

    fig.suptitle("FairnessAudit-AI  |  how it all works", fontsize=15,
                 fontweight="bold")
    os.makedirs("docs", exist_ok=True)
    out = os.path.join("docs", "how-it-works.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
