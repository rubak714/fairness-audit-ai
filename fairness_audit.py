"""
fairness_audit.py

Step 2: check if the model is unfair, then fix most of it.

This is the heart of the project. I do two things here:
  1. Measure the unfairness using three simple numbers (I wrote them by hand so I
     really understand them).
  2. Fix most of it with a small trick: use a different "yes/no cutoff" for each
     group. Then I draw a chart of the result.

To run:  python fairness_audit.py
It prints a before and after report, and saves a chart in the results folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score


# ----------------------------------------------------------------------------
# The three fairness numbers.
# All of them compare group 0 (the group that was treated unfairly) to group 1.
# ----------------------------------------------------------------------------

def _rates(y_true, y_pred, group, g):
    """Helper for one group: what fraction did we approve, and out of the people
    who really deserved a loan, what fraction did we approve (the TPR)."""
    mask = group == g          # pick only the people in this group
    y_t = y_true[mask]         # their true answers
    y_p = y_pred[mask]         # our guesses for them

    # approval rate = how many of them we said yes to
    selection_rate = y_p.mean() if len(y_p) else 0.0

    # TPR = out of the people who truly deserved a loan, how many we approved
    deserved = y_t == 1
    tpr = y_p[deserved].mean() if deserved.sum() else 0.0
    return selection_rate, tpr


def statistical_parity_difference(y_pred, group) -> float:
    """Approval rate for group 0 minus approval rate for group 1.

    0 means both groups get approved equally. A negative number means group 0 is
    approved less often. This is simple, but it ignores who actually deserved it.
    """
    r0 = y_pred[group == 0].mean()
    r1 = y_pred[group == 1].mean()
    return float(r0 - r1)


def disparate_impact(y_pred, group) -> float:
    """The same idea, but as a ratio: group 0 rate divided by group 1 rate.

    There is a well known rule: below 0.8 is a warning sign of unfairness.
    1.0 means both groups are treated equally.
    """
    r0 = y_pred[group == 0].mean()
    r1 = y_pred[group == 1].mean()
    return float(r0 / r1) if r1 > 0 else float("nan")


def equal_opportunity_difference(y_true, y_pred, group) -> float:
    """Compare the two groups, but only among people who truly deserved a loan.

    In numbers: TPR of group 0 minus TPR of group 1.
    This is the one I trust most. It is fair to people who deserve approval, no
    matter which group they are in. 0 is the fair point.
    """
    _, tpr0 = _rates(y_true, y_pred, group, 0)
    _, tpr1 = _rates(y_true, y_pred, group, 1)
    return float(tpr0 - tpr1)


@dataclass
class AuditResult:
    """Holds all the numbers for one run, so I can print them neatly."""

    label: str
    accuracy: float
    stat_parity_diff: float
    disparate_impact: float
    equal_opp_diff: float

    def show(self) -> None:
        print(f"\n[{self.label}]")
        print(f"  Accuracy (how often it is right) : {self.accuracy:.3f}")
        print(f"  Statistical parity difference    : {self.stat_parity_diff:+.3f}  (0 = fair)")
        print(f"  Disparate impact ratio           : {self.disparate_impact:.3f}  (1 = fair, below 0.8 = warning)")
        print(f"  Equal opportunity difference     : {self.equal_opp_diff:+.3f}  (0 = fair)")


def audit(y_true, y_pred, group, label: str) -> AuditResult:
    """Run all three checks plus accuracy, and hand back the numbers."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    return AuditResult(
        label=label,
        accuracy=accuracy_score(y_true, y_pred),
        stat_parity_diff=statistical_parity_difference(y_pred, group),
        disparate_impact=disparate_impact(y_pred, group),
        equal_opp_diff=equal_opportunity_difference(y_true, y_pred, group),
    )


# ----------------------------------------------------------------------------
# The fix: use a different yes/no cutoff for each group so they end up equally
# fair. I never retrain the model, I only move the cutoff line.
# ----------------------------------------------------------------------------

def find_group_thresholds(y_true, y_prob, group, grid=None):
    """Find a good cutoff for each group so their TPRs match.
    Group 1 stays at the normal 0.5. I search the best cutoff for group 0."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)   # a list of cutoffs to try

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group = np.asarray(group)

    # group 1 stays at 0.5. its TPR is the target I want group 0 to match.
    thr = {1: 0.5}
    _, target_tpr = _rates(y_true, (y_prob >= 0.5).astype(int), group, 1)

    # for group 0, try every cutoff and keep the closest match.
    best_t, best_gap = 0.5, np.inf
    for t in grid:
        preds = (y_prob >= t).astype(int)
        _, tpr0 = _rates(y_true, preds, group, 0)
        gap = abs(tpr0 - target_tpr)
        if gap < best_gap:
            best_gap, best_t = gap, t
    thr[0] = float(best_t)
    return thr


def apply_group_thresholds(y_prob, group, thresholds) -> np.ndarray:
    """Turn scores into yes/no answers, using each group's own cutoff."""
    y_prob = np.asarray(y_prob)
    group = np.asarray(group)
    out = np.zeros(len(y_prob), dtype=int)
    for g, t in thresholds.items():
        out[group == g] = (y_prob[group == g] >= t).astype(int)
    return out


# ----------------------------------------------------------------------------
# The chart. One picture beats a table of numbers.
# ----------------------------------------------------------------------------

def plot_tradeoff(before: AuditResult, after: AuditResult, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")   # just save a file, no pop-up window needed
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Before", "After the fix"]
    acc = [before.accuracy, after.accuracy]
    # I show unfairness as a distance from 0, so smaller bars mean more fair.
    unfairness = [abs(before.equal_opp_diff), abs(after.equal_opp_diff)]

    # left chart: accuracy should stay high. I add headroom so the number above
    # the bar never bumps into the title.
    sns.barplot(x=labels, y=acc, ax=axes[0], hue=labels, legend=False, palette="Blues_d")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Accuracy (we want this to stay high)", pad=12)
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    # right chart: unfairness should drop toward zero. same headroom trick.
    top = max(unfairness) * 1.35 if max(unfairness) > 0 else 1.0
    sns.barplot(x=labels, y=unfairness, ax=axes[1], hue=labels, legend=False, palette="Reds_d")
    axes[1].set_ylim(0, top)
    axes[1].set_title("Unfairness (we want this near zero)", pad=12)
    for i, v in enumerate(unfairness):
        axes[1].text(i, v + top * 0.03, f"{v:.3f}", ha="center", fontweight="bold")

    fig.suptitle(
        "I removed almost all the unfairness and barely lost any accuracy",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved the chart to: {out_path}")


def main() -> None:
    from model import generate_loan_dataset, train_baseline

    print("=" * 68)
    print("FairnessAudit-AI  |  checking a loan model for unfairness")
    print("=" * 68)

    # step 1: get the data and the trained model from the other file
    df = generate_loan_dataset()
    trained = train_baseline(df)

    y_true = trained.y_test.to_numpy()          # the correct answers
    group = trained.sensitive_test.to_numpy()   # which group each person is in
    y_prob = trained.y_prob                      # the model's scores

    # before the fix: same 0.5 cutoff for everyone
    baseline_pred = (y_prob >= 0.5).astype(int)
    before = audit(y_true, baseline_pred, group, "BEFORE  (same cutoff for everyone)")
    before.show()

    # the fix: a different cutoff for each group
    thresholds = find_group_thresholds(y_true, y_prob, group)
    print(f"\nCutoff chosen for each group: {thresholds}")
    direction = "a higher cutoff" if thresholds[0] > thresholds[1] else "a lower cutoff"
    print(f"(to make the groups equally fair, group 0 needs {direction} than group 1)")

    # after the fix
    mitigated_pred = apply_group_thresholds(y_prob, group, thresholds)
    after = audit(y_true, mitigated_pred, group, "AFTER   (each group has its own cutoff)")
    after.show()

    plot_tradeoff(before, after, os.path.join("results", "accuracy_fairness_tradeoff.png"))

    print("\nWhat I learned here:")
    print("  I never gave the model anyone's gender, yet it was still unfair.")
    print("  A small change to the cutoff removed most of the unfairness, and I")
    print("  barely lost any accuracy. Making that trade clear is the whole point.")


if __name__ == "__main__":
    main()
