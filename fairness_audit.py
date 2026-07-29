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
    return {0: 0.5, 1: 0.5}   # placeholder, I fill in the real search next
