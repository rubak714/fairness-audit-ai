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

import numpy as np


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
