"""
Tests for fairness_audit.py.

I use tiny hand-made inputs where I already know the right answer. That way I am
testing the maths, not the model.
"""

import numpy as np

from fairness_audit import statistical_parity_difference


def test_statistical_parity_is_zero_when_groups_are_equal():
    # both groups approved at the same rate, so the difference must be 0
    y_pred = np.array([1, 0, 1, 0])
    group = np.array([0, 0, 1, 1])
    assert statistical_parity_difference(y_pred, group) == 0.0
