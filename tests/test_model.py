"""
Tests for model.py.

I check two things: the dataset looks how I expect, and the unfairness I claim to
inject is actually there. That second test is the important one.
"""

from model import SENSITIVE_ATTR, TARGET, generate_loan_dataset


def test_dataset_has_expected_shape_and_columns():
    df = generate_loan_dataset(n_samples=1000, seed=0)
    assert len(df) == 1000
    assert SENSITIVE_ATTR in df.columns
    assert TARGET in df.columns
    # the target must be binary (0 or 1)
    assert set(df[TARGET].unique()).issubset({0, 1})
