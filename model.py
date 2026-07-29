"""
model.py

Step 1: make a dataset and train a first AI model.

I picked a loan example. The AI has to guess "approve this loan" or "reject it".
This is a good example because a wrong guess hurts a real person.

Instead of downloading data, I create it here in code. Two reasons:
  1. The whole project runs on my own computer, no internet needed.
  2. I hide the unfairness myself, so later I can check if my test finds it.

To run just this file:  python model.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# I keep these names in one place so I do not spell them wrong in other files.
SENSITIVE_ATTR = "gender"   # the personal detail I want to be fair about
TARGET = "approved"         # the thing the AI tries to guess
RANDOM_STATE = 42           # a fixed number so the results are the same every run


def generate_loan_dataset(n_samples: int = 5000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Make a pretend loan dataset that hides a real problem.

    My idea in plain words:
      - There is a fair reason to approve someone (good income, good credit,
        steady job). This part has nothing to do with gender.
      - Then I add a small unfair penalty to one group. This copies real life,
        where past loan decisions were themselves unfair. So the unfairness is
        hidden inside the answers, not in an obvious column.
    """
    rng = np.random.default_rng(seed)

    # the sensitive detail. 0 and 1 are just two groups. no meaning to the numbers.
    gender = rng.integers(0, 2, size=n_samples)

    # the fair information.
    age = rng.normal(40, 12, n_samples).clip(18, 80)
    income = rng.normal(55_000, 20_000, n_samples).clip(12_000, 250_000)
    employment_years = rng.normal(8, 6, n_samples).clip(0, 45)
    credit_score = rng.normal(660, 90, n_samples).clip(300, 850)
    loan_amount = rng.normal(18_000, 9_000, n_samples).clip(1_000, 100_000)
    debt_to_income = (loan_amount / income).clip(0, 3)

    # a hidden "true score" of how safe the loan is. built ONLY from fair info.
    # higher number means safer loan.
    z = (
        0.000045 * (income - 55_000)
        + 0.012 * (credit_score - 660)
        + 0.05 * (employment_years - 8)
        - 1.4 * (debt_to_income - 0.33)
        + rng.normal(0, 1, n_samples)   # a little randomness, like real life
    )

    # here is the unfair part. group 0 gets pushed down for no fair reason.
    # this is the bias I want my test to catch later.
    historical_penalty = np.where(gender == 0, -0.9, 0.0)

    # turn the score into a yes/no answer.
    approval_logit = z + historical_penalty
    approval_prob = 1 / (1 + np.exp(-approval_logit))   # squashes score into 0..1
    approved = (rng.uniform(0, 1, n_samples) < approval_prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age.round(0),
            "income": income.round(0),
            "employment_years": employment_years.round(1),
            "credit_score": credit_score.round(0),
            "loan_amount": loan_amount.round(0),
            "debt_to_income": debt_to_income.round(3),
            SENSITIVE_ATTR: gender,
            TARGET: approved,
        }
    )
    return df


@dataclass
class TrainedModel:
    """A simple box that holds the trained model and the test data together,
    so the next file can grab everything in one go."""

    model: GradientBoostingClassifier
    X_test: pd.DataFrame
    y_test: pd.Series
    sensitive_test: pd.Series
    y_prob: np.ndarray            # the model's guessed chance of approval (0 to 1)
    feature_names: list[str]


def train_baseline(df: pd.DataFrame, seed: int = RANDOM_STATE) -> TrainedModel:
    """Train a basic model, using the fair information only.

    Important beginner point:
    I do NOT give the model the gender column. Many people think "if the model
    never sees gender, it cannot be unfair". This project shows that is wrong.
    The other columns are linked to the unfair answers, so the model still learns
    the unfairness in a sneaky, indirect way. My test in the next file proves it.
    """
    # use every column except the answer and the sensitive detail
    feature_names = [c for c in df.columns if c not in (TARGET, SENSITIVE_ATTR)]
    X = df[feature_names]           # the inputs the model learns from
    y = df[TARGET]                  # the correct answers
    sensitive = df[SENSITIVE_ATTR]  # kept aside, only used to check fairness

    # split into a part to learn from and a part to test on
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=seed, stratify=y
    )

    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)     # this line is where the model learns

    # the model's guessed chance of approval for each test person
    y_prob = model.predict_proba(X_test)[:, 1]

    return TrainedModel(
        model=model,
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        sensitive_test=s_test.reset_index(drop=True),
        y_prob=y_prob,
        feature_names=feature_names,
    )


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, roc_auc_score

    data = generate_loan_dataset()
    print("Dataset size (rows, columns):", data.shape)
    print("\nApproval rate for each group (this is where the unfairness hides):")
    print(data.groupby(SENSITIVE_ATTR)[TARGET].mean().round(3))

    trained = train_baseline(data)
    y_pred = (trained.y_prob >= 0.5).astype(int)   # 0.5 means "50% sure or more, approve"
    print("\nHow good is the basic model?")
    print("  Accuracy (how often it is right):", round(accuracy_score(trained.y_test, y_pred), 3))
    print("  ROC AUC (another quality score) :", round(roc_auc_score(trained.y_test, trained.y_prob), 3))
    print("\nIt looks accurate. But the next file asks a harder question:")
    print("accurate for whom?")
