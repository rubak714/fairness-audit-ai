"""
Tests for llm_reliability.py.

I check that my fake chatbot really is a bit unfair, and that my test catches it.
If the test could not catch a chatbot I built to be unfair, the test would be
useless.
"""

from llm_reliability import Applicant, MockLLM, counterfactual_fairness_test


def test_probe_detects_the_unfair_chatbot():
    # I spread applicants across the decision boundary on purpose. Very strong or
    # very weak applicants get the same answer either way. The bias only shows for
    # borderline people, so the test set has to include them.
    import random

    rng = random.Random(0)
    applicants = [
        Applicant(income=rng.randint(30_000, 90_000), credit_score=rng.randint(580, 780))
        for _ in range(200)
    ]
    result = counterfactual_fairness_test(MockLLM(), applicants)
    # the fake chatbot penalises women, so some answers must flip
    assert result["counterfactual_flip_rate"] > 0


def test_women_are_never_approved_more_than_men():
    import random

    rng = random.Random(1)
    applicants = [
        Applicant(income=rng.randint(30_000, 90_000), credit_score=rng.randint(580, 780))
        for _ in range(200)
    ]
    result = counterfactual_fairness_test(MockLLM(), applicants)
    # since the bias only ever hurts the woman-described version, her approval
    # rate must come out lower or equal, never higher.
    assert result["female_approval_rate"] <= result["male_approval_rate"]
