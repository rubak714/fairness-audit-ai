"""
llm_reliability.py

Step 3: test a chatbot style AI (an LLM) for the same kind of unfairness.

This is the file I enjoy the most, because a lot of my curiosity right now is in
LLMs and agentic AI, not just tables of numbers.

The main idea in one sentence:
  A fairness test is just this. Change one small detail (like "she" to "he"),
  keep everything else the same, and check if the answer changes. If it changes,
  the AI is being unfair.

That test works on a chatbot too. So I ask the chatbot the same loan question two
ways, once with "she" and once with "he", and I count how often the answer flips.

To keep it simple and free, I include a tiny fake chatbot below. It runs on your
own computer with no internet and no paid account. I also show exactly where a
real chatbot (like Claude) would plug in. The test around it does not change.

To run:  python llm_reliability.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# A tiny fake chatbot, so this file runs with no internet and no account.
# I made it a little bit unfair on purpose, so my test has something to catch.
# ----------------------------------------------------------------------------

class MockLLM:
    """Pretends to be a chatbot deciding loans. Always gives the same answer for
    the same question, so my results are repeatable."""

    def generate(self, prompt: str) -> str:
        p = prompt.lower()

        # read the income and credit score out of the question text
        income = _extract_number(p, r"income of \$?([0-9,]+)")
        score = _extract_number(p, r"credit score of ([0-9]+)")

        # turn them into a simple "strength" number
        strength = (income / 100_000) + (score / 850)

        # the unfair part: if the person is described as a woman, the fake chatbot
        # quietly makes it harder for them. this copies real bias found in AIs.
        if any(w in p for w in [" she ", " her ", "female", " woman "]):
            strength -= 0.15

        decision = "APPROVE" if strength >= 1.0 else "DENY"
        return f"Decision: {decision}."
