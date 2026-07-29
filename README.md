# FairnessAudit-AI

A small toolkit I am building to check AI models for unfairness, fix some of it,
and run the same check on chatbot AIs (LLMs).

Alongside my cloud background, I love learning and doing hands-on research in
LLMs, agentic AI, MCP servers, and trustworthy AI. This project is that passion
in action.

> New to these words? There is a plain-English **Glossary** at the bottom. Every
> hard term is explained there in one line.

## How it all works (the big picture)

This diagram is the whole project on one screen. Data comes in on the left. The
audit and the fix sit in the middle. A human makes the final call on the right,
because a person should decide important things, not a machine. The dotted branch
shows the same fairness idea reused on a chatbot AI.

```mermaid
flowchart TB
    subgraph DATA["1. DATA  (model.py)"]
        A["Make loan applicants<br/>from fair features"]
        B["Hide unfair penalty<br/>inside the answers"]
        A --> B
    end

    subgraph MODEL["2. MODEL  (model.py)"]
        C["Train the model<br/>WITHOUT gender"]
        D["Scores from 0 to 1<br/>for each person"]
        C --> D
    end

    subgraph AUDIT["3. AUDIT  (fairness_audit.py)"]
        E["Measure 3 fairness numbers:<br/>parity, impact, opportunity"]
        F["Unfair? Give each group<br/>its own yes/no cutoff"]
        G["Draw before/after chart"]
        E --> F --> G
    end

    subgraph LLM["4. SAME TEST ON A CHATBOT  (llm_reliability.py)"]
        H["Ask 'she' and 'he'<br/>versions of one question"]
        I["Count how often the<br/>answer flips"]
        H --> I
    end

    DATA --> MODEL --> AUDIT
    MODEL -. same fairness idea .-> LLM
    AUDIT --> J(("Human<br/>reviewer"))
    LLM --> J
    J --> K["Fairer decision<br/>for a real person"]

    classDef human fill:#ffe9c7,stroke:#e08a00,stroke-width:2px,color:#000;
    class J human;
```

## What the project shows, in one line each

1. Hiding someone's gender from the model does not make it fair. It still learns
   to be unfair in an indirect way.
2. I can measure that unfairness with three simple numbers.
3. I can reduce the unfairness with a small change, and barely lose any accuracy.
4. The same fairness check also works on a chatbot AI.

## The files

| File | What it does, simply |
| --- | --- |
| `model.py` | Makes a pretend loan dataset. Some unfairness is hidden inside it on purpose. Then it trains a basic AI model. |
| `fairness_audit.py` | Measures the unfairness with three numbers. Then fixes most of it. Then draws a chart. |
| `llm_reliability.py` | Tests a chatbot style AI. It changes "she" to "he" in a question and checks if the answer changes. |
| `tests/` | Small unit tests that check the data, the metrics, and the chatbot probe. |

## How to run it

You type these lines one at a time in a terminal:

```bash
pip install -r requirements.txt

python model.py             # step 1: make the data and train the model
python fairness_audit.py    # step 2: measure the unfairness, then fix it
python llm_reliability.py   # step 3: test the chatbot AI
```

It all runs on your own computer. You do not need the internet. You do not need a
paid AI account. To run the tests: `pip install -r requirements-dev.txt` then `pytest`.

## The three fairness numbers, in plain words

I wrote these three checks myself, by hand, so I truly understand them.

1. **Statistical parity.** Do both groups get approved at the same rate? Easy to
   understand. But it ignores who actually deserved a loan.
2. **Disparate impact.** The same idea, but written as a ratio. A famous rule
   says that if one group gets under 80% of what the other gets, that is a
   warning sign.
3. **Equal opportunity.** Out of the people who would truly repay the loan, do
   both groups get approved equally? I trust this one the most. It is fair to
   people who deserve the loan, no matter their group.

There is no single "correct" fairness number. Picking one is a choice about
values. Being honest about that choice is part of doing this the right way.

Work in progress. More coming soon.
