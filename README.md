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

Work in progress. More coming soon.
