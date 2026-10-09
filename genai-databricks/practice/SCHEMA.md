# Practice question format

Questions live in `questions-0N.yaml` (one file per exam domain). `scripts/render_questions.py`
turns them into `../practice-questions.md` (for reading on GitHub) and the website's interactive
practice exam. Never edit the generated Markdown by hand.

```yaml
- id: D3-007                 # D<domain>-<3-digit number>, unique
  domain: 3                  # 1..6, matches the exam outline
  objective: "Implement LLM guardrails to prevent negative outcomes"   # verbatim from the exam guide
  difficulty: 2              # 1 recall, 2 apply, 3 scenario/trade-off
  question: |
    A Generative AI Engineer ... Which approach ...?
  choices:                   # 4 choices for single answer, 5 for "Which TWO"
    A: ...
    B: ...
    C: ...
    D: ...
  answer: [B]                # one letter, or two letters for multiple-selection ("Which TWO ...?")
  explanation: |
    Why B is right, from first principles.
    Why each distractor is wrong (A: ..., C: ..., D: ...).
  refs:                      # official docs that confirm the answer
    - https://docs.databricks.com/...
```
