---
name: review-my-code
description: Review a student's own lab, exercise, or capstone code like a senior ML engineer would — correctness, ML pitfalls (leakage, wrong split, metric misuse), clarity — giving feedback and questions instead of rewritten code. Use when a student asks for feedback or a review of code they wrote.
argument-hint: <file path or "my last changes">
---

# Code review for learning

Target: $ARGUMENTS (default: the student's uncommitted changes, `git diff`).

## Steps

1. **Read the code and run it.** Run the file (`python3 <file>`) and, for labs, confirm whether
   the final asserts pass. Record exact output for any failure.
2. **Review in this priority order**, citing `file:line` for every point:
   1. *Correctness* — wrong math, wrong shapes, off-by-one, numerical instability
      (e.g. `log(0)`, unstable softmax), wrong gradient sign.
   2. *ML methodology* — leakage (fitting a scaler on test data, future information in features),
      wrong split for time-dependent data, metric that does not match the goal, missing baseline,
      no seed. Link the relevant section of `modules/04-evaluation-and-data.md` or
      `system-design/11-data-and-training-infrastructure.md`.
   3. *Verification* — is there a check that would catch a bug (gradient check, shape asserts,
      comparison with a closed-form or library answer)?
   4. *Clarity* — naming, comments on non-obvious math, vectorization over Python loops.
3. **Write the review as**: a 2-line summary verdict; then at most 7 findings, each as
   *what you observed → why it matters → a question or hint that leads to the fix*.
   Do not provide the corrected code.
4. **Pick one finding to go deeper on** — the one with the most learning value — and ask the
   student how they would fix it.

## Rules

- Do not rewrite or edit the student's file. Feedback only. If they then ask for a fix to a
  graded deliverable, follow the hint ladder in `AGENTS.md`.
- Praise specific good decisions (e.g. "using a time-based split here is exactly right because…")
  so the student knows what to keep doing.
