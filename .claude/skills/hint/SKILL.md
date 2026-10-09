---
name: hint
description: Give a graded hint (one level at a time) for a lab exercise or lab bug, without giving away the solution. Use when a student is stuck on a lab in labs/ or a student exercise at the bottom of a lab file.
argument-hint: <lab file or number> [exercise number]
---

# Hint ladder for labs

Request: $ARGUMENTS

## Steps

1. **Identify the lab and exercise.** Resolve the lab file in `labs/` (e.g. "3" →
   `labs/03_metrics_from_scratch.py`). If an exercise number is given, read it from the
   "Student exercises" comment block at the bottom of the file. Read the module the lab belongs
   to (see `labs/README.md`) for the relevant section.
2. **See the student's current state.** Run the lab: `python3 labs/<file>`. Read their changes
   (`git diff -- labs/<file>` if in git). Note the failing assert or error, if any.
3. **Find the level.** Check `.study/hints.md` for earlier hints on this exercise to continue from
   the next level. If none, start at L0.
4. **Give exactly one level**, then stop and let the student work:
   - **L0 — Ask back:** ask what they expect the code to do and what they observed.
   - **L1 — Concept:** name the concept and the exact module section to reread.
   - **L2 — Location:** point to the function/line/term that is wrong or missing.
   - **L3 — Shape:** pseudocode or a worked numeric example of the step, not runnable code.
   - **L4 — Minimal code:** only if the student explicitly asks after L3. Write the smallest
     unblocking snippet, explain each line, and ask one question that checks they understood it.
5. **Log it.** Append one line to `.study/hints.md` (create the folder/file if missing):
   `YYYY-MM-DD | <lab> | <exercise> | L<n> | <one-line summary of the hint>`.
   `.study/` is git-ignored; it is the student's private log.

## Rules

- Never paste a full solution to an exercise. Never edit the lab file yourself unless at L4,
  and then only the minimal snippet the student asked for.
- Never weaken or delete an `assert` to make a lab pass; if an assert fails, the hint is about
  why it fails.
- If the problem is environment/tooling (missing NumPy, Python version, shapes in an error
  message unrelated to the exercise), just fix or explain it directly — that is not the learning goal.
