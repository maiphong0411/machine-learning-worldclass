---
name: explain-back
description: Feynman-technique session where the student explains a concept in their own words and Claude acts as a curious novice, probing for gaps and misconceptions. Use when a student wants to test whether they really understand something.
argument-hint: <topic>
---

# Explain it back (Feynman technique)

Topic: $ARGUMENTS (ask for one if empty).

## Steps

1. **Read the course's treatment** of the topic (search `modules/`, `system-design/`,
   `case-studies/`) so you know what a complete explanation must contain. Privately list the
   3–6 essential ideas (e.g. for bias–variance: definition of each term, the decomposition,
   how model capacity trades them, how data size affects variance, how to diagnose each).
2. **Ask the student to explain the topic** as if to a smart first-year student, in their own
   words, with one example. Tell them not to look at the notes.
3. **Play the curious novice.** Ask "why?" and "what would happen if…?" questions, one at a time,
   aimed at the essential ideas they skipped or got wrong. Do not lecture during this phase.
4. **Debrief** after 4–8 exchanges:
   - which essential ideas they explained well (be specific),
   - which were missing or wrong, with a short first-principles correction and the module section,
   - one sentence they could use as a crisp summary.
5. Suggest the next step: `/quiz-me` on the weak ideas, or the lab that exercises them.
