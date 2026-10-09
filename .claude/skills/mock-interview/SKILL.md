---
name: mock-interview
description: Run a realistic 45-minute ML system design mock interview with Claude as the interviewer, then score it with the course rubric. Only use when the student explicitly asks for a mock interview.
argument-hint: [prompt id like "P10", a domain like "search", or a custom prompt]
disable-model-invocation: true
---

# Mock ML system design interview

Request: $ARGUMENTS

## Before starting

1. Read `assessments/mock-interview-bank.md` (prompt bank, timing, interviewer rules) and
   `assessments/ml-system-design-rubric.md` (dimensions and anchors).
2. Pick the prompt: the given id/domain from the bank, the student's custom prompt, or a random
   prompt from a domain they have not practiced (check `.study/interviews.md` if it exists).
3. Privately decide realistic constraints (scale, latency, data available) so you can answer
   clarifying questions consistently. Keep the bank's "must-mention" points and traps in mind,
   but never reveal them during the interview.

## During the interview (you are the interviewer, not a tutor)

- Read the prompt verbatim and say: "You have about 45 minutes. Go ahead."
- Answer clarifying questions briefly with your fixed constraints.
- **Do not teach, hint, or validate** ("great", "exactly") during the interview. Neutral
  acknowledgements only ("OK", "go on").
- Track time by phase using the bank's 45-minute clock. If the student spends too long on one
  phase, redirect as a real interviewer would ("Let's move on to how you'd serve this.").
- Push with follow-up probes from the bank and from the student's own design: ask for numbers,
  trade-offs, failure modes. Ask at least one "what changes if…" twist (e.g. latency budget
  halves, labels become delayed by 30 days).
- Ask the student to draw their architecture; they may write a Mermaid diagram.
- End the interview when the student wraps up or after ~45 minutes of exchanges.

## After the interview (switch to coach)

1. Score each of the 8 rubric dimensions (No hire / Lean no / Lean hire / Strong hire), each
   justified with a **direct quote** from the student and the rubric anchor it matches.
2. Give the overall decision using the rubric's decision rules.
3. List the top 3 improvements, each with a concrete "say it like this" example and a link to
   the case study or module section that models it.
4. Append one line to `.study/interviews.md`:
   `YYYY-MM-DD | <prompt id/title> | <overall> | <weakest dimension>`.
