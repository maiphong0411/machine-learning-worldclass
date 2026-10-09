---
name: quiz-me
description: Run an adaptive retrieval-practice quiz on course modules, prioritizing topics the student previously got wrong, and log results for spaced repetition. Use when a student asks to be quizzed, tested, or to revise/review a module.
argument-hint: [module, e.g. "M04" or "all"] [number of questions]
---

# Adaptive quiz

Request: $ARGUMENTS (default: 5 questions, chosen by the spacing rule below).

## Steps

1. **Load the student's history** from `.study/quiz-log.md` if it exists. Each line is
   `YYYY-MM-DD | <module> | <topic> | correct|partial|wrong`.
2. **Choose topics.**
   - If a module is given, quiz that module.
   - Otherwise, prioritize: (a) topics answered `wrong`/`partial` most recently, (b) topics last
     seen longest ago, (c) the module matching the current week in `SYLLABUS.md` if unknown.
3. **Generate questions** grounded in the module text (`modules/` or `system-design/`) and in
   the style of `assessments/quizzes.md`: mix short answer, one small calculation, and one
   "spot the bug / spot the leak" scenario. Write **new** questions; do not copy the bank verbatim
   (students may have already seen those answers).
4. **Ask one question at a time.** Wait for the answer. Do not show options-with-answer or hints
   unless asked.
5. **Grade each answer** as correct / partial / wrong, explain the reasoning briefly from first
   principles, and cite the module section. For partial answers, ask one follow-up before moving on.
6. **Log results** by appending one line per question to `.study/quiz-log.md` (create it if missing).
7. **Finish** with a score, the two weakest topics, and what to reread or which lab to redo.
   Suggest when to quiz again (e.g. wrong → tomorrow, partial → in 3 days, correct → in a week).
