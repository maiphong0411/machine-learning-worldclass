# Module authoring template

Every module (M01–M12) uses exactly these sections, in this order. Case studies use the
template at the bottom of this file. Consistency is deliberate: students learn the
*shape* of reasoning, which is the same shape the ML system design interview rewards.

---

## Module template

```markdown
# Mxx — <Title>

> **One-sentence big idea.** (What a student should still remember in five years.)

**Prerequisites:** links to earlier modules · **Lab:** link · **Time:** lecture hours

## Learning objectives
- 4–6 bullets, each starting with a measurable verb (derive, implement, compare, choose, diagnose).

## 1. The problem
A concrete real-world situation that this module's ideas solve. No formulas yet.

## 2. First principles
Build the idea from the ground up. Ask "what is the simplest thing that could work, and
why does it fail?" then fix it step by step until the standard algorithm appears.
State every assumption explicitly.

## 3. The algorithm(s)
Math (LaTeX with $...$ / $$...$$), then pseudocode, then complexity (time/memory, train/inference).

## 4. Diagrams
At least two Mermaid diagrams: one for *data/compute flow*, one for *decision logic or
intuition* (e.g. a flowchart of when to use it). Diagrams sit next to the text they explain.

## 5. Code
Short, readable NumPy (from scratch) — and the one-line library equivalent
(scikit-learn / PyTorch) so students see what the library hides.

## 6. Real-world applications
2–4 named industry uses, each with: the task, why this algorithm, and the constraint that matters.

## 7. System-design hook
How this topic shows up in an ML system design interview: what an interviewer probes,
what a strong answer sounds like, typical trade-offs.

## 8. Pitfalls & debugging
Failure modes, how to detect them, how to fix them.

## 9. Exercises
- Conceptual (3+), derivation/math (2+), coding (2+), design (1+). Mark difficulty ★/★★/★★★.

## 10. Interview questions
5–8 questions with concise model answers (in <details> blocks).

## Further reading
3–5 canonical references (papers, book chapters, engineering blogs).
```

---

## Case study template

```markdown
# Case study N — <System>

> **Interview prompt:** "Design a system that ..." (exactly how an interviewer would phrase it)

## 1. Clarify requirements   (functional, non-functional: scale, latency, freshness, cost)
## 2. Frame as an ML problem (business objective → ML objective → labels → input/output)
## 3. Data                   (sources, labeling, sampling, imbalance, privacy)
## 4. Features               (table: feature, type, source, online/offline, freshness)
## 5. Model                  (baseline → production model → why; multi-stage if needed)
## 6. Training               (loss, splits, negatives, retraining cadence)
## 7. Evaluation             (offline metrics, online metrics, guardrails, A/B design)
## 8. Serving architecture   (Mermaid diagram; latency budget table)
## 9. Monitoring & iteration (drift, feedback loops, failure modes)
## 10. Trade-offs & extensions
## 11. Interviewer follow-ups (with model answers)
```

---

## Style rules

- **Mermaid:** quote any node label containing punctuation, e.g. `A["Score (0–1)"]`;
  use `<br/>` for line breaks; keep diagrams under ~20 nodes; one idea per diagram.
- **Math:** define every symbol on first use. Prefer intuition sentence → formula → worked number.
- **Real-world claims:** name the company/system only where it is publicly documented;
  otherwise say "a large e-commerce platform".
- **Tone:** explain *why* before *how*. If a step feels like magic, it needs another sentence.
