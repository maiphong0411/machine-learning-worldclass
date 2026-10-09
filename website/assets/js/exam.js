// Interactive practice exam for the Databricks Generative AI Engineer Associate track.
// Modes: timed mock exam (45 questions / 90 minutes, weighted like the real exam), domain
// drills with instant feedback, and a "review my mistakes" deck. Attempt history and the
// mistakes deck are kept in this browser only (localStorage).
(function () {
  "use strict";
  const ROOT = new URL("../../", document.currentScript.src);
  const DATA_URL = new URL("assets/genai-questions.json", ROOT).href;
  const KEY_HISTORY = "mlwc:genai:history";
  const KEY_MISTAKES = "mlwc:genai:mistakes";
  // Questions per domain in a 45-question mock, matching the exam weights (14/14/30/22/8/12%).
  const MOCK_PLAN = { 1: 6, 2: 6, 3: 14, 4: 10, 5: 4, 6: 5 };
  const MOCK_MINUTES = 90;
  const TARGET = 80; // percent; Databricks does not publish a passing score, so aim high.

  let bank = null;
  let host = null;
  let state = null; // { mode, items:[{q, picked:Set, flagged}], index, deadline, timer, checked:Set }

  // ---------- storage ----------
  function read(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch (_) { return fallback; }
  }
  function write(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* storage blocked */ }
  }

  // ---------- helpers ----------
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    });
    children.flat(Infinity).forEach((c) => node.append(c instanceof Node ? c : document.createTextNode(c)));
    return node;
  }
  // Question text is plain text with `inline code` and blank-line paragraphs.
  function rich(text) {
    const wrap = el("div", { class: "exam-rich" });
    text.trim().split(/\n\s*\n/).forEach((para) => {
      const p = el("p");
      para.split(/(`[^`]+`)/).forEach((part) => {
        if (part.startsWith("`") && part.endsWith("`") && part.length > 1) p.append(el("code", {}, part.slice(1, -1)));
        else p.append(part);
      });
      wrap.append(p);
    });
    return wrap;
  }
  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  function isCorrect(item) {
    const want = item.q.answer;
    return item.picked.size === want.length && want.every((a) => item.picked.has(a));
  }
  function domainName(d) { return bank.domains[String(d)].name; }
  function fmtTime(ms) {
    const s = Math.max(0, Math.round(ms / 1000));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  // ---------- setup screen ----------
  function renderSetup() {
    stopTimer();
    state = null;
    host.innerHTML = "";
    const mistakes = read(KEY_MISTAKES, []);
    const counts = {};
    bank.questions.forEach((q) => { counts[q.domain] = (counts[q.domain] || 0) + 1; });

    const domainSelect = el("select", { class: "exam-select" },
      Object.keys(bank.domains).map((d) =>
        el("option", { value: d }, `${d}. ${domainName(d)} (${counts[d] || 0} questions)`)));

    host.append(
      el("div", { class: "exam-modes" },
        card("Full mock exam", `45 questions, ${MOCK_MINUTES} minutes, weighted like the real exam. ` +
          "No feedback until you submit.", "Start mock exam", () => startMock()),
        card("Domain drill", "All questions from one domain, with the answer and explanation after each one.",
          "Start drill", () => startDrill(Number(domainSelect.value)), domainSelect),
        card("Review my mistakes", mistakes.length
          ? `${mistakes.length} question(s) you got wrong. Answer them correctly to clear them.`
          : "Questions you get wrong are collected here.",
          "Review mistakes", () => startReview(), null, !mistakes.length)),
      historyTable());
  }

  function card(title, text, button, onclick, extra, disabled) {
    const btn = el("button", { class: "md-button md-button--primary", onclick }, button);
    if (disabled) btn.disabled = true;
    return el("div", { class: "exam-card" }, el("h3", {}, title), el("p", {}, text), extra || "", btn);
  }

  function historyTable() {
    const history = read(KEY_HISTORY, []);
    if (!history.length) return el("p", { class: "exam-muted" }, "Your mock exam results will appear here.");
    const domains = Object.keys(bank.domains);
    return el("div", {},
      el("h3", {}, "Your mock exam history"),
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "Date"), el("th", {}, "Score"),
          domains.map((d) => el("th", { title: domainName(d) }, `D${d}`)), el("th", {}, "Time"))),
        el("tbody", {}, history.slice(-10).reverse().map((h) =>
          el("tr", {}, el("td", {}, h.date), el("td", {}, `${h.score}%`),
            domains.map((d) => el("td", {}, h.byDomain[d] == null ? "–" : `${h.byDomain[d]}%`)),
            el("td", {}, h.time))))));
  }

  // ---------- starting a session ----------
  function startMock() {
    const items = [];
    Object.entries(MOCK_PLAN).forEach(([d, n]) => {
      shuffle(bank.questions.filter((q) => q.domain === Number(d))).slice(0, n).forEach((q) => items.push(q));
    });
    begin("mock", shuffle(items), Date.now() + MOCK_MINUTES * 60000);
  }
  function startDrill(domain) {
    begin("drill", shuffle(bank.questions.filter((q) => q.domain === domain)));
  }
  function startReview() {
    const ids = new Set(read(KEY_MISTAKES, []));
    begin("review", shuffle(bank.questions.filter((q) => ids.has(q.id))));
  }
  function begin(mode, questions, deadline) {
    state = {
      mode, index: 0, deadline, started: Date.now(), checked: new Set(),
      items: questions.map((q) => ({ q, picked: new Set(), flagged: false })),
    };
    if (deadline) {
      state.timer = setInterval(() => {
        if (Date.now() >= state.deadline) submit(true);
        else updateClock();
      }, 1000);
      window.addEventListener("beforeunload", warnOnLeave);
    }
    renderQuestion();
  }
  function stopTimer() {
    if (state && state.timer) clearInterval(state.timer);
    window.removeEventListener("beforeunload", warnOnLeave);
  }
  function warnOnLeave(e) { e.preventDefault(); e.returnValue = ""; }

  // ---------- question screen ----------
  function updateClock() {
    const clock = host.querySelector(".exam-clock");
    if (clock && state.deadline) {
      const left = state.deadline - Date.now();
      clock.textContent = `⏱ ${fmtTime(left)}`;
      clock.classList.toggle("exam-clock--low", left < 10 * 60000);
    }
  }

  function renderQuestion() {
    host.innerHTML = "";
    const item = state.items[state.index];
    const q = item.q;
    const multi = q.answer.length > 1;
    const checked = state.checked.has(state.index);

    const header = el("div", { class: "exam-header" },
      el("span", {}, `Question ${state.index + 1} of ${state.items.length}`),
      el("span", { class: "exam-muted" }, `Domain ${q.domain} · ${domainName(q.domain)}`),
      state.deadline ? el("span", { class: "exam-clock" }) : "");

    const choices = el("div", { class: "exam-choices", role: multi ? "group" : "radiogroup" },
      Object.entries(q.choices).map(([letter, text]) => {
        const input = el("input", { type: multi ? "checkbox" : "radio", name: "choice", value: letter });
        input.checked = item.picked.has(letter);
        input.disabled = checked;
        input.addEventListener("change", () => {
          if (!multi) item.picked.clear();
          if (input.checked) item.picked.add(letter); else item.picked.delete(letter);
          renderNav();
        });
        const label = el("label", { class: "exam-choice" }, input, el("b", {}, `${letter}. `), text);
        if (checked) {
          if (q.answer.includes(letter)) label.classList.add("exam-choice--right");
          else if (item.picked.has(letter)) label.classList.add("exam-choice--wrong");
        }
        return label;
      }));

    const flag = el("button", { class: "md-button", onclick: () => { item.flagged = !item.flagged; renderQuestion(); } },
      item.flagged ? "★ Flagged" : "☆ Flag for review");
    const prev = el("button", { class: "md-button", onclick: () => go(-1) }, "← Previous");
    const next = el("button", { class: "md-button", onclick: () => go(1) }, "Next →");
    prev.disabled = state.index === 0;
    next.disabled = state.index === state.items.length - 1;
    const actions = [prev, next, flag];
    if (state.mode !== "mock" && !checked) {
      const check = el("button", { class: "md-button md-button--primary", onclick: () => checkAnswer() }, "Check answer");
      check.disabled = item.picked.size === 0;
      actions.splice(2, 0, check);
    }
    actions.push(el("button", { class: "md-button md-button--primary exam-submit", onclick: () => submit(false) },
      state.mode === "mock" ? "Submit exam" : "Finish"));

    host.dataset.qid = q.id;
    host.append(header,
      el("p", { class: "exam-objective" }, `${q.id} · Objective: ${q.objective}`),
      rich(q.question),
      multi ? el("p", { class: "exam-muted" }, "Select TWO answers.") : "",
      choices,
      checked ? explanation(q, isCorrect(item)) : "",
      el("div", { class: "exam-actions" }, actions),
      el("div", { class: "exam-nav" }));
    renderNav();
    updateClock();
  }

  function renderNav() {
    const nav = host.querySelector(".exam-nav");
    if (!nav) return;
    nav.innerHTML = "";
    state.items.forEach((it, i) => {
      let cls = "exam-nav__item";
      if (i === state.index) cls += " exam-nav__item--current";
      if (it.picked.size) cls += " exam-nav__item--answered";
      if (it.flagged) cls += " exam-nav__item--flagged";
      if (state.checked.has(i)) cls += isCorrect(it) ? " exam-nav__item--right" : " exam-nav__item--wrong";
      nav.append(el("button", { class: cls, title: `Question ${i + 1}`, onclick: () => { state.index = i; renderQuestion(); } },
        String(i + 1)));
    });
    const check = host.querySelector(".exam-actions .md-button--primary:not(.exam-submit)");
    if (check) check.disabled = state.items[state.index].picked.size === 0;
  }

  function go(step) {
    state.index = Math.min(state.items.length - 1, Math.max(0, state.index + step));
    renderQuestion();
  }

  function checkAnswer() {
    state.checked.add(state.index);
    recordMistake(state.items[state.index]);
    renderQuestion();
  }

  function explanation(q, right) {
    return el("div", { class: `exam-explain ${right ? "exam-explain--right" : "exam-explain--wrong"}` },
      el("p", {}, el("b", {}, right ? "✅ Correct. " : `❌ Not quite. Correct answer: ${q.answer.join(", ")}. `)),
      rich(q.explanation),
      el("p", { class: "exam-refs" }, "Docs: ",
        q.refs.map((r, i) => [i ? " · " : "", el("a", { href: r, target: "_blank", rel: "noopener" }, `[${i + 1}]`)])));
  }

  function recordMistake(item) {
    const ids = new Set(read(KEY_MISTAKES, []));
    if (isCorrect(item)) ids.delete(item.q.id); else ids.add(item.q.id);
    write(KEY_MISTAKES, [...ids]);
  }

  // ---------- results ----------
  function submit(timeUp) {
    const unanswered = state.items.filter((it) => !it.picked.size).length;
    if (!timeUp && unanswered && !confirm(`${unanswered} question(s) unanswered. Submit anyway?`)) return;
    stopTimer();
    state.items.forEach((it, i) => { if (!state.checked.has(i)) recordMistake(it); });

    const byDomain = {};
    Object.keys(bank.domains).forEach((d) => {
      const items = state.items.filter((it) => it.q.domain === Number(d));
      if (items.length) byDomain[d] = Math.round((100 * items.filter(isCorrect).length) / items.length);
    });
    const right = state.items.filter(isCorrect).length;
    const score = Math.round((100 * right) / Math.max(1, state.items.length));
    const time = fmtTime(Date.now() - state.started);
    if (state.mode === "mock") {
      const history = read(KEY_HISTORY, []);
      history.push({ date: new Date().toISOString().slice(0, 10), score, byDomain, time });
      write(KEY_HISTORY, history.slice(-50));
    }

    host.innerHTML = "";
    const weakest = Object.entries(byDomain).sort((a, b) => a[1] - b[1])[0];
    host.append(
      el("div", { class: `exam-score ${score >= TARGET ? "exam-score--good" : "exam-score--low"}` },
        el("div", { class: "exam-score__big" }, `${score}%`),
        el("div", {}, `${right} of ${state.items.length} correct · ${time}${timeUp ? " · time ran out" : ""}`),
        el("div", { class: "exam-muted" }, score >= TARGET
          ? "At or above the 80% target we recommend before booking the real exam."
          : `Below the 80% target. Focus next on Domain ${weakest[0]} (${domainName(weakest[0])}).`)),
      el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "Domain"), el("th", {}, "Exam weight"), el("th", {}, "Your score"))),
        el("tbody", {}, Object.entries(byDomain).map(([d, s]) => el("tr", {},
          el("td", {}, `${d}. ${domainName(d)}`), el("td", {}, `${bank.domains[d].weight}%`),
          el("td", {}, el("span", { class: "exam-bar" }, el("span", { style: `width:${s}%` })), ` ${s}%`))))),
      el("div", { class: "exam-actions" },
        el("button", { class: "md-button md-button--primary", onclick: renderSetup }, "Back to practice menu")),
      el("h3", {}, "Review every question"),
      ...state.items.map((it, i) => el("details", { class: "exam-review" },
        el("summary", {}, `${isCorrect(it) ? "✅" : "❌"} ${i + 1}. ${it.q.id} — ${it.q.objective}`),
        rich(it.q.question),
        el("ul", {}, Object.entries(it.q.choices).map(([l, t]) => el("li", {
          class: it.q.answer.includes(l) ? "exam-choice--right" : it.picked.has(l) ? "exam-choice--wrong" : "",
        }, `${l}. ${t}${it.picked.has(l) ? "  ← your answer" : ""}`))),
        explanation(it.q, isCorrect(it)))));
    window.scrollTo(0, 0);
  }

  // ---------- init ----------
  async function init() {
    host = document.getElementById("genai-exam");
    if (!host || host.dataset.mounted) return;
    host.dataset.mounted = "1";
    try {
      bank = await fetch(DATA_URL).then((r) => r.json());
      renderSetup();
    } catch (err) {
      host.textContent = "Could not load the question bank: " + err;
    }
  }
  if (window.document$) window.document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
