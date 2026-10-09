// Personal progress tracking, stored only in this browser (localStorage).
// Adds a "Mark as done" toggle to every course page, and renders the dashboard on progress.md.
(function () {
  "use strict";
  const KEY = "mlwc:done";
  // The site root is two levels above this script (<root>/assets/js/progress.js).
  const ROOT = new URL("../../", document.currentScript.src);

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (_) { return {}; }
  }
  function save(done) {
    try { localStorage.setItem(KEY, JSON.stringify(done)); } catch (_) { /* storage blocked */ }
  }
  function pageId(href) {
    const path = new URL(href, location.href).pathname;
    return decodeURIComponent(path.slice(ROOT.pathname.length)).replace(/index\.html$/, "") || "/";
  }

  function addToggle() {
    const article = document.querySelector("article.md-content__inner");
    const id = pageId(location.href);
    if (!article || id === "/" || id === "progress/" || article.querySelector(".done-toggle")) return;
    const label = document.createElement("label");
    label.className = "done-toggle";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = Boolean(load()[id]);
    box.addEventListener("change", () => {
      const done = load();
      if (box.checked) done[id] = new Date().toISOString().slice(0, 10);
      else delete done[id];
      save(done);
    });
    label.append(box, document.createTextNode(" Mark this page as done"));
    article.prepend(label);
  }

  function renderDashboard() {
    const host = document.getElementById("progress-dashboard");
    if (!host) return;
    const done = load();
    // Read the course structure from the site navigation so this never goes out of date.
    const sections = [];
    document.querySelectorAll(".md-nav--primary > .md-nav__list > .md-nav__item").forEach((item) => {
      const title = (item.querySelector(":scope > label, :scope > a") || {}).textContent;
      const links = [...item.querySelectorAll("a.md-nav__link[href]")]
        .filter((a) => !a.getAttribute("href").startsWith("#"))
        .map((a) => ({ id: pageId(a.href), href: a.href, text: a.textContent.trim() }))
        .filter((l, i, all) => l.id !== "progress/" && all.findIndex((m) => m.id === l.id) === i);
      if (links.length && title) sections.push({ title: title.trim(), links });
    });

    const all = sections.flatMap((s) => s.links);
    const count = all.filter((l) => done[l.id]).length;
    host.innerHTML = "";
    const summary = document.createElement("p");
    summary.className = "progress-summary";
    summary.textContent = `${count} of ${all.length} pages done (${Math.round((100 * count) / Math.max(all.length, 1))}%)`;
    const bar = document.createElement("progress");
    bar.max = all.length;
    bar.value = count;
    host.append(summary, bar);

    sections.forEach((s) => {
      const n = s.links.filter((l) => done[l.id]).length;
      const h = document.createElement("h3");
      h.textContent = `${s.title} — ${n}/${s.links.length}`;
      const ul = document.createElement("ul");
      ul.className = "progress-list";
      s.links.forEach((l) => {
        const li = document.createElement("li");
        li.textContent = done[l.id] ? "✅ " : "⬜ ";
        const a = document.createElement("a");
        a.href = l.href;
        a.textContent = l.text;
        li.append(a);
        if (done[l.id]) li.append(document.createTextNode(` (${done[l.id]})`));
        ul.append(li);
      });
      host.append(h, ul);
    });

    const copy = document.getElementById("progress-copy");
    if (copy) {
      copy.onclick = () => {
        const lines = sections.map((s) =>
          `${s.title}: ${s.links.filter((l) => done[l.id]).map((l) => l.text).join(", ") || "-"}`);
        const text = `ML course progress: ${count}/${all.length} pages\n` + lines.join("\n");
        navigator.clipboard.writeText(text).then(
          () => { copy.textContent = "Copied!"; },
          () => { prompt("Copy your progress summary:", text); });
      };
    }
  }

  function init() { addToggle(); renderDashboard(); }
  if (window.document$) window.document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
