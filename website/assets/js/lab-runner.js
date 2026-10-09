// In-browser lab runner: edit a lab and run it with Python + NumPy via Pyodide.
// Python runs in a Web Worker so long labs never freeze the page, and Stop can kill it.
(function () {
  "use strict";
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";

  const WORKER_SRC = `
    importScripts("${PYODIDE}pyodide.js");
    const ready = (async () => {
      self.pyodide = await loadPyodide({ indexURL: "${PYODIDE}" });
      await self.pyodide.loadPackage("numpy");
    })();
    self.onmessage = async (e) => {
      try { await ready; } catch (err) {
        postMessage({ type: "done", ok: false, error: "Could not load Python: " + err });
        return;
      }
      const py = self.pyodide;
      py.setStdout({ batched: (s) => postMessage({ type: "out", text: s }) });
      py.setStderr({ batched: (s) => postMessage({ type: "out", text: s }) });
      const globals = py.toPy({ __name__: "__main__" });
      try {
        await py.runPythonAsync(e.data.code, { globals });
        postMessage({ type: "done", ok: true });
      } catch (err) {
        postMessage({ type: "done", ok: false, error: String(err.message || err) });
      } finally {
        globals.destroy();
      }
    };`;

  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch (_) { /* storage blocked: runner still works, edits just aren't kept */ }
    return null;
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  }

  async function mount(root) {
    const src = new URL(root.dataset.src, location.href).href;
    const key = "mlwc:lab:" + root.dataset.lab;
    const original = await fetch(src).then((r) => r.text());

    const editor = el("textarea", "lab-editor");
    editor.spellcheck = false;
    editor.value = store(key) || original;
    editor.addEventListener("input", () => store(key, editor.value));
    editor.addEventListener("keydown", (e) => {
      if (e.key !== "Tab") return;
      e.preventDefault();
      editor.setRangeText("    ", editor.selectionStart, editor.selectionEnd, "end");
    });

    const run = el("button", "md-button md-button--primary", "▶ Run");
    const stop = el("button", "md-button", "■ Stop");
    const reset = el("button", "md-button", "↺ Reset code");
    const status = el("span", "lab-status", store(key) ? "Your edits are restored." : "");
    const bar = el("div", "lab-toolbar");
    bar.append(run, stop, reset, status);
    const output = el("pre", "lab-output");
    stop.disabled = true;
    root.append(bar, editor, output);

    let worker = null;
    const startWorker = () => {
      worker = new Worker(URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" })));
    };

    const finish = (ok, message) => {
      run.disabled = false;
      stop.disabled = true;
      root.dataset.state = ok ? "pass" : "fail";
      status.textContent = message;
    };

    run.addEventListener("click", () => {
      if (!worker) startWorker();
      output.textContent = "";
      root.dataset.state = "running";
      status.textContent = "Running… (the first run downloads Python + NumPy, ~10 MB)";
      run.disabled = true;
      stop.disabled = false;
      const started = performance.now();
      worker.onmessage = (e) => {
        const msg = e.data;
        if (msg.type === "out") {
          output.textContent += msg.text + "\n";
          output.scrollTop = output.scrollHeight;
          return;
        }
        const secs = ((performance.now() - started) / 1000).toFixed(1);
        if (msg.ok) {
          finish(true, `✅ All checks passed (${secs}s).`);
        } else {
          output.textContent += "\n" + msg.error;
          output.scrollTop = output.scrollHeight;
          const failed = /AssertionError/.test(msg.error) ? "an assert failed" : "an error occurred";
          finish(false, `❌ Not passing yet: ${failed}. Read the traceback at the bottom of the output.`);
        }
      };
      worker.postMessage({ code: editor.value });
    });

    stop.addEventListener("click", () => {
      worker.terminate();
      worker = null;
      finish(false, "Stopped.");
    });

    reset.addEventListener("click", () => {
      if (!confirm("Discard your edits and restore the original lab code?")) return;
      editor.value = original;
      store(key, null);
      status.textContent = "Original code restored.";
    });
  }

  function init() {
    document.querySelectorAll(".lab-runner:not([data-mounted])").forEach((root) => {
      root.dataset.mounted = "1";
      mount(root).catch((err) => {
        root.textContent = "Could not load the lab code: " + err;
      });
    });
  }

  if (window.document$) window.document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
