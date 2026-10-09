// md-ellipsis: also typeset math in navigation/table-of-contents labels (headings like "AUC in O(n log n)").
window.MathJax = {
  tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]], processEscapes: true, processEnvironments: true },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex|md-ellipsis" },
};
if (window.document$) {
  window.document$.subscribe(() => {
    if (window.MathJax.typesetPromise) {
      window.MathJax.startup.output.clearCache();
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      window.MathJax.typesetPromise();
    }
  });
}
