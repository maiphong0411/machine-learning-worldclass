"""Assemble the course into a MkDocs source tree (website/_src) for GitHub Pages.

The repository is written to read well on GitHub; this script adapts it for the website
without changing the source files:

* README.md files become index.md (the site's section landing pages).
* Relative links are re-resolved: links to course pages stay relative, links to labs/*.py go
  to the interactive lab-runner pages, and links to anything else (AGENTS.md, .claude/skills/)
  go to GitHub.
* <details> blocks get markdown="1" so the answers inside render math and formatting.
* One runner page is generated per lab, so students can edit and run it in the browser.

Usage:  python3 website/build.py && mkdocs build -f website/mkdocs.yml --strict
"""

import ast
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "website" / "_src"
REPO_URL = "https://github.com/maiphong0411/machine-learning-worldclass"

SECTIONS = ["modules", "system-design", "case-studies", "assessments", "docs", "labs"]
TOP_LEVEL = ["README.md", "SYLLABUS.md"]
# Repo-relative markdown files that are part of the site.
PAGES = TOP_LEVEL + [str(p.relative_to(ROOT)) for s in SECTIONS for p in sorted((ROOT / s).glob("*.md"))]
LABS = sorted((ROOT / "labs").glob("*.py"))

LINK = re.compile(r"(\]\()([^)\s#]+)(#[^)\s]*)?(\))")


def site_path(repo_path: str) -> str:
    """Where a repo markdown file lives in the site source tree."""
    p = Path(repo_path)
    return str(p.with_name("index.md")) if p.name == "README.md" else repo_path


def resolve(target: str, from_file: str) -> str | None:
    """Map a link target in `from_file` to a site-relative path, or None if not on the site."""
    repo_target = os.path.normpath(os.path.join(os.path.dirname(from_file), target))
    abs_target = ROOT / repo_target
    if abs_target.is_dir() and (abs_target / "README.md").exists():
        repo_target = os.path.join(repo_target, "README.md")
    if repo_target in PAGES:
        return site_path(repo_target)
    if repo_target.startswith("labs/") and repo_target.endswith(".py"):
        return repo_target[:-3] + ".md"
    return None


def rewrite_links(text: str, from_file: str) -> str:
    def sub(m: re.Match) -> str:
        target, anchor = m.group(2), m.group(3) or ""
        if re.match(r"^[a-z]+:", target) or target.startswith("/"):
            return m.group(0)
        resolved = resolve(target, from_file)
        if resolved is None:
            repo_target = os.path.normpath(os.path.join(os.path.dirname(from_file), target))
            kind = "tree" if (ROOT / repo_target).is_dir() else "blob"
            return f"{m.group(1)}{REPO_URL}/{kind}/main/{repo_target}{anchor}{m.group(4)}"
        rel = os.path.relpath(resolved, os.path.dirname(site_path(from_file)) or ".")
        return f"{m.group(1)}{rel}{anchor}{m.group(4)}"

    # Leave fenced code blocks untouched.
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    return "".join(p if p.startswith("```") else LINK.sub(sub, p) for p in parts)


CONTENTS_LIST = re.compile(r"^## Contents\n\n(?:(?:\d+\.|-|\s) .*\n)+\n?(?:---\n\n)?", re.M)


def adapt(text: str, from_file: str) -> str:
    text = rewrite_links(text, from_file)
    # A hand-written "Contents" link list repeats the site's own table of contents.
    text = CONTENTS_LIST.sub("", text)
    # Only real <details> blocks (at line start), not ones quoted in `inline code`.
    return re.sub(r"^<details>", '<details markdown="1">', text, flags=re.M)


def lab_page(lab: Path) -> str:
    doc = ast.get_docstring(ast.parse(lab.read_text())) or ""
    title = doc.strip().splitlines()[0].strip() if doc.strip() else lab.stem
    return f"""# {title}

!!! tip "Run it in your browser"
    Edit the code below and press **Run**. It runs with Python + NumPy inside your browser
    (via Pyodide). Nothing is installed or sent anywhere. The lab passes when all asserts
    pass. Stuck? Use `/hint {lab.stem[:2]}` in Claude Code, or the
    [hint ladder](../docs/ai-assisted-learning.md#2-the-six-study-modes).

<div class="lab-runner" data-src="../{lab.name}" data-lab="{lab.stem}"></div>

[Download {lab.name}]({lab.name}) · [View on GitHub]({REPO_URL}/blob/main/labs/{lab.name})
"""


def main() -> None:
    if SRC.exists():
        shutil.rmtree(SRC)
    for page in PAGES:
        out = SRC / site_path(page)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(adapt((ROOT / page).read_text(), page))
    for lab in LABS:
        shutil.copy(lab, SRC / "labs" / lab.name)
        (SRC / "labs" / f"{lab.stem}.md").write_text(lab_page(lab))
    shutil.copytree(ROOT / "website" / "assets", SRC / "assets")
    shutil.copy(ROOT / "website" / "progress.md", SRC / "progress.md")
    print(f"Assembled {len(PAGES)} pages and {len(LABS)} lab runners into {SRC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
