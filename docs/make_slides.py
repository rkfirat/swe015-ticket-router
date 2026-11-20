"""Builds docs/slides.pdf (beamer via tectonic) from results/ and docs/slides_content.py.

    python docs/make_slides.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS))
import slides_content as C  # noqa: E402


def tex_escape(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("<", r"\textless{}"), (">", r"\textgreater{}")]:
        s = s.replace(a, b)
    return s


def results_table() -> str:
    df = pd.read_csv(ROOT / "results" / "results.csv")
    cols = ["model", "precision", "accuracy", "in-scope acc", "OOS recall", "size (MB)", "latency (ms)"]
    head = " & ".join(rf"\textbf{{{tex_escape(c)}}}" for c in cols) + r" \\ \midrule"
    rows = "\n".join(" & ".join(tex_escape(r[c]) for c in cols) + r" \\" for _, r in df.iterrows())
    return (r"\resizebox{\linewidth}{!}{\begin{tabular}{llrrrrr}\toprule " + head + "\n" + rows
            + r"\bottomrule\end{tabular}}")


def frame(title: str, body: str) -> str:
    return f"\\begin{{frame}}{{{title}}}\n{body}\n\\end{{frame}}\n"


def main():
    hw = (ROOT / "results" / "results.md").read_text().split("\n")[0]
    fill = {"@@RESULTS@@": results_table(), "@@HARDWARE@@": tex_escape(hw)}
    body = "".join(frame(t, b) for t, b in C.FRAMES)
    for k, v in fill.items():
        body = body.replace(k, v)
    tex = C.PREAMBLE + "\\begin{document}\n" + body + "\\end{document}\n"
    (DOCS / "slides.tex").write_text(tex)
    subprocess.run(["tectonic", "-X", "compile", "--outdir", str(DOCS), str(DOCS / "slides.tex")], check=True)


if __name__ == "__main__":
    main()
