"""Convert markdown manuscript artifacts to Microsoft Word .docx with:
  - Cambria Math 1.15 line spacing via the reference template
  - Embedded figures (PNG) at the end with captions
  - Native Word equations from $$...$$ LaTeX (Office Math Markup Language)
  - Inline tables converted to native Word tables

Operates on the standard project layout:
  notebooks/<paper-slug>/<paper-slug>_main_text.md
  notebooks/<paper-slug>/<paper-slug>_appendix.md
  notebooks/<paper-slug>/cover_letter.md
  packaging/<paper-slug>/figures/figureN_*.png
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

REFERENCE_DOC = Path("/Users/sanjaybasu/.claude/templates/sanjay_paper_reference.docx")

FIGURE_CAPTION_PATTERN = re.compile(r"^- (Figure \d+\.\s+.*?)\s*$", re.MULTILINE)
TABLE_CAPTION_PATTERN = re.compile(r"^- (Table \d+\.\s+.*?)\s*$", re.MULTILINE)


def discover_figures(figures_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for png in sorted(figures_dir.glob("figure*.png")):
        m = re.match(r"figure(\d+)", png.stem)
        if not m:
            continue
        out[f"Figure {int(m.group(1))}"] = png
    return out


def discover_tables(tables_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for md in sorted(tables_dir.glob("table*.md")):
        m = re.match(r"table(\d+)", md.stem)
        if not m:
            continue
        out[f"Table {int(m.group(1))}"] = md
    return out


def append_figures_and_tables(md_text: str, figures: dict[str, Path], tables: dict[str, Path]) -> str:
    body = md_text.rstrip()

    fig_caps = FIGURE_CAPTION_PATTERN.findall(md_text)
    tbl_caps = TABLE_CAPTION_PATTERN.findall(md_text)

    out = [body, "", "---", "", "## Tables", ""]
    for cap in tbl_caps:
        m = re.match(r"(Table \d+)\.", cap)
        if not m:
            continue
        key = m.group(1)
        out.append(f"### {cap}")
        out.append("")
        if key in tables:
            md = tables[key].read_text().strip()
            out.append(md)
            out.append("")
        else:
            out.append(f"_(Table file not found at {key})_")
            out.append("")

    out.extend(["", "---", "", "## Figures", ""])
    for cap in fig_caps:
        m = re.match(r"(Figure \d+)\.", cap)
        if not m:
            continue
        key = m.group(1)
        out.append(f"### {cap}")
        out.append("")
        if key in figures:
            rel = figures[key]
            out.append(f"![{cap}]({rel})")
            out.append("")
        else:
            out.append(f"_(Figure file not found at {key})_")
            out.append("")

    return "\n".join(out)


def convert(md_path: Path, docx_path: Path, reference_doc: Path | None = None) -> None:
    cmd = [
        "pandoc",
        str(md_path),
        "-o", str(docx_path),
        "--from", "markdown+raw_html+tex_math_dollars+pipe_tables",
        "--to", "docx",
        "--standalone",
    ]
    if reference_doc and reference_doc.exists():
        cmd += ["--reference-doc", str(reference_doc)]
    subprocess.run(cmd, check=True)


def build_paper(paper_slug: str, root: Path = Path("/Users/sanjaybasu/waymark-local")) -> dict[str, Path]:
    notebooks = root / "notebooks" / paper_slug
    packaging = root / "packaging" / paper_slug
    figures_dir = packaging / "figures"
    tables_dir = packaging / "tables"

    figures = discover_figures(figures_dir)
    tables = discover_tables(tables_dir)

    main_md = notebooks / f"{paper_slug}_main_text.md"
    appendix_md = notebooks / f"{paper_slug}_appendix.md"
    cover_md = notebooks / "cover_letter.md"

    out: dict[str, Path] = {}
    if main_md.exists():
        rebuilt = append_figures_and_tables(main_md.read_text(), figures, tables)
        tmp = notebooks / "_tmp_main_for_docx.md"
        tmp.write_text(rebuilt)
        docx = notebooks / f"{paper_slug}_main_text.docx"
        convert(tmp, docx, REFERENCE_DOC)
        tmp.unlink()
        out["main_text"] = docx
        print(f"[ok] main_text -> {docx}")
    if appendix_md.exists():
        rebuilt = append_figures_and_tables(appendix_md.read_text(), figures, tables)
        tmp = notebooks / "_tmp_appendix_for_docx.md"
        tmp.write_text(rebuilt)
        docx = notebooks / f"{paper_slug}_appendix.docx"
        convert(tmp, docx, REFERENCE_DOC)
        tmp.unlink()
        out["appendix"] = docx
        print(f"[ok] appendix -> {docx}")
    if cover_md.exists():
        docx = notebooks / "cover_letter.docx"
        convert(cover_md, docx, REFERENCE_DOC)
        out["cover_letter"] = docx
        print(f"[ok] cover_letter -> {docx}")
    title_page_md = notebooks / "title_page.md"
    if title_page_md.exists():
        docx = notebooks / "title_page.docx"
        convert(title_page_md, docx, REFERENCE_DOC)
        out["title_page"] = docx
        print(f"[ok] title_page -> {docx}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="paper slug, e.g. medicaid-unwinding-chronic-disease-fairness")
    parser.add_argument("--root", default="/Users/sanjaybasu/waymark-local")
    args = parser.parse_args()
    build_paper(args.slug, Path(args.root))


if __name__ == "__main__":
    main()
