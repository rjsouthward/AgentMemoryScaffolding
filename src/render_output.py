"""
Utilities for extracting markdown tables from free-form text.

This module exposes a single function `extract_markdown_table` that scans an
input string and returns the first GitHub-style markdown pipe table it finds.

Speed-optimized: single pass with simple regex checks; no external deps.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Union
import sys
import json
from src.dataclass import RetrievedDocument, DocumentType

import re
from typing import Optional


_PIPE_LINE_START = re.compile(r"^\s*\|")
_ALIGN_CELL = re.compile(r"^:?-{3,}:?$")


def _is_alignment_row(line: str) -> bool:
    """
    Return True if the line is a markdown table alignment row.

    Example valid rows:
    |:---|---:|:----:|
    :---|---|
    """
    s = line.strip()
    if not s:
        return False
    # Allow with or without starting/ending pipes; split on '|'
    parts = s.strip("|").split("|")
    if not parts or all(p.strip() == "" for p in parts):
        return False
    for cell in parts:
        if not _ALIGN_CELL.fullmatch(cell.strip().replace(" ", "")):
            return False
    return True


def extract_markdown_table(text: str) -> Optional[str]:
    """
    Extract the first markdown pipe table from the given text.

    The function looks for a header line that starts with '|' followed by an
    alignment row (e.g., |:---|---:|). It then captures all subsequent lines
    that start with '|' as part of the table, stopping at the first line that
    does not belong to the table.

    Returns the table as a string with original line breaks, or None if no
    table is found.
    """
    if not text:
        return None

    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n - 1:  # need at least header + alignment
        line = lines[i]
        if _PIPE_LINE_START.match(line):
            # Candidate header line; next line must be an alignment row
            next_line = lines[i + 1] if i + 1 < n else ""
            if _is_alignment_row(next_line):
                # Collect contiguous table lines starting at header
                start = i
                j = i + 2
                while j < n and _PIPE_LINE_START.match(lines[j]):
                    j += 1
                # Return the block [start:j]
                return "\n".join(lines[start:j]).rstrip("\n")
            # If the next line is not an alignment row, continue scanning
        i += 1

    return None


def _coerce_documents(items: List[Union[dict, RetrievedDocument]]) -> List[RetrievedDocument]:
    out: List[RetrievedDocument] = []
    for it in items or []:
        if isinstance(it, RetrievedDocument):
            out.append(it)
        else:
            out.append(RetrievedDocument.from_dict(it))
    return out


def _extract_tables_for_docs(docs: List[RetrievedDocument]) -> Dict[int, str]:
    """Return mapping from 1-based citation index -> extracted table string."""
    tables: Dict[int, str] = {}
    for i, doc in enumerate(docs, start=1):
        if getattr(doc, "document_type", None) == DocumentType.DOCUMENT_TYPE_DATATALK:
            text = "\n\n".join(getattr(doc, "excerpts", []) or [])
            tbl = extract_markdown_table(text)
            if tbl:
                tables[i] = tbl
    return tables


def render_output_markdown(data: Union[dict, str]) -> str:
    """
    Render markdown from input with fields: topic, writeup, cited_documents.

    - Title: set to topic as H1
    - Body: writeup, with [n] citations replaced by [table k] when the nth
      cited_document is a Datatalk doc and a markdown table is extractable
    - Bibliography: appended after the body
      - First list all non-Datatalk citations (that appear in the writeup) with URL
      - Then list all table citations in order of first appearance, showing the table
    """
    if isinstance(data, str):
        data = json.loads(data)

    topic = (data.get("topic") or "").strip()
    writeup = data.get("writeup") or ""
    cited_docs = _coerce_documents(data.get("cited_documents", []))

    # Find all [n] citations in writeup in order
    citation_pattern = re.compile(r"\[(\d+)\]")
    matches = list(citation_pattern.finditer(writeup))
    cited_indices_in_text: List[int] = []
    seen: set[int] = set()
    for m in matches:
        idx = int(m.group(1))
        if 1 <= idx <= len(cited_docs) and idx not in seen:
            cited_indices_in_text.append(idx)
            seen.add(idx)

    # Map citation index -> table text if Datatalk with extractable table
    table_by_index = _extract_tables_for_docs(cited_docs)

    # Assign table indices by first appearance order in writeup
    table_index_map: Dict[int, int] = {}
    table_order: List[Tuple[int, int]] = []  # (table_idx, citation_idx)
    next_table_num = 1
    for idx in cited_indices_in_text:
        if idx in table_by_index and idx not in table_index_map:
            table_index_map[idx] = next_table_num
            table_order.append((next_table_num, idx))
            next_table_num += 1

    # Replace citations in body where applicable with anchors/links
    def _repl(m: re.Match[str]) -> str:
        i = int(m.group(1))
        # Table link
        if i in table_index_map:
            tnum = table_index_map[i]
            return f'<a href="#table-{tnum}">[table {tnum}]</a>'
        # Non-datatalk link to bibliography
        if 1 <= i <= len(cited_docs):
            doc = cited_docs[i - 1]
            if getattr(doc, "document_type", None) != DocumentType.DOCUMENT_TYPE_DATATALK:
                return f'<a href="#ref-{i}">[{i}]</a>'
        # Otherwise leave as-is
        return m.group(0)

    body = citation_pattern.sub(_repl, writeup)

    # Build bibliography sections
    lines: List[str] = []
    if topic:
        lines.append(f"# {topic}")
        lines.append("")

    if body:
        lines.append(body.rstrip())
        lines.append("")

    # Bibliography header
    lines.append("## Bibliography")

    # First: non-Datatalk citations with URL, in order of first appearance
    has_any_standard = False
    for idx in cited_indices_in_text:
        doc = cited_docs[idx - 1]
        if getattr(doc, "document_type", None) != DocumentType.DOCUMENT_TYPE_DATATALK:
            url = getattr(doc, "url", "") or ""
            title = getattr(doc, "title", None)
            if url:
                has_any_standard = True
                title_line = title or url
                lines.append(f"<a id=\"ref-{idx}\"></a>[{idx}] {title_line}")
                lines.append(f"Retrieved from <{url}>")
                lines.append("")
    if not has_any_standard:
        lines.append("(no non-datatalk citations)")

    # Then: table citations
    if table_order:
        lines.append("")
        lines.append("## Tables")
        for tnum, cidx in table_order:
            doc = cited_docs[cidx - 1]
            title = getattr(doc, "title", None)
            header = (
                f"Table {tnum} (from "
                f"<a href=\"#ref-{cidx}\">[{cidx}]</a>)"
            )
            if title:
                header += f": {title}"
            lines.append(f"<a id=\"table-{tnum}\"></a>")
            lines.append(f"### {header}")
            lines.append("")
            lines.append(table_by_index[cidx])
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_output_markdown_from_path(path: str) -> str:
    with open(path, "r") as f:
        data = json.load(f)
    return render_output_markdown(data)


# ---- LaTeX rendering ----

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def _latex_escape(text: str) -> str:
    if not text:
        return ""
    # Fast path: iterate and replace only if needed
    out = []
    for ch in text:
        out.append(_LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


def _markdown_table_to_latex(md_table: str) -> str:
    """Convert a simple GitHub-style pipe table to LaTeX tabular."""
    if not md_table:
        return ""
    lines = [ln.strip() for ln in md_table.splitlines() if ln.strip()]
    if len(lines) < 2:
        return _latex_escape(md_table)
    header = lines[0].strip("|")
    align = lines[1].strip("|")
    body_lines = lines[2:]

    def split_row(row: str) -> List[str]:
        return [c.strip() for c in row.strip("|").split("|")]

    header_cells = split_row(header)
    align_cells = split_row(align)
    col_count = max(len(header_cells), len(align_cells))

    # Determine column alignment spec
    specs: List[str] = []
    for i in range(col_count):
        a = align_cells[i] if i < len(align_cells) else ""
        s = a.replace(" ", "")
        if s.startswith(":") and s.endswith(":"):
            specs.append("c")
        elif s.endswith(":"):
            specs.append("r")
        else:
            specs.append("l")

    colspec = "".join(specs) or "l"

    def cells_to_row(cells: List[str]) -> str:
        # Ensure length matches columns
        cells = (cells + [""] * col_count)[:col_count]
        return " & ".join(_latex_escape(c) for c in cells) + r" \\"  # end row

    rows = []
    rows.append(cells_to_row(header_cells))
    rows.append(r"\hline")
    for bl in body_lines:
        rows.append(cells_to_row(split_row(bl)))

    return (
        f"\\begin{{tabular}}{{{colspec}}}\n"
        + "\n".join(rows)
        + "\n\\end{tabular}"
    )


def _markdown_to_latex_text(md_text: str) -> str:
    """Very small mapping of Markdown headings to LaTeX; escape other text."""
    out_lines: List[str] = []
    for line in (md_text or "").splitlines():
        s = line.lstrip()
        if s.startswith("### "):
            out_lines.append(r"\subsubsection*{" + _latex_escape(s[4:].strip()) + "}")
        elif s.startswith("## "):
            out_lines.append(r"\subsection*{" + _latex_escape(s[3:].strip()) + "}")
        elif s.startswith("# "):
            out_lines.append(r"\section*{" + _latex_escape(s[2:].strip()) + "}")
        else:
            out_lines.append(_latex_escape(line))
    return "\n".join(out_lines)


def render_output_latex(data: Union[dict, str]) -> str:
    if isinstance(data, str):
        data = json.loads(data)

    topic = (data.get("topic") or "").strip()
    writeup = data.get("writeup") or ""
    cited_docs = _coerce_documents(data.get("cited_documents", []))

    # Find citations in writeup
    citation_pattern = re.compile(r"\[(\d+)\]")
    matches = list(citation_pattern.finditer(writeup))
    cited_indices_in_text: List[int] = []
    seen: set[int] = set()
    for m in matches:
        idx = int(m.group(1))
        if 1 <= idx <= len(cited_docs) and idx not in seen:
            cited_indices_in_text.append(idx)
            seen.add(idx)

    table_by_index = _extract_tables_for_docs(cited_docs)

    table_index_map: Dict[int, int] = {}
    table_order: List[Tuple[int, int]] = []
    next_table_num = 1
    for idx in cited_indices_in_text:
        if idx in table_by_index and idx not in table_index_map:
            table_index_map[idx] = next_table_num
            table_order.append((next_table_num, idx))
            next_table_num += 1

    def _repl(m: re.Match[str]) -> str:
        i = int(m.group(1))
        if i in table_index_map:
            tnum = table_index_map[i]
            return f"<<<LINK:table-{tnum}:[table {tnum}]>>>"
        if 1 <= i <= len(cited_docs):
            doc = cited_docs[i - 1]
            if getattr(doc, "document_type", None) != DocumentType.DOCUMENT_TYPE_DATATALK:
                return f"<<<LINK:ref-{i}:[{i}]>>>"
        return m.group(0)

    replaced = citation_pattern.sub(_repl, writeup)

    parts: List[str] = []
    parts.append(r"\documentclass[11pt]{article}")
    parts.append(r"\usepackage[margin=1in]{geometry}")
    parts.append(r"\usepackage[hidelinks]{hyperref}")
    parts.append("")
    parts.append(r"\begin{document}")
    if topic:
        parts.append(r"\title{" + _latex_escape(topic) + "}")
        parts.append(r"\maketitle")
        parts.append("")

    # Convert markdown-like text to latex then resolve link tokens to hyperref
    latex_body = _markdown_to_latex_text(replaced)
    latex_body = re.sub(r"<<<LINK:([^:>]+):(.+?)>>>", r"\\hyperlink{\1}{\2}", latex_body)
    parts.append(latex_body)
    parts.append("")

    # Bibliography
    parts.append(r"\section*{Bibliography}")
    std_any = False
    for idx in cited_indices_in_text:
        doc = cited_docs[idx - 1]
        if getattr(doc, "document_type", None) != DocumentType.DOCUMENT_TYPE_DATATALK:
            url = getattr(doc, "url", "") or ""
            title = getattr(doc, "title", None)
            if url:
                std_any = True
                title_line = title or url
                parts.append(
                    r"\hypertarget{ref-" + str(idx) + r"}{}" +
                    "[" + str(idx) + "] " + _latex_escape(title_line) + r"\\ " +
                    "Retrieved from " + r"\url{" + url + r"}"
                )
                parts.append("")
    if not std_any:
        parts.append("(no non-datatalk citations)")

    # Tables
    if table_order:
        parts.append(r"\section*{Tables}")
        for tnum, cidx in table_order:
            doc = cited_docs[cidx - 1]
            title = getattr(doc, "title", None)
            header = f"Table {tnum} (from [{cidx}])"
            if title:
                header += f": {title}"
            parts.append(r"\hypertarget{table-" + str(tnum) + r"}{}" + "\n" +
                         r"\subsection*{" + _latex_escape(header) + "}")
            parts.append("")
            parts.append(_markdown_table_to_latex(table_by_index[cidx]))
            parts.append("")

    parts.append(r"\end{document}")
    return "\n".join(parts).rstrip() + "\n"


def render_output_latex_from_path(path: str) -> str:
    with open(path, "r") as f:
        data = json.load(f)
    return render_output_latex(data)


if __name__ == "__main__":
    # Optional CLI usage: python -m brainstorm.render_output [--latex] [json_path]
    args = sys.argv[1:]
    json_path = next((a for a in args if not a.startswith("-")), "output.json")
    use_latex = any(a == "--latex" for a in args)
    if use_latex:
        print(render_output_latex_from_path(json_path))
    else:
        print(render_output_markdown_from_path(json_path))
