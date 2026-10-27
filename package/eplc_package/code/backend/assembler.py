"""One-click assembly: generated sections -> complete .docx in official format.

The exported document mirrors the official CDC UP template conventions:
  - section headings at the template's heading levels,
  - template guidance ([bracketed instructions]) rendered in italic gray
    (the official convention is blue italic; color is approximated for
    portability across Word readers),
  - <angle-bracket fields> rendered bold, unresolved ones flagged with
    [MISSING: ...],
  - generated draft text as normal body paragraphs.

assemble(template_name, drafts, project_info, out_path) walks the template's
sections IN TEMPLATE ORDER and fills each with the corresponding draft
(drafts keyed by section_heading). Missing drafts are rendered as
[MISSING: section not generated] so nothing is silently dropped.
"""

from __future__ import annotations

import glob
import json
import os
import sys

from .common import package_root, load_config
from .generator import _field_key


def load_template(template_name: str) -> dict:
    for f in sorted(glob.glob(os.path.join(package_root(), "data_parsed", "*.json"))):
        doc = json.load(open(f))
        if doc.get("template_name") == template_name or template_name in f:
            return doc
    raise ValueError(f"template {template_name!r} not found in data_parsed/")


def fill_fields(text: str, project_info: dict) -> str:
    """Replace known <fields> in body text; unknown ones stay as [MISSING: ...]."""
    import re

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        key = _field_key(raw)
        val = project_info.get(key) or project_info.get(raw)
        return str(val) if val else f"[MISSING: {raw}]"

    return re.sub(r"<[^<>\n]{1,60}>", repl, text)


def assemble(
    template_name: str,
    drafts: dict[str, dict],
    project_info: dict | None = None,
    out_path: str | None = None,
    include_instructions: bool = True,
) -> str:
    """Assemble drafts into a .docx. Returns the absolute output path."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    project_info = project_info or {}
    doc_tpl = load_template(template_name)
    sections = doc_tpl.get("sections", [])

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Title page
    title = document.add_heading(doc_tpl.get("template_name", template_name), level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if project_info.get("project_name"):
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(project_info["project_name"])
        run.bold = True
        run.font.size = Pt(16)

    instr_color = RGBColor(0x59, 0x59, 0x59)
    missing_color = RGBColor(0xC0, 0x39, 0x2B)

    for sec in sections:
        heading = sec.get("section_heading", "")
        level = max(1, min(3, int(sec.get("section_level") or 1)))
        document.add_heading(heading, level=level)

        instruction = (sec.get("instruction") or "").strip()
        if instruction and include_instructions:
            p = document.add_paragraph()
            run = p.add_run("[Instruction] " + " ".join(instruction.split()))
            run.italic = True
            run.font.color.rgb = instr_color
            run.font.size = Pt(9)

        draft = drafts.get(heading)
        if draft is None:
            p = document.add_paragraph()
            run = p.add_run("[MISSING: section not generated — "
                            "run the generator for this section first]")
            run.bold = True
            run.font.color.rgb = missing_color
        else:
            text = fill_fields(draft["draft"], project_info)
            for para in text.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                p = document.add_paragraph()
                # bold/color the [MISSING: ...] markers inside the paragraph
                _render_with_missing(p, para, missing_color)

        # section-level fields list as a quick checklist
        fields = sec.get("fields") or []
        if fields:
            p = document.add_paragraph()
            run = p.add_run("Fields: " + ", ".join(fields))
            run.italic = True
            run.font.size = Pt(9)
            run.font.color.rgb = instr_color

    if out_path is None:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in template_name)
        out_path = os.path.join(package_root(), "code", "backend", f"{safe}_draft.docx")
    document.save(out_path)
    return os.path.abspath(out_path)


def _render_with_missing(paragraph, text: str, missing_color) -> None:
    """Add runs to a paragraph, highlighting [MISSING: ...] markers in red bold."""
    import re

    pos = 0
    for m in re.finditer(r"\[MISSING:[^\]]*\]", text):
        if m.start() > pos:
            paragraph.add_run(text[pos : m.start()])
        run = paragraph.add_run(m.group(0))
        run.bold = True
        run.font.color.rgb = missing_color
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Assemble generated sections into a .docx")
    ap.add_argument("--template", required=True, help="template_name, e.g. CDC_UP_Business_Case_LITE_Template")
    ap.add_argument("--drafts", required=True, help="JSON file: {section_heading: draft-dict}")
    ap.add_argument("--project", default="{}", help="JSON dict of project facts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    drafts = json.load(open(args.drafts))
    out = assemble(args.template, drafts, json.loads(args.project), args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
