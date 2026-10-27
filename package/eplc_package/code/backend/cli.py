"""Unified backend CLI for EPLC Assistant.

Full chain:
  1. python -m backend.cli build            # build library A + B indexes (cached, incremental)
  2. python -m backend.cli ask "..."         # policy Q&A with citations
  3. python -m backend.cli section ...       # draft one template section
  4. python -m backend.cli assemble ...      # stitch drafts -> .docx

Run with cwd anywhere; use:  cd <package root>/code && python -m backend.cli ...
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="backend.cli", description="EPLC Assistant backend")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build retrieval indexes")
    b.add_argument("--lib", choices=["A", "B", "all"], default="all")
    b.add_argument("--force", action="store_true")

    a = sub.add_parser("ask", help="policy Q&A")
    a.add_argument("question")
    a.add_argument("--lib", default="A,B")
    a.add_argument("--artifact", default=None)
    a.add_argument("--doc-type", default=None)
    a.add_argument("--phase", default=None)
    a.add_argument("--json", action="store_true", dest="as_json")

    s = sub.add_parser("section", help="draft one template section")
    s.add_argument("--template", required=True)
    s.add_argument("--heading", required=True)
    s.add_argument("--project", default="{}")
    s.add_argument("--out", default=None, help="write draft JSON to file")

    asm = sub.add_parser("assemble", help="assemble drafts into .docx")
    asm.add_argument("--template", required=True)
    asm.add_argument("--drafts", required=True)
    asm.add_argument("--project", default="{}")
    asm.add_argument("--out", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "build":
        from .index import build_library

        libs = ["A", "B"] if args.lib == "all" else [args.lib]
        for lib in libs:
            r = build_library(lib, force=args.force, verbose=True)
            print(f"library {lib}: {r['chunks']} chunks ({'cache' if r['cached'] else 'built'})")
        return 0

    if args.cmd == "ask":
        from .generator import SectionGenerator

        gen = SectionGenerator(verbose=True)
        out = gen.answer_policy_question(args.question)
        hits = out.pop("hits")
        print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])
        for h in hits[:8]:
            print(f"  [{h['library']}] {h['score']:.3f} {h['citation']}")
        return 0

    if args.cmd == "section":
        from .generator import SectionGenerator

        gen = SectionGenerator(verbose=True)
        out = gen.generate_section(args.template, args.heading,
                                   project_info=json.loads(args.project))
        if args.out:
            json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
            print(f"draft written to {args.out}")
        else:
            print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "assemble":
        from .assembler import assemble

        drafts = json.load(open(args.drafts))
        out = assemble(args.template, drafts, json.loads(args.project), args.out)
        print(f"wrote {out}")
        return 0

    ap.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
