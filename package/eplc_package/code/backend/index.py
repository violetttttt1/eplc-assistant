"""Dual-library indexing for EPLC Assistant.

Library A (templates): 50 CDC UP docx templates (data_parsed/) +
                         10 xls templates (data_parsed_xls/).
Library B (hhs):         16 HHS docs (data_parsed_hhs/), framework + policies.

Chunk rule (locked in by the retrieval AB test): section-level, split only
when a section exceeds 512 tokens. Token counts are approximated the same way
as the AB harness (len(text)/4), so behaviour is comparable to the published
numbers (Recall@5=0.653 clean set).

Every chunk carries: chunk_id, text, and metadata with at least
doc_id, title, library, doc_type, section_path. Citation fields (url /
snapshot_url) are included for HHS docs.

Embedding cache: index_cache/<lib>/chunks.json + embeddings.npy + manifest.json.
Rebuilds are incremental: if the corpus fingerprint (sha256 over the source
files' bytes) matches the manifest, the cache is reused without touching the API.

Determinism: chunk order follows file order (sorted filenames) and JSON order;
identical inputs -> identical chunk_ids, identical numpy arrays.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

from .common import package_root, load_config, sha256_files
from .embeddings import HFEmbedder, EmbeddingError

# ---------- token approximation (same convention as the AB harness) ----------

def approx_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def split_by_tokens(text: str, max_tok: int, overlap_tok: int = 0) -> list[str]:
    """Sentence-aware greedy packing, copied from code/chunk_ab_test.py."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, cur_tok = [], [], 0

    def toks(s: str) -> int:
        return max(1, round(len(s) / 4))

    for s in sents:
        st = toks(s)
        if cur_tok + st > max_tok and cur:
            chunks.append(" ".join(cur))
            if overlap_tok > 0:
                tail, ttok = [], 0
                for x in reversed(cur):
                    ttok += toks(x)
                    tail.insert(0, x)
                    if ttok >= overlap_tok:
                        break
                cur, cur_tok = tail[:], sum(toks(x) for x in tail)
            else:
                cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += st
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


def section_text_docx(sec: dict) -> str:
    """Retrieval text for a docx-template section."""
    parts = []
    if sec.get("instruction"):
        parts.append(sec["instruction"])
    if sec.get("boilerplate"):
        parts.append(sec["boilerplate"])
    fields = sec.get("fields") or []
    if fields:
        parts.append("Fields to complete: " + ", ".join(fields))
    return " ".join(parts).strip()


# ---------- chunk builders ----------

def _chunk_id(lib: str, source: str, seq: int) -> str:
    return f"{lib}:{source}:{seq:05d}"


def chunks_from_docx_templates(lib: str, max_tok: int, preamble_as_chunk: bool,
                               metadata_augment: bool = False) -> list[dict]:
    root = package_root()
    files = sorted(glob.glob(os.path.join(root, "data_parsed", "*.json")))
    if not files:
        raise FileNotFoundError("no parsed docx templates found in data_parsed/")
    if metadata_augment:
        from .artifact_opt import augment_chunk_text, load_alias_table
        alias_table = load_alias_table()
    chunks = []
    for f in files:
        doc = json.load(open(f))
        tname = doc.get("template_name") or os.path.basename(f)
        source = os.path.basename(f)
        seq = 0
        for sec in doc.get("sections", []):
            heading = sec.get("section_heading", "")
            if heading == "(preamble)" and not preamble_as_chunk:
                continue
            txt = section_text_docx(sec)
            if metadata_augment:
                # E3 winner (ablation: overall R@5 0.637 -> 0.976, artifact 0.588 -> 0.941):
                # prepend "Document: <canonical>. Section: <path>." before chunking.
                txt = augment_chunk_text(sec, tname, txt, alias_table)
            if not txt:
                continue
            meta = {
                "doc_id": tname,
                "title": tname,
                "library": lib,
                "doc_type": doc.get("doc_type", "template"),
                "artifact": tname,
                "section_path": sec.get("section_path", heading),
                "section_heading": heading,
                "source_file": source,
            }
            if approx_tokens(txt) <= max_tok:
                chunks.append({**meta, "chunk_id": _chunk_id(lib, source, seq), "text": txt})
                seq += 1
            else:
                for i, part in enumerate(split_by_tokens(txt, max_tok)):
                    m = {**meta, "chunk_id": _chunk_id(lib, source, seq),
                         "text": part, "part_index": i + 1}
                    chunks.append(m)
                    seq += 1
    return chunks


def chunks_from_xls_templates(lib: str, max_tok: int) -> list[dict]:
    root = package_root()
    xdir = os.path.join(root, "data_parsed_xls")
    files = sorted(glob.glob(os.path.join(xdir, "*.json")))
    files = [f for f in files if os.path.basename(f) != "_summary.json"]
    if not files:
        raise FileNotFoundError("no parsed xls templates found in data_parsed_xls/")
    chunks = []
    for f in files:
        doc = json.load(open(f))
        source = os.path.basename(f)
        kind = doc.get("kind", "log")
        title = doc.get("title") or doc.get("template", "")
        seq = 0

        def add(text: str, **meta):
            nonlocal seq
            m = {"doc_id": title, "title": title, "library": lib,
                 "doc_type": "xls_" + kind, "artifact": title,
                 "source_file": source, **meta}
            m["chunk_id"] = _chunk_id(lib, source, seq)
            m["text"] = text
            chunks.append(m)
            seq += 1

        if kind == "log":
            cols = doc.get("columns", [])
            hdr = doc.get("header_fields", [])
            hdr_txt = "; ".join(
                f"{h.get('name')}{' (required)' if h.get('required') else ''}" for h in hdr
            )
            col_names = ", ".join(c.get("name", "") for c in cols)
            add(
                f"{title} (structured {kind} template). Header fields: {hdr_txt}. "
                f"Columns: {col_names}.",
                section_path=f"{title} > Overview",
            )
            examples = doc.get("example_rows", [])
            for c in cols:
                cname = c.get("name", "")
                lines = [
                    f"Column '{cname}' ({c.get('col_letter', '?')}) in {title}.",
                ]
                if c.get("instruction"):
                    lines.append("Instructions: " + c["instruction"])
                vals = c.get("allowed_values") or []
                if vals:
                    lines.append("Allowed dropdown values: " + ", ".join(str(v) for v in vals))
                ex_vals = sorted({str(r.get(cname, "")).strip() for r in examples
                                  if str(r.get(cname, "")).strip()})
                if ex_vals:
                    lines.append("Example values: " + "; ".join(ex_vals[:5]))
                add(" ".join(lines), section_path=f"{title} > Column {cname}")
        elif kind == "schedule":
            tasks = doc.get("tasks", [])
            add(
                f"{title} (project schedule template): {len(tasks)} WBS tasks with "
                "duration, baseline/expected/actual start and finish dates, and work estimates.",
                section_path=f"{title} > Overview",
            )
            for t in tasks:
                bits = [f"WBS {t.get('wbs_id')}: {t.get('task_name')}."]
                if t.get("duration") is not None:
                    bits.append(f"Duration: {t['duration']} {t.get('duration_unit', 'days')}.")
                if t.get("expected_start"):
                    bits.append(f"Expected start: {t['expected_start']}.")
                if t.get("expected_finish"):
                    bits.append(f"Expected finish: {t['expected_finish']}.")
                if t.get("baseline_work") is not None:
                    bits.append(f"Baseline work: {t['baseline_work']} {t.get('baseline_work_unit', 'hrs')}.")
                add(" ".join(bits), section_path=f"{title} > WBS {t.get('wbs_id')}",
                    wbs_id=str(t.get("wbs_id")))
        elif kind == "stage_gate":
            phases = doc.get("phases", [])
            add(
                f"{title} (stage-gate review assessment): {len(phases)} phases, "
                "each with review-area questionnaires and a red/yellow/green dashboard.",
                section_path=f"{title} > Overview",
            )
            for p in phases:
                phase = p.get("phase", "")
                for area in p.get("areas", []):
                    aname = area.get("area", "")
                    qs = [q for q in area.get("questions", []) if not q.get("is_placeholder")]
                    if not qs:
                        continue
                    qtxt = " ".join(
                        f"[{q.get('number')}] {q.get('question')}" for q in qs
                    )
                    text = (
                        f"Stage-gate review questions, phase '{phase}', review area '{aname}': "
                        f"{qtxt}"
                    )
                    # A whole phase-area questionnaire can exceed 512 tokens; split it.
                    if approx_tokens(text) > max_tok:
                        for i, part in enumerate(split_by_tokens(text, max_tok)):
                            add(part, section_path=f"{title} > {phase} > {aname} (part {i + 1})",
                                phase=phase, review_area=aname, part_index=i + 1)
                    else:
                        add(text, section_path=f"{title} > {phase} > {aname}",
                            phase=phase, review_area=aname)
        else:
            add(f"{title} (kind={kind}).", section_path=f"{title} > Overview")
    return chunks


def chunks_from_hhs(lib: str, max_tok: int) -> list[dict]:
    root = package_root()
    hdir = os.path.join(root, "data_parsed_hhs")
    files = sorted(glob.glob(os.path.join(hdir, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    if not files:
        raise FileNotFoundError("no parsed HHS docs found in data_parsed_hhs/")
    chunks = []
    for f in files:
        doc = json.load(open(f))
        source = os.path.basename(f)
        doc_id = doc.get("doc_id", source)
        title = doc.get("title", source)
        seq = 0
        for sec in doc.get("sections", []):
            body = sec.get("body", "") or ""
            heading = sec.get("heading", "")
            text = f"{heading}\n{body}".strip() if heading else body
            if not text:
                continue
            meta = {
                "doc_id": doc_id,
                "title": title,
                "library": lib,
                "doc_type": doc.get("doc_type", "policy"),
                "covers": doc.get("covers", []),
                "section_path": sec.get("section_path", heading),
                "section_heading": heading,
                "url": doc.get("url"),
                "snapshot_url": doc.get("snapshot_url"),
                "source_file": source,
            }
            if approx_tokens(text) > max_tok:
                for i, part in enumerate(split_by_tokens(text, max_tok)):
                    m = {**meta, "chunk_id": _chunk_id(lib, source, seq),
                         "text": part, "part_index": i + 1}
                    chunks.append(m)
                    seq += 1
            else:
                chunks.append({**meta, "chunk_id": _chunk_id(lib, source, seq), "text": text})
                seq += 1
    return chunks


# ---------- index build / load ----------

def _cache_paths(lib: str) -> tuple[str, str, str]:
    root = package_root()
    d = os.path.join(root, "code", "backend", "index_cache", lib)
    return (os.path.join(d, "chunks.json"),
            os.path.join(d, "embeddings.npy"),
            os.path.join(d, "manifest.json"))


def corpus_files(lib: str) -> list[str]:
    root = package_root()
    if lib == "A":
        a = sorted(glob.glob(os.path.join(root, "data_parsed", "*.json")))
        b = sorted(glob.glob(os.path.join(root, "data_parsed_xls", "*.json")))
        return a + b
    if lib == "B":
        return sorted(
            f for f in glob.glob(os.path.join(root, "data_parsed_hhs", "*.json"))
            if not os.path.basename(f).startswith("_")
        )
    raise ValueError(f"unknown library {lib!r}; expected 'A' or 'B'")


def build_library(lib: str, force: bool = False, verbose: bool = False) -> dict:
    """Build (or reuse cached) the index for library lib. Returns a summary dict."""
    cfg = load_config()
    max_tok = cfg["chunking"]["max_tokens"]
    cpath, epath, mpath = _cache_paths(lib)
    files = corpus_files(lib)
    if not files:
        raise FileNotFoundError(f"no corpus files for library {lib}")

    fp = sha256_files(files)
    meta_aug = cfg.get("indexing", {}).get("metadata_augmentation", False)
    if not force and os.path.exists(mpath):
        manifest = json.load(open(mpath))
        if (manifest.get("corpus_fingerprint") == fp
                and manifest.get("metadata_augmentation", False) == meta_aug
                and os.path.exists(epath)):
            chunks = json.load(open(cpath))
            emb = np.load(epath)
            if len(chunks) == emb.shape[0]:
                if verbose:
                    print(f"library {lib}: cache hit ({len(chunks)} chunks)")
                return {"library": lib, "chunks": len(chunks), "cached": True,
                        "cache_dir": os.path.dirname(cpath)}

    # Build chunks.
    if lib == "A":
        chunks = chunks_from_docx_templates(
            lib, max_tok, cfg["chunking"]["preamble_as_chunk"],
            metadata_augment=meta_aug)
        chunks += chunks_from_xls_templates(lib, max_tok)
    else:
        chunks = chunks_from_hhs(lib, max_tok)
    if not chunks:
        raise EmbeddingError(f"no chunks produced for library {lib}")

    texts = [c["text"] for c in chunks]
    if verbose:
        print(f"library {lib}: {len(chunks)} chunks -> embedding", flush=True)
    emb_cfg = cfg["embedding"]
    embedder = HFEmbedder(
        model=emb_cfg["model"],
        query_prefix=emb_cfg["query_prefix"],
        batch_size=emb_cfg["batch_size"],
        max_retries=emb_cfg["max_retries"],
        timeout_s=emb_cfg["timeout_s"],
        verbose=verbose,
    )
    emb = embedder.embed(texts, kind="doc")
    if emb.shape != (len(chunks), 1024):
        raise EmbeddingError(f"unexpected embedding shape {emb.shape}")

    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    json.dump(chunks, open(cpath, "w"), ensure_ascii=False, indent=1)
    np.save(epath, emb)
    json.dump({
        "library": lib,
        "num_chunks": len(chunks),
        "corpus_fingerprint": fp,
        "model": emb_cfg["model"],
        "chunk_strategy": cfg["chunking"]["strategy"],
        "max_tokens": max_tok,
        "metadata_augmentation": meta_aug,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [os.path.basename(f) for f in files],
    }, open(mpath, "w"), ensure_ascii=False, indent=1)
    if verbose:
        print(f"library {lib}: indexed {len(chunks)} chunks")
    return {"library": lib, "chunks": len(chunks), "cached": False,
            "cache_dir": os.path.dirname(cpath)}


def load_index(lib: str) -> tuple[list[dict], np.ndarray]:
    cpath, epath, mpath = _cache_paths(lib)
    if not (os.path.exists(cpath) and os.path.exists(epath)):
        raise FileNotFoundError(
            f"index for library {lib} not built yet. Run: "
            f"python -m backend.cli build --lib {lib}"
        )
    return json.load(open(cpath)), np.load(epath)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build EPLC retrieval indexes")
    ap.add_argument("--lib", choices=["A", "B", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="rebuild even if cache is fresh")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    libs = ["A", "B"] if args.lib == "all" else [args.lib]
    for lib in libs:
        s = build_library(lib, force=args.force, verbose=args.verbose)
        print(f"library {lib}: {s['chunks']} chunks ({'cache' if s['cached'] else 'built'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
