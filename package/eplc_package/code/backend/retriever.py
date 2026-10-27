"""Dual-library retrieval for EPLC Assistant.

Retrieval contract:
  - Query is embedded with the BGE query prefix; documents were embedded
    without it (embedding consistency, design doc 6.1).
  - Each library is searched independently with its own top-k, then results
    are merged, deduplicated, and truncated to merge_top_k.
  - Metadata filters: artifact (template title), doc_type, phase.
    Phase matches stage-gate phase metadata or an HHS `covers` entry.
  - Deterministic: ties broken by chunk_id; identical inputs -> identical hits.
  - Every hit carries a citation: title + section_path + url/snapshot_url.
"""

from __future__ import annotations

import sys

import numpy as np

from .common import load_config
from .embeddings import HFEmbedder
from .index import load_index


def format_citation(meta: dict) -> str:
    title = meta.get("title") or meta.get("doc_id") or "?"
    path = meta.get("section_path") or ""
    url = meta.get("url") or meta.get("snapshot_url") or ""
    cite = f"{title} — {path}".strip(" —")
    if url:
        cite += f" <{url}>"
    return cite


def _matches(meta: dict, filters: dict) -> bool:
    for key, want in filters.items():
        if want in (None, "", []):
            continue
        if key == "artifact":
            got = str(meta.get("artifact") or "")
            if want.lower() not in got.lower():
                return False
        elif key == "doc_type":
            got = str(meta.get("doc_type") or "")
            if got.lower() != str(want).lower():
                return False
        elif key == "phase":
            hit = False
            ph = str(meta.get("phase") or "")
            if want.lower() in ph.lower():
                hit = True
            for c in meta.get("covers") or []:
                if want.lower() in str(c).lower():
                    hit = True
                    break
            if want.lower() in str(meta.get("section_path") or "").lower():
                hit = True
            if not hit:
                return False
        else:
            if str(meta.get(key) or "").lower() != str(want).lower():
                return False
    return True


class DualRetriever:
    def __init__(self, verbose: bool = False) -> None:
        cfg = load_config()
        self.cfg = cfg
        rcfg = cfg["retrieval"]
        self.top_k_lib = rcfg["top_k_per_library"]
        self.merge_k = rcfg["merge_top_k"]
        ecfg = cfg["embedding"]
        self.embedder = HFEmbedder(
            model=ecfg["model"],
            query_prefix=ecfg["query_prefix"],
            batch_size=ecfg["batch_size"],
            max_retries=ecfg["max_retries"],
            timeout_s=ecfg["timeout_s"],
            verbose=verbose,
        )
        self.verbose = verbose
        self._indexes: dict[str, tuple[list[dict], np.ndarray]] = {}

    def _get_index(self, lib: str):
        if lib not in self._indexes:
            self._indexes[lib] = load_index(lib)
        return self._indexes[lib]

    def search(
        self,
        query: str,
        libraries: tuple[str, ...] = ("A", "B"),
        top_k_per_library: int | None = None,
        merge_top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search both libraries; return hits with text, metadata, score, citation."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        filters = filters or {}
        k_lib = top_k_per_library or self.top_k_lib
        k_merge = merge_top_k or self.merge_k
        rcfg = self.cfg["retrieval"]
        use_alias = rcfg.get("alias_expansion", False)
        use_rrf = rcfg.get("hybrid_rrf", False)
        rerank_n = int(rcfg.get("rerank_top_n", 0) or 0)

        if use_alias:
            # E1 (ablation: no gain on BGE, default off) — expand via alias table.
            from .artifact_opt import expand_query_aliases
            query = expand_query_aliases(query)

        qvec = self.embedder.embed([query], kind="query")[0]
        pooled: list[dict] = []
        for lib in libraries:
            chunks, emb = self._get_index(lib)
            idxs = [i for i, c in enumerate(chunks) if _matches(c, filters)]
            if not idxs:
                continue
            sub = emb[idxs]  # already L2-normalized
            scores = sub @ qvec  # cosine similarity
            dense_order = np.argsort(-scores, kind="stable")
            if use_rrf:
                # E4 (ablation: hurt overall, default off) — RRF fuse BM25 + dense.
                from .artifact_opt import BM25, rrf_fuse
                bm25 = BM25([chunks[i]["text"] for i in idxs])
                bm25_order = np.argsort(-bm25.scores(query), kind="stable")
                order_pos = rrf_fuse([dense_order, bm25_order])[:k_lib]
            elif rerank_n > 0:
                # E5 (ablation: hurt, default off) — cross-encoder rerank of top-N.
                from .artifact_opt import rerank_pairs
                cand = dense_order[:rerank_n]
                pairs = [(query, chunks[idxs[int(oi)]]["text"]) for oi in cand]
                rs = rerank_pairs(pairs)
                reranked = sorted(range(len(cand)), key=lambda j: -rs[j])
                order_pos = [int(cand[j]) for j in reranked[:k_lib]]
            else:
                order_pos = dense_order[:k_lib]
            for rank, oi in enumerate(order_pos):
                gi = idxs[int(oi)]
                pooled.append({
                    "chunk_id": chunks[gi]["chunk_id"],
                    "text": chunks[gi]["text"],
                    "metadata": {k: v for k, v in chunks[gi].items()
                                 if k not in ("chunk_id", "text")},
                    "score": float(scores[int(oi)]),
                    "library": lib,
                    "rank_in_library": int(rank),
                })

        # Deterministic merge: (-score, chunk_id); dedupe by chunk_id.
        pooled.sort(key=lambda h: (-h["score"], h["chunk_id"]))
        seen, merged = set(), []
        for h in pooled:
            if h["chunk_id"] in seen:
                continue
            seen.add(h["chunk_id"])
            h["citation"] = format_citation(h["metadata"])
            merged.append(h)
            if len(merged) >= k_merge:
                break
        return merged


def main(argv: list[str] | None = None) -> int:
    import argparse, json

    ap = argparse.ArgumentParser(description="EPLC policy Q&A over the dual index")
    ap.add_argument("query", help="question to answer")
    ap.add_argument("--lib", default="A,B", help="comma-separated libraries, e.g. A,B")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--doc-type", default=None)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    libs = tuple(x.strip() for x in args.lib.split(",") if x.strip())
    filters = {"artifact": args.artifact, "doc_type": args.doc_type, "phase": args.phase}
    retr = DualRetriever(verbose=True)
    hits = retr.search(args.query, libraries=libs, filters=filters)
    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=1))
    else:
        for h in hits:
            print(f"[{h['library']}] score={h['score']:.3f} {h['citation']}")
            print(h["text"][:400].replace("\n", " "))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
