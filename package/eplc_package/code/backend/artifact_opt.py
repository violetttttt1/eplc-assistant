"""Artifact retrieval optimizations for EPLC Assistant (template library).

What this module holds
----------------------
Techniques validated by the ablation in eval/ARTIFACT_OPT_TABLE.md
(experiments script: code/run_artifact_opt.py):

  E1  query-side artifact alias expansion  (expand_query_aliases)
  E3  doc-side metadata augmentation        (augment_chunk_text) — applied at INDEX time
  E4  hybrid BM25 + dense retrieval, RRF    (BM25, rrf_fuse)
  E5  cross-encoder rerank of top-N         (rerank_pairs)
      artifact alias table                  (code/backend/artifact_aliases.json)

Switches (config.yaml -> retrieval:):
  alias_expansion: true/false   # E1: append "[Relevant document: ...]" to the query
  hybrid_rrf: true/false        # E4: fuse BM25 and dense rankings with RRF (k=60)
  rerank_top_n: 0               # E5: 0 = off; N > 0 reranks top-N dense hits per library

E2 (LLM query rewriting) was attempted but no text-generation endpoint was
reachable from the build VM; documented in ARTIFACT_OPT_TABLE.md as future work.

Determinism: alias expansion and RRF are deterministic. Rerank scores come from
the HF Inference API cross-encoder and are deterministic for fixed inputs.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.request

ALIAS_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "artifact_aliases.json")
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANK_URL = ("https://router.huggingface.co/hf-inference/models/"
               + RERANK_MODEL + "/pipeline/text-classification")

_table = None


def load_alias_table() -> dict:
    """Load the artifact alias table (cached)."""
    global _table
    if _table is None:
        with open(ALIAS_TABLE_PATH, encoding="utf-8") as f:
            _table = json.load(f)
    return _table


# ---------- E1: query-side alias expansion ----------

def _norm(s: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()) + " "


def expand_query_aliases(query: str, table: dict | None = None) -> str:
    """Append canonical artifact names detected via the alias table.

    Returns the original query unchanged when no alias matches.
    """
    table = table or load_alias_table()
    ql = _norm(query)
    matched = []
    for tname, v in table["artifacts"].items():
        for a in v["aliases"]:
            if (" " + a + " ") in ql:
                matched.append(v["canonical"])
                break
    if not matched:
        return query
    canons = sorted(set(matched))
    return query + " [Relevant document: " + "; ".join(canons) + "]"


# ---------- E3: doc-side metadata augmentation (index time) ----------

def augment_chunk_text(section: dict, template_name: str,
                       base_text: str, table: dict | None = None) -> str:
    """Prepend 'Document: <canonical>. Section: <path>.' to chunk text.

    Use when building the index (see index.py chunk builders); the augmented
    corpus must be (re-)embedded for the effect to apply.
    """
    table = table or load_alias_table()
    v = table["artifacts"].get(template_name)
    canon = v["canonical"] if v else template_name
    head = f"Document: {canon}. Section: {section.get('section_path', '')}."
    return head + " " + base_text


# ---------- E4: BM25 + RRF hybrid ----------

def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    """Minimal BM25 over a fixed corpus (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tok(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - f + 0.5) / (f + 0.5))
                    for t, f in df.items()}

    def scores(self, query: str):
        import numpy as np
        qts = _tok(query)
        out = np.zeros(self.N)
        for i, d in enumerate(self.docs):
            dl = len(d)
            tf: dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in qts:
                if t not in self.idf:
                    continue
                f = tf.get(t, 0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf[t] * f * (self.k1 + 1) / denom
            out[i] = s
        return out


def rrf_fuse(rank_lists: list, k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over ordered index lists. Returns fused order."""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for r, ci in enumerate(ranks, 1):
            scores[ci] = scores.get(ci, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=lambda ci: -scores[ci])


# ---------- E5: cross-encoder rerank ----------

def rerank_pairs(pairs: list[tuple[str, str]], batch: int = 16,
                 timeout_s: int = 180) -> list[float]:
    """Score (query, doc) pairs with bge-reranker-v2-m3 via HF Inference API.

    Auth follows the same pattern as embeddings.py (HF_TOKEN env, else the
    workspace custom.huggingface surrogate). Higher = more relevant.
    """
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    import dynamic_credentials as dc

    out: list[float] = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        payload = {"inputs": [q + " [SEP] " + d[:4000] for q, d in chunk]}
        req = urllib.request.Request(
            _RERANK_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        # same auth helper as embeddings.HFEmbedder._apply_auth
        token = os.environ.get("HF_TOKEN")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        else:
            dc.add_surrogate_to_request(
                req, "custom.huggingface",
                allowed_hosts=("router.huggingface.co", "api-inference.huggingface.co"))
        resp = urllib.request.urlopen(req, timeout=timeout_s)
        res = json.loads(dc.read_response_body(resp).decode("utf-8"))
        for r in res:
            items = r if isinstance(r, list) else [r]
            out.append(max(x["score"] for x in items))
    return out
