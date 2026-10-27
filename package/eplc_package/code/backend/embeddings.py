"""BGE-large-en-v1.5 embeddings via the HuggingFace Inference API.

Deterministic contract: identical input texts -> identical output vectors
(the API is deterministic for this model; no randomness anywhere here).

Auth order:
  1. HF_TOKEN environment variable (Bearer header) — used by Violet when she
     runs this locally with her own token.
  2. The workspace `custom.huggingface` credential via the surrogate helper
     (this VM's flow, same one the retrieval AB tests used).
The token is NEVER printed, logged, or persisted. If the raw value ever
appears in a traceback, we redact Authorization headers before printing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

ALLOWED_HOSTS = ("router.huggingface.co", "api-inference.huggingface.co")
MODEL_DIM = 1024  # BAAI/bge-large-en-v1.5


class EmbeddingError(RuntimeError):
    """Raised when embedding fails after all retries/endpoints."""


def _endpoint_urls(model: str) -> list[str]:
    return [
        f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction",
        f"https://api-inference.huggingface.co/models/{model}",
    ]


def _load_surrogate_helper():
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    import dynamic_credentials as dc  # type: ignore

    return dc


def _redact(msg: str) -> str:
    # Cheap insurance: never let an Authorization header value leak into logs.
    return msg.replace("Bearer ", "Bearer <redacted>")


class HFEmbedder:
    """Batched HF Inference embeddings with retry + failover endpoints."""

    def __init__(
        self,
        model: str,
        query_prefix: str,
        batch_size: int = 16,
        max_retries: int = 6,
        timeout_s: int = 180,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.verbose = verbose

    # ---------- auth ----------

    def _apply_auth(self, req: urllib.request.Request) -> None:
        token = os.environ.get("HF_TOKEN")
        if token:
            req.add_header("Authorization", "Bearer " + token)
            return
        # Fall back to the workspace surrogate credential (no raw token in code).
        try:
            dc = _load_surrogate_helper()
        except Exception as e:
            raise EmbeddingError(
                "No HF_TOKEN in environment and the surrogate credential helper "
                f"is unavailable ({e}). Set HF_TOKEN or configure custom.huggingface."
            ) from None
        dc.add_surrogate_to_request(req, "custom.huggingface", allowed_hosts=ALLOWED_HOSTS)

    # ---------- core ----------

    def _post(self, url: str, payload: dict) -> list:
        data = json.dumps(payload).encode("utf-8")
        last_err = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            self._apply_auth(req)
            try:
                resp = urllib.request.urlopen(req, timeout=self.timeout_s)
                dc = _load_surrogate_helper()
                raw = dc.read_response_body(resp).decode("utf-8")
                out = json.loads(raw)
                if isinstance(out, dict) and "error" in out:
                    raise EmbeddingError(f"API error: {out['error']}")
                return out
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")[:300]
                last_err = f"HTTP {e.code}: {body}"
                if e.code in (429, 500, 502, 503):
                    wait = min(60, 5 * (attempt + 1))
                    if self.verbose:
                        print(f"    retry in {wait}s ({_redact(last_err)[:90]})", flush=True)
                    time.sleep(wait)
                    continue
                raise EmbeddingError(_redact(last_err)) from None
            except EmbeddingError:
                raise
            except Exception as e:  # network blip
                last_err = repr(e)
                time.sleep(min(60, 5 * (attempt + 1)))
        raise EmbeddingError(_redact(f"failed after {self.max_retries} retries: {last_err}"))

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed one API-sized list of texts. Returns (n, 1024) row-normalized."""
        err = None
        for url in _endpoint_urls(self.model):
            try:
                out = self._post(url, {"inputs": texts, "options": {"wait_for_model": True}})
                vecs = []
                for item in out:
                    arr = np.array(item, dtype=np.float64)
                    vec = arr[0] if arr.ndim == 2 else arr  # CLS-pool if token-level
                    vecs.append(vec / (np.linalg.norm(vec) + 1e-12))
                return np.stack(vecs)
            except Exception as e:
                err = e
                if self.verbose:
                    print(f"  endpoint {url.split('/')[2]} failed: {e}; trying next", flush=True)
        raise EmbeddingError(f"all endpoints failed: {err}")

    def embed(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        """Embed texts; kind='query' applies the BGE instruction prefix, 'doc' does not.

        Raises:
            EmbeddingError: if `texts` is empty or all endpoints fail.
        """
        if kind not in ("doc", "query"):
            raise ValueError(f"kind must be 'doc' or 'query', got {kind!r}")
        if not texts:
            raise EmbeddingError("embed() called with an empty text list")
        prepared = [
            (self.query_prefix + t if kind == "query" else t) for t in texts
        ]
        vecs = []
        total = len(prepared)
        for i in range(0, total, self.batch_size):
            batch = prepared[i : i + self.batch_size]
            vecs.append(self._embed_batch(batch))
            if self.verbose:
                print(f"    embedded {min(i + self.batch_size, total)}/{total}", flush=True)
        return np.vstack(vecs)
