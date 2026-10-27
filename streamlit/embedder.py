"""OpenRouter embeddings for the EPLC Assistant Streamlit demo.

Replaces the HuggingFace BGE embedder (credits exhausted) with
``openai/text-embedding-3-small`` at 1024 dimensions, so the existing
(n, 1024) index contract holds and previously built chunk lists can be
re-embedded without changing any retrieval code.

Interface mirrors ``backend.embeddings.HFEmbedder.embed(texts, kind)``:
``kind`` is accepted for compatibility but no instruction prefix is
applied -- that prefix is BGE-specific and this model does not use one.
All vectors are L2-normalized, matching the retriever's assumption that
index vectors are normalized (scores are plain dot products).

Auth: the API key is passed in (never hard-coded, never logged). The
Streamlit app resolves it from ``st.secrets`` with an environment-variable
fallback; local runs use the Secure Vault surrogate via ``run_local.py``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import numpy as np

API_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_MODEL = "openai/text-embedding-3-small"
DEFAULT_DIMENSIONS = 1024


class EmbeddingError(RuntimeError):
    """Raised when embedding fails after all retries."""


def _redact(msg: str) -> str:
    return msg.replace("Bearer ", "Bearer <redacted>")


class OpenRouterEmbedder:
    """Batched OpenRouter embeddings with retry. Returns L2-normalized vectors."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = 64,
        max_retries: int = 6,
        timeout_s: int = 180,
    ) -> None:
        if not api_key:
            raise EmbeddingError("OpenRouterEmbedder needs an API key")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        # Cumulative usage across all embed() calls (for cost reporting).
        self.total_prompt_tokens = 0
        self.total_cost_usd = 0.0
        self.last_usage: dict = {}

    def _post(self, texts: list[str]) -> dict:
        payload = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        }
        data = json.dumps(payload).encode("utf-8")
        last_err = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                API_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.api_key,
                    "HTTP-Referer": "eplc-assistant",
                    "X-Title": "EPLC Assistant",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:300]
                last_err = f"HTTP {e.code}: {body}"
                if e.code in (429, 500, 502, 503):
                    time.sleep(min(60, 5 * (attempt + 1)))
                    continue
                raise EmbeddingError(_redact(last_err)) from None
            except Exception as e:  # network blip
                last_err = repr(e)
                time.sleep(min(60, 5 * (attempt + 1)))
        raise EmbeddingError(_redact(f"failed after {self.max_retries} retries: {last_err}"))

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        body = self._post(texts)
        usage = body.get("usage") or {}
        self.total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        cost = usage.get("cost")
        if cost is not None:
            self.total_cost_usd += float(cost)
        self.last_usage = usage
        items = sorted(body.get("data") or [], key=lambda d: d.get("index", 0))
        if len(items) != len(texts):
            raise EmbeddingError(
                f"expected {len(texts)} embeddings, got {len(items)}"
            )
        vecs = []
        for item in items:
            vec = np.array(item["embedding"], dtype=np.float64)
            if vec.shape != (self.dimensions,):
                raise EmbeddingError(
                    f"unexpected embedding dim {vec.shape}, expected ({self.dimensions},)"
                )
            n = np.linalg.norm(vec)
            vecs.append(vec / (n + 1e-12))
        return np.stack(vecs)

    def embed(self, texts: list[str], kind: str = "doc") -> np.ndarray:
        """Embed texts. ``kind`` is accepted for HFEmbedder compatibility
        ('query'/'doc') but no prefix is applied for this model."""
        if kind not in ("doc", "query"):
            raise ValueError(f"kind must be 'doc' or 'query', got {kind!r}")
        if not texts:
            raise EmbeddingError("embed() called with an empty text list")
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            vecs.append(self._embed_batch(texts[i : i + self.batch_size]))
        return np.vstack(vecs)
