"""Non-UI wiring for the EPLC Assistant Streamlit demo.

Imports the package's own retrieval/generation code (no copies) and swaps
in the OpenRouter embedder + OpenRouter chat provider:
  - retrieval: DualRetriever with .embedder replaced by OpenRouterEmbedder
    and _indexes pre-populated from the Streamlit re-embedded cache
    (chunks read from the package cache, embeddings from streamlit/index_cache).
  - generation: SectionGenerator with a stdlib-urllib OpenRouter chat
    provider (qwen/qwen-plus, same model as the FastAPI demo).

Libraries A+B only. The per-user private library P is out of scope for the
public demo. No login, no invite codes.

The API key is passed in by the caller (app.py resolves st.secrets first,
then environment, then the Secure Vault surrogate). It is never printed
or written to disk here.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo-relative root first (works on Streamlit Cloud when the repo keeps
# the package/ + streamlit/ sibling layout), VM workspace path as fallback.
_CANDIDATES = [
    os.path.dirname(HERE),
    os.path.expanduser("~/workspace/eplc-assistant"),
]
EPLC_ROOT = next(
    (c for c in _CANDIDATES
     if os.path.isdir(os.path.join(c, "package", "eplc_package"))),
    _CANDIDATES[1],
)
PACKAGE_CODE = os.path.join(EPLC_ROOT, "package", "eplc_package", "code")
PACKAGE_CACHE = os.path.join(
    EPLC_ROOT, "package", "eplc_package", "code", "backend", "index_cache"
)
STREAMLIT_CACHE = os.path.join(HERE, "index_cache")
DATA_PARSED = os.path.join(EPLC_ROOT, "package", "eplc_package", "data_parsed")
HELPER = "/opt/muse/skills/custom-connector-setup/bin/dynamic_credentials.py"

for _p in (PACKAGE_CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedder import OpenRouterEmbedder  # noqa: E402

CHAT_MODEL = "qwen/qwen-plus"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def resolve_api_key(explicit: str | None = None) -> str:
    """st.secrets value > env var > local Secure Vault surrogate. Never printed.

    The surrogate helper only exists on the developer VM. On Streamlit Cloud
    (or any machine without it) a missing key raises a clear error instead
    of a confusing file-not-found error for a VM-internal path.
    """
    if explicit:
        return explicit
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if os.path.exists(HELPER):
        spec = importlib.util.spec_from_file_location("dynamic_credentials", HELPER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.dynamic_credential_entry("custom.openrouter")["surrogate"]
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to the app's Secrets "
        "(Streamlit Cloud: app Settings -> Secrets) or set the "
        "OPENROUTER_API_KEY environment variable for local runs."
    )


# ---------------------------------------------------------------- provider


class OpenRouterChatProvider:
    """Minimal LLMProvider-compatible chat provider (stdlib urllib)."""

    name = "openrouter"

    def __init__(self, api_key: str, model: str = CHAT_MODEL,
                 temperature: float = 0.2, max_tokens: int = 1500) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage: dict = {}
        self.last_latency_s: float = 0.0
        self.last_cost_usd: float | None = None
        self.last_used_reasoning_fallback: bool = False

    def generate(self, prompt: str, system: str | None = None) -> str:
        from backend.generator import ProviderError

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "HTTP-Referer": "eplc-assistant",
                "X-Title": "EPLC Assistant",
            },
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise ProviderError(
                f"OpenRouter HTTP {e.code}: "
                f"{e.read().decode('utf-8', 'replace')[:300]}"
            ) from e
        except urllib.error.URLError as e:
            raise ProviderError(f"OpenRouter request failed: {e}") from e
        self.last_latency_s = time.time() - t0
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()
        usage = body.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        self.last_cost_usd = usage.get("cost")
        self.last_used_reasoning_fallback = False
        if not content:
            reasoning = (msg.get("reasoning") or "").strip()
            if reasoning:
                content = reasoning
                self.last_used_reasoning_fallback = True
        return content


# ---------------------------------------------------------------- wiring


def build_retriever(api_key: str):
    """DualRetriever over the re-embedded A/B indexes (no HF anywhere)."""
    from backend.retriever import DualRetriever

    retr = DualRetriever()  # constructs HFEmbedder object only; no API call
    retr.embedder = OpenRouterEmbedder(api_key=api_key)
    indexes = {}
    for lib in ("A", "B"):
        chunks = json.load(
            open(os.path.join(PACKAGE_CACHE, lib, "chunks.json"), encoding="utf-8")
        )
        emb = np.load(os.path.join(STREAMLIT_CACHE, lib, "embeddings.npy"))
        if emb.shape[0] != len(chunks):
            raise ValueError(
                f"library {lib}: {len(chunks)} chunks but embeddings {emb.shape}"
            )
        indexes[lib] = (chunks, emb)
    retr._indexes = indexes
    return retr


def build_generator(api_key: str | None = None):
    """SectionGenerator wired to OpenRouter retrieval + generation."""
    from backend.generator import SectionGenerator

    key = resolve_api_key(api_key)
    return SectionGenerator(
        retriever=build_retriever(key),
        provider=OpenRouterChatProvider(api_key=key),
    )


# ---------------------------------------------------------------- templates


def list_templates() -> dict[str, dict]:
    """{template_name: {"num_sections": n, "num_fields": n}} from data_parsed."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(os.path.join(DATA_PARSED, "*.json"))):
        try:
            doc = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = doc.get("template_name")
        if not name:
            continue
        secs = doc.get("sections", [])
        out[name] = {
            "num_sections": len(secs),
            "num_fields": sum(len(s.get("fields") or []) for s in secs),
            "path": f,
        }
    return out


def load_template_sections(template_name: str) -> list[dict]:
    from backend.assembler import load_template

    return load_template(template_name).get("sections", [])


# ---------------------------------------------------------------- drafting


def run_draft(gen, template_name: str, project_info: dict,
              progress_cb=None) -> dict:
    """Generate every section in template order, then assemble a .docx.

    Mirrors app/api/app/tasks.py run_draft_job for shared templates
    (previous-sections summary capped at 4000 chars). Returns
    {"docx_path", "docx_bytes", "num_sections", "drafts", "flagged"} where
    flagged lists sections whose draft contains [MISSING: ...] markers.
    """
    from backend.assembler import assemble

    sections = load_template_sections(template_name)
    if not sections:
        raise ValueError(f"template {template_name!r} has no sections")
    drafts: dict[str, dict] = {}
    prev_summary = ""
    total = len(sections)
    for i, sec in enumerate(sections):
        heading = sec.get("section_heading") or f"section-{i}"
        if progress_cb:
            progress_cb(i + 1, total, heading)
        out = gen.generate_section(
            template_name,
            heading,
            project_info=project_info,
            previous_sections_summary=prev_summary,
        )
        drafts[heading] = out
        prev_summary = (prev_summary + "\n\n" + out.get("draft", ""))[-4000:]

    import tempfile

    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_"
                   for c in template_name)
    tmpdir = tempfile.mkdtemp(prefix="eplc_demo_")
    out_path = os.path.join(tmpdir, f"{safe}_draft.docx")
    docx_path = assemble(template_name, drafts, project_info, out_path)
    flagged = [h for h, d in drafts.items() if "[MISSING:" in (d.get("draft") or "")]
    return {
        "docx_path": docx_path,
        "docx_bytes": os.path.getsize(docx_path),
        "num_sections": total,
        "drafts": drafts,
        "flagged": flagged,
    }
