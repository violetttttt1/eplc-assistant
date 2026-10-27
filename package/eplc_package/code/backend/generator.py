"""Section-by-section generation for EPLC Assistant.

Flow (matches design doc 4.2 / 3.3):
  1. generate_section(template_name, section_heading, project_info):
     retrieve related context -> LLM drafts THAT section only.
  2. Missing user info is marked with [MISSING: <name>] — never invented.
  3. Policy Q&A: if the best retrieval score is below the refusal threshold,
     the assistant refuses rather than fabricating (federal compliance).

LLMProvider is an abstract interface. MockProvider is fully deterministic and
runs the whole chain with no API key. Real providers (OpenAI / Anthropic)
exist as thin wrappers that read their key from the environment; they raise a
clear error until model selection is done and keys are provisioned — nothing
is hard-coded.
"""

from __future__ import annotations

import os
import sys


class ProviderError(RuntimeError):
    """Clear, actionable LLM provider errors."""


class LLMProvider:
    """Abstract LLM interface."""

    name = "base"

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic stand-in: drafts from section metadata + user fields + context.

    It NEVER invents project facts (unknown fields -> [MISSING: ...]) and NEVER
    invents policy (it only echoes retrieved snippets verbatim when asked).
    """

    name = "mock"

    def generate(self, prompt: str, system: str | None = None) -> str:
        # The mock does not parse the prompt; real callers use the typed
        # helpers below (draft_section / answer_question) for structure.
        return prompt

    # ----- typed helpers (used by the pipeline) -----

    def draft_section(
        self,
        template_name: str,
        section: dict,
        project_info: dict,
        context_hits: list[dict],
        previous_sections_summary: str = "",
    ) -> str:
        heading = section.get("section_heading", "")
        instruction = (section.get("instruction") or "").strip()
        fields = section.get("fields") or []
        lines: list[str] = []

        proj = project_info.get("project_name", "")
        proj_clause = f" for {proj}" if proj else ""
        lines.append(f"This section of the {template_name}{proj_clause} is drafted below.")

        filled, missing = [], []
        for f in fields:
            key = _field_key(f)
            val = project_info.get(key) or project_info.get(f)
            if val:
                filled.append(f"{f}: {val}")
            else:
                missing.append(f"[MISSING: {f}]")
        if filled:
            lines.append("Provided values: " + "; ".join(filled))
        if missing:
            lines.append("Still required: " + "; ".join(missing))

        if instruction:
            lines.append("Per the template guidance: " + " ".join(instruction.split())[:500])

        if context_hits:
            lines.append("Relevant reference material:")
            for h in context_hits[:3]:
                snippet = " ".join(h["text"].split())[:280]
                lines.append(f"- [{h['citation']}] {snippet}")
        elif heading.lower() not in ("(preamble)",):
            lines.append("No directly related reference material was retrieved for this section.")

        if previous_sections_summary:
            lines.append(f"Consistency note with prior sections: {previous_sections_summary[:300]}")

        lines.append(
            "Note: this draft was produced without a real LLM (mock provider). "
            "It contains no invented project facts and no invented policy statements."
        )
        return "\n\n".join(lines)

    def answer_question(self, question: str, hits: list[dict]) -> str:
        parts = [f"Question: {question}\n"]
        if not hits:
            return (
                "I cannot answer from the retrieved context.\n"
                "No sufficiently relevant passages were found in the EPLC template "
                "or HHS policy libraries. Per federal-compliance policy I will not "
                "fabricate an answer. [MISSING: relevant source material]"
            )
        parts.append("Based on the retrieved material:")
        for h in hits:
            snippet = " ".join(h["text"].split())[:400]
            parts.append(f"- {snippet}\n  Source: {h['citation']}")
        parts.append(
            "Note: evidence-only answer assembled by the mock provider; "
            "verify against the cited sources before use."
        )
        return "\n\n".join(parts)


def _field_key(field: str) -> str:
    """Normalize '<Project Name>' -> 'project_name' so users can pass flat dicts."""
    return field.strip().strip("<>").lower().replace(" ", "_").replace("/", "_")


class OpenRouterProvider(LLMProvider):
    """Real LLM via OpenRouter, using the stored custom.openrouter credential.

    Default model is qwen/qwen-plus — the 2026-09-27 bake-off winner
    (26 models; QA 0.833 tied-top, lowest TCO, fastest p95, $0.009/doc).

    Lessons baked in:
    - reasoning fallback: thinking models may return empty content with the
      answer in the reasoning field; fall back to it (flagged) instead of
      silently scoring an empty string (bake-off bug #1).
    - usage + latency are recorded on the instance (last_usage, last_latency_s,
      last_used_reasoning_fallback) so the pipeline can report real cost.
    """

    name = "openrouter"
    DEFAULT_MODEL = "qwen/qwen-plus"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage: dict = {}
        self.last_latency_s: float = 0.0
        self.last_used_reasoning_fallback: bool = False
        self.last_cost_usd: float | None = None

    def _or_call(self):
        import importlib.util

        p = os.path.expanduser("~/workspace/skills/openrouter/bin/chat.py")
        if not os.path.exists(p):
            raise ProviderError(
                f"OpenRouter skill not found at {p}; cannot authenticate."
            )
        spec = importlib.util.spec_from_file_location("or_skill_chat", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # noqa
        return mod.call

    def generate(self, prompt: str, system: str | None = None) -> str:
        import time

        call = self._or_call()
        t0 = time.time()
        try:
            body = call(
                self.model, prompt,
                system=system,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except RuntimeError as e:
            raise ProviderError(f"OpenRouter call failed: {e}") from e
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
        # OpenRouter reports per-request cost in usage.cost for some providers
        self.last_cost_usd = usage.get("cost")

        self.last_used_reasoning_fallback = False
        if not content:
            reasoning = (msg.get("reasoning") or "").strip()
            if reasoning:
                content = reasoning
                self.last_used_reasoning_fallback = True
        return content


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def _client(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProviderError(
                "OPENAI_API_KEY is not set. Real LLM access is pending the model "
                "selection step; use the mock provider or set the key in your environment."
            )
        try:
            import openai  # type: ignore
        except ImportError as e:
            raise ProviderError(
                "the 'openai' package is not installed (pip install openai)"
            ) from e
        return openai.OpenAI()

    def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._client()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=self.model, messages=msgs, temperature=0.2)
        return (resp.choices[0].message.content or "").strip()


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        self.model = model

    def _client(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Real LLM access is pending the model "
                "selection step; use the mock provider or set the key in your environment."
            )
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ProviderError(
                "the 'anthropic' package is not installed (pip install anthropic)"
            ) from e
        return anthropic.Anthropic()

    def generate(self, prompt: str, system: str | None = None) -> str:
        client = self._client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


def provider_from_config(cfg: dict) -> LLMProvider:
    name = cfg["generation"]["provider"].lower()
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openrouter":
        g = cfg["generation"]
        return OpenRouterProvider(
            model=g.get("model"),
            temperature=g.get("temperature", 0.2),
            max_tokens=g.get("max_tokens", 1500),
        )
    raise ProviderError(
        f"unknown generation provider {name!r} in config.yaml "
        "(expected one of: mock, openai, anthropic, openrouter)"
    )


# ---------- pipeline ----------

EVIDENCE_ONLY_SYSTEM = (
    "You draft federal EPLC project documents. Rules: use ONLY the provided "
    "reference material and user-supplied facts. Mark anything unknown as "
    "[MISSING: <name>]. Never invent policy statements, dates, names, or "
    "numbers. If the material does not support an answer, say so explicitly. "
    "Compliance classifications (project tier, Exhibit 300 obligations, ATO, "
    "FedRAMP or other certifications, named directives/memoranda) must come "
    "ONLY from user-supplied facts or the reference material -- if not "
    "established there, write [NEEDS PM CONFIRMATION: <item>] and never "
    "assert them by default. Use the canonical terminology provided exactly "
    "as given; never re-expand an acronym on your own."
)


def _load_terminology() -> dict:
    """Canonical glossary; missing file -> empty (never fatal)."""
    import json

    p = os.path.join(os.path.dirname(__file__), "terminology.json")
    try:
        return json.load(open(p)).get("terms", {})
    except (OSError, ValueError):
        return {}


def _terminology_block() -> str:
    terms = _load_terminology()
    if not terms:
        return ""
    lines = ["Canonical terminology -- use these exact expansions, never invent your own:"]
    lines += [f"- {k} = {v}" for k, v in terms.items()]
    return "\n".join(lines)


def check_terminology(draft: str) -> list[str]:
    """Post-generation scan for known-wrong expansions (audit fix)."""
    import json

    p = os.path.join(os.path.dirname(__file__), "terminology.json")
    try:
        wrong = json.load(open(p)).get("wrong_expansions_flag", [])
    except (OSError, ValueError):
        return []
    low = draft.lower()
    return [w for w in wrong if w.lower() in low]


def find_section(template_name: str, section_heading: str) -> dict:
    """Locate a section's metadata in the parsed docx templates."""
    import glob, json

    from .common import package_root

    for f in sorted(glob.glob(os.path.join(package_root(), "data_parsed", "*.json"))):
        doc = json.load(open(f))
        if doc.get("template_name") == template_name or template_name in f:
            for sec in doc.get("sections", []):
                if sec.get("section_heading", "").lower() == section_heading.lower():
                    return {"template": doc, "section": sec}
    raise ValueError(
        f"section {section_heading!r} not found in template {template_name!r}"
    )


class SectionGenerator:
    """Per-section generation with retrieval grounding."""

    def __init__(self, retriever=None, provider: LLMProvider | None = None, verbose: bool = False):
        from .common import load_config

        self.cfg = load_config()
        if retriever is None:
            from .retriever import DualRetriever

            retriever = DualRetriever(verbose=verbose)
        self.retriever = retriever
        self.provider = provider or provider_from_config(self.cfg)
        self.verbose = verbose

    def _build_prompt(
        self,
        template_name: str,
        section: dict,
        project_info: dict,
        hits: list[dict],
        prev_summary: str = "",
        registry: str = "",
    ) -> tuple[str, str]:
        ctx = "\n\n".join(
            f"[Source: {h['citation']}]\n{' '.join(h['text'].split())[:1200]}"
            for h in hits[: self.cfg["generation"]["max_context_chunks"]]
        )
        term_block = _terminology_block()
        user = (
            f"Template: {template_name}\n"
            f"Section: {section.get('section_heading')} "
            f"({section.get('section_path')})\n"
            f"Template guidance: {' '.join((section.get('instruction') or '').split())[:800]}\n"
            f"Fields in this section: {', '.join(section.get('fields') or [])}\n"
            f"Project facts provided: {project_info}\n"
        )
        if term_block:
            user += f"\n{term_block}\n"
        if registry:
            user += (
                "\nDocument registry -- entities already defined earlier in THIS "
                f"document. Use these EXACT definitions; do not redefine them:\n{registry[:3000]}\n"
            )
        if prev_summary:
            user += f"Summary of previous sections (for consistency): {prev_summary[:400]}\n"
        user += (
            "\nReference material:\n" + (ctx or "(none retrieved)") + "\n\n"
            "Draft this section now. Mark every unknown fact as [MISSING: <name>]."
        )
        return EVIDENCE_ONLY_SYSTEM, user

    def generate_section(
        self,
        template_name: str,
        section_heading: str,
        project_info: dict | None = None,
        previous_sections_summary: str = "",
        registry: str = "",
    ) -> dict:
        project_info = project_info or {}
        found = find_section(template_name, section_heading)
        section = found["section"]

        query = (
            f"{template_name} {section.get('section_heading')} "
            f"{' '.join((section.get('instruction') or '').split())[:300]}"
        )
        hits = self.retriever.search(
            query,
            filters={"artifact": template_name},
            top_k_per_library=self.cfg["retrieval"]["top_k_per_library"],
            merge_top_k=self.cfg["generation"]["max_context_chunks"],
        )
        if self.verbose:
            print(f"  retrieved {len(hits)} hits for '{section_heading}'", flush=True)

        if isinstance(self.provider, MockProvider):
            text = self.provider.draft_section(
                template_name, section, project_info, hits, previous_sections_summary
            )
        else:
            system, user = self._build_prompt(
                template_name, section, project_info, hits, previous_sections_summary,
                registry=registry,
            )
            text = self.provider.generate(user, system=system)
        return {
            "template_name": template_name,
            "section_heading": section.get("section_heading"),
            "section_path": section.get("section_path"),
            "section_level": section.get("section_level", 1),
            "instruction": section.get("instruction"),
            "fields": section.get("fields") or [],
            "draft": text,
            "citations": [h["citation"] for h in hits],
            "provider": self.provider.name,
        }

    def answer_policy_question(self, question: str) -> dict:
        """Q&A flow A: retrieve -> refuse-or-answer with citations."""
        hits = self.retriever.search(question)
        threshold = self.cfg["retrieval"]["refusal_min_score"]
        best = hits[0]["score"] if hits else 0.0
        if best < threshold or not hits:
            answer = (
                "Not specified in the retrieved context. I will not fabricate a "
                "federal-policy answer — please rephrase or point me to a source."
            )
            return {"question": question, "answer": answer, "refused": True,
                    "hits": [], "provider": self.provider.name}
        if isinstance(self.provider, MockProvider):
            answer = self.provider.answer_question(question, hits)
        else:
            ctx = "\n\n".join(
                f"[Source: {h['citation']}]\n{' '.join(h['text'].split())[:1200]}"
                for h in hits
            )
            answer = self.provider.generate(
                f"Question: {question}\n\nReference material:\n{ctx}\n\n"
                "Answer using only the material above, with citations. "
                "If it is insufficient, say so explicitly.",
                system=EVIDENCE_ONLY_SYSTEM,
            )
        return {"question": question, "answer": answer, "refused": False,
                "hits": hits, "provider": self.provider.name}


def main(argv: list[str] | None = None) -> int:
    import argparse, json

    ap = argparse.ArgumentParser(description="Generate one EPLC section / answer a policy question")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("section", help="draft one section")
    g.add_argument("--template", required=True)
    g.add_argument("--heading", required=True)
    g.add_argument("--project", default="{}", help="JSON dict of project facts")
    q = sub.add_parser("ask", help="policy Q&A")
    q.add_argument("question")
    args = ap.parse_args(argv)

    gen = SectionGenerator(verbose=True)
    if args.cmd == "section":
        out = gen.generate_section(args.template, args.heading,
                                   project_info=json.loads(args.project))
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        out = gen.answer_policy_question(args.question)
        hits = out.pop("hits")
        print(json.dumps(out, ensure_ascii=False, indent=1)[:2000])
        for h in hits:
            print(f"  [{h['library']}] {h['score']:.3f} {h['citation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
