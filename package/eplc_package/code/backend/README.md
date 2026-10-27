# EPLC Assistant — Unified Backend

Single backend for the EPLC Assistant, replacing the scattered experiment scripts.
Follows the architecture in `EPLC_Assistant_Design_Doc_v2.md` (section 四):
offline indexing → dual-library retrieval → per-section generation → one-click
`.docx` assembly.

## Locked-in technical decisions (from the retrieval AB test)

- Embeddings: `BAAI/bge-large-en-v1.5` via HuggingFace Inference API.
  **Same model for indexing and querying**; documents get NO prefix, queries get
  `Represent this sentence for searching relevant passages: `.
- Chunking: **section-level + 512 token cap** (split only oversized sections).
- Token is read from `HF_TOKEN`; on this workspace the `custom.huggingface`
  surrogate is used as a fallback. The raw token is never printed, logged, or
  persisted.

## Modules

| File | Purpose |
|---|---|
| `config.yaml` | Model name, prefixes, top-k, chunking limits, corpus paths — all knobs here |
| `common.py` | Package root, config loader, corpus fingerprinting |
| `embeddings.py` | HF Inference API client: batching, retry+failover, query/doc prefix split |
| `index.py` | Builds library A and B indexes; chunk-level; metadata travels with chunks; embeddings cached as `.npy`; incremental rebuild via corpus fingerprint |
| `retriever.py` | Dual-library retrieval: per-library top-k, metadata filters (`--phase`, `--doc-type`, `--artifact`), merge + dedupe, citation per hit (`title — section_path <url>`) |
| `generator.py` | Per-section generation: retrieval-grounded drafting, `[MISSING: ...]` markers for unknown facts, refusal below the similarity threshold; `LLMProvider` abstraction + deterministic `MockProvider` |
| `assembler.py` | One-click assembly: sections in template order → `.docx` via python-docx (heading levels, italic-gray instructions, bold field placeholders, red `[MISSING]` markers) |
| `cli.py` | Single entry point for the whole chain |

## Run flow

All commands run from `<package root>/code` (e.g. `cd ~/workspace/eplc-assistant/package/eplc_package/code`).

```bash
# 1. Build indexes (run once; later runs reuse the cache unless sources change)
python3 -m backend.cli build            # libraries A + B
python3 -m backend.cli build --lib A --force   # force rebuild one library

# 2. Policy Q&A (flow A)
python3 -m backend.cli ask "What are the EPLC stage gate reviews?"
python3 -m backend.cli ask "deliverables in the Design phase" --phase Design

# 3. Draft one template section (flow B)
python3 -m backend.cli section \
  --template CDC_UP_Business_Case_LITE_Template \
  --heading "Business Need" \
  --project '{"project_name":"CaseFlow","project_manager":"Violet"}' \
  --out /tmp/section_business_need.json

# 4. Assemble drafts into .docx
python3 -m backend.cli assemble \
  --template CDC_UP_Business_Case_LITE_Template \
  --drafts /tmp/all_drafts.json \
  --project '{"project_name":"CaseFlow"}' \
  --out /tmp/caseflow_draft.docx
```

## Determinism & failure behaviour

- Same corpus → same chunk_ids, same order (sorted filenames, JSON order),
  same embeddings → same retrieval ranking (ties broken by `chunk_id`).
- The mock provider is fully deterministic: same inputs → same draft.
- Rebuilds are incremental: unchanged sources hit the `.npy` cache, no API calls.
- Clear errors: missing index → tells you the exact build command; no query text →
  `ValueError`; API failure after retries → `EmbeddingError` with the HTTP status
  (never the token); unknown provider in config → `ProviderError`.

## Libraries

- **A — templates**: 50 CDC UP `.docx` templates (`data_parsed/`) + 10 `.xls`
  templates (`data_parsed_xls/`): 8 log-type (one chunk per column with
  instructions / dropdown values / examples), schedule (one chunk per WBS task),
  stage-gate (one chunk per phase × review area, placeholders excluded).
- **B — hhs**: 16 official HHS docs (`data_parsed_hhs/`): EPLC Framework + 8
  Practices Guides + 7 policies, with `url` / `snapshot_url` / `covers` metadata
  for citations.

## What real-LLM integration still needs

The `LLMProvider` interface is ready; `OpenAIProvider` and `AnthropicProvider`
are thin wrappers that read `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the
environment and raise a clear `ProviderError` if unset. Real integration is
blocked on the pending **model selection step** (候选横向测评: quality, cost,
latency, context window, compliance/data-residency) — after that, set
`generation.provider` in `config.yaml` to `openai`/`anthropic` and supply the
key. No key is hard-coded anywhere; nothing will silently call a paid API.

Update 2026-09-27: the pending bake-off now has its funding path — an
OpenRouter key (`custom.openrouter`, $5 limit) lives in the Secure Vault and
`~/workspace/skills/openrouter/` provides `bin/key_info.py` / `bin/chat.py`.
After the bake-off picks a winner, add a thin `OpenRouterProvider` (mirroring
`OpenAIProvider`, reading the key from env or the vault skill) and set
`generation.provider` in `config.yaml`.
