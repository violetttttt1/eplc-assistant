# EPLC Assistant — Demo

Policy Q&A and section-by-section drafting for federal EPLC project
documents, grounded in two shared libraries:

- **Library A:** CDC EPLC document templates
- **Library B:** HHS EPLC framework and policy documents

This is a public demo — no login required. Answers are grounded in the
cited sources; anything unknown is marked `[MISSING: ...]` and never
invented. If the best retrieval score is below the refusal threshold
(0.55), the assistant refuses rather than guessing.

## How it works

- **Retrieval:** dual-library dense search. Document embeddings were built
  with `openai/text-embedding-3-small` (1024 dims) and live under
  `streamlit/index_cache/`; the query is embedded with the same model at
  request time.
- **Generation:** `qwen/qwen-plus` via OpenRouter.
- **Drafting:** pick a shared CDC template, generate it section by section,
  and download the result as a `.docx` in the template's format. Unfilled
  fields are flagged, never invented.

## Run locally

```bash
pip install -r streamlit/requirements.txt
export OPENROUTER_API_KEY="sk-or-..."
streamlit run streamlit/app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (public).
2. At [share.streamlit.io](https://share.streamlit.io), create a new app:
   pick the repo, branch `main`, main file `streamlit/app.py`.
3. Under **Advanced settings → Secrets**, add:

```toml
OPENROUTER_API_KEY = "sk-or-..."
```

4. Deploy. The app URL is fixed (`*.streamlit.app`) and does not depend
   on any tunnel.

## Layout

- `streamlit/app.py` — the Streamlit UI (English, no login).
- `streamlit/eplc_demo.py` — wiring: retrieval + generation + drafting.
- `streamlit/embedder.py` — OpenRouter embeddings client.
- `streamlit/index_cache/` — prebuilt embeddings for libraries A and B.
- `package/eplc_package/code/backend/` — retrieval, generation, and
  docx-assembly code plus `config.yaml`.
- `package/eplc_package/code/backend/index_cache/` — chunk texts
  (`chunks.json`) for libraries A and B.
- `package/eplc_package/data_parsed/` — parsed CDC template definitions.

## Notes

- No secrets are committed to this repo. The OpenRouter key is supplied
  only via Streamlit secrets (Cloud) or an environment variable (local).
- The original BGE-based index and the earlier Next.js demo are not part
  of this deployment.
