"""EPLC Assistant — public demo (English).

Policy Q&A and section-by-section drafting over two shared libraries:
  - Library A: CDC EPLC document templates
  - Library B: HHS EPLC framework and policy documents

Public demo: no login, no invite codes. Answers are grounded in the cited
sources; anything unknown is marked [MISSING: ...] and never invented.
Retrieval uses OpenRouter embeddings; generation uses qwen/qwen-plus
via OpenRouter.

Secrets: the OpenRouter key comes from Streamlit secrets
(.streamlit/secrets.toml, key OPENROUTER_API_KEY) with an environment
variable fallback for local runs. The key is never shown in the UI.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eplc_demo  # noqa: E402

st.set_page_config(page_title="EPLC Assistant — Demo", layout="wide")
st.title("EPLC Assistant")
st.markdown(
    "Policy Q&A and section-by-section drafting for federal EPLC project "
    "documents, grounded in the CDC template library and HHS policy library. "
    "This is a public demo — no login required."
)


@st.cache_resource(show_spinner="Loading retrieval indexes…")
def get_generator(_api_key):
    return eplc_demo.build_generator(api_key=_api_key)


def resolve_key():
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return None


API_KEY = resolve_key()
try:
    GEN = get_generator(API_KEY)
except Exception as e:
    st.error(f"Could not start the demo backend: {e}")
    st.stop()

SECTION = st.sidebar.radio(
    "Navigate", ["Policy Q&A", "Drafting", "Templates"], index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n"
    "- Retrieval: dual-library dense search over CDC templates (A) and "
    "HHS policies (B).\n"
    "- Answers carry citations with retrieval scores. If the best score "
    "is below the refusal threshold (0.55), the assistant refuses rather "
    "than guessing.\n"
    "- Drafts mark every unknown fact as `[MISSING: ...]`."
)

# ------------------------------------------------------------------ Q&A

if SECTION == "Policy Q&A":
    st.header("Policy Q&A")
    st.markdown(
        "Ask a question about the EPLC process, stage gates, or federal "
        "project documentation. The answer is generated only from retrieved "
        "passages, with citations."
    )
    question = st.text_area("Question", height=80, placeholder="e.g. What are the phases of the HHS EPLC framework?")
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving passages and generating an answer…"):
                try:
                    out = GEN.answer_policy_question(question.strip())
                except Exception as e:
                    st.error(f"Question failed: {e}")
                    st.stop()
            if out.get("refused"):
                st.warning(
                    "Not specified in the retrieved context. The assistant "
                    "refused to answer rather than fabricate a federal-policy "
                    "answer — please rephrase or point to a source."
                )
                st.caption(
                    "Refusal rule: the best retrieval score was below the "
                    "0.55 threshold, so no answer was generated."
                )
            else:
                st.markdown(out["answer"])
                st.subheader("Sources")
                for h in out.get("hits", []):
                    st.markdown(
                        f"- **[{h['library']}]** score `{h['score']:.3f}` — "
                        f"{h['citation']}"
                    )
                    with st.expander("Show passage"):
                        st.markdown(h["text"][:2000])

# --------------------------------------------------------------- drafting

elif SECTION == "Drafting":
    st.header("Drafting")
    st.markdown(
        "Pick a shared template, name the project, and generate it "
        "section by section. Unfilled fields are flagged, never invented, "
        "and the result downloads as a .docx in the template's format."
    )
    templates = eplc_demo.list_templates()
    names = sorted(templates)
    template_name = st.selectbox("Template", names)
    project_name = st.text_input("Project name", placeholder="e.g. CaseFlow Modernization")
    facts_raw = st.text_area(
        "Additional project facts (optional, one per line as Key: value)",
        height=100,
        placeholder="Project sponsor: Office of the CIO\nTarget completion: 2027-06-30",
    )
    if st.button("Generate draft", type="primary"):
        if not project_name.strip():
            st.warning("Please enter a project name.")
        else:
            project_info = {"project_name": project_name.strip()}
            for line in facts_raw.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k.strip() and v.strip():
                        project_info[k.strip()] = v.strip()
            progress = st.progress(0.0)
            status = st.empty()
            try:
                result = eplc_demo.run_draft(
                    GEN, template_name, project_info,
                    progress_cb=lambda cur, tot, h: (
                        progress.progress(cur / tot),
                        status.markdown(f"Generating section {cur}/{tot}: **{h}**"),
                    ),
                )
            except Exception as e:
                st.error(f"Draft failed: {e}")
                st.stop()
            status.markdown("Draft complete.")
            st.success(
                f"Generated {result['num_sections']} sections. "
                f"{len(result['flagged'])} section(s) contain [MISSING: ...] "
                "markers for facts you did not provide."
            )
            if result["flagged"]:
                st.markdown("**Sections needing input:**")
                for h in result["flagged"]:
                    st.markdown(f"- {h}")
            with open(result["docx_path"], "rb") as fh:
                st.download_button(
                    "Download .docx",
                    data=fh.read(),
                    file_name=f"{template_name}_draft.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

# -------------------------------------------------------------- templates

else:
    st.header("Templates")
    st.markdown(
        "Read-only browsing of the shared CDC template library (library A). "
        "Each template lists its sections, heading levels, and fields."
    )
    templates = eplc_demo.list_templates()
    names = sorted(templates)
    template_name = st.selectbox("Template", names)
    meta = templates[template_name]
    st.markdown(
        f"**{template_name}** — {meta['num_sections']} sections, "
        f"{meta['num_fields']} fields."
    )
    try:
        sections = eplc_demo.load_template_sections(template_name)
    except Exception as e:
        st.error(f"Could not load template: {e}")
        st.stop()
    for sec in sections:
        heading = sec.get("section_heading", "")
        level = sec.get("section_level", 1)
        fields = sec.get("fields") or []
        instr = " ".join((sec.get("instruction") or "").split())[:300]
        with st.expander(f"{'  ' * (level - 1)}{heading}"):
            if instr:
                st.markdown(f"*{instr}*")
            if fields:
                st.markdown("**Fields:** " + ", ".join(fields))
            st.caption(f"Section path: {sec.get('section_path', '')}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Demo note: retrieval embeddings and generation run through OpenRouter. "
    "Verify cited sources before relying on any answer."
)
