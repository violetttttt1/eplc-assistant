"""EPLC Assistant — public demo (English).

Grounded guidance and drafting support for IT project managers, over two
shared libraries:
  - Library A: CDC EPLC document templates
  - Library B: HHS EPLC framework and policy documents

Layout follows the original EPLC_AI_Assistant frontend: Learn / Ask /
Create. Public demo: no login, no invite codes. Answers are grounded in
the cited sources; anything unknown is marked [MISSING: ...] and never
invented. If the best retrieval score is below the 0.55 refusal
threshold, the assistant refuses rather than guessing.

Retrieval uses OpenRouter embeddings; generation uses qwen/qwen-plus
via OpenRouter.

Secrets: the OpenRouter key comes from Streamlit secrets
(.streamlit/secrets.toml, key OPENROUTER_API_KEY) with an environment
variable fallback for local runs. The key is never shown in the UI.
"""

import os
import re
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eplc_demo  # noqa: E402
import template_guide  # noqa: E402

st.set_page_config(
    page_title="EPLC Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
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


def _render_citations(hits):
    """Expander of retrieved passages, like the original's Sources used."""
    hits = tuple(hits or ())
    if not hits:
        return
    with st.expander(f"Sources used ({len(hits)})"):
        for h in hits:
            st.markdown(
                f"**[{h['library']}] {h['citation']}**  ·  "
                f"score `{h['score']:.3f}`"
            )
            excerpt = " ".join((h.get("text") or "").split())[:800]
            if excerpt:
                st.caption(excerpt)


def _render_citation_strings(citations):
    """Expander for generate_section's citation strings."""
    citations = tuple(citations or ())
    if not citations:
        return
    with st.expander(f"Sources used ({len(citations)})"):
        for c in citations:
            st.markdown(f"- {c}")


def _parse_project_facts(project_name, facts_raw, instructions):
    """Project name + one-per-line Key: value facts + optional instructions."""
    project_info = {"project_name": project_name.strip()}
    for line in facts_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip() and v.strip():
                project_info[k.strip()] = v.strip()
    if instructions.strip():
        project_info["Additional drafting instructions"] = instructions.strip()
    return project_info


# ------------------------------------------------------------------ sidebar

st.session_state.setdefault("page", "Learn")
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.title("EPLC Assistant")
    st.caption(
        "Grounded guidance and drafting support for IT project managers."
    )
    st.divider()
    page = st.radio(
        "Navigation",
        ("Learn", "Ask", "Create"),
        index=("Learn", "Ask", "Create").index(st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = page
    st.divider()
    st.caption(
        "AI output is a drafting aid, not an official compliance determination."
    )

# ------------------------------------------------------------------- Learn

if page == "Learn":
    st.title("Work with EPLC guidance, one decision at a time")
    st.write(
        "The assistant retrieves relevant passages from two shared libraries "
        "— CDC document templates (library A) and HHS EPLC policy documents "
        "(library B) — before it answers a policy question or drafts a "
        "document section."
    )

    qna_column, draft_column = st.columns(2)
    with qna_column:
        st.subheader("Ask a policy question")
        st.markdown(
            """
1. Ask a specific question and name the phase when possible.
2. Review the answer and its retrieved source excerpts.
3. If no relevant source is found, the assistant will say so instead of
   filling the gap with general knowledge.
"""
        )

    with draft_column:
        st.subheader("Draft a document section")
        st.markdown(
            """
1. Choose a phase and template.
2. Name one section and provide known project facts.
3. Review the draft, missing-information checklist, and sources.
"""
        )

    st.info(
        "This is a public demo. Avoid entering sensitive project information."
    )

# --------------------------------------------------------------------- Ask

elif page == "Ask":
    st.title("Ask an EPLC policy question")
    st.caption(
        "Answers are limited to passages retrieved from the CDC template "
        "library (A) and the HHS policy library (B). If the best retrieval "
        "score is below 0.55, the assistant refuses rather than guessing."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("warning"):
                st.warning(message["warning"])
            _render_citations(message.get("hits", ()))

    question = st.chat_input(
        "For example: What are the Development Phase exit criteria?"
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving EPLC sources and preparing an answer…"):
                    out = GEN.answer_policy_question(question.strip())
            except Exception as e:
                st.error(f"Question failed: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Question failed: {e}"}
                )
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
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": out["answer"],
                        "warning": (
                            "Refused: best retrieval score below the 0.55 "
                            "threshold."
                        ),
                        "hits": [],
                    }
                )
            else:
                st.markdown(out["answer"])
                _render_citations(out.get("hits", []))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": out["answer"],
                        "hits": out.get("hits", []),
                    }
                )

# ------------------------------------------------------------------ Create

else:
    st.title("Create a grounded document section")
    st.caption(
        "The assistant drafts one section at a time and does not invent "
        "missing project facts."
    )

    phase = st.selectbox("EPLC phase", tuple(template_guide.PHASE_TEMPLATES))
    st.caption(template_guide.PHASE_DESCRIPTIONS[phase])
    template_name = st.selectbox(
        "Document template", template_guide.templates_for_phase(phase)
    )
    st.markdown(f"*{template_guide.purpose(template_name)}*")

    try:
        sections = eplc_demo.load_template_sections(template_name)
    except Exception as e:
        st.error(f"Could not load template: {e}")
        st.stop()
    headings = [s.get("section_heading") for s in sections if s.get("section_heading")]
    # "(preamble)" is boilerplate, not a draftable section; fall back to it
    # only for the few templates that have nothing else.
    selectable = [h for h in headings if h != "(preamble)"] or headings
    with st.expander(f"Sections in this template ({len(headings)})"):
        for h in headings:
            st.markdown(f"- {h}")

    # Plain widgets (not an st.form) so both action buttons below can share
    # the same inputs; a form would only release values on its own submit.
    section = st.selectbox("Section", selectable)
    project_name = st.text_input(
        "Project name", placeholder="e.g. CaseFlow Modernization"
    )
    facts_raw = st.text_area(
        "Known project details",
        height=150,
        placeholder=(
            "One fact per line as Key: value.\n"
            "Project sponsor: Office of the CIO\n"
            "Target completion: 2027-06-30"
        ),
    )
    instructions = st.text_area(
        "Additional drafting instructions (optional)",
        height=80,
        placeholder="For example: Use a formal tone and emphasize security controls.",
    )

    if st.button("Generate grounded draft", type="primary"):
        if not project_name.strip():
            st.warning("Please enter a project name.")
        else:
            project_info = _parse_project_facts(
                project_name, facts_raw, instructions
            )
            try:
                with st.spinner("Retrieving template guidance and drafting the section…"):
                    out = GEN.generate_section(
                        template_name, section, project_info=project_info
                    )
            except Exception as e:
                st.error(f"Draft failed: {e}")
                st.stop()

            st.subheader("Draft")
            st.write(out["draft"])

            st.subheader("Information to confirm")
            missing = re.findall(r"\[MISSING:[^\]]*\]", out["draft"] or "")
            if missing:
                for m in dict.fromkeys(missing):
                    st.markdown(f"- {m}")
            else:
                st.markdown(
                    "No missing information flagged — every field in this "
                    "section was covered by the facts you provided."
                )
            _render_citation_strings(out.get("citations") or [])

    st.divider()
    st.markdown("Need the whole document instead of one section?")
    if st.button("Generate full document (.docx)"):
        if not project_name.strip():
            st.warning("Please enter a project name.")
        else:
            project_info = _parse_project_facts(
                project_name, facts_raw, instructions
            )
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
