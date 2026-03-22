"""
InciteResearch — Streamlit Web UI
Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv

    if os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
    elif os.path.exists(".env"):
        load_dotenv(".env", override=True)
except Exception:
    pass

_page_icon_path = os.path.join(os.path.dirname(__file__), "assets", "pig_avatar.png")
_page_icon = "🔬"
if os.path.exists(_page_icon_path):
    try:
        from PIL import Image

        _page_icon = Image.open(_page_icon_path)
    except Exception:
        _page_icon = _page_icon_path

def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False


def _topic_to_search_queries(topic: str) -> list[str]:
    t = (topic or "").strip()
    if not t:
        return []
    if _is_ascii(t):
        return [t]
    try:
        import json, re
        from langchain_core.messages import HumanMessage, SystemMessage
        from utils.llm_client import get_llm

        llm = get_llm(temperature=0.2)
        resp = llm.invoke([
            SystemMessage(content=(
                "Rewrite the user's research topic into 2 concise English academic search queries.\n"
                "Rules:\n"
                "- Preserve the exact task and scope. Do not add new tasks.\n"
                "- Use common English terminology and abbreviations.\n"
                "Output a strict JSON array of strings only."
            )),
            HumanMessage(content=t),
        ])
        m = re.search(r"\[.*\]", resp.content, re.DOTALL)
        if m:
            arr = json.loads(m.group())
            out = []
            for x in arr:
                s = str(x or "").strip()
                if s and s not in out:
                    out.append(s)
            if out:
                return out[:2]
    except Exception:
        pass
    return [t]


def _build_profile_from_topic_and_insight(topic: str, insight: str) -> dict:
    t = (topic or "").strip()
    ins = (insight or "").strip()
    if not t:
        return {"friction_points": [], "motivation": "", "research_taste": ""}
    try:
        import json, re
        from langchain_core.messages import HumanMessage, SystemMessage
        from utils.llm_client import get_llm

        llm = get_llm(temperature=0.2)
        resp = llm.invoke([
            SystemMessage(content=(
                "You summarize a compact researcher profile from a topic and a user insight.\n"
                "Priority rule:\n"
                "- The Topic is the primary objective and must dominate.\n"
                "- The Insight is secondary: treat it as inspiration/constraints/mechanisms, never as a new standalone task.\n"
                "Hard constraints:\n"
                "- Do NOT introduce new domains/tasks/datasets not implied by the Topic.\n"
                "- Each friction point must explicitly relate to improving the Topic objective.\n"
                "Output strict JSON only (no extra text):\n"
                "{\n"
                '  "friction_points": ["...","..."],\n'
                '  "motivation": "one sentence",\n'
                '  "research_taste": "one sentence"\n'
                "}"
            )),
            HumanMessage(content=json.dumps({
                "topic": t,
                "insight": ins,
            }, ensure_ascii=False)),
        ])
        m = re.search(r"\{.*\}", resp.content, re.DOTALL)
        if m:
            profile = json.loads(m.group())
            fp = profile.get("friction_points", [])
            fp = [str(x).strip() for x in (fp or []) if str(x).strip()]
            return {
                "friction_points": fp[:6],
                "motivation": str(profile.get("motivation", "") or "").strip(),
                "research_taste": str(profile.get("research_taste", "") or "").strip(),
            }
    except Exception:
        pass
    return {"friction_points": [ins] if ins else [], "motivation": "", "research_taste": ""}


st.set_page_config(
    page_title="InciteResearch",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 InciteResearch — Your Phase-0 AI research partner")
st.caption("A LangGraph-driven assistant from idea discovery to draft writing")

with st.sidebar:
    st.header("⚙️ Settings")
    default_provider = (os.environ.get("RESEARCH_AGENT_LLM_PROVIDER") or os.environ.get("LLM_PROVIDER") or "gemini").strip().lower()
    if default_provider not in ("gemini", "openai", "anthropic"):
        default_provider = "gemini"
    provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "openai", "anthropic"],
        index=["gemini", "openai", "anthropic"].index(default_provider),
        key="settings_provider",
    )
    os.environ["RESEARCH_AGENT_LLM_PROVIDER"] = provider

    if provider == "gemini":
        api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""), key="settings_gemini_api_key")
        model = st.text_input("Gemini Model", value=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"), key="settings_gemini_model")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        if model:
            os.environ["GEMINI_MODEL"] = model
    elif provider == "anthropic":
        api_key = st.text_input("Anthropic API Key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""), key="settings_anthropic_api_key")
        model = st.text_input("Anthropic Model", value=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), key="settings_anthropic_model")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        if model:
            os.environ["ANTHROPIC_MODEL"] = model
    else:
        api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""), key="settings_openai_api_key")
        model = st.text_input("OpenAI Model", value=os.environ.get("OPENAI_MODEL", "gpt-4o"), key="settings_openai_model")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if model:
            os.environ["OPENAI_MODEL"] = model

    if provider == "gemini":
        st.caption(f"Effective: provider=gemini | model={os.environ.get('GEMINI_MODEL','')}")
    elif provider == "openai":
        st.caption(f"Effective: provider=openai | model={os.environ.get('OPENAI_MODEL','')}")
    else:
        st.caption(f"Effective: provider=anthropic | model={os.environ.get('ANTHROPIC_MODEL','')}")
    st.caption(
        "Effective: paper_min="
        + str(os.environ.get("RESEARCH_AGENT_PAPER_MIN", ""))
        + " | paper_limit="
        + str(os.environ.get("RESEARCH_AGENT_PAPER_LIMIT", ""))
        + " | rerank_candidates="
        + str(os.environ.get("RESEARCH_AGENT_RERANK_CANDIDATES", ""))
    )

    st.divider()
    st.header("📍 Current stage")
    phase_labels = {
        "input": "0) Input",
        "running": "1) Idea discovery",
        "idea_select": "2) Idea selection",
        "refining": "3) Idea refinement",
        "idea_discovery": "1) Idea discovery",
        "idea_refine": "2) Idea refinement",
        "experiment": "3) Experiment support",
        "writing": "4) Writing",
        "review": "5) Review / rebuttal",
        "done": "✅ Done",
    }
    current_phase = st.session_state.get("phase", "input")
    phase_order = list(phase_labels.keys())

    def _phase_index(p: str):
        try:
            return phase_order.index(p)
        except ValueError:
            return None

    current_idx = _phase_index(current_phase)
    for phase, label in phase_labels.items():
        if phase == current_phase:
            st.markdown(f"**→ {label}**")
        elif current_idx is not None and _phase_index(phase) is not None and _phase_index(phase) < current_idx:
            st.markdown(f"~~{label}~~ ✓")
        else:
            st.markdown(f"  {label}")

if "state" not in st.session_state:
    st.session_state.state = {}
if "phase" not in st.session_state:
    st.session_state.phase = "input"

if st.session_state.phase == "input":
    st.header("Step 1: Tell me your research direction")

    col1, col2 = st.columns([1, 1])
    with col1:
        topic = st.text_area(
            "🎯 Research direction",
            placeholder="e.g., medical image segmentation with diffusion models",
            height=100,
        )
    with col2:
        user_insight = st.text_area(
            "💡 Your intuition / experience (required)",
            placeholder="e.g., prior work ignores anatomical priors; adding priors may help.\n\nIt can be vague; this is the most important input.",
            height=100,
        )

    st.info("💡 Your intuition matters: literature-only synthesis often produces shallow combinations.")

    if st.button("🚀 Start", type="primary", disabled=not (topic and user_insight and api_key)):
        if not api_key:
            st.error("Please provide an API key in the sidebar first.")
        else:
            st.session_state.state = {
                "topic": topic,
                "user_insight": user_insight,
                "phase": "idea_discovery",
                "user_language": "Auto",
            }
            st.session_state.phase = "running"
            st.rerun()

elif st.session_state.phase == "running":
    state = st.session_state.state
    topic = state.get("topic", "")

    st.header(f"Analyzing: {topic}")

    with st.spinner("🧭 Building researcher profile (topic-first)..."):
        try:
            profile = _build_profile_from_topic_and_insight(topic, state.get("user_insight", ""))
            if not state.get("friction_points"):
                state["friction_points"] = profile.get("friction_points", []) or []
            if not state.get("motivation"):
                state["motivation"] = profile.get("motivation", "") or ""
            if not state.get("research_taste"):
                state["research_taste"] = profile.get("research_taste", "") or ""
        except Exception as e:
            st.warning(f"Profile step had an issue: {e}. Continuing.")

    with st.spinner("🔎 Searching papers..."):
        try:
            from agents.idea_agent import paper_search_node
            state = paper_search_node(state)
        except Exception as e:
            st.warning(f"Paper search had an issue: {e}. Continuing with no papers.")

    with st.spinner("🧠 Synthesizing directions (topic-first, insight-secondary)..."):
        try:
            from agents.idea_agent import direction_synthesis_node
            state = direction_synthesis_node(state)
            if "raw_ideas" not in state:
                dirs = state.get("candidate_directions", []) or []
                state["raw_ideas"] = [
                    (d.get("name") or d.get("one_line") or "").strip()
                    for d in dirs
                    if isinstance(d, dict)
                ] or []
            st.session_state.state = state
            st.session_state.phase = "idea_select"
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(str(e))

elif st.session_state.phase == "idea_select":
    state = st.session_state.state
    st.header("Step 2: Pick a direction to refine")

    papers = state.get("raw_papers", []) or []
    if papers:
        with st.expander(f"📚 Related papers (expand) — {len(papers)}"):
            for p in papers[:12]:
                score = p.get("relevance_score", None)
                score_txt = f" — score {score}" if score is not None else ""
                st.markdown(f"**{p.get('title','')}** ({p.get('year','')}){score_txt}")
                reason = (p.get("relevance_reason", "") or "").strip()
                if reason:
                    st.caption(f"reason: {reason}")
                if p.get("abstract"):
                    st.caption((p.get("abstract") or "")[:220] + "...")

    if state.get("paper_summary"):
        with st.expander("📊 Literature summary (AI)"):
            st.markdown(state["paper_summary"])

    ideas = state.get("raw_ideas", [])
    if ideas:
        st.subheader("Generated candidates")
        selected = st.radio("Pick one:", ideas, index=0)

        custom = st.text_area("Or write your own (can be rough):", height=80)
        final_idea = custom.strip() if custom.strip() else selected

        if st.button("✅ Confirm and refine", type="primary"):
            state["selected_idea"] = final_idea
            st.session_state.state = state
            st.session_state.phase = "refining"
            st.rerun()

elif st.session_state.phase == "refining":
    state = st.session_state.state
    st.header("Step 3: Refine + build story arc")

    with st.spinner("🔬 Evaluating novelty and feasibility..."):
        from agents.refine_agent import gap_analysis_node, story_arc_node, proposal_node
        state = gap_analysis_node(state)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Novelty", f"{state.get('novelty_score', 0):.0%}")
    with col2:
        st.metric("Feasibility", f"{state.get('feasibility_score', 0):.0%}")

    if state.get("novelty_score", 0) < 0.6:
        st.warning("⚠️ Low novelty; consider adjusting the direction.")
    if state.get("feasibility_score", 0) < 0.5:
        st.warning("⚠️ Low feasibility; consider simplifying the method.")

    with st.spinner("📖 Building story arc..."):
        state = story_arc_node(state)

    st.subheader("📖 Story arc")
    st.markdown(state.get("story_arc", ""))

    with st.spinner("📋 Generating proposal..."):
        state = proposal_node(state)

    st.subheader("📋 Proposal draft")
    st.markdown(state.get("research_proposal", ""))

    st.session_state.state = state

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Continue to experiments", type="primary"):
            st.session_state.phase = "experiment"
            st.rerun()
    with col2:
        if st.button("🔄 Back to selection"):
            st.session_state.phase = "idea_select"
            st.rerun()

elif st.session_state.phase == "experiment":
    state = st.session_state.state
    st.header("Step 4: Experiment support")

    with st.spinner("🔍 Searching baseline repositories..."):
        from agents.experiment_agent import github_search_node, code_modify_node, experiment_node
        state = github_search_node(state)

    st.subheader("🛠️ Recommended baselines")
    st.markdown(state.get("baseline_code", ""))

    with st.spinner("⚙️ Generating code modification guidance..."):
        state = code_modify_node(state)

    st.subheader("💻 Core code modification guidance")
    st.markdown(state.get("core_modification", ""))

    st.divider()
    st.subheader("📊 Paste your experiment results")
    results = st.text_area(
        "Paste results (tables or numbers):",
        placeholder="e.g.,\nOurs: 85.3 mAP on COCO\nBaseline: 82.1 mAP\n...",
        height=150,
    )

    if st.button("📈 Analyze results", disabled=not results):
        state["experiment_results"] = results
        with st.spinner("Analyzing..."):
            state = experiment_node(state)
        st.markdown(state.get("ablation_plan", ""))

    st.session_state.state = state

    if st.button("✅ Continue to writing", type="primary"):
        st.session_state.phase = "writing"
        st.rerun()

elif st.session_state.phase == "writing":
    state = st.session_state.state
    st.header("Step 5: Writing")

    with st.spinner("📑 Generating outline..."):
        from agents.writing_agent import outline_node, draft_node, polish_node
        state = outline_node(state)

    st.subheader("Outline")
    st.markdown(state.get("paper_outline", ""))

    if st.button("✍️ Generate full draft", type="primary"):
        progress = st.progress(0)
        sections = ["abstract", "introduction", "related_work", "method", "experiments", "conclusion"]
        drafts = {}

        from langchain_core.messages import HumanMessage, SystemMessage
        from utils.llm_client import get_llm
        llm = get_llm(temperature=0.6)

        for i, section in enumerate(sections):
            with st.spinner(f"Writing {section}..."):
                resp = llm.invoke([
                    SystemMessage(content=f"You are a top-tier conference paper writing expert. Write the {section} section in the user's language (match the language used in the input)."),
                    HumanMessage(content=f"Outline: {state.get('paper_outline','')[:600]}\nStory arc: {state.get('story_arc','')[:400]}\nWrite the {section} section."),
                ])
                drafts[section] = resp.content
            progress.progress((i + 1) / len(sections))

        state["draft_sections"] = drafts
        state["full_draft"] = "\n\n".join(
            f"## {s.upper()}\n\n{drafts[s]}" for s in sections if s in drafts
        )
        st.session_state.state = state

        st.success("✅ Draft complete!")
        st.download_button(
            "📥 Download draft (Markdown)",
            data=state["full_draft"],
            file_name="paper_draft.md",
            mime="text/markdown",
        )

        for section, content in drafts.items():
            with st.expander(f"📄 {section.upper()}"):
                st.markdown(content)
