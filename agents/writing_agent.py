"""
Phase 4 — Writing Agent
Includes a reviewer role-play loop to keep the user in control of writing.
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

from utils.state import ResearchState
from utils.llm_client import get_llm
from tools.literature_tools import search_semantic_scholar


def _llm(temperature=0.5):
    return get_llm(temperature=temperature)


SECTION_ORDER = ["abstract", "introduction", "related_work", "method", "experiments", "conclusion"]


def outline_node(state: ResearchState) -> ResearchState:
    llm = _llm()
    story = state.get("story_arc", "")
    proposal = state.get("research_proposal", "")
    topic = state.get("topic", "")
    direction = state.get("selected_direction", {})
    broken_assumption = direction.get("broken_assumption", state.get("broken_assumption", ""))

    sim_papers = search_semantic_scholar(topic, limit=5)
    sim_text = "\n".join(f"- {p.get('title','')} ({p.get('year','')})" for p in sim_papers[:5])

    resp = llm.invoke([
        SystemMessage(content=(
            "Generate a paper outline.\n"
            "Key principles:\n"
            "- The first two paragraphs of Introduction must make the broken assumption and its importance crystal clear.\n"
            "- Related Work must explicitly identify which works rely on the broken assumption.\n"
            "Format: Markdown. Each section should include 3–5 bullet points.\n"
            "Write in the same language as the user's latest message."
        )),
        HumanMessage(content=(
            f"Story arc:\n{story}\n\n"
            f"Broken assumption:\n{broken_assumption}\n\n"
            f"Proposal:\n{proposal[:500]}\n\n"
            f"Reference papers:\n{sim_text}"
        )),
    ])

    print(f"  ✓ Outline generated | reference papers: {len(sim_papers)}")
    return {**state, "similar_papers": sim_papers, "paper_outline": resp.content, "phase": "writing"}


def draft_node(state: ResearchState) -> ResearchState:
    llm = _llm(temperature=0.6)
    outline = state.get("paper_outline", "")
    story = state.get("story_arc", "")
    proposal = state.get("research_proposal", "")
    results = state.get("experiment_results", "(to be filled)")
    direction = state.get("selected_direction", {})
    broken_assumption = direction.get("broken_assumption", state.get("broken_assumption", ""))

    section_extra = {
        "abstract": "≤250 words. Must include: the broken assumption + key quantitative results.",
        "introduction": (
            "Paragraph 1: why the problem matters (concrete scenario)\n"
            "Paragraph 2: what assumption prior work relies on and why it fails\n"
            "Paragraph 3: our insight and approach\n"
            "Paragraph 4: contributions (bullets)"
        ),
        "related_work":  "Explicitly mark which works rely on the broken assumption; do not just enumerate.",
        "method":        "For each key design decision, explicitly state: “Because X, we must do Y.”",
        "experiments":   f"Main results table + ablations (each ablation maps to one design decision).\nData: {results}",
        "conclusion":    "Summary + limitations (new limitations created after breaking the assumption) + future work",
    }

    drafts = {}
    for section in SECTION_ORDER:
        resp = llm.invoke([
            SystemMessage(content=(
                f"Write the {section} section. {section_extra.get(section, '')}\n"
                "Write in the same language as the user's latest message.\n"
                "Every sentence should carry information; remove filler."
            )),
            HumanMessage(content=(
                f"Outline: {outline[:600]}\n"
                f"Story arc: {story[:400]}\n"
                f"Broken assumption: {broken_assumption}\n"
                f"Proposal: {proposal[:400]}"
            )),
        ])
        drafts[section] = resp.content
        print(f"  ✓ {section} done")

    full = "\n\n".join(f"## {s.upper()}\n\n{drafts[s]}" for s in SECTION_ORDER if s in drafts)
    return {**state, "draft_sections": drafts, "full_draft": full}


def polish_node(state: ResearchState) -> ResearchState:
    llm = _llm(temperature=0.3)
    draft = state.get("full_draft", "")
    direction = state.get("selected_direction", {})
    broken_assumption = direction.get("broken_assumption", "")

    reviewer_resp = llm.invoke([
        SystemMessage(content=(
            "You are a strict but fair NeurIPS Area Chair.\n"
            "Your job is to provide feedback, not to rewrite.\n"
            "The author decides how to revise; you do not draft the text.\n\n"
            "Format:\n"
            "**Overall** (2 sentences: strength + biggest issue)\n\n"
            "**Must-fix issues** (3 items; explain why each matters)\n\n"
            "**Suggested improvements** (3 optional items)\n\n"
            "**Assumption-breaking checklist**:\n"
            "- Does the Introduction make the broken assumption clear?\n"
            "- Does Related Work correctly identify works relying on that assumption?\n"
            "- Do experiments include ablations that directly test “assumption X is wrong”?\n\n"
            "**5 expected reviewer concerns + your suggested responses**"
        )),
        HumanMessage(content=(
            f"Paper (first 3000 chars):\n{draft[:3000]}\n\n"
            f"Core claim (broken assumption): {broken_assumption}"
        )),
    ])

    dialogue_entry = {"role": "reviewer", "content": reviewer_resp.content, "accepted": None}
    existing = state.get("reviewer_dialogue", [])

    return {
        **state,
        "full_draft": draft + f"\n\n---\n## REVIEWER FEEDBACK (Round {len(existing)+1})\n\n{reviewer_resp.content}",
        "reviewer_dialogue": existing + [dialogue_entry],
        "phase": "review",
    }
