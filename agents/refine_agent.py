"""
Phase 2 — Idea Refinement Agent
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

from utils.state import ResearchState
from utils.llm_client import get_llm
from tools.literature_tools import fetch_fulltext_excerpt, load_paper_library, save_paper_library
from prompts.refine_prompts import (
    ASSUMPTION_BREAKING_PROMPT,
    PROBLEM_REFRAMING_PROMPT,
    STORY_ARC_PROMPT,
    NECESSITY_CHECK_PROMPT,
    PROPOSAL_PROMPT,
)


def _get_llm(temperature=0.5):
    return get_llm(temperature=temperature)


def _auto_pdf_enabled() -> bool:
    import os
    v = (os.environ.get("RESEARCH_AGENT_AUTO_READ_PDF") or os.environ.get("RESEARCH_AGENT_AUTO_PDF") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return False


def _auto_pdf_max() -> int:
    import os
    try:
        return max(0, int(os.environ.get("RESEARCH_AGENT_AUTO_READ_PDF_MAX", os.environ.get("RESEARCH_AGENT_AUTO_PDF_MAX", "3"))))
    except Exception:
        return 3


def _auto_pick_papers(llm, topic: str, selected_idea: str, paper_summary: str, raw_papers: list, max_n: int) -> list[str]:
    if max_n <= 0 or not raw_papers:
        return []
    items = []
    for p in raw_papers[:30]:
        items.append({
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "year": p.get("year", ""),
            "has_pdf": bool(p.get("pdf_url")),
            "abstract": (p.get("abstract", "") or "")[:500],
        })
    import json, re
    resp = llm.invoke([
        SystemMessage(content=(
            "You are deciding whether to read PDF full-text excerpts to more rigorously extract implicit assumptions and reduce hallucinations.\n"
            "Only read when abstracts/summaries are insufficient to support the inference.\n"
            "Output a strict JSON array of paper_id (max N). If not needed, output [].\n"
            "Only choose items with has_pdf=true."
        )),
        HumanMessage(content=json.dumps({
            "topic": topic,
            "selected_idea": selected_idea[:800],
            "paper_summary": paper_summary[:800],
            "N": max_n,
            "papers": items,
        }, ensure_ascii=False)),
    ])
    m = re.search(r"\[.*\]", resp.content, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group())
    except Exception:
        return []
    ids = []
    for x in arr:
        if isinstance(x, str) and x.strip():
            ids.append(x.strip())
    return ids[:max_n]


def assumption_breaking_node(state: ResearchState) -> ResearchState:
    llm = _get_llm(temperature=0.6)
    library = {**load_paper_library(), **dict(state.get("paper_library") or {})}
    raw_papers = state.get("raw_papers") or []
    if _auto_pdf_enabled():
        picked = _auto_pick_papers(
            llm,
            topic=state.get("topic", ""),
            selected_idea=state.get("selected_idea", ""),
            paper_summary=state.get("paper_summary", ""),
            raw_papers=raw_papers,
            max_n=_auto_pdf_max(),
        )
        if picked:
            by_id = {p.get("paper_id"): p for p in raw_papers if isinstance(p, dict)}
            updated = 0
            for pid in picked:
                entry = dict(library.get(pid) or {})
                if entry.get("fulltext_excerpt"):
                    continue
                paper = by_id.get(pid) or entry
                if not isinstance(paper, dict):
                    continue
                res = fetch_fulltext_excerpt(paper)
                if not res.get("ok"):
                    library[pid] = {**entry, "paper_id": pid, "fulltext_status": "unavailable", "pdf_url": res.get("pdf_url")}
                    continue
                library[pid] = {
                    **entry,
                    "paper_id": pid,
                    "pdf_url": res.get("pdf_url") or paper.get("pdf_url"),
                    "pdf_path": res.get("pdf_path"),
                    "fulltext_excerpt": res.get("text_excerpt", ""),
                    "fulltext_status": "cached",
                }
                updated += 1
            if updated:
                save_paper_library(library)
                print(f"  ✓ Auto-read and cached full-text excerpts for {updated} papers")

    fulltext_blocks = []
    for p in (raw_papers[:20] if isinstance(raw_papers, list) else []):
        pid = p.get("paper_id") if isinstance(p, dict) else None
        if not pid:
            continue
        excerpt = (library.get(pid) or {}).get("fulltext_excerpt", "")
        if excerpt:
            fulltext_blocks.append(f"({pid})\n{excerpt[:1400]}")
    fulltext_text = "\n\n".join(fulltext_blocks)

    fulltext_part = f"Available full-text excerpts (may be partial):\n{fulltext_text}\n\n" if fulltext_text else ""
    prompt_text = (
        f"Idea:\n{state.get('selected_idea', '')}\n\n"
        f"Literature summary (background of existing methods):\n{state.get('paper_summary', '')[:800]}\n\n"
        + fulltext_part +
        "Extract implicit assumptions and select the most promising one to break.\n"
        "Write in the same language as the user's latest message.\n"
        "Output strict JSON:\n"
        "{\n"
        '  "hidden_assumptions": ["assumption 1", "assumption 2", "assumption 3"],\n'
        '  "broken_assumption": "the assumption to break (one sentence)",\n'
        '  "breaking_rationale": "why it can be broken and what the world looks like after breaking it",\n'
        '  "novelty_score": 0.0-1.0,\n'
        '  "feasibility_score": 0.0-1.0\n'
        "}"
    )

    response = llm.invoke([
        SystemMessage(content=ASSUMPTION_BREAKING_PROMPT),
        HumanMessage(content=prompt_text),
    ])

    import json, re
    content = response.content
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            assumptions = data.get("hidden_assumptions", [])
            broken = data.get("broken_assumption", "")
            rationale = data.get("breaking_rationale", "")
            novelty = float(data.get("novelty_score", 0.7))
            feasibility = float(data.get("feasibility_score", 0.7))
            analysis_text = content
        except Exception:
            assumptions, broken, analysis_text = [], content, content
            novelty, feasibility = 0.7, 0.7
    else:
        assumptions, broken, analysis_text = [], content, content
        novelty, feasibility = 0.7, 0.7

    print(f"  ✓ Implicit assumptions: {len(assumptions)} | novelty: {novelty:.0%} | feasibility: {feasibility:.0%}")

    return {
        **state,
        "hidden_assumptions": assumptions,
        "broken_assumption": broken,
        "assumption_analysis": analysis_text,
        "novelty_score": novelty,
        "feasibility_score": feasibility,
    }


def problem_reframing_node(state: ResearchState) -> ResearchState:
    llm = _get_llm(temperature=0.65)
    response = llm.invoke([
        SystemMessage(content=PROBLEM_REFRAMING_PROMPT),
        HumanMessage(content=(
            f"Idea:\n{state.get('selected_idea', '')}\n\n"
            f"Broken assumption:\n{state.get('broken_assumption', '')}\n\n"
            f"Researcher context (reference only):\n{state.get('seed_insight', state.get('user_insight', ''))}\n\n"
            "From the angle of “are we solving the right problem?”, propose 1–2 problem reframings.\n"
            "If reframing is unnecessary, explain why.\n"
            "Write in the same language as the user's latest message."
        )),
    ])

    return {
        **state,
        "reframed_problem": response.content,
    }


def story_arc_node(state: ResearchState) -> ResearchState:
    llm = _get_llm(temperature=0.6)

    assumption_context = state.get("assumption_analysis", "")
    reframing_context = state.get("reframed_problem", "")
    combined_context = ""
    if assumption_context:
        combined_context += f"**Broken assumption:**\n{assumption_context}\n\n"
    if reframing_context:
        combined_context += f"**Problem reframing:**\n{reframing_context}\n\n"

    response = llm.invoke([
        SystemMessage(content=STORY_ARC_PROMPT),
        HumanMessage(content=(
            f"Idea:\n{state.get('selected_idea', '')}\n\n"
            f"{combined_context}"
            f"Motivation (why this matters to the researcher):\n"
            f"{state.get('motivation', state.get('seed_insight', state.get('user_insight', '')))}\n\n"
            "Build a complete, persuasive paper story arc.\n"
            "Write in the same language as the user's latest message.\n\n"
            "Format:\n"
            "🔴 **Problem**: Why is this problem important and hard? (concrete, with data)\n"
            "❌ **Broken Assumption**: Which fundamental assumption do existing methods get wrong?\n"
            "💡 **Insight**: After breaking the assumption, what is the core insight?\n"
            "🔧 **Method**: Given the insight, why must the method be designed this way?\n"
            "📊 **Validation**: What experiments could falsify/validate the insight?\n"
            "🌟 **Impact**: If true, what does it change for the field?"
        )),
    ])

    return {
        **state,
        "story_arc": response.content,
    }


def necessity_check_node(state: ResearchState) -> ResearchState:
    llm = _get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=NECESSITY_CHECK_PROMPT),
        HumanMessage(content=(
            f"Story arc:\n{state.get('story_arc', '')}\n\n"
            f"Method description in the proposal (if any):\n{state.get('research_proposal', '')[:400]}\n\n"
            "Run the following three tests strictly and point out anything optional or floating.\n"
            "Write in the same language as the user's latest message.\n\n"
            "**(1) Necessity test**\n"
            "Given Problem + Insight, is there a simpler alternative than the proposed method?\n"
            "If yes, explain why the simpler alternative is insufficient and where the method is irreplaceable.\n\n"
            "**(2) Sufficiency test**\n"
            "For each core component, can you trace it to a clear causal chain: “because [reason], we must have [component]”?\n"
            "List components and verify one by one.\n\n"
            "**(3) Counterexample test**\n"
            "Could minor tweaks on the baseline (bigger model, more data, etc.) reach the same effect?\n"
            "If yes, the contribution is weak; suggest how to strengthen it.\n\n"
            "End with an overall verdict: does the story close the loop (yes/no), and the single biggest thing to strengthen."
        )),
    ])

    print("  ✓ Story loop check complete")

    return {
        **state,
        "method_necessity_check": response.content,
    }


def proposal_node(state: ResearchState) -> ResearchState:
    llm = _get_llm(temperature=0.4)
    response = llm.invoke([
        SystemMessage(content=PROPOSAL_PROMPT),
        HumanMessage(content=(
            f"Story arc:\n{state.get('story_arc', '')}\n\n"
            f"Necessity-check results:\n{state.get('method_necessity_check', '')}\n\n"
            f"Novelty: {state.get('novelty_score', 0.7):.0%} | Feasibility: {state.get('feasibility_score', 0.7):.0%}\n\n"
            "Write a detailed research proposal draft (Markdown).\n"
            "Write in the same language as the user's latest message.\n"
            "## 1. Method overview (one paragraph; emphasize why it must be designed this way)\n"
            "## 2. Core algorithm (pseudocode; highlight the essential difference vs baseline)\n"
            "## 3. Baselines (what to compare to and why)\n"
            "## 4. Datasets & metrics (minimal set that can falsify/validate the core claim)\n"
            "## 5. Ablations (each component maps to one ablation)\n"
            "## 6. Expected results & risks (if the assumption is wrong, what would we observe?)"
        )),
    ])

    print(f"  ✓ Proposal draft generated ({len(response.content)} chars)")

    return {
        **state,
        "research_proposal": response.content,
        "phase": "idea_refine",
    }


gap_analysis_node = assumption_breaking_node
