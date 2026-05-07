"""
Phase 3 — Experiment Agent
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

from utils.state import ResearchState
from utils.llm_client import get_llm
from tools.code_tools import search_github_repos, search_papers_with_code


def _get_llm(temperature=0.3):
    return get_llm(temperature=temperature)


def github_search_node(state: ResearchState) -> ResearchState:
    llm = _get_llm()
    proposal = state.get("research_proposal", "")
    topic = state.get("topic", "")

    kw_resp = llm.invoke([
        SystemMessage(content=(
            "Extract 3 technical keywords best suited for GitHub search.\n"
            "Output a JSON array only."
        )),
        HumanMessage(content=f"Proposal summary:\n{proposal[:500]}"),
    ])
    import json, re
    match = re.search(r'\[.*?\]', kw_resp.content, re.DOTALL)
    tech_keywords = json.loads(match.group()) if match else [topic]

    repos = []
    for kw in tech_keywords[:2]:
        repos += search_github_repos(kw, limit=5)
        repos += search_papers_with_code(kw, limit=3)

    seen, unique_repos = set(), []
    for r in repos:
        key = r.get("url", r.get("name", ""))
        if key and key not in seen:
            seen.add(key)
            unique_repos.append(r)

    repo_text = "\n".join(
        f"- {r.get('name','')}: {r.get('description','')} ⭐{r.get('stars',0)} {r.get('url','')}"
        for r in unique_repos[:15]
    )
    rec_resp = llm.invoke([
        SystemMessage(content=(
            "You are a software engineer who is good at finding the right open-source baseline.\n"
            "Core principle: only change the core algorithm; reuse framework/dataloading/evaluation as-is.\n"
            "For each recommended repo, explain:\n"
            "1) what can be reused\n"
            "2) what core code must be modified\n"
            "3) expected modification difficulty\n"
            "Write in the same language as the user's latest message."
        )),
        HumanMessage(content=(
            f"Proposal:\n{proposal[:600]}\n\n"
            f"Candidate repos:\n{repo_text}\n\n"
            "Recommend the top 3 repos as baselines and describe the adaptation strategy."
        )),
    ])

    print(f"  ✓ Found {len(unique_repos)} candidate repos; recommended top 3")

    return {
        **state,
        "github_repos": unique_repos[:15],
        "baseline_code": rec_resp.content,
        "phase": "experiment",
    }


def code_modify_node(state: ResearchState) -> ResearchState:
    llm = _get_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=(
            "You are an experiment engineering expert.\n"
            "Write in the same language as the user's latest message.\n"
            "Output format (Markdown):\n"
            "## Core files to modify\n"
            "## Key function rewrite (before → after pseudocode)\n"
            "## New modules to add\n"
            "## Hyperparameter suggestions\n"
            "## Quick validation steps (make it run on a small subset first)"
        )),
        HumanMessage(content=(
            f"Proposal (method core):\n{state.get('research_proposal','')[:800]}\n\n"
            f"Recommended baseline repo:\n{state.get('baseline_code','')[:600]}\n\n"
            "Give concrete modification guidance. Focus on minimal changes to get experiments running."
        )),
    ])

    return {
        **state,
        "core_modification": response.content,
    }


def experiment_node(state: ResearchState) -> ResearchState:
    llm = _get_llm(temperature=0.3)
    results = state.get("experiment_results", "")

    if not results:
        response = llm.invoke([
            SystemMessage(content="Generate an experiment logging template as a Markdown table. Write in the same language as the user's latest message."),
            HumanMessage(content=(
                f"Proposal: {state.get('research_proposal','')[:400]}\n\n"
                "Include:\n"
                "- compared methods\n"
                "- datasets\n"
                "- metrics\n"
                "- results table (blank to fill)\n"
                "- ablation variants"
            )),
        ])
        ablation = response.content
        print("  ✓ Generated an experiment template (waiting for results)")
        return {**state, "ablation_plan": ablation}

    analysis_resp = llm.invoke([
        SystemMessage(content=(
            "You are a strict paper reviewer evaluating whether the experimental evidence is publishable.\n"
            "Write in the same language as the user's latest message.\n"
            "Criteria:\n"
            "1) Is the improvement over baselines meaningful? (often >1% is not enough by itself)\n"
            "2) Are ablations complete and aligned with the method narrative?\n"
            "3) Do we need more datasets / settings to validate robustness?\n"
            "4) Publishability grade:\n"
            "   A: top-tier conference\n"
            "   B: strong journal\n"
            "   C: needs more experiments\n"
            "   D: reconsider direction"
        )),
        HumanMessage(content=(
            f"Results:\n{results}\n\n"
            f"Proposal:\n{state.get('research_proposal','')[:400]}"
        )),
    ])

    print("  ✓ Experiment analysis complete")
    return {
        **state,
        "experiment_results": results,
        "ablation_plan": analysis_resp.content,
    }
