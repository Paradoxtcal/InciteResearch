"""
InciteResearch — ResearchState
"""

from __future__ import annotations
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    phase: str
    human_feedback: str
    error: Optional[str]
    user_language: str

    topic: str
    friction_points: list
    motivation: str
    constraints: dict
    research_taste: str
    dialogue_turns: int

    keywords: list
    raw_papers: list
    paper_summary: str
    candidate_directions: list
    selected_direction: dict
    paper_library: dict
    paper_fulltext_request: list

    hidden_assumptions: list
    broken_assumption: str
    reframed_problem: str
    assumption_analysis: str
    story_arc: str
    method_necessity_check: str
    novelty_score: float
    feasibility_score: float
    research_proposal: str

    baseline_code: str
    core_modification: str
    github_repos: list

    session_id: str
    dialogue_turns: int
    checkpoints: list
    metadata: dict

    user_insight: str
    raw_ideas: list
    selected_idea: str
    seed_insight: str
    experiment_results: str
    ablation_plan: str
    similar_papers: list
    paper_outline: str
    draft_sections: dict
    full_draft: str
    reviewer_dialogue: list
