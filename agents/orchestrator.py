"""
InciteResearch — LangGraph Orchestrator
Scope: Phase 0–2 (from friction points to core algorithm code).

Phase 0  elicit          Socratic elicitation -> researcher profile
Phase 1  idea_discovery  assumption mining -> 3 candidate directions
Phase 2  idea_refine     select + refine direction
         assumption_breaking / problem_reframing
         story_arc / necessity_check / proposal
         generate_core_code  (core deliverable for an AI coding assistant)
"""

from __future__ import annotations
import uuid
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from utils.state import ResearchState
from utils.memory import get_checkpointer
from agents.dialogue_agent import elicit_node
from agents.idea_agent import idea_node, idea_refine_node, paper_expand_node
from agents.refine_agent import (
    assumption_breaking_node,
    problem_reframing_node,
    story_arc_node,
    necessity_check_node,
    proposal_node,
)
from agents.method_agent import generate_core_code_node


def _pass(state):
    return state

human_elicit_continue  = _pass
human_direction_select = _pass
human_proposal_review  = _pass


def route_elicit(state: ResearchState) -> str:
    if state.get("friction_points") and state.get("dialogue_turns", 0) >= 2:
        return "idea_discovery"
    return "human_elicit_continue"


def route_after_idea(state: ResearchState) -> str:
    if state.get("candidate_directions") and not state.get("selected_direction"):
        return "human_direction_select"
    return "idea_refine"


def route_after_proposal(state: ResearchState) -> str:
    approved = state.get("human_feedback", "").lower() in ("approved", "yes", "ok")
    if state.get("research_proposal") and not approved:
        return "human_proposal_review"
    return "generate_core_code"


def build_graph(checkpointer=None):
    if checkpointer is None:
        try:
            checkpointer = get_checkpointer()
        except Exception:
            checkpointer = MemorySaver()

    g = StateGraph(ResearchState)

    g.add_node("elicit",                 elicit_node)
    g.add_node("human_elicit_continue",  human_elicit_continue)
    g.add_node("idea_discovery",         idea_node)
    g.add_node("human_direction_select", human_direction_select)
    g.add_node("paper_expand",           paper_expand_node)
    g.add_node("paper_expand_post_proposal", paper_expand_node)
    g.add_node("idea_refine",            idea_refine_node)
    g.add_node("assumption_breaking",    assumption_breaking_node)
    g.add_node("problem_reframing",      problem_reframing_node)
    g.add_node("story_arc",              story_arc_node)
    g.add_node("necessity_check",        necessity_check_node)
    g.add_node("proposal",               proposal_node)
    g.add_node("human_proposal_review",  human_proposal_review)
    g.add_node("generate_core_code",     generate_core_code_node)

    g.set_entry_point("elicit")
    g.add_conditional_edges("elicit", route_elicit)
    g.add_edge("human_elicit_continue",  "elicit")

    g.add_conditional_edges("idea_discovery", route_after_idea)
    g.add_edge("human_direction_select", "paper_expand")
    g.add_edge("paper_expand",           "idea_refine")

    g.add_edge("idea_refine",         "assumption_breaking")
    g.add_edge("assumption_breaking", "problem_reframing")
    g.add_edge("problem_reframing",   "story_arc")
    g.add_edge("story_arc",           "necessity_check")
    g.add_edge("necessity_check",     "proposal")

    g.add_conditional_edges("proposal", route_after_proposal)
    g.add_edge("human_proposal_review", "paper_expand_post_proposal")
    g.add_edge("paper_expand_post_proposal", "generate_core_code")
    g.add_edge("generate_core_code",    END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "human_elicit_continue",
            "human_direction_select",
            "human_proposal_review",
        ],
    )


def start_dialogue(topic: str, session_id: str | None = None, user_language: str | None = None):
    """
    Start a session and return (graph, config, first_question).
    """
    graph = build_graph()
    sid = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": sid}}

    initial: ResearchState = {
        "topic": topic,
        "phase": "eliciting",
        "dialogue_turns": 0,
        "session_id": sid,
        "checkpoints": [],
        "metadata": {},
        "user_language": user_language or "Auto",
    }

    first_msg = ""
    for event in graph.stream(initial, config, stream_mode="values"):
        msgs = event.get("messages", [])
        for m in reversed(msgs):
            if isinstance(m, dict):
                if m.get("role") == "assistant":
                    first_msg = m.get("content", "")
                    break
                continue
            msg_type = getattr(m, "type", "")
            if msg_type == "ai":
                first_msg = getattr(m, "content", "") or ""
                break
        if first_msg:
            break

    return graph, config, first_msg


def resume_session(session_id: str):
    """Resume an existing session by session_id."""
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    return graph, config
