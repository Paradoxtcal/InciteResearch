"""
Method Agent
Generates core algorithm code meant to be handed to an AI coding assistant.
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

from utils.state import ResearchState
from utils.llm_client import get_llm


def _llm(temperature=0.3):
    return get_llm(temperature=temperature)


CORE_CODE_SYSTEM = """You translate research ideas into implementable code notes.

Your output will be handed to an AI coding assistant (e.g., Cursor / Claude Code).
The goal is to make intent unambiguous, not to produce a runnable project.

Output rules:
1) Use Python + PyTorch style; pseudocode is allowed when clearly marked
2) Only comment on key design decisions; use # WHY: to tie decisions to the narrative
3) Mark key differences vs baseline with # DIFF: (not line-by-line)
4) Annotate integration points with a single line like: # FILE: models/xxx.py  # FUNCTION: forward()
5) Do not write main(), training loops, or dataloaders

Tone: a technical memo to a smart colleague."""


def generate_core_code_node(state: ResearchState) -> ResearchState:
    llm = _llm()
    direction = state.get("selected_direction", {})
    broken = direction.get("broken_assumption", state.get("broken_assumption", ""))
    story = state.get("story_arc", "")
    proposal = state.get("research_proposal", "")
    necessity = state.get("method_necessity_check", "")

    baseline_resp = llm.invoke([
        SystemMessage(content=(
            "You are an implementation-focused research engineer.\n"
            "Recommend the most suitable baseline framework/repo for this research direction.\n"
            "Output strict JSON only:\n"
            '{"framework": "Framework name (e.g., MMTracking, MMDetection, Transformers, etc.)",\n'
            ' "repo_url": "GitHub URL",\n'
            ' "key_files": ["file paths to modify"],\n'
            ' "key_functions": ["functions/classes to modify"],\n'
            ' "why": "Why this baseline over alternatives"}'
        )),
        HumanMessage(content=(
            f"Topic: {state.get('topic', '')}\n"
            f"Broken assumption: {broken}\n"
            f"Proposal summary: {proposal[:400]}"
        )),
    ])

    import json, re
    bp_match = re.search(r'\{.*\}', baseline_resp.content, re.DOTALL)
    try:
        baseline_pointer = json.loads(bp_match.group()) if bp_match else {}
    except Exception:
        baseline_pointer = {"framework": baseline_resp.content}

    key_files = baseline_pointer.get("key_files", ["model.py"])
    key_funcs = baseline_pointer.get("key_functions", ["forward()"])

    code_resp = llm.invoke([
        SystemMessage(content=CORE_CODE_SYSTEM),
        HumanMessage(content=(
            f"# Background\n"
            f"Broken assumption: {broken}\n\n"
            f"Narrative (method section):\n{_extract_method_section(story)}\n\n"
            f"Necessity check (what components are truly necessary):\n{necessity[:600]}\n\n"
            f"# Integration targets\n"
            f"Framework: {baseline_pointer.get('framework', 'TBD')}\n"
            f"Files: {key_files}\n"
            f"Functions: {key_funcs}\n\n"
            f"# Task\n"
            f"Write the core algorithm implementation in Python.\n"
            f"Focus on the module that breaks the assumption; for everything else, state “reuse baseline’s XXX”.\n"
            f"Only add # WHY for key design decisions and # DIFF for key differences vs baseline.\n"
            f"End with a # AI CODING PROMPT telling Cursor/Claude Code how to integrate this into the baseline."
        )),
    ])

    core_code = code_resp.content

    integration_resp = llm.invoke([
        SystemMessage(content=(
            "Generate an integration instruction for Cursor / Claude Code (English, directly copy-pastable).\n"
            "Format:\n"
            "```\n"
            "I have a baseline implementation of [X] from [repo].\n"
            "I need to modify it to implement [core idea].\n"
            "The key change is: [one sentence].\n"
            "Please modify [file] → [function] as follows:\n"
            "[core change description]\n"
            "Keep everything else unchanged.\n"
            "```"
        )),
        HumanMessage(content=(
            f"Framework: {baseline_pointer.get('framework', '')}\n"
            f"Broken assumption: {broken}\n"
            f"Core code:\n{core_code[:800]}"
        )),
    ])

    integration_prompt = integration_resp.content

    print(f"  ✓ Core code generated ({len(core_code)} chars) | baseline: {baseline_pointer.get('framework', '?')}")

    return {
        **state,
        "baseline_code": json.dumps(baseline_pointer, ensure_ascii=False),
        "core_modification": core_code,
        "github_repos": [baseline_pointer] if baseline_pointer.get("repo_url") else [],
        "metadata": {
            **state.get("metadata", {}),
            "integration_prompt": integration_prompt,
            "baseline_pointer": baseline_pointer,
        },
    }


def _extract_method_section(story_arc: str) -> str:
    lines = story_arc.split("\n")
    in_method = False
    method_lines = []
    for line in lines:
        if "Method" in line or "🔧" in line:
            in_method = True
        elif any(x in line for x in ["📊", "🌟", "Validation", "Impact"]) and in_method:
            break
        if in_method:
            method_lines.append(line)
    return "\n".join(method_lines) if method_lines else story_arc[:400]
