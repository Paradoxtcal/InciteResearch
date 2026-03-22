"""
Socratic Elicitor
Turn vague instincts into a usable researcher profile via 1–3 rounds of concrete questions.
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

from utils.state import ResearchState
from utils.llm_client import get_llm


def _llm(temperature=0.7):
    return get_llm(temperature=temperature)


ELICITOR_SYSTEM = """You are a research thinking partner. Your goal is to help a researcher surface the research intuitions they cannot yet articulate.

Operating principles:
1) Ask concrete, friction-inducing questions. Do not ask abstract questions like “what is your insight?”
2) Every answer is valid input, including “I don't know” or “something feels off but I can't explain.”
3) After 1–3 turns, summarize into a structured researcher profile.
4) The user's original research direction (Topic) is the primary objective and must not drift. Never replace the Topic with a different task/domain.

Style: curious, equal-footing, and never rushing.
Language: reply in the same language as the user's latest message.
"""


def elicit_node(state: ResearchState) -> ResearchState:
    llm = _llm()
    turns = state.get("dialogue_turns", 0)
    topic = state.get("topic", "Unspecified")
    messages = state.get("messages", [])
    feedback = state.get("human_feedback", "")

    if turns == 0:
        response = llm.invoke([
            SystemMessage(content=ELICITOR_SYSTEM),
            HumanMessage(content=(
                f"The researcher says their direction is: {topic}\n\n"
                "In 1–2 sentences, acknowledge you understood. Then ask the first question.\n"
                "The question must be concrete enough to immediately recall a specific paper or class of methods.\n"
                "Do not explain what you are doing; just continue the conversation."
            )),
        ])
        return {
            **state,
            "phase": "eliciting",
            "dialogue_turns": 1,
            "messages": [{"role": "assistant", "content": response.content}],
        }

    if turns == 1:
        prev_answer = feedback or "(no answer)"
        response = llm.invoke([
            SystemMessage(content=ELICITOR_SYSTEM),
            HumanMessage(content=(
                f"Research topic: {topic}\n"
                f"Answer to the previous question: {prev_answer}\n\n"
                "Analyze the answer:\n"
                "- If it contains a concrete friction point (an assumption, a class of methods, a specific dissatisfaction), dig deeper.\n"
                "- If it is vague, switch angle and ask a different concrete question.\n"
                "- If it implies resource constraints, ask about constraints.\n\n"
                "Ask the next question in at most 2 sentences."
            )),
        ])
        return {
            **state,
            "dialogue_turns": 2,
            "human_feedback": "",
            "messages": [{"role": "user", "content": prev_answer},
                         {"role": "assistant", "content": response.content}],
        }

    all_dialogue = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in (state.get("messages") or [])
        if isinstance(m, dict)
    )
    if feedback:
        all_dialogue += f"\nuser: {feedback}"

    profile_response = llm.invoke([
        SystemMessage(content=(
            "Summarize a researcher profile from the dialogue.\n"
            "Priority rule:\n"
            "- The original Topic is the primary objective and must dominate.\n"
            "- The user's answers are secondary: treat them as constraints/mechanisms, never as a new standalone task.\n"
            "Hard constraints:\n"
            "- Do NOT introduce a different domain/task/dataset not implied by the original Topic.\n"
            "- refined_topic must be a paraphrase/clarification of the original Topic, not a different problem.\n"
            "Output strict JSON only. Do not add extra text.\n"
            "Use the user's language for all string fields (match the user's latest message).\n"
            "{\n"
            '  "friction_points": ["friction point 1", "friction point 2"],\n'
            '  "motivation": "Why this matters (one sentence)",\n'
            '  "constraints": {"compute": "...", "timeline": "...", "other": "..."},\n'
            '  "research_taste": "What kinds of work they prefer (one sentence, inferred)",\n'
            '  "refined_topic": "A more precise version of the original topic"\n'
            "}"
        )),
        HumanMessage(content=f"Original topic: {topic}\n\nDialogue:\n{all_dialogue}"),
    ])

    import json, re
    content = profile_response.content
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            profile = json.loads(match.group())
            friction = profile.get("friction_points", [])
            motivation = profile.get("motivation", "")
            constraints = profile.get("constraints", {})
            taste = profile.get("research_taste", "")
            refined_topic = profile.get("refined_topic", topic)
        except Exception:
            friction, motivation, constraints, taste = [content], "", {}, ""
            refined_topic = topic
    else:
        friction, motivation, constraints, taste = [content], "", {}, ""
        refined_topic = topic

    print(f"  ✓ Researcher profile built | friction points: {len(friction)}")

    return {
        **state,
        "phase": "idea_discovery",
        "dialogue_turns": turns + 1,
        "friction_points": friction,
        "motivation": motivation,
        "constraints": constraints,
        "research_taste": taste,
        "refined_topic": refined_topic,
        "human_feedback": "",
        "messages": [{"role": "user", "content": feedback or ""},
                     {"role": "assistant", "content": f"[Researcher profile built; moving to idea discovery]"}],
    }
