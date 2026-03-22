"""
Phase 1 — Idea Discovery Agent
Design: friction points -> assumption mining -> 3 candidate directions (each breaks a distinct assumption).
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

import os
import json

from utils.state import ResearchState
from utils.llm_client import get_llm
from tools.literature_tools import search_semantic_scholar, search_arxiv_recent
from tools.literature_tools import fetch_fulltext_excerpt
from tools.literature_tools import load_paper_library, save_paper_library


def _llm(temperature=0.6):
    return get_llm(temperature=temperature)

def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False

def _dedupe_phrases(items: list, max_n: int = 24) -> list[str]:
    out: list[str] = []
    seen = set()
    for x in items or []:
        s = str(x or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def _extract_topic_anchors(llm, topic: str, friction_points: list, keywords: list) -> list[str]:
    import re
    base = [str(topic or "").strip()] + [str(x or "").strip() for x in (keywords or [])] + [str(x or "").strip() for x in (friction_points or [])]
    base = [x for x in base if x]
    resp = llm.invoke([
        SystemMessage(content=(
            "Extract 8–12 anchor phrases that define the user's intended research direction and objective.\n"
            "Rules:\n"
            "- Anchors must be grounded in the user's input (topic + friction points) and preserve intent.\n"
            "- Include both Chinese and English aliases if the user input is Chinese.\n"
            "- Anchors should be short phrases suitable for search queries and drift checking.\n"
            "- Do not introduce new diseases, modalities, or tasks not stated by the user.\n"
            "Output a strict JSON array of strings only."
        )),
        HumanMessage(content=json.dumps({
            "topic": topic,
            "friction_points": friction_points,
            "keywords": keywords,
            "user_input_phrases": base[:16],
        }, ensure_ascii=False)),
    ])
    m = re.search(r"\[.*\]", resp.content, re.DOTALL)
    arr = []
    if m:
        try:
            arr = json.loads(m.group())
        except Exception:
            arr = []
    anchors = _dedupe_phrases([topic] + (keywords or []) + arr, max_n=24)
    return anchors


def _contains_anchor(text: str, anchors: list[str]) -> bool:
    t = (text or "")
    tl = t.lower()
    for a in anchors or []:
        s = (a or "").strip()
        if not s:
            continue
        if _is_ascii(s):
            if s.lower() in tl:
                return True
        else:
            if s in t:
                return True
    return False


def _validate_directions(directions: list, anchors: list[str], primary_anchors: list[str]) -> bool:
    if not isinstance(directions, list) or len(directions) != 3:
        return False
    for d in directions:
        if not isinstance(d, dict):
            return False
        name = str(d.get("name", "") or "")
        one = str(d.get("one_line", "") or "")
        if not _contains_anchor(name, primary_anchors) or not _contains_anchor(one, primary_anchors):
            return False
        if (not _contains_anchor(name, anchors)) or (not _contains_anchor(one, anchors)):
            return False
    return True

def _auto_pdf_enabled() -> bool:
    v = (os.environ.get("RESEARCH_AGENT_AUTO_READ_PDF") or os.environ.get("RESEARCH_AGENT_AUTO_PDF") or "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return False


def _auto_pdf_max() -> int:
    try:
        return max(0, int(os.environ.get("RESEARCH_AGENT_AUTO_READ_PDF_MAX", os.environ.get("RESEARCH_AGENT_AUTO_PDF_MAX", "4"))))
    except Exception:
        return 4


def _select_papers_for_fulltext(llm, topic: str, friction_points: list, papers: list[dict], max_n: int) -> list[str]:
    if max_n <= 0 or not papers:
        return []
    items = []
    for p in papers[:30]:
        items.append({
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "year": p.get("year", ""),
            "has_pdf": bool(p.get("pdf_url")),
            "abstract": (p.get("abstract", "") or "")[:500],
        })
    resp = llm.invoke([
        SystemMessage(content=(
            "You are deciding whether to read full-text PDF excerpts to reduce hallucinations and topic drift.\n"
            "Select papers only if at least one is true:\n"
            "1) The abstract is clearly insufficient to judge implicit assumptions / experimental setup / key definitions\n"
            "2) The candidate directions depend on a technical detail that requires full text verification\n"
            "3) Multiple abstracts conflict and need full text to resolve\n\n"
            "Output a strict JSON array of paper_id to read (max N). If not needed, output [].\n"
            "Only choose items with has_pdf=true."
        )),
        HumanMessage(content=json.dumps({
            "topic": topic,
            "friction_points": friction_points,
            "N": max_n,
            "papers": items,
        }, ensure_ascii=False)),
    ])
    import re
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


def paper_expand_node(state: ResearchState) -> ResearchState:
    paper_ids = state.get("paper_fulltext_request") or []
    if not paper_ids:
        return state

    library = {**load_paper_library(), **dict(state.get("paper_library") or {})}
    by_id = {p.get("paper_id"): p for p in (state.get("raw_papers") or []) if isinstance(p, dict)}

    updated = 0
    for pid in paper_ids:
        if not pid:
            continue
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
        print(f"  ✓ Cached full-text excerpts for {updated} papers (on demand)")

    save_paper_library(library)
    return {**state, "paper_library": library}


def idea_node(state: ResearchState) -> ResearchState:
    state = paper_search_node(state)
    state = direction_synthesis_node(state)
    papers = state.get("raw_papers", []) or []
    directions = state.get("candidate_directions", []) or []
    print(f"  ✓ Papers: {len(papers)} | candidate directions: {len(directions)}")
    return state


def paper_search_node(state: ResearchState) -> ResearchState:
    llm = _llm()
    topic = state.get("topic", "")
    friction_points = state.get("friction_points", [])

    llm_ok = True
    translated_queries = []
    try:
        kw_resp = llm.invoke([
            SystemMessage(content=(
                "Extract up to 8 academic search keywords/phrases. Output a JSON array only.\n"
                "Hard constraint: keywords must be tightly related to the research topic; do not drift to unrelated fields.\n"
                "If the topic is not in English, include its most common English translation / aliases.\n"
                "Always include the topic itself (or its best English name) as one of the items."
            )),
            HumanMessage(content=f"Topic: {topic}\nFriction points: {friction_points}"),
        ])
        import re
        match = re.search(r'\[.*?\]', kw_resp.content, re.DOTALL)
        keywords = json.loads(match.group()) if match else [topic]
        bad_terms = (
            "recommendation", "recommender", "cold start", "bandit", "ctr",
            "finance", "stock", "trading", "navigation", "embodied", "llm",
        )
        kw_text = " ".join([str(x) for x in (keywords or [])]).lower()
        if any(x in kw_text for x in bad_terms):
            kw_resp = llm.invoke([
                SystemMessage(content=(
                    "Your previous keywords likely drifted. Re-extract 8 English academic search keywords/phrases.\n"
                    "Hard constraint: stay strictly on-topic; do not include unrelated fields (e.g., recommender systems, finance, navigation, LLM).\n"
                    "If the topic is not in English, include its common English translation / aliases.\n"
                    "Always include the topic itself (or its best English name) as one of the items.\n"
                    "Output a JSON array only."
                )),
                HumanMessage(content=f"Topic: {topic}\nFriction points: {friction_points}"),
            ])
            match = re.search(r'\[.*?\]', kw_resp.content, re.DOTALL)
            keywords = json.loads(match.group()) if match else keywords

        anchors = _extract_topic_anchors(llm, topic, friction_points, keywords)
        need_query_rewrite = (not any(_is_ascii(a) for a in anchors or [])) or (_is_ascii(str(topic or "")) and (len(str(topic or "")) > 80 or str(topic or "").count(" ") > 14))
        if need_query_rewrite:
            resp = llm.invoke([
                SystemMessage(content=(
                    "Rewrite the user's topic into 1–2 concise English academic search queries.\n"
                    "Rules:\n"
                    "- Preserve the exact task and scope. Do not add new tasks.\n"
                    "- Use common English terminology and abbreviations.\n"
                    "Output a strict JSON array of strings only."
                )),
                HumanMessage(content=str(topic or "").strip()),
            ])
            m = re.search(r"\[.*\]", resp.content, re.DOTALL)
            if m:
                try:
                    translated_queries = json.loads(m.group())
                except Exception:
                    translated_queries = []
            translated_queries = [str(x or "").strip() for x in (translated_queries or []) if str(x or "").strip()]
            if translated_queries:
                anchors = _dedupe_phrases(anchors + translated_queries, max_n=24)
                keywords = _dedupe_phrases(keywords + translated_queries, max_n=12)
    except Exception as e:
        print(f"  [WARNING] Keyword/anchor extraction failed; proceeding with topic-only search. {e}")
        llm_ok = False
        keywords = [topic]
        anchors = [topic]

    library = {**load_paper_library(), **dict(state.get("paper_library") or {})}

    try:
        total_target = int(os.environ.get("RESEARCH_AGENT_PAPER_LIMIT", "60"))
    except Exception:
        total_target = 60
    try:
        min_target = int(os.environ.get("RESEARCH_AGENT_PAPER_MIN", "12"))
    except Exception:
        min_target = 12

    terms = _dedupe_phrases(
        [a for a in anchors if a and _is_ascii(a)] + [k for k in (keywords or []) if k and _is_ascii(k)],
        max_n=12,
    )

    papers = []
    if terms:
        use_n = min(6, len(terms))
        per_kw = max(6, total_target // max(1, use_n))
        q = " ".join(terms[: min(3, use_n)]).strip()
        if q:
            papers += search_semantic_scholar(q, limit=min(80, max(per_kw, 12)))
        for kw in terms[: min(4, use_n)]:
            papers += search_semantic_scholar(kw, limit=min(60, max(per_kw, 10)))
        for kw in terms[:use_n]:
            papers += search_arxiv_recent(kw, limit=min(80, max(per_kw, 12)))

    if not papers and translated_queries:
        for q in translated_queries[:2]:
            if q:
                papers += search_semantic_scholar(q, limit=min(60, max(total_target, 10)))
                papers += search_arxiv_recent(q, limit=min(60, max(total_target, 10)))

    seen, unique = set(), []
    for p in papers:
        t = (p.get("title", "") or "").strip().lower()
        if t and t not in seen:
            seen.add(t)
            unique.append(p)

    if len(unique) < min_target:
        try:
            from tools.literature_tools import quick_literature_scan

            if translated_queries:
                supplement_queries = translated_queries[:2]
            elif terms:
                supplement_queries = [" ".join(terms[:3]).strip()]
            else:
                supplement_queries = [str(topic or "").strip()]

            for sq in [x for x in supplement_queries if x]:
                papers += quick_literature_scan(sq, top_k=min(40, max(min_target * 2, 20)))
            seen2, merged = set(), []
            for p in unique + papers:
                t = (p.get("title", "") or "").strip().lower()
                if t and t not in seen2:
                    seen2.add(t)
                    merged.append(p)
            unique = merged
            if len(unique) >= min_target:
                print(f"  ✓ Supplemented papers via quick scan | now: {len(unique)}")
        except Exception as e:
            print(f"  [WARNING] Paper supplementation failed: {e}")

    unique_unfiltered = list(unique)
    anchors_ascii_l = [a.lower() for a in terms if a]
    if anchors_ascii_l and unique:
        filtered = []
        for p in unique:
            hay = (p.get("title", "") + " " + (p.get("abstract", "") or "")).lower()
            if any(a in hay for a in anchors_ascii_l[:12]):
                filtered.append(p)
        if filtered and len(filtered) >= max(4, min_target // 3):
            unique = filtered
        else:
            print("  [WARNING] Papers do not match anchors strongly; keeping top results to avoid low-paper mode.")
            unique = unique_unfiltered[: min(max(min_target, 12), len(unique_unfiltered))]

    paper_rerank = []
    if unique and llm_ok:
        try:
            try:
                import re
                is_zh = bool(re.search(r"[\u4e00-\u9fff]", f"{topic} {' '.join([str(x) for x in (friction_points or [])])}"))
            except Exception:
                is_zh = False

            try:
                rerank_candidates = int(os.environ.get("RESEARCH_AGENT_RERANK_CANDIDATES", "30"))
            except Exception:
                rerank_candidates = 30
            n = min(len(unique), max(min_target, rerank_candidates))
            candidates = unique[:n]

            items = []
            for i, p in enumerate(candidates, 1):
                items.append({
                    "idx": i,
                    "paper_id": p.get("paper_id", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year", ""),
                    "abstract": (p.get("abstract", "") or "")[:700],
                    "source": p.get("source", ""),
                })

            resp = llm.invoke([
                SystemMessage(content=(
                    "You are an expert literature triage system.\n"
                    "Given a research Topic and friction points, score each paper's relevance.\n\n"
                    "Scoring rules:\n"
                    "- score is an integer 0–100.\n"
                    "- 90–100: directly on the exact task + setting.\n"
                    "- 70–89: same task, slightly different setting or a key sub-problem.\n"
                    "- 40–69: adjacent but potentially useful.\n"
                    "- 0–39: likely irrelevant.\n\n"
                    "Hard constraints:\n"
                    "- Do not introduce new tasks/domains. Judge only based on given metadata.\n"
                    "- Be strict: do not give high scores to generic ML papers.\n\n"
                    "Output strict JSON array only (no extra text). Each element:\n"
                    '{ "idx": 1, "paper_id": "...", "score": 0, "reason": "one short sentence" }\n'
                    + ("Use Chinese for reason." if is_zh else "Use English for reason.")
                )),
                HumanMessage(content=json.dumps({
                    "topic": topic,
                    "friction_points": friction_points,
                    "keywords": keywords,
                    "anchors": anchors,
                    "papers": items,
                }, ensure_ascii=False)),
            ])
            import re
            m = re.search(r"\[.*\]", resp.content, re.DOTALL)
            if m:
                paper_rerank = json.loads(m.group())
        except Exception as e:
            print(f"  [WARNING] LLM rerank failed; skipping rerank. {e}")

    if paper_rerank:
        by_id = {}
        by_idx = {}
        for r in paper_rerank:
            if not isinstance(r, dict):
                continue
            pid = str(r.get("paper_id", "") or "").strip()
            if pid:
                by_id[pid] = r
            try:
                idx = int(r.get("idx", 0) or 0)
            except Exception:
                idx = 0
            if idx > 0:
                by_idx[idx] = r

        enriched = []
        for i, p in enumerate(unique, 1):
            pid = str(p.get("paper_id", "") or "").strip()
            r = by_id.get(pid) or by_idx.get(i)
            if isinstance(r, dict):
                try:
                    score = int(r.get("score", 0) or 0)
                except Exception:
                    score = 0
                reason = str(r.get("reason", "") or "").strip()
                enriched.append({**p, "relevance_score": score, "relevance_reason": reason})
            else:
                enriched.append({**p, "relevance_score": 0, "relevance_reason": ""})

        enriched.sort(key=lambda x: int(x.get("relevance_score", 0) or 0), reverse=True)
        unique = enriched

    if unique:
        unique = unique[: min(len(unique), max(min_target, min(total_target, len(unique))))]

    for p in unique:
        pid = p.get("paper_id")
        if not pid:
            continue
        prev = dict(library.get(pid) or {})
        library[pid] = {
            **prev,
            "paper_id": pid,
            "title": p.get("title", ""),
            "abstract": p.get("abstract", "") or "",
            "year": p.get("year", ""),
            "authors": p.get("authors", []) or [],
            "url": p.get("url", ""),
            "pdf_url": p.get("pdf_url"),
            "relevance_score": p.get("relevance_score", prev.get("relevance_score", 0)),
            "relevance_reason": p.get("relevance_reason", prev.get("relevance_reason", "")),
            "source": p.get("source", ""),
        }
    save_paper_library(library)

    if llm_ok and _auto_pdf_enabled() and not (state.get("paper_fulltext_request") or []):
        max_n = _auto_pdf_max()
        picked = _select_papers_for_fulltext(llm, topic, friction_points, unique, max_n=max_n)
        if picked:
            by_id = {p.get("paper_id"): p for p in unique if isinstance(p, dict)}
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

    if unique and llm_ok:
        paper_text = "\n".join(
            f"- ({p.get('paper_id','')}) [{p.get('year','')}] {p.get('title','')}. {p.get('abstract','')[:180]}"
            for p in unique[:20]
        )
        fulltext_blocks = []
        for p in unique[:20]:
            pid = p.get("paper_id")
            if not pid:
                continue
            excerpt = (library.get(pid) or {}).get("fulltext_excerpt", "")
            if excerpt:
                fulltext_blocks.append(f"({pid})\n{excerpt[:1200]}")
        fulltext_text = "\n\n".join(fulltext_blocks)
        summary_resp = llm.invoke([
            SystemMessage(content=(
                "You are a critical research reviewer. Analyze the papers and surface implicit assumptions in the field.\n"
                "Only reason from the provided papers/excerpts. Do not invent papers, metrics, or results.\n"
                "Write in the same language as the user's latest message.\n"
                "Output:\n"
                "**State of the field** (2–3 sentences)\n"
                "**Implicit assumptions** (4–5; be as specific as possible)\n"
                "**Link to friction points** (which assumptions map to which friction points)"
            )),
            HumanMessage(content=(
                f"Topic: {topic}\n"
                f"Friction points: {friction_points}\n\n"
                f"Papers:\n{paper_text}"
                + (f"\n\nAvailable full-text excerpts (may be partial):\n{fulltext_text}" if fulltext_text else "")
            )),
        ])
        paper_summary = summary_resp.content
    else:
        paper_summary = ""

    return {
        **state,
        "phase": "idea_discovery",
        "keywords": keywords,
        "topic_anchors": anchors,
        "raw_papers": unique,
        "paper_rerank": paper_rerank,
        "paper_summary": paper_summary,
        "paper_library": library,
    }


def direction_synthesis_node(state: ResearchState) -> ResearchState:
    llm = _llm()
    topic = state.get("topic", "")
    friction_points = state.get("friction_points", [])
    motivation = state.get("motivation", "")
    taste = state.get("research_taste", "")
    keywords = state.get("keywords", []) or []
    anchors = state.get("topic_anchors", []) or []
    paper_summary = state.get("paper_summary", "") or ""
    insight = (state.get("human_feedback") or state.get("user_insight") or "").strip()

    primary_anchors = _dedupe_phrases([topic] + [str(x or "").strip() for x in (keywords or [])[:3]], max_n=6)
    try:
        import re
        is_zh = bool(re.search(r"[\u4e00-\u9fff]", f"{topic} {insight}"))
    except Exception:
        is_zh = False

    def _gen_directions(strict: bool):
        strict_line = (
            "Every string field (name, broken_assumption, reframed_problem, rationale, one_line) must include at least one anchor phrase verbatim.\n"
            if strict else
            "All directions must remain within the user input anchors; do not drift.\n"
        )
        return llm.invoke([
            SystemMessage(content=(
                "You are a top-tier research strategy advisor.\n"
                "Based on friction points and implicit assumptions, propose 3 research directions.\n\n"
                "Priority rule:\n"
                "- The Topic/primary objective dominates.\n"
                "- The Insight is secondary: treat it as inspiration/constraints/mechanisms, never as a new standalone task.\n\n"
                "Hard constraints:\n"
                "- Absolute rule: directions must be strictly constrained by the user's input anchors.\n"
                "- Absolute rule: each direction's name and one_line MUST include at least one PRIMARY_ANCHOR verbatim.\n"
                f"- {strict_line}"
                "- If you mention any sub-problem, it must be explicitly tied to improving the Topic objective.\n"
                "- For each direction, the rationale must cite at least one keyword and map it to a friction point.\n\n"
                "Requirements:\n"
                "1) Each direction breaks a different assumption; directions must be truly independent\n"
                "2) Each direction directly addresses at least one friction point\n"
                "3) Consider the user's taste and constraints\n"
                "4) No “add module X to baseline Y” mashups; each direction must be disruptive\n\n"
                "Output a strict JSON array of 3 elements, each with:\n"
                "name, broken_assumption, reframed_problem, rationale, one_line\n"
                "Use the same language as the user's latest message for all string fields."
            )),
            HumanMessage(content=json.dumps({
                "topic": topic,
                "insight_secondary": insight,
                "friction_points": friction_points,
                "motivation": motivation,
                "research_taste": taste,
                "keywords": keywords,
                "primary_anchors_must_use": primary_anchors,
                "anchors_must_use": anchors,
                "paper_summary": paper_summary[:800],
            }, ensure_ascii=False)),
        ])

    direction_resp = _gen_directions(strict=False)
    content = direction_resp.content
    match = re.search(r'\[.*\]', content, re.DOTALL)
    if match:
        try:
            directions = json.loads(match.group())
        except Exception:
            directions = []
    else:
        directions = []

    if not _validate_directions(directions, anchors, primary_anchors):
        print("  [WARNING] Candidate directions are not tightly coupled to the Topic; regenerating with strict anchoring.")
        direction_resp = _gen_directions(strict=True)
        content = direction_resp.content
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                directions = json.loads(match.group())
            except Exception:
                directions = []
        if not _validate_directions(directions, anchors, primary_anchors):
            print("  [WARNING] Still not coupled after strict regeneration; falling back to Topic-anchored directions.")
            base = str(topic or "").strip() or "User topic"
            fps = [str(x).strip() for x in (friction_points or []) if str(x).strip()]
            fps = fps[:3]
            a = [str(x).strip() for x in (primary_anchors or []) if str(x).strip()]
            if not a:
                a = [base]
            while len(a) < 3:
                a.append(a[-1])
            directions = []
            for i in range(3):
                ai = a[i]
                fi = fps[i] if i < len(fps) else ""
                directions.append({
                    "name": f"{base} - Candidate direction {i + 1} ({ai})",
                    "broken_assumption": f"(auto-fallback) Direction generation violated constraints. Topic={base}; PrimaryAnchor={ai}.",
                    "reframed_problem": f"{base} (PrimaryAnchor={ai})",
                    "rationale": (f"{base} (PrimaryAnchor={ai})" + (f"; Friction={fi}" if fi else "")),
                    "one_line": f"{base}: direction {i + 1} around {ai}" + (f" ({fi})" if fi else ""),
                })

    return {
        **state,
        "phase": "idea_discovery",
        "candidate_directions": directions,
        "raw_ideas": [str(d.get("one_line", "") or "") for d in (directions or [])],
    }


def idea_refine_node(state: ResearchState) -> ResearchState:
    llm = _llm(temperature=0.5)
    direction = state.get("selected_direction") or {}
    if not direction and state.get("candidate_directions"):
        direction = state["candidate_directions"][0]

    resp = llm.invoke([
        SystemMessage(content=(
            "Expand the selected research direction into a refined description.\n"
            "Write in the same language as the user's latest message.\n"
            "Format:\n"
            "**Problem**: concrete problem (with data context)\n"
            "**Broken Assumption**: what assumption is broken and why it fails\n"
            "**Key Insight**: the core insight after breaking the assumption\n"
            "**Proposed Approach**: high-level approach (no need for low-level details)\n"
            "**Why Now**: why this is the right time to do it"
        )),
        HumanMessage(content=(
            f"Selected direction:\n{json.dumps(direction, ensure_ascii=False, indent=2)}\n\n"
            f"Motivation: {state.get('motivation', '')}\n"
            f"Literature context: {state.get('paper_summary', '')[:400]}"
        )),
    ])

    refined = {**direction, "refined_description": resp.content}
    return {
        **state,
        "selected_direction": refined,
        "selected_idea": resp.content,
    }
