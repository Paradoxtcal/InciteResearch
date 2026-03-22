"""
InciteResearch — CLI
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime, timezone

UI = {
    "you": "You",
    "topic_prompt": "What research direction are you thinking about?",
    "intro": "Let's start with what you're trying to do. It doesn't need to be clear yet.",
    "searching": "Searching papers + mining assumptions...",
    "direction_pick": "Which direction do you pick? (1/2/3, or describe your own idea)",
    "papers_header": "Candidate papers (title/abstract). Full text is skipped by default; pick only if needed:",
    "pick_fulltext": "Which papers should we read full text for (PDF only)? e.g., 1,3 / all / Enter to skip",
    "pick_fulltext_before_code": "Before generating core code, which papers should we read full text for? e.g., 2,5 / all / Enter to skip",
    "proposal_ready": "A research proposal draft is ready.",
    "proposal_hint": "(Type anything to approve, or describe what to change.)",
    "building_story": "Building the story arc...",
    "generating_code": "Generating core algorithm code...",
    "done_files": "✅  Done! Files saved:",
    "direction_selected": "✓ Selected direction: {name}",
    "need_llm_config": "❌ Configure at least one LLM provider: GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY",
    "rate_limit_wait": "❌ LLM rate-limited/quota exceeded: wait about {s:.0f}s and retry.",
    "rate_limit_generic": "❌ LLM rate-limited/quota exceeded: retry later or switch model/key.",
    "resume_hint": "   You can run with --resume {sid} to continue the same session.",
    "directions_found": "Found 3 candidate directions (each breaks a different assumption):",
    "broken_assumption": "Broken assumption: {v}",
    "one_line": "One-liner: {v}",
    "file_core": "  · core_algorithm.py    ← hand to Cursor / Claude Code",
    "file_proposal": "  · research_proposal.md ← story + checks + proposal",
    "file_cursor": "  · cursor_prompt.txt    ← paste into Cursor",
    "file_state": "  · research_state.json  ← resume state",
}


def _is_internal_marker(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if s.startswith("[") and s.endswith("]"):
        low = s.lower()
        if ("researcher profile" in low) or ("idea discovery" in low):
            return True
    return False


def _safe_json_state(state: dict) -> dict:
    if not isinstance(state, dict):
        return {}
    def _coerce(x):
        if isinstance(x, (str, int, float, bool, type(None))):
            return x
        if isinstance(x, dict):
            out = {}
            for kk, vv in x.items():
                if kk == "metadata":
                    continue
                out[str(kk)] = _coerce(vv)
            return out
        if isinstance(x, (list, tuple)):
            return [_coerce(i) for i in x]
        msg_type = getattr(x, "type", None)
        content = getattr(x, "content", None)
        if msg_type is not None and content is not None:
            return {"type": str(msg_type), "content": str(content)}
        role = getattr(x, "role", None)
        if role is not None and content is not None:
            return {"role": str(role), "content": str(content)}
        return str(x)

    clean = {k: _coerce(v) for k, v in state.items() if k != "metadata"}
    clean["_autosaved_at_utc"] = datetime.now(timezone.utc).isoformat()
    return clean


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _autosave_root() -> Path:
    base = (os.environ.get("RESEARCH_AGENT_AUTOSAVE_DIR") or "outputs").strip()
    return Path(base)


def _snapshot_write(session_id: str, kind: str, ext: str, content: str) -> None:
    try:
        if not content:
            return
        sid = (session_id or "unknown").strip() or "unknown"
        out_dir = _autosave_root() / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        pattern = f"*_{kind}.{ext}"
        latest = None
        for p in sorted(out_dir.glob(pattern)):
            latest = p
        if latest and latest.exists():
            try:
                if latest.read_text(encoding="utf-8") == content:
                    return
            except Exception:
                pass

        name = f"{_utc_stamp()}_{kind}.{ext}"
        (out_dir / name).write_text(content, encoding="utf-8")
    except Exception:
        pass


def _write_if_changed(path: str, content: str) -> None:
    try:
        p = Path(path)
        if p.exists():
            prev = p.read_text(encoding="utf-8")
            if prev == content:
                return
        p.write_text(content, encoding="utf-8")
    except Exception:
        pass


def _autosave_partial(state: dict) -> None:
    if not isinstance(state, dict):
        return
    session_id = str(state.get("session_id") or "")

    proposal = state.get("research_proposal", "") or ""
    story = state.get("story_arc", "") or ""
    necessity = state.get("method_necessity_check", "") or ""
    parts = [story, necessity, proposal]
    parts = [p.strip() for p in parts if (p or "").strip()]
    if parts:
        proposal_text = "\n\n---\n\n".join(parts) + "\n"
        _write_if_changed("research_proposal.md", proposal_text)
        _snapshot_write(session_id, "research_proposal", "md", proposal_text)

    code = state.get("core_modification", "") or ""
    if code.strip():
        header = (
            f"# InciteResearch — Core Algorithm\n"
            f"# Direction: {state.get('selected_direction', {}).get('name', '')}\n"
            f"# Broken assumption: {state.get('broken_assumption', '')}\n\n"
        )
        code_text = header + code
        _write_if_changed("core_algorithm.py", code_text)
        _snapshot_write(session_id, "core_algorithm", "py", code_text)

    integration = (state.get("metadata", {}) or {}).get("integration_prompt", "") or ""
    if integration.strip():
        _write_if_changed("cursor_prompt.txt", integration)
        _snapshot_write(session_id, "cursor_prompt", "txt", integration)

    try:
        state_text = json.dumps(_safe_json_state(state), ensure_ascii=False, indent=2) + "\n"
    except Exception:
        state_text = json.dumps({"_autosaved_at_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2) + "\n"
    _write_if_changed("research_state.json", state_text)


def _autosave_from_graph(graph, config) -> None:
    try:
        st = graph.get_state(config).values
    except Exception:
        return
    _autosave_partial(st)


def _has_real_key(v: str | None) -> bool:
    if not v:
        return False
    s = v.strip()
    if not s:
        return False
    low = s.lower()
    if "your-key" in low or "your key" in low:
        return False
    if low in ("changeme", "replace-me", "replace_with_your_key", "todo"):
        return False
    return True


def _handle_rate_limit_error(err: Exception, config: dict | None = None) -> bool:
    msg = str(err)
    low = msg.lower()
    if ("resource_exhausted" not in low) and ("too many requests" not in low) and (" 429 " not in f" {low} ") and ("quota exceeded" not in low):
        return False
    import re, time
    retry_s = None
    m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
    if m:
        try:
            retry_s = float(m.group(1))
        except Exception:
            retry_s = None
    if retry_s is None:
        m = re.search(r"retryDelay['\"]?\s*:\s*['\"]([0-9]+)s['\"]", msg)
        if m:
            try:
                retry_s = float(m.group(1))
            except Exception:
                retry_s = None
    sid = None
    if isinstance(config, dict):
        sid = (config.get("configurable") or {}).get("thread_id")
    if retry_s:
        print("\n" + UI["rate_limit_wait"].format(s=retry_s))
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        time.sleep(min(120.0, retry_s + 0.5))
    else:
        print("\n" + UI["rate_limit_generic"])
        if sid:
            print(UI["resume_hint"].format(sid=sid))
    return True


def _is_approved(v: str | None) -> bool:
    return (v or "").strip().lower() in ("approved", "yes", "ok")


def _handle_llm_config_error(err: Exception, config: dict | None = None) -> bool:
    msg = str(err or "")
    if not msg:
        return False
    low = msg.lower()
    sid = None
    if isinstance(config, dict):
        sid = (config.get("configurable") or {}).get("thread_id")
    if "invalid gemini_api_key" in low:
        print("\n❌ Invalid GEMINI_API_KEY. Update GEMINI_API_KEY in your environment (or .env) and retry.")
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if "invalid openai_api_key" in low:
        print("\n❌ Invalid OPENAI_API_KEY. Update OPENAI_API_KEY in your environment (or .env) and retry.")
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if "invalid anthropic_api_key" in low:
        print("\n❌ Invalid ANTHROPIC_API_KEY. Update ANTHROPIC_API_KEY in your environment (or .env) and retry.")
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if "gemini_api_key is required" in low:
        print("\n❌ GEMINI_API_KEY is required when provider=gemini. Set it and retry, or unset RESEARCH_AGENT_LLM_PROVIDER to auto-pick another provider.")
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if "openai_api_key is required" in low:
        print("\n❌ OPENAI_API_KEY is required when provider=openai. Set it and retry, or switch provider.")
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if "anthropic_api_key is required" in low:
        print("\n❌ ANTHROPIC_API_KEY is required when provider=anthropic. Set it and retry, or switch provider.")
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if "ollama is not reachable" in low:
        print("\n❌ " + msg)
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if "no llm provider available" in low or "missing dependency" in low:
        print("\n❌ " + msg)
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if "network error while calling llm provider" in low:
        print("\n❌ " + msg)
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if (
        "server disconnected without sending a response" in low
        or "remoteprotocolerror" in low
        or "readtimeout" in low
        or "connecttimeout" in low
        or "timed out" in low
    ):
        print("\n❌ Network error while calling the LLM provider. Please retry.")
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    if ("not_found" in low or "not found" in low) and "model" in low:
        print("\n❌ " + msg)
        if sid:
            print(UI["resume_hint"].format(sid=sid))
        return True
    return False


def check_env():
    try:
        from dotenv import load_dotenv
        if Path(".env").exists():
            load_dotenv(".env", override=True)
        elif Path(".env.example").exists():
            load_dotenv(".env.example", override=False)
    except Exception:
        pass

    provider = (os.environ.get("RESEARCH_AGENT_LLM_PROVIDER") or "").strip().lower()
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if provider in ("gemini", "google") and not _has_real_key(gemini_key):
        print("❌ GEMINI_API_KEY is required when provider=gemini. Set a valid key in .env or your shell.")
        sys.exit(1)
    if provider in ("openai",) and not _has_real_key(openai_key):
        print("❌ OPENAI_API_KEY is required when provider=openai. Set a valid key in .env or your shell.")
        sys.exit(1)
    if provider in ("anthropic", "claude") and not _has_real_key(anthropic_key):
        print("❌ ANTHROPIC_API_KEY is required when provider=anthropic. Set a valid key in .env or your shell.")
        sys.exit(1)

    has_any_key = any(
        _has_real_key(os.environ.get(k))
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not has_any_key and not provider:
        print("❌ Configure at least one LLM provider: GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY")
        sys.exit(1)


def run():
    print("\n" + "="*58)
    print("  InciteResearch")
    print("  Your Phase-0 AI research partner: turn vague instincts into core algorithms")
    print("="*58)
    print(f"\n{UI['intro']}\n")
    provider = (os.environ.get("RESEARCH_AGENT_LLM_PROVIDER") or "").strip().lower() or "auto"
    model = (os.environ.get("GEMINI_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("ANTHROPIC_MODEL") or os.environ.get("OLLAMA_MODEL") or "").strip()
    if model:
        print(f"LLM: provider={provider} model={model}\n")

    topic = input(UI["topic_prompt"] + "\n> ").strip() or "TBD"

    from agents.orchestrator import start_dialogue, build_graph
    try:
        graph, config, q = start_dialogue(topic, user_language="Auto")
    except Exception as e:
        if _handle_llm_config_error(e):
            return
        raise
    _autosave_from_graph(graph, config)
    print(f"\nAI: {q}\n")
    last_ai = q or ""
    last_saved = {"proposal": "", "story": "", "dirs": 0}

    while True:
        try:
            ans = input(f"{UI['you']}: ").strip()
        except KeyboardInterrupt:
            _autosave_from_graph(graph, config)
            print("\n")
            break
        if not ans:
            continue

        graph.update_state(config, {
            "human_feedback": ans,
            "messages": [{"role": "user", "content": ans}],
        })
        _autosave_from_graph(graph, config)

        stopped_at = None
        while True:
            try:
                for event in graph.stream(None, config, stream_mode="values"):
                    msgs = event.get("messages", [])
                    for m in reversed(msgs):
                        if isinstance(m, dict):
                            if m.get("role") == "assistant":
                                c = m.get("content", "")
                                if c and c != last_ai and not _is_internal_marker(c):
                                    print(f"\nAI: {c}\n")
                                    last_ai = c
                                break
                            continue
                        msg_type = getattr(m, "type", "")
                        if msg_type == "ai":
                            c = getattr(m, "content", "") or ""
                            if c and c != last_ai and not _is_internal_marker(str(c)):
                                print(f"\nAI: {c}\n")
                                last_ai = c
                            break

                    if event.get("friction_points") and not event.get("candidate_directions"):
                        print(f"\n{UI['searching']}\n")

                    if event.get("candidate_directions") and not event.get("selected_direction"):
                        stopped_at = "direction_select"
                        _print_directions(event["candidate_directions"])
                        if isinstance(event.get("candidate_directions"), list):
                            last_saved["dirs"] = max(last_saved["dirs"], len(event.get("candidate_directions") or []))
                        _autosave_from_graph(graph, config)
                        break

                    if event.get("research_proposal") and not event.get("core_modification"):
                        feedback = event.get("human_feedback") or graph.get_state(config).values.get("human_feedback", "")
                        if not _is_approved(feedback):
                            stopped_at = "proposal_review"
                            print("\n" + "─"*50)
                            print(UI["proposal_ready"] + "\n")
                            print(event["research_proposal"][:800])
                            print("\n" + UI["proposal_hint"])
                            rp = event.get("research_proposal") or ""
                            if rp and rp != last_saved["proposal"]:
                                last_saved["proposal"] = rp
                                _autosave_from_graph(graph, config)
                            break

                    if event.get("core_modification"):
                        stopped_at = "done"
                        _save_outputs(event)
                        break
                break
            except Exception as e:
                _autosave_from_graph(graph, config)
                if _handle_rate_limit_error(e, config=config):
                    continue
                if _handle_llm_config_error(e, config=config):
                    return
                raise

        if stopped_at == "direction_select":
            directions = graph.get_state(config).values.get("candidate_directions", [])
            choice = input("\n" + UI["direction_pick"] + "\n> ").strip()
            if choice in ("1", "2", "3") and len(directions) >= int(choice):
                sel = directions[int(choice) - 1]
            else:
                sel = {"name": choice, "broken_assumption": choice,
                       "one_line": choice, "rationale": "", "reframed_problem": ""}
            raw_papers = graph.get_state(config).values.get("raw_papers", []) or []
            if raw_papers:
                _print_papers(raw_papers[:10])
                pick = input("\n" + UI["pick_fulltext"] + "\n> ").strip()
                paper_ids = _parse_paper_picks(pick, raw_papers[:10])
            else:
                paper_ids = []
            graph.update_state(config, {
                "selected_direction": sel,
                "paper_fulltext_request": paper_ids,
                "human_feedback": "approved",
            })
            _autosave_from_graph(graph, config)
            print("\n" + UI["direction_selected"].format(name=sel.get("name", choice)))
            print(UI["building_story"] + "\n")

        elif stopped_at == "proposal_review":
            raw_papers = graph.get_state(config).values.get("raw_papers", []) or []
            if raw_papers:
                _print_papers(raw_papers[:10])
                pick = input("\n" + UI["pick_fulltext_before_code"] + "\n> ").strip()
                paper_ids = _parse_paper_picks(pick, raw_papers[:10])
            else:
                paper_ids = []
            try:
                feedback = input(f"{UI['you']}: ").strip() or "approved"
            except KeyboardInterrupt:
                print("\n")
                break
            graph.update_state(config, {"human_feedback": feedback, "paper_fulltext_request": paper_ids})
            _autosave_from_graph(graph, config)
            print("\n" + UI["generating_code"] + "\n")

        elif stopped_at == "done":
            break

        while True:
            try:
                for event in graph.stream(None, config, stream_mode="values"):
                    if event.get("core_modification"):
                        _save_outputs(event)
                        stopped_at = "done"
                        break
                    if event.get("candidate_directions") and not event.get("selected_direction"):
                        stopped_at = "direction_select"
                        _print_directions(event["candidate_directions"])
                        break
                    if event.get("research_proposal") and not event.get("core_modification"):
                        feedback = event.get("human_feedback") or graph.get_state(config).values.get("human_feedback", "")
                        if not _is_approved(feedback):
                            stopped_at = "proposal_review"
                            break
                    if event.get("story_arc"):
                        sa = event.get("story_arc") or ""
                        if sa and sa != last_saved["story"]:
                            last_saved["story"] = sa
                            _autosave_from_graph(graph, config)
                break
            except Exception as e:
                _autosave_from_graph(graph, config)
                if _handle_rate_limit_error(e, config=config):
                    continue
                if _handle_llm_config_error(e, config=config):
                    return
                raise



def _print_directions(directions):
    print("\n" + "─"*50)
    print(UI["directions_found"] + "\n")
    for i, d in enumerate(directions, 1):
        print(f"{i}. {d.get('name', f'Direction {i}')}") 
        print("   " + UI["broken_assumption"].format(v=d.get("broken_assumption", "")))
        print("   " + UI["one_line"].format(v=d.get("one_line", "")) + "\n")

def _print_papers(papers):
    print("\n" + "─"*50)
    print(UI["papers_header"] + "\n")
    for i, p in enumerate(papers, 1):
        title = p.get("title", "")
        year = p.get("year", "")
        src = p.get("source", "")
        pdf_mark = " PDF" if p.get("pdf_url") else ""
        score = p.get("relevance_score", None)
        reason = (p.get("relevance_reason", "") or "").strip()
        score_txt = f" | score={score}" if score is not None else ""
        print(f"{i}. [{year}] {title} ({src}{pdf_mark}{score_txt})")
        if reason:
            print(f"   reason: {reason}")


def _parse_paper_picks(text: str, papers: list[dict]) -> list[str]:
    if not text:
        return []
    t = text.strip().lower()
    if t in ("all", "*"):
        picks = []
        for p in papers:
            if not p.get("pdf_url"):
                continue
            pid = p.get("paper_id")
            if pid:
                picks.append(pid)
        return picks
    items = [x.strip() for x in text.split(",") if x.strip()]
    picks = []
    for it in items:
        if it.isdigit():
            idx = int(it)
            if 1 <= idx <= len(papers):
                pid = papers[idx - 1].get("paper_id")
                if pid:
                    picks.append(pid)
    return picks


def _jsonify(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        out = {}
        for k, vv in v.items():
            try:
                kk = str(k)
            except Exception:
                kk = repr(k)
            out[kk] = _jsonify(vv)
        return out
    if isinstance(v, (list, tuple, set)):
        return [_jsonify(x) for x in v]
    try:
        model_dump = getattr(v, "model_dump", None)
        if callable(model_dump):
            return _jsonify(model_dump())
    except Exception:
        pass
    try:
        to_dict = getattr(v, "dict", None)
        if callable(to_dict):
            return _jsonify(to_dict())
    except Exception:
        pass
    try:
        return {"__type__": type(v).__name__, "__repr__": repr(v)}
    except Exception:
        return {"__type__": type(v).__name__}


def _save_outputs(state: dict):
    print("\n" + "="*58)
    print(UI["done_files"])

    code = state.get("core_modification", "")
    with open("core_algorithm.py", "w", encoding="utf-8") as f:
        f.write(f"# InciteResearch — Core Algorithm\n")
        f.write(f"# Direction: {state.get('selected_direction', {}).get('name', '')}\n")
        f.write(f"# Broken assumption: {state.get('broken_assumption', '')}\n\n")
        f.write(code)
    print(UI["file_core"])

    proposal = state.get("research_proposal", "")
    story = state.get("story_arc", "")
    necessity = state.get("method_necessity_check", "")
    with open("research_proposal.md", "w", encoding="utf-8") as f:
        parts = [story, necessity, proposal]
        parts = [p.strip() for p in parts if (p or "").strip()]
        f.write("\n\n---\n\n".join(parts) + ("\n" if parts else ""))
    print(UI["file_proposal"])

    integration = state.get("metadata", {}).get("integration_prompt", "")
    if integration:
        with open("cursor_prompt.txt", "w", encoding="utf-8") as f:
            f.write(integration)
        print(UI["file_cursor"])

    clean = _jsonify(state)
    with open("research_state.json", "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(UI["file_state"])
    print("="*58 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress-pdfs", action="store_true")
    parser.add_argument("--pdf-dir", default=".")
    parser.add_argument("--resume", help="Resume session ID")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        if Path(".env").exists():
            load_dotenv(".env", override=True)
        elif Path(".env.example").exists():
            load_dotenv(".env.example", override=False)
    except Exception:
        pass

    if args.compress_pdfs:
        from scripts.compress_pdfs import compress_and_zip
        compress_and_zip(input_dir=args.pdf_dir)
        sys.exit(0)

    check_env()
    run()
