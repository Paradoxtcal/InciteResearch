import threading
import os
import re
import time
import argparse
import json
import pickle
import statistics
import yaml
from copy import copy
from pathlib import Path
from datetime import date

from .agents import *

# Legacy global for optional AgentRxiv parallel mode (set in __main__)
GLOBAL_AGENTRXIV: type | None = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def infer_research_topic_from_proposal_md(
    proposal_text: str,
    model_str: str,
    *,
    openai_api_key=None,
    gemini_api_key=None,
    anthropic_api_key=None,
) -> str:
    """One English line: concrete method / empirical approach, for agents and literature queries."""
    from . import inference

    excerpt = (proposal_text or "").strip()
    if len(excerpt) > 14000:
        excerpt = excerpt[:14000] + "\n[... truncated ...]"
    system_prompt = (
        "You read a research proposal and output exactly one short line in English. "
        "The line must name the concrete method, empirical approach, or system the authors "
        "intend to build, train, or evaluate — not a vague theme, field label, or paper title alone. "
        "No markdown, no quotation marks, no numbering: plain text, at most 220 characters."
    )
    user_prompt = "Proposal text:\n\n" + excerpt
    raw = inference.query_model(
        model_str=model_str,
        prompt=user_prompt,
        system_prompt=system_prompt,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        anthropic_api_key=anthropic_api_key,
        print_cost=False,
    )
    line = (raw or "").strip().splitlines()[0].strip() if raw else ""
    if not line:
        raise ValueError("Could not derive a research topic line from the proposal (empty model output).")
    return line[:500]


def resolve_lab_llm_model_id() -> str:
    """
    research_lab inference model id: same basis as the main InciteResearch CLI (.env).
    Uses RESEARCH_AGENT_LLM_PROVIDER when set; otherwise auto-picks from available keys
    (Gemini, OpenAI, Anthropic, DeepSeek) in the same order as utils.llm_client.get_llm.
    """
    provider = (os.getenv("RESEARCH_AGENT_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "").strip().lower()

    def _has_g() -> bool:
        return bool((os.getenv("GEMINI_API_KEY") or "").strip())

    def _has_o() -> bool:
        return bool((os.getenv("OPENAI_API_KEY") or "").strip())

    def _has_a() -> bool:
        return bool((os.getenv("ANTHROPIC_API_KEY") or os.getenv("OFOX_API_KEY") or "").strip())

    def _has_d() -> bool:
        return bool((os.getenv("DEEPSEEK_API_KEY") or "").strip())

    def _anthropic_inference_id(raw: str) -> str:
        s = raw.strip().lower().replace("_", "-")
        if "claude-4-6" in s or "sonnet-4.6" in s:
            return "claude-4-6-sonnet"
        if s.startswith("claude-3-5") or s.startswith("claude-3.5"):
            return "claude-3-5-sonnet"
        return raw.strip()

    if provider in ("gemini", "google"):
        if not _has_g():
            raise ValueError("RESEARCH_AGENT_LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        return (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash").strip()
    if provider in ("openai",):
        if not _has_o():
            raise ValueError("RESEARCH_AGENT_LLM_PROVIDER=openai requires OPENAI_API_KEY")
        return (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()
    if provider in ("deepseek", "deep-seek"):
        if not _has_d() and not _has_o():
            raise ValueError("RESEARCH_AGENT_LLM_PROVIDER=deepseek requires DEEPSEEK_API_KEY or OPENAI_API_KEY (compatible gateway).")
        return (os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
    if provider in ("anthropic", "claude"):
        if not _has_a():
            raise ValueError("RESEARCH_AGENT_LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY (or OFOX_API_KEY)")
        mid = (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022").strip()
        return _anthropic_inference_id(mid)
    if provider in ("ollama",):
        return (os.getenv("OLLAMA_MODEL") or "llama3").strip()

    if _has_g():
        return (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash").strip()
    if _has_o():
        return (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()
    if _has_a():
        mid = (os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022").strip()
        return _anthropic_inference_id(mid)
    if _has_d():
        return (os.getenv("OPENAI_MODEL") or "deepseek-chat").strip()
    raise ValueError(
        "research_lab needs the same LLM credentials as main.py: set GEMINI_API_KEY, OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY (or OFOX), or DEEPSEEK_API_KEY, plus the matching *_MODEL / OPENAI_MODEL if needed."
    )


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = val
    return out


def lit_entries_from_research_state(raw_papers: list, need: int) -> list[dict]:
    """
    Map InciteResearch `raw_papers` records into laboratory `lit_review`-compatible
    dicts so the agent loop can be skipped when enough items are available.
    """
    if not raw_papers or need <= 0:
        return []
    ranked: list[dict] = [p for p in raw_papers if isinstance(p, dict)]

    def _score(p: dict) -> int:
        try:
            return int(p.get("relevance_score") or 0)
        except Exception:
            return 0

    ranked.sort(key=_score, reverse=True)
    out: list[dict] = []
    for p in ranked:
        if len(out) >= need:
            break
        arxiv_id = (p.get("arxiv_id") or "").strip()
        if not arxiv_id:
            pid = str(p.get("paper_id") or p.get("title") or "unknown")
            pid = pid.replace("\n", " ")[:160]
            arxiv_id = f"incite:{pid}"
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        reason = (p.get("relevance_reason") or "").strip()
        authors = p.get("authors") if isinstance(p.get("authors"), list) else []
        auth_str = ", ".join(str(a) for a in authors[:16]) if authors else ""
        # full_text / summary feed "Current Literature Review" in agent prompts (not just titles).
        full_text = (abstract[:20000] if abstract else "")[:20000]
        summary_parts = []
        if title:
            summary_parts.append(title)
        if auth_str:
            summary_parts.append(f"Authors: {auth_str}")
        if abstract:
            summary_parts.append(abstract)
        if reason:
            summary_parts.append(f"InciteResearch relevance note: {reason}")
        summary = "\n\n".join(summary_parts).strip()[:24000]
        pid = str(p.get("paper_id") or "").strip()
        out.append({
            "paper_id": pid,
            "arxiv_id": arxiv_id,
            "title": title[:500] if title else "",
            "authors": auth_str[:800],
            "relevance_reason": reason[:4000],
            "abstract_text": full_text,
            "full_text": full_text,
            "summary": summary[:16000],
            "pdf_excerpt": "",
        })
    return out


def enrich_lit_entries_with_pdf_excerpts(
    entries: list[dict],
    state: dict | None,
    *,
    enabled: bool,
    max_pages: int,
    max_chars: int,
) -> list[dict]:
    """
    Match pre-filled review rows to InciteResearch state: use paper_library.fulltext_excerpt
    when present; otherwise call tools.literature_tools.fetch_fulltext_excerpt (same as main.py),
    with page/char caps to limit prompt size. Abstracts in entries are always kept.
    """
    if not enabled or not entries:
        return [dict(x) for x in entries]
    if not state:
        return [dict(x) for x in entries]
    try:
        from tools.literature_tools import fetch_fulltext_excerpt
    except Exception as exc:
        print(f"[literature-prefill] pdf-excerpt skipped (import tools.literature_tools failed): {exc}")
        return [dict(x) for x in entries]

    lib = state.get("paper_library") or {}
    if not isinstance(lib, dict):
        lib = {}
    raw_by_pid: dict[str, dict] = {}
    for p in state.get("raw_papers") or []:
        if isinstance(p, dict) and p.get("paper_id"):
            raw_by_pid[str(p["paper_id"]).strip()] = p

    out: list[dict] = []
    for e in entries:
        ne = dict(e)
        pid = str(ne.get("paper_id") or "").strip()
        if not pid:
            out.append(ne)
            continue
        lib_ent = lib.get(pid) if isinstance(lib.get(pid), dict) else {}
        cached = (lib_ent.get("fulltext_excerpt") or "").strip()
        if cached:
            ne["pdf_excerpt"] = cached[: max_chars] if len(cached) > max_chars else cached
            print(
                f"[literature-prefill] paper_library fulltext_excerpt: {pid[:20]}… → {len(ne['pdf_excerpt'])} chars (cap {max_chars})"
            )
            out.append(ne)
            continue
        paper = raw_by_pid.get(pid)
        if not paper:
            out.append(ne)
            continue
        try:
            res = fetch_fulltext_excerpt(
                paper,
                max_pages=max_pages,
                max_chars=max_chars,
            )
        except Exception as ex:
            print(f"[literature-prefill] fetch_fulltext_excerpt failed for {pid[:20]}…: {ex}")
            out.append(ne)
            continue
        excerpt = (res.get("text_excerpt") or "").strip()
        if excerpt:
            ne["pdf_excerpt"] = excerpt
            print(
                f"[literature-prefill] PDF excerpt fetched: {pid[:20]}… → {len(excerpt)} chars "
                f"(pages≤{max_pages})"
            )
        else:
            print(
                f"[literature-prefill] no PDF text for {pid[:20]}… "
                f"(pdf_url={bool((paper.get('pdf_url') or '').strip())})"
            )
        out.append(ne)
    return out


def incite_state_context_notes(state: dict | None, lit_target: int) -> list[dict]:
    """Extra task notes so --refine-method agents see main-session topic, method lock, and paper list."""
    if not state:
        return []
    notes: list[dict] = []
    sa = (state.get("story_arc") or "").strip()
    if sa:
        if len(sa) > 12000:
            sa = sa[:12000] + "\n[... truncated ...]"
        notes.append({
            "phases": ["literature review", "plan formulation"],
            "note": "InciteResearch main session (story_arc):\n" + sa,
        })
    lm = state.get("locked_method_spec")
    if isinstance(lm, dict):
        lm = json.dumps(lm, ensure_ascii=False, indent=2)
    elif lm is not None:
        lm = str(lm).strip()
    else:
        lm = ""
    if lm:
        if len(lm) > 12000:
            lm = lm[:12000] + "\n[... truncated ...]"
        notes.append({
            "phases": ["plan formulation"],
            "note": "InciteResearch locked_method_spec:\n" + lm,
        })
    raw = [p for p in (state.get("raw_papers") or []) if isinstance(p, dict)]
    if raw:
        def _sc(p: dict) -> int:
            try:
                return int(p.get("relevance_score") or 0)
            except Exception:
                return 0

        raw = sorted(raw, key=_sc, reverse=True)
        show = max(int(lit_target or 3), 8)
        lines = []
        for i, p in enumerate(raw[:show], 1):
            t = (p.get("title") or "?").replace("\n", " ")[:220]
            lines.append(f"{i}. {t}")
        notes.append({
            "phases": ["literature review", "plan formulation"],
            "note": (
                "Index of papers in research_state.json (by relevance). "
                "The prompt block “Current Literature Review” contains each item’s abstract from raw_papers plus "
                "an optional PDF excerpt (first pages, capped in literature-prefill) when a PDF was available — not title-only.\n"
                + "\n".join(lines)
            ),
        })
    return notes


def _plan_length_saturation_accept(lengths: list, gate: dict, min_plan_chars: int) -> tuple:
    """
    Stop when PLAN length has plateaued near the running max: the last two absolute
    step sizes are small relative to earlier |Δ| in this phase (median step scale), and
    the current length sits within a margin of the historical peak (allows mild shrink).
    Thresholds are derived from the sequence and peak only — no YAML stall/peak fractions.
    """
    if not gate.get("saturation_enable", True):
        return False, ""
    # Need two recent steps and at least one earlier transition to define "typical" step size.
    if len(lengths) < 4:
        return False, ""
    cur = int(lengths[-1])
    peak = int(max(lengths))
    lo_req = max(
        int(gate["convergence_min_chars"]),
        int(min_plan_chars * float(gate["accept_as_converged_fraction_of_min_plan"])),
    )
    if cur < lo_req or peak <= 0:
        return False, ""
    r1 = abs(int(lengths[-1]) - int(lengths[-2]))
    r2 = abs(int(lengths[-2]) - int(lengths[-3]))
    prior_steps = [
        abs(int(lengths[i]) - int(lengths[i - 1]))
        for i in range(1, len(lengths) - 2)
    ]
    if not prior_steps:
        return False, ""
    med = float(statistics.median(prior_steps))
    # Typical edit size for this run; floor relative to peak so tiny-med runs still behave.
    scale = max(med, peak * 0.01, 1.0)
    stall_eps = 0.2
    if r1 > stall_eps * scale or r2 > stall_eps * scale:
        return False, ""
    # Stay within a modest band of the best length seen (not a user-tuned fraction).
    near_margin = min(peak * 0.08, max(scale * 0.25, peak * 0.02, 500.0))
    if (peak - cur) > near_margin:
        return False, ""
    return (
        True,
        f"plateau vs median prior step {med:.0f} (scale {scale:.0f}, peak {peak}, "
        f"r1={r1}, r2={r2}, near_margin={near_margin:.0f})",
    )


class LaboratoryWorkflow:
    def __init__(
        self,
        research_topic,
        openai_api_key,
        max_steps=100,
        num_papers_lit_review=5,
        agent_model_backbone=None,
        notes=None,
        human_in_loop_flag=None,
        compile_pdf=True,
        mlesolver_max_steps=3,
        papersolver_max_steps=5,
        paper_index=0,
        except_if_fail=False,
        parallelized=False,
        lab_dir=None,
        lab_index=0,
        agentRxiv=False,
        agentrxiv_papers=5,
        min_plan_chars=None,
        prefilled_lit_review_entries=None,
        use_progress_bar=True,
        arxiv_summary_count=None,
        plan_formulation_gate=None,
        arxiv_paper_exp_time=None,
    ):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (str or dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        """
        if agent_model_backbone is None:
            raise ValueError("agent_model_backbone is required")
        if min_plan_chars is None:
            raise ValueError("min_plan_chars is required")
        if arxiv_summary_count is None:
            raise ValueError("arxiv_summary_count is required")
        if plan_formulation_gate is None:
            raise ValueError("plan_formulation_gate is required")
        if arxiv_paper_exp_time is None:
            raise ValueError("arxiv_paper_exp_time is required")
        notes = notes or []
        self.agentRxiv = agentRxiv
        self.max_prev_papers = 10
        self.parallelized = parallelized
        self.notes = notes
        self.lab_dir = lab_dir
        self.lab_index = lab_index
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.paper_index = paper_index
        self.openai_api_key = openai_api_key
        self.except_if_fail = except_if_fail
        self.research_topic = research_topic
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review

        self.print_cost = True
        self.review_override = True # should review be overridden?
        self.review_ovrd_steps = 0 # review steps so far
        self.arxiv_paper_exp_time = int(arxiv_paper_exp_time)
        self.reference_papers = list()

        ##########################################
        ####### COMPUTE BUDGET PARAMETERS ########
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0 # num steps to take if overridden
        self.arxiv_num_summaries = int(arxiv_summary_count)
        # Reject underspecified plans and force long-form proposals.
        self.min_plan_chars = int(min_plan_chars)
        self.prefilled_lit_review_entries = list(prefilled_lit_review_entries or [])
        self._plan_submission_lengths = []
        self._plan_last_body: str | None = None
        self.use_progress_bar = bool(use_progress_bar)
        self.num_agentrxiv_papers = agentrxiv_papers
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps
        self._plan_gate = dict(plan_formulation_gate)

        self.phases = [
            ("literature review", ["literature review"]),
            ("plan formulation", ["plan formulation"]),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        self.phase_models = dict()
        if type(agent_model_backbone) == str:
            for phase, subtasks in self.phases:
                for subtask in subtasks:
                    self.phase_models[subtask] = agent_model_backbone
        elif type(agent_model_backbone) == dict:
            # todo: check if valid
            self.phase_models = agent_model_backbone

        self.human_in_loop_flag = human_in_loop_flag

        self.statistics_per_phase = {
            "literature review":      {"time": 0.0, "steps": 0.0,},
            "plan formulation":       {"time": 0.0, "steps": 0.0,},
            "data preparation":       {"time": 0.0, "steps": 0.0,},
            "running experiments":    {"time": 0.0, "steps": 0.0,},
            "results interpretation": {"time": 0.0, "steps": 0.0,},
            "report writing":         {"time": 0.0, "steps": 0.0,},
            "report refinement":      {"time": 0.0, "steps": 0.0,},
        }

        self.save = True
        self.verbose = True
        self.reviewers = ReviewersAgent(model=self.model_backbone, notes=self.notes, openai_api_key=self.openai_api_key)
        self.phd = PhDStudentAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.postdoc = PostdocAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.professor = ProfessorAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.ml_engineer = MLEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.sw_engineer = SWEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)


    def set_model(self, model):
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def save_state(self, phase):
        """
        Save state for phase
        @param phase: (str) phase string
        @return: None
        """
        with open(f"state_saves/Paper{self.paper_index}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_agent_attr(self, attr, obj):
        """
        Set attribute for all agents
        @param attr: (str) agent attribute
        @param obj: (object) object attribute
        @return: None
        """
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def reset_agents(self):
        """
        Reset all agent states
        @return: None
        """
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def _lab_log(self, msg: str):
        if self.use_progress_bar:
            try:
                from tqdm import tqdm
                tqdm.write(str(msg))
            except Exception:
                print(msg)
        else:
            print(msg)

    def _try_prefill_lit_review(self) -> bool:
        """
        If InciteResearch already retrieved enough papers, skip the arXiv tool-calling loop
        and reuse the same summaries as the formal literature review context.
        """
        entries = self.prefilled_lit_review_entries
        if not entries:
            return False
        need = max(1, int(self.num_papers_lit_review))
        if len(entries) < need:
            self.phd.lit_review = [dict(x) for x in entries]
            return False
        self.phd.lit_review = [dict(x) for x in entries[:need]]
        lit_review_sum = self.phd.format_review()
        self.set_agent_attr("lit_review_sum", lit_review_sum)
        self.reset_agents()
        self.statistics_per_phase["literature review"]["steps"] = 0
        return True

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        phase_iter = self.phases
        if self.use_progress_bar and tqdm is not None:
            phase_iter = tqdm(self.phases, desc="Research phases", unit="phase")
        for phase, subtasks in phase_iter:
            phase_start_time = time.time()  # Start timing the phase
            show_long_banner = self.verbose and not self.use_progress_bar
            if show_long_banner:
                print(f"{'*'*50}\nBeginning phase: {phase}\n{'*'*50}")
            elif self.use_progress_bar:
                self._lab_log(f"--- Phase: {phase} ---")
            for subtask in subtasks:
                if self.agentRxiv:
                    if show_long_banner:
                        print(f"{'&' * 30}\n[Lab #{self.lab_index} Paper #{self.paper_index}] Beginning subtask: {subtask}\n{'&' * 30}")
                else:
                    if show_long_banner:
                        print(f"{'&'*30}\nBeginning subtask: {subtask}\n{'&'*30}")
                    elif self.use_progress_bar:
                        self._lab_log(f"  - Subtask: {subtask}")
                if type(self.phase_models) == dict:
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else: self.set_model(f"{DEFAULT_LLM_BACKBONE}")
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "literature review":
                    repeat = True
                    while repeat: repeat = self.literature_review()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "plan formulation":
                    repeat = True
                    while repeat: repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
                    save_to_file(f"./{self.lab_dir}", "proposal.md", self.phd.plan)
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "data preparation":
                    repeat = True
                    while repeat: repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "running experiments":
                    repeat = True
                    while repeat: repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "results interpretation":
                    repeat = True
                    while repeat: repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report writing":
                    repeat = True
                    while repeat: repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (subtask not in self.phase_status or not self.phase_status[subtask]) and subtask == "report refinement":
                    return_to_exp_phase = self.report_refinement()

                    if not return_to_exp_phase:
                        if self.save: self.save_state(subtask)
                        return

                    self.set_agent_attr("second_round", return_to_exp_phase)
                    self.set_agent_attr("prev_report", copy(self.phd.report))
                    self.set_agent_attr("prev_exp_results", copy(self.phd.exp_results))
                    self.set_agent_attr("prev_results_code", copy(self.phd.results_code))
                    self.set_agent_attr("prev_interpretation", copy(self.phd.interpretation))

                    self.phase_status["plan formulation"] = False
                    self.phase_status["data preparation"] = False
                    self.phase_status["running experiments"] = False
                    self.phase_status["results interpretation"] = False
                    self.phase_status["report writing"] = False
                    self.phase_status["report refinement"] = False
                    self.perform_research()
                if self.save: self.save_state(subtask)
                # Calculate and print the duration of the phase
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                self._lab_log(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report)
        print("Reviews:", reviews)
        if self.human_in_loop_flag["report refinement"]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            input("Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) for go back (n) for complete project: ")
        else:
            review_prompt = f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n Type y and nothing else to go back, type n and nothing else for complete project."
            self.phd.phases.append("report refinement")
            if self.review_override:
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.phd.inference(
                    research_topic=self.research_topic, phase="report refinement", feedback=review_prompt, step=0)
            if len(response) == 0:
                raise Exception("Model did not respond")
            response = response.lower().strip()[0]
            if response == "n":
                if self.verbose: print("*"*40, "\n", "REVIEW COMPLETE", "\n", "*"*40)
                return False
            elif response == "y":
                self.set_agent_attr("reviewer_response", f"Provided are reviews from a set of three reviewers: {reviews}.")
                return True
            else: raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        report_notes = [_note["note"] for _note in self.ml_engineer.notes if "report writing" in _note["phases"]]
        report_notes = f"Notes for the task objective: {report_notes}\n" if len(report_notes) > 0 else ""
        # instantiate mle-solver
        try:
            from papersolver import PaperSolver
        except ImportError as exc:
            raise RuntimeError(
                "The `papersolver` module is not bundled; install or provide it to use report writing."
            ) from exc
        self.reference_papers = []
        solver = PaperSolver(notes=report_notes, max_steps=self.papersolver_max_steps, plan=self.phd.plan, exp_code=self.phd.results_code, exp_results=self.phd.exp_results, insights=self.phd.interpretation, lit_review=self.phd.lit_review, ref_papers=self.reference_papers, topic=self.research_topic, openai_api_key=self.openai_api_key, llm_str=self.model_backbone["report writing"], compile_pdf=self.compile_pdf, save_loc=self.lab_dir)
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.papersolver_max_steps): solver.solve()
        # get best report results
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]
        match = re.search(r'\\title\{([^}]*)\}', report)
        if match: report_title = match.group(1).replace(" ", "_")
        else: report_title = "\n".join([str(random.randint(0, 10)) for _ in range(10)])
        if self.agentRxiv: shutil.copyfile(self.lab_dir + "/tex/temp.pdf", f"uploads/{report_title}.pdf")
        if self.verbose: print(f"Report writing completed, reward function score: {score}")
        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry: return retry
        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme()
        save_to_file(f"./{self.lab_dir}", "readme.md", readme)
        save_to_file(f"./{self.lab_dir}", "report.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            resp = self.postdoc.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose: print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry: return retry
                self.set_agent_attr("interpretation", interpretation)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False
            resp = self.phd.inference(self.research_topic, "results interpretation", feedback=dialogue, step=_i)
            if self.verbose: print("PhD Student: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose: print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        """
        Perform running experiments phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        experiment_notes = [_note["note"] for _note in self.ml_engineer.notes if "running experiments" in _note["phases"]]
        experiment_notes = f"Notes for the task objective: {experiment_notes}\n" if len(experiment_notes) > 0 else ""
        # instantiate mle-solver
        from .mlesolver import MLESolver
        solver = MLESolver(dataset_code=self.ml_engineer.dataset_code, notes=experiment_notes, insights=self.ml_engineer.lit_review_sum, max_steps=self.mlesolver_max_steps, plan=self.ml_engineer.plan, openai_api_key=self.openai_api_key, llm_str=self.model_backbone["running experiments"])
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.mlesolver_max_steps-1):
            solver.solve()
        # get best code results
        code = "\n".join(solver.best_codes[0][0])
        # regenerate figures from top code
        #execute_code(code)
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose: print(f"Running experiments completed, reward function score: {score}")
        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("data preparation", code)
            if retry: return retry
        save_to_file(f"./{self.lab_dir}/src", "run_experiments.py", code)
        save_to_file(f"./{self.lab_dir}/src", "experiment_output.log", exp_results)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        # reset agent state
        self.reset_agents()
        return False

    def data_preparation(self):
        """
        Perform data preparation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        ml_feedback = str()
        ml_dialogue = str()
        swe_feedback = str()
        ml_command = str()
        hf_engine = HFDataSearch()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            if ml_feedback != "":
                ml_feedback_in = "Feedback provided to the ML agent: " + ml_feedback
            else: ml_feedback_in = ""
            resp = self.sw_engineer.inference(self.research_topic, "data preparation", feedback=f"{ml_dialogue}\nFeedback from previous command: {swe_feedback}\n{ml_command}{ml_feedback_in}", step=_i)
            swe_feedback = str()
            swe_dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                swe_dialogue = f"\nThe following is dialogue produced by the SW Engineer: {dialogue}\n"
                if self.verbose: print("#"*40, f"\nThe following is dialogue produced by the SW Engineer: {dialogue}", "\n", "#"*40)
            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_code(final_code, timeout=60)
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
                swe_feedback += f"\nCode Response: {code_resp}\n"
                if "[CODE EXECUTION ERROR]" in code_resp:
                    swe_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry: return retry
                    save_to_file(f"./{self.lab_dir}/src", "load_data.py", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    # reset agent state
                    self.reset_agents()
                    self.statistics_per_phase["data preparation"]["steps"] = _i
                    return False

            if ml_feedback != "":
                ml_feedback_in = "Feedback from previous command: " + ml_feedback
            else:
                ml_feedback_in = ""
            resp = self.ml_engineer.inference(
                self.research_topic, "data preparation",
                feedback=f"{swe_dialogue}\n{ml_feedback_in}", step=_i)
            #if self.verbose: print("ML Engineer: ", resp, "\n~~~~~~~~~~~")
            ml_feedback = str()
            ml_dialogue = str()
            ml_command = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ml_dialogue = f"\nThe following is dialogue produced by the ML Engineer: {dialogue}\n"
                if self.verbose: print("#" * 40, f"\nThe following is dialogue produced by the ML Engineer: {dialogue}", "#" * 40, "\n")
            if "```python" in resp:
                code = extract_prompt(resp, "python")
                code = self.ml_engineer.dataset_code + "\n" + code
                code_resp = execute_code(code, timeout=120)
                ml_command = f"Code produced by the ML agent:\n{code}"
                ml_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose: print("!"*100, "\n", f"CODE RESPONSE: {code_resp}")
            if "```SEARCH_HF" in resp:
                hf_query = extract_prompt(resp, "SEARCH_HF")
                hf_res = "\n".join(hf_engine.results_str(hf_engine.retrieve_ds(hf_query)))
                ml_command = f"HF search command produced by the ML agent:\n{hf_query}"
                ml_feedback += f"Huggingface results: {hf_res}\n"
        raise Exception("Max tries during phase: Data Preparation")

    def plan_formulation(self):
        """
        Perform plan formulation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        self._plan_submission_lengths.clear()
        self._plan_last_body = None
        for _i in range(max_tries):
            if not self.use_progress_bar:
                print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
            resp = self.postdoc.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose and not self.use_progress_bar:
                print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose and not self.use_progress_bar:
                    print("#"*40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#"*40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                plan_len = count_plan_chars(plan)
                if (
                    self._plan_last_body is not None
                    and self._plan_submission_lengths
                    and plan_len < self._plan_submission_lengths[-1]
                ):
                    plan = self._plan_last_body
                    plan_len = count_plan_chars(plan)
                    self._lab_log(
                        f"[PLAN] Length regressed vs previous draft; keeping prior PLAN ({plan_len} chars) as final."
                    )
                    if self.human_in_loop_flag["plan formulation"]:
                        retry = self.human_in_loop("plan formulation", plan)
                        if retry:
                            return retry
                    self.set_agent_attr("plan", plan)
                    self.reset_agents()
                    self.statistics_per_phase["plan formulation"]["steps"] = _i
                    return False

                self._plan_submission_lengths.append(plan_len)
                self._plan_last_body = plan
                last3 = self._plan_submission_lengths[-3:]
                converged = False
                span = 0
                tol = 0
                if len(last3) >= 3:
                    lo, hi = min(last3), max(last3)
                    span = hi - lo
                    tol = max(
                        self._plan_gate["convergence_span_floor"],
                        int(self._plan_gate["convergence_span_frac"] * hi),
                    ) if hi else self._plan_gate["convergence_span_floor"]
                    converged = span <= tol
                long_enough = plan_len >= self.min_plan_chars
                frac = self._plan_gate["accept_as_converged_fraction_of_min_plan"]
                min_converged_len = int(self.min_plan_chars * frac)
                converged_accept = converged and plan_len >= max(
                    self._plan_gate["convergence_min_chars"],
                    min_converged_len,
                )
                saturation_accept, saturation_note = _plan_length_saturation_accept(
                    self._plan_submission_lengths, self._plan_gate, self.min_plan_chars
                )
                if not long_enough and not converged_accept and not saturation_accept:
                    dialogue = (
                        f"The submitted PLAN body (only the text inside the ```PLAN fenced block, after strip()) is too short: {plan_len} chars "
                        f"(required minimum {self.min_plan_chars}). The reviewer measures Unicode code points on the extracted block — not tokens, not words.\n"
                        "Rewrite and expand the PLAN, or keep iterating until three consecutive PLAN submissions have stable lengths (similar character counts), "
                        "or until length edits shrink versus earlier steps while staying near the best length seen in this phase.\n"
                        "Include concrete sections: Problem Statement, Motivation, Method Overview, "
                        "System Architecture, Algorithmic Steps, Data Pipeline, Implementation Plan, "
                        "Evaluation Protocol, Ablations, Risks/Failure Modes, and Expected Outcomes."
                    )
                    extra = f"[PLAN] Rejected: extracted body {plan_len} chars / threshold {self.min_plan_chars}"
                    if len(last3) >= 3:
                        extra += f"; last-three span {span} (tolerance +/-{tol})"
                    self._lab_log(extra)
                    continue
                if converged_accept and not long_enough:
                    self._lab_log(
                        f"[PLAN] Three consecutive submissions have similar length (span <= {tol}); accepting as sufficiently concrete ({plan_len} chars)."
                    )
                if saturation_accept and not long_enough and not converged_accept:
                    self._lab_log(
                        f"[PLAN] Saturation / plateau stop ({saturation_note}); accepting {plan_len} chars (target min {self.min_plan_chars})."
                    )
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry:
                        return retry
                self.set_agent_attr("plan", plan)
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            resp = self.phd.inference(self.research_topic, "plan formulation", feedback=dialogue, step=_i)
            if self.verbose and not self.use_progress_bar:
                print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the PhD student: {dialogue}"
                if self.verbose and not self.use_progress_bar:
                    print("#"*40, "\n", "PhD Dialogue:", dialogue, "#"*40, "\n")
        if self.except_if_fail:
            raise Exception("Max tries during phase: Plan Formulation")
        plan = "No plan specified."
        if self.human_in_loop_flag["plan formulation"]:
            retry = self.human_in_loop("plan formulation", plan)
            if retry:
                return retry
        self.set_agent_attr("plan", plan)
        self.reset_agents()
        return False

    def literature_review(self):
        """
        Perform literature review phase
        @return: (bool) whether to repeat the phase
        """
        if self._try_prefill_lit_review():
            self._lab_log(
                "[Literature] Loaded InciteResearch research_state.json papers into agent context "
                f"({len(self.phd.lit_review)} entries); skipping arXiv tool loop — "
                '"Current Literature Review" in prompts is built from these papers.'
            )
            return False
        arx_eng = ArxivSearch()
        max_tries = self.max_steps # lit review often requires extra steps
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_tries, desc="Literature review (arXiv)", leave=False, disable=not self.use_progress_bar)
        except Exception:
            pbar = None
        try:
            # get initial response from PhD agent
            resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.4)
            if self.verbose and not self.use_progress_bar:
                print(resp, "\n~~~~~~~~~~~")
            # iterate until max num tries to complete task is exhausted
            for _i in range(max_tries):
                if pbar is not None:
                    pbar.update(1)
                elif not self.use_progress_bar:
                    print(f"@@ Lab #{self.lab_index} Paper #{self.paper_index} @@")
                feedback = str()
                # grab summary of papers from arxiv
                if "```SUMMARY" in resp:
                    query = extract_prompt(resp, "SUMMARY")
                    papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                    if self.agentRxiv:
                        if GLOBAL_AGENTRXIV.num_papers() > 0:
                            papers += GLOBAL_AGENTRXIV.search_agentrxiv(query, self.num_agentrxiv_papers,)
                    feedback = f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"

                # grab full text from arxiv ID
                elif "```FULL_TEXT" in resp:
                    query = extract_prompt(resp, "FULL_TEXT")
                    if self.agentRxiv and "AgentRxiv" in query:
                        full_text = GLOBAL_AGENTRXIV.retrieve_full_text(query,)
                    else:
                        full_text = arx_eng.retrieve_full_paper_text(query)
                    # expiration timer so that paper does not remain in context too long
                    arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n" + full_text + "```"
                    feedback = arxiv_paper

                # if add paper, extract and add to lit review, provide feedback
                elif "```ADD_PAPER" in resp:
                    query = extract_prompt(resp, "ADD_PAPER")
                    if self.agentRxiv and "AgentRxiv" in query:
                        feedback, text = self.phd.add_review(query, arx_eng, agentrxiv=True, GLOBAL_AGENTRXIV=GLOBAL_AGENTRXIV)
                    else:
                        feedback, text = self.phd.add_review(query, arx_eng)
                    if len(self.reference_papers) < self.num_ref_papers:
                        self.reference_papers.append(text)

                # completion condition
                if len(self.phd.lit_review) >= self.num_papers_lit_review:
                    # generate formal review
                    lit_review_sum = self.phd.format_review()
                    # if human in loop -> check if human is happy with the produced review
                    if self.human_in_loop_flag["literature review"]:
                        retry = self.human_in_loop("literature review", lit_review_sum)
                        # if not happy, repeat the process with human feedback
                        if retry:
                            self.phd.lit_review = []
                            return retry
                    # otherwise, return lit review and move on to next stage
                    if self.verbose and not self.use_progress_bar:
                        print(self.phd.lit_review_sum)
                    # set agent
                    self.set_agent_attr("lit_review_sum", lit_review_sum)
                    # reset agent state
                    self.reset_agents()
                    self.statistics_per_phase["literature review"]["steps"] = _i
                    return False
                resp = self.phd.inference(self.research_topic, "literature review", feedback=feedback, step=_i + 1, temp=0.4)
                if self.verbose and not self.use_progress_bar:
                    print(resp, "\n~~~~~~~~~~~")
        finally:
            if pbar is not None:
                pbar.close()
        if self.except_if_fail:
            raise Exception("Max tries during phase: Literature Review")
        else:
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # generate formal review
                lit_review_sum = self.phd.format_review()
                # if human in loop -> check if human is happy with the produced review
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    # if not happy, repeat the process with human feedback
                    if retry:
                        self.phd.lit_review = []
                        return retry
                # otherwise, return lit review and move on to next stage
                if self.verbose: print(self.phd.lit_review_sum)
                # set agent
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = _i
                return False

    def human_in_loop(self, phase, phase_prod):
        """
        Get human feedback for phase output
        @param phase: (str) current phase
        @param phase_prod: (str) current phase result
        @return: (bool) whether to repeat the loop
        """
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        # repeat until a valid answer is provided
        while y_or_no not in ["y", "n"]:
            y_or_no = input("\n\n\nAre you happy with the presented content? Respond Y or N: ").strip().lower()
            # if person is happy with feedback, move on to next stage
            if y_or_no == "y": pass
            # if not ask for feedback and repeat
            elif y_or_no == "n":
                # ask the human for feedback
                notes_for_agent = input("Please provide notes for the agent so that they can try again and improve performance: ")
                # reset agent state
                self.reset_agents()
                # add suggestions to the notes
                self.notes.append({
                    "phases": [phase],
                    "note": notes_for_agent})
                return True
            else: print("Invalid response, type Y or N")
        return False

class AgentRxiv:
    def __init__(self, lab_index=0):
        self.lab_index = lab_index
        self.server_thread = None
        self.initialize_server()
        self.pdf_text = dict()
        self.summaries = dict()

    def initialize_server(self):
        # Calculate the port dynamically
        port = 5000 + self.lab_index
        # Start the server on the computed port using a lambda to pass the port value
        self.server_thread = threading.Thread(target=lambda: self.run_server(port))
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(5)  # allow time for the server to start up

    @staticmethod
    def num_papers():
        return len(os.listdir("uploads"))

    def retrieve_full_text(self, arxiv_id):
        try:
            return self.pdf_text[arxiv_id]
        except Exception:
            return "Paper ID not found?"

    def search_agentrxiv(self, search_query, num_papers):
        # Disabled for simplified pipeline
        return "Search disabled"

    def run_server(self, port):
        pass


def parse_arguments():
    _pkg = Path(__file__).resolve().parent
    _default_yaml = str(_pkg / "configs" / "MATH_agentlab.yaml")
    parser = argparse.ArgumentParser(description="Research lab workflow (literature + plan formulation)")

    parser.add_argument(
        '--yaml-location',
        type=str,
        default=_default_yaml,
        help='Path to experiment YAML (merged over research_lab/configs/builtin_defaults.yaml).',
    )
    parser.add_argument(
        '--incite-state',
        type=str,
        default=None,
        help='Path to InciteResearch research_state.json (merges raw_papers into literature review).',
    )
    parser.add_argument(
        '--incite-proposal-md',
        type=str,
        default=None,
        help='Path to InciteResearch research_proposal.md (appended as plan-formulation note).',
    )

    return parser.parse_args()


def parse_yaml(yaml_file_loc):
    builtin_path = Path(__file__).resolve().parent / "configs" / "builtin_defaults.yaml"
    builtin: dict = {}
    if builtin_path.is_file():
        with open(builtin_path, "r", encoding="utf-8") as bf:
            builtin = yaml.safe_load(bf) or {}
    with open(yaml_file_loc, "r", encoding="utf-8") as file:
        user_data = yaml.safe_load(file) or {}
    agentlab_data = _deep_merge_dict(builtin, user_data)
    class YamlDataHolder:
        def __init__(self): pass
    parser = YamlDataHolder()
    if "copilot_mode" in agentlab_data: parser.copilot_mode = agentlab_data["copilot_mode"]
    else: parser.copilot_mode = False
    if 'load-previous' in agentlab_data: parser.load_previous = agentlab_data["load-previous"]
    else: parser.load_previous = False
    if "research-topic" in agentlab_data:
        _rt = agentlab_data["research-topic"]
        parser.research_topic = _rt if _rt is not None and str(_rt).strip() else None
    else:
        parser.research_topic = None
    if 'api-key' in agentlab_data: parser.api_key = agentlab_data["api-key"]
    if 'deepseek-api-key' in agentlab_data: parser.deepseek_api_key = agentlab_data["deepseek-api-key"]
    if 'compile-latex' in agentlab_data: parser.compile_latex = agentlab_data["compile-latex"]
    else: parser.compile_latex = True
    if 'language' in agentlab_data: parser.language = agentlab_data["language"]
    else: parser.language = "English"
    if 'num-papers-lit-review' in agentlab_data: parser.num_papers_lit_review = agentlab_data["num-papers-lit-review"]
    else: parser.num_papers_lit_review = 5
    if 'mlesolver-max-steps' in agentlab_data: parser.mlesolver_max_steps = agentlab_data["mlesolver-max-steps"]
    else: parser.mlesolver_max_steps = 3
    if 'papersolver-max-steps' in agentlab_data: parser.papersolver_max_steps = agentlab_data["papersolver-max-steps"]
    else: parser.papersolver_max_steps = 5
    if 'task-notes' in agentlab_data: parser.task_notes = agentlab_data["task-notes"]
    else: parser.task_notes = []
    if 'num-papers-to-write' in agentlab_data: parser.num_papers_to_write = agentlab_data["num-papers-to-write"]
    else: parser.num_papers_to_write = 100
    if 'parallel-labs' in agentlab_data: parser.parallel_labs = agentlab_data["parallel-labs"]
    else: parser.parallel_labs = False
    if 'num-parallel-labs' in agentlab_data: parser.num_parallel_labs = agentlab_data["num-parallel-labs"]
    else: parser.num_parallel_labs = 8
    if 'except-if-fail' in agentlab_data: parser.except_if_fail = agentlab_data["except-if-fail"]
    else: parser.except_if_fail = False
    if 'agentRxiv' in agentlab_data: parser.agentRxiv = agentlab_data["agentRxiv"]
    else: parser.agentRxiv = False
    if 'construct-agentRxiv' in agentlab_data: parser.construct_agentRxiv = agentlab_data["construct-agentRxiv"]
    else: parser.construct_agentRxiv = False
    if 'agentrxiv-papers' in agentlab_data: parser.agentrxiv_papers = agentlab_data["agentrxiv-papers"]
    else:  parser.agentrxiv_papers = 5
    parser.min_plan_chars = agentlab_data["min-plan-chars"]

    def _yaml_bool(v, default=True):
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("false", "0", "no", "off"):
            return False
        if s in ("true", "1", "yes", "on"):
            return True
        return default

    lp = agentlab_data.get("literature-prefill") or {}
    parser.literature_prefill = {
        "pdf_excerpt": _yaml_bool(lp.get("pdf-excerpt"), True),
        "pdf_max_pages": int(lp.get("pdf-max-pages", 8)),
        "pdf_max_chars": int(lp.get("pdf-max-chars", 20000)),
    }

    pf = agentlab_data.get("plan-formulation") or {}
    parser.plan_formulation_gate = {
        "convergence_min_chars": int(pf["convergence-min-chars"]),
        "convergence_span_floor": int(pf["convergence-span-tolerance-floor"]),
        "convergence_span_frac": float(pf["convergence-span-tolerance-frac"]),
        "accept_as_converged_fraction_of_min_plan": float(pf["accept-as-converged-fraction-of-min-plan"]),
        "saturation_enable": _yaml_bool(pf.get("saturation-enable"), True),
    }
    arxiv_cfg = agentlab_data.get("arxiv") or {}
    parser.arxiv_summary_count = int(arxiv_cfg["summary-count-per-query"])
    parser.arxiv_paper_exp_time = int(arxiv_cfg["paper-expiration-time-steps"])
    if "research-output-subdir" in agentlab_data:
        parser.research_output_subdir = agentlab_data["research-output-subdir"]
    else:
        raise ValueError("experiment yaml must define research-output-subdir")

    if 'lab-index' in agentlab_data: parser.lab_index = agentlab_data["lab-index"]
    else: parser.lab_index = 0
    return parser


def main():
    global GLOBAL_AGENTRXIV
    try:
        from dotenv import load_dotenv
        _repo_root = Path(__file__).resolve().parent.parent
        _root_env = _repo_root / ".env"
        if _root_env.exists():
            load_dotenv(_root_env, override=True)
        _lab_env = Path(__file__).resolve().parent / ".env"
        if _lab_env.exists():
            load_dotenv(_lab_env, override=False)
    except Exception:
        pass

    user_args = parse_arguments()
    yaml_to_use = user_args.yaml_location
    args = parse_yaml(yaml_to_use)

    llm_backend = resolve_lab_llm_model_id()
    human_mode =  args.copilot_mode.lower() == "true" if type(args.copilot_mode) == str else args.copilot_mode
    compile_pdf = args.compile_latex.lower() == "true" if type(args.compile_latex) == str else args.compile_latex
    load_previous = args.load_previous.lower() == "true" if type(args.load_previous) == str else args.load_previous
    parallel_labs = args.parallel_labs.lower() == "true" if type(args.parallel_labs) == str else args.parallel_labs
    except_if_fail = args.except_if_fail.lower() == "true" if type(args.except_if_fail) == str else args.except_if_fail
    agentRxiv = args.agentRxiv.lower() == "true" if type(args.agentRxiv) == str else args.agentRxiv
    construct_agentRxiv = args.construct_agentRxiv.lower() == "true" if type(args.construct_agentRxiv) == str else args.construct_agentRxiv
    lab_index = int(args.lab_index) if type(args.construct_agentRxiv) == str else args.lab_index

    try: num_papers_to_write = int(args.num_papers_to_write.lower()) if type(args.num_papers_to_write) == str else args.num_papers_to_write
    except Exception: raise Exception("args.num_papers_lit_review must be a valid integer!")
    try: num_papers_lit_review = int(args.num_papers_lit_review.lower()) if type(args.num_papers_lit_review) == str else args.num_papers_lit_review
    except Exception: raise Exception("args.num_papers_lit_review must be a valid integer!")
    try: papersolver_max_steps = int(args.papersolver_max_steps.lower()) if type(args.papersolver_max_steps) == str else args.papersolver_max_steps
    except Exception: raise Exception("args.papersolver_max_steps must be a valid integer!")
    try: mlesolver_max_steps = int(args.mlesolver_max_steps.lower()) if type(args.mlesolver_max_steps) == str else args.mlesolver_max_steps
    except Exception: raise Exception("args.mlesolver_max_steps must be a valid integer!")
    try: min_plan_chars = int(args.min_plan_chars.lower()) if type(args.min_plan_chars) == str else args.min_plan_chars
    except Exception: raise Exception("args.min_plan_chars must be a valid integer!")
    research_output_subdir = args.research_output_subdir
    plan_formulation_gate = args.plan_formulation_gate
    arxiv_summary_count = args.arxiv_summary_count
    arxiv_paper_exp_time = args.arxiv_paper_exp_time
    if parallel_labs:
        num_parallel_labs = int(args.num_parallel_labs)
        print("="*20 , f"RUNNING {num_parallel_labs} LABS IN PARALLEL", "="*20)
    else: num_parallel_labs = 0

    prefilled_lit = None
    incite_state_snapshot = None
    if user_args.incite_state:
        try:
            st_path = Path(user_args.incite_state).expanduser()
            incite_state_snapshot = json.loads(st_path.read_text(encoding="utf-8"))
            raw = incite_state_snapshot.get("raw_papers") or []
            prefilled_lit = lit_entries_from_research_state(raw, max(num_papers_lit_review, 1))
            if prefilled_lit:
                print(
                    f"[InciteResearch] Loaded {len(prefilled_lit)} paper entries from state "
                    f"(literature review target: {num_papers_lit_review})."
                )
            else:
                print(
                    "[InciteResearch] research_state.json has no raw_papers (or list empty); "
                    "lab will run the arXiv literature loop unless you add papers in main.py first."
                )
            if prefilled_lit and incite_state_snapshot and getattr(args, "literature_prefill", None):
                _lp = args.literature_prefill
                prefilled_lit = enrich_lit_entries_with_pdf_excerpts(
                    prefilled_lit,
                    incite_state_snapshot,
                    enabled=bool(_lp.get("pdf_excerpt")),
                    max_pages=int(_lp.get("pdf_max_pages", 8)),
                    max_chars=int(_lp.get("pdf_max_chars", 20000)),
                )
        except Exception as e:
            print(f"[WARNING] Failed to read --incite-state; falling back to full literature agent loop: {e}")
            prefilled_lit = None
            incite_state_snapshot = None

    api_key = (os.getenv('OPENAI_API_KEY') or args.api_key) if (hasattr(args, 'api_key') or os.getenv('OPENAI_API_KEY')) else None
    deepseek_api_key = (os.getenv('DEEPSEEK_API_KEY') or args.deepseek_api_key) if (hasattr(args, 'deepseek_api_key') or os.getenv('DEEPSEEK_API_KEY')) else None
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key is not None and os.getenv("OPENAI_API_KEY") is None:
        os.environ["OPENAI_API_KEY"] = str(api_key)
    if deepseek_api_key is not None and os.getenv("DEEPSEEK_API_KEY") is None:
        os.environ["DEEPSEEK_API_KEY"] = str(deepseek_api_key)

    if not api_key and not deepseek_api_key and not anthropic_api_key: raise ValueError("API key must be provided via --api-key / -deepseek-api-key or the OPENAI_API_KEY / DEEPSEEK_API_KEY / ANTHROPIC_API_KEY environment variable.")

    resolved_api_key = api_key or deepseek_api_key or anthropic_api_key
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    incite_proposal_body = None
    if user_args.incite_proposal_md:
        _prop_path = Path(user_args.incite_proposal_md).expanduser()
        if _prop_path.is_file():
            _pb = _prop_path.read_text(encoding="utf-8").strip()
            if _pb:
                incite_proposal_body = _pb

    yaml_rt = getattr(args, "research_topic", None)
    if incite_proposal_body:
        research_topic = infer_research_topic_from_proposal_md(
            incite_proposal_body,
            llm_backend,
            openai_api_key=resolved_api_key,
            gemini_api_key=gemini_api_key,
            anthropic_api_key=anthropic_api_key,
        )
        print(f"[research lab] Topic line derived from proposal (model {llm_backend}): {research_topic}")
    elif yaml_rt is not None and str(yaml_rt).strip():
        research_topic = str(yaml_rt).strip()
    elif human_mode:
        research_topic = input("Please describe the research idea for the lab to run: ")
    else:
        raise ValueError(
            "No research topic source: pass a non-empty --incite-proposal-md file, "
            "or set research-topic in the experiment YAML, or enable copilot-mode for interactive input."
        )

    task_notes_LLM = list()
    task_notes = args.task_notes
    for _task in task_notes:
        for _note in task_notes[_task]:
            task_notes_LLM.append({"phases": [_task.replace("-", " ")], "note": _note})

    for _ctx in incite_state_context_notes(incite_state_snapshot, num_papers_lit_review):
        task_notes_LLM.append(_ctx)

    if incite_proposal_body:
        task_notes_LLM.append({
            "phases": ["plan formulation"],
            "note": "InciteResearch research_proposal.md (full proposal; refine into an executable method plan):\n" + incite_proposal_body,
        })

    if args.language != "English":
        task_notes_LLM.append(
            {"phases": ["literature review", "plan formulation", "data preparation", "running experiments", "results interpretation", "report writing", "report refinement"],
            "note": f"You should always write in the following language to converse and to write the report {args.language}"},
        )

    human_in_loop = {
        "literature review":      human_mode,
        "plan formulation":       human_mode,
        "data preparation":       human_mode,
        "running experiments":    human_mode,
        "results interpretation": human_mode,
        "report writing":         human_mode,
        "report refinement":      human_mode,
    }

    agent_models = {
        "literature review":      llm_backend,
        "plan formulation":       llm_backend,
        "data preparation":       llm_backend,
        "running experiments":    llm_backend,
        "report writing":         llm_backend,
        "results interpretation": llm_backend,
        "paper refinement":       llm_backend,
    }
    if parallel_labs:
        remove_figures()
        GLOBAL_AGENTRXIV = AgentRxiv()
        remove_directory(f"{research_output_subdir}")
        os.mkdir(os.path.join(".", f"{research_output_subdir}"))
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if not compile_pdf: raise Exception("PDF compilation must be used with agentRxiv!")
        def run_lab(parallel_lab_index):
            time_str = str()
            time_now = time.time()
            for _paper_index in range(num_papers_to_write):
                lab_dir = os.path.join(research_output_subdir, f"research_dir_lab{parallel_lab_index}_paper{_paper_index}")
                os.mkdir(lab_dir)
                os.mkdir(os.path.join(lab_dir, "src"))
                os.mkdir(os.path.join(lab_dir, "tex"))
                lab_instance = LaboratoryWorkflow(
                    parallelized=True,
                    research_topic=research_topic,
                    notes=task_notes_LLM,
                    agent_model_backbone=agent_models,
                    human_in_loop_flag=human_in_loop,
                    openai_api_key=resolved_api_key,
                    compile_pdf=compile_pdf,
                    num_papers_lit_review=num_papers_lit_review,
                    papersolver_max_steps=papersolver_max_steps,
                    mlesolver_max_steps=mlesolver_max_steps,
                    paper_index=_paper_index,
                    lab_index=parallel_lab_index,
                    except_if_fail=except_if_fail,
                    lab_dir=lab_dir,
                    agentRxiv=True,
                    agentrxiv_papers=args.agentrxiv_papers,
                    min_plan_chars=min_plan_chars,
                    prefilled_lit_review_entries=prefilled_lit,
                    arxiv_summary_count=arxiv_summary_count,
                    plan_formulation_gate=plan_formulation_gate,
                    arxiv_paper_exp_time=arxiv_paper_exp_time,
                )
                lab_instance.perform_research()
                time_str += str(time.time() - time_now) + " | "
                with open(f"agent_times_{parallel_lab_index}.txt", "w") as f:
                    f.write(time_str)
                time_now = time.time()

        with ThreadPoolExecutor(max_workers=num_parallel_labs) as executor:
            futures = [executor.submit(run_lab, lab_idx) for lab_idx in range(num_parallel_labs)]
            for future in as_completed(futures):
                try: future.result()
                except Exception as e: print(f"Error in lab: {e}")

        from . import inference
        print("="*40)
        print("Token Consumption and Cost Estimation:")
        print(f"Tokens In: {inference.TOKENS_IN}")
        print(f"Tokens Out: {inference.TOKENS_OUT}")
        print(f"Cache Read Tokens: {inference.TOKENS_CACHE_READ}")
        print(f"Cache Creation Tokens: {inference.TOKENS_CACHE_CREATE}")
        print(f"Total Cost Estimation: ${inference.curr_cost_est():.4f}")
        print("="*40)
        raise NotImplementedError("Todo: implement parallel labs")
    else:
        # remove previous files
        remove_figures()
        if agentRxiv: GLOBAL_AGENTRXIV = AgentRxiv(lab_index)
        if not agentRxiv:
            remove_directory(f"{research_output_subdir}")
            os.mkdir(os.path.join(".", f"{research_output_subdir}"))
        # make src and research directory
        if not os.path.exists("state_saves"): os.mkdir(os.path.join(".", "state_saves"))
        time_str = str()
        time_now = time.time()
        for _paper_index in range(num_papers_to_write):
            lab_direct = f"{research_output_subdir}/research_dir_{_paper_index}_lab_{lab_index}"
            os.mkdir(os.path.join(".", lab_direct))
            os.mkdir(os.path.join(f"./{lab_direct}", "src"))
            os.mkdir(os.path.join(f"./{lab_direct}", "tex"))
            lab = LaboratoryWorkflow(
                research_topic=research_topic,
                notes=task_notes_LLM,
                agent_model_backbone=agent_models,
                human_in_loop_flag=human_in_loop,
                openai_api_key=resolved_api_key,
                compile_pdf=compile_pdf,
                num_papers_lit_review=num_papers_lit_review,
                papersolver_max_steps=papersolver_max_steps,
                mlesolver_max_steps=mlesolver_max_steps,
                paper_index=_paper_index,
                except_if_fail=except_if_fail,
                agentRxiv=False,
                lab_index=lab_index,
                lab_dir=f"./{lab_direct}",
                min_plan_chars=min_plan_chars,
                prefilled_lit_review_entries=prefilled_lit,
                arxiv_summary_count=arxiv_summary_count,
                plan_formulation_gate=plan_formulation_gate,
                arxiv_paper_exp_time=arxiv_paper_exp_time,
            )
            lab.perform_research()
            time_str += str(time.time() - time_now) + " | "
            with open(f"agent_times_{lab_index}.txt", "w") as f:
                f.write(time_str)
            time_now = time.time()
            
        from . import inference
        print("="*40)
        print("Token Consumption and Cost Estimation:")
        print(f"Tokens In: {inference.TOKENS_IN}")
        print(f"Tokens Out: {inference.TOKENS_OUT}")
        print(f"Cache Read Tokens: {inference.TOKENS_CACHE_READ}")
        print(f"Cache Creation Tokens: {inference.TOKENS_CACHE_CREATE}")
        print(f"Total Cost Estimation: ${inference.curr_cost_est():.4f}")
        print("="*40)


if __name__ == "__main__":
    main()


"""
@@@@@@@@@@@@@@@ CHECKLIST @@@@@@@@@@@@@@@ 
Practical:
----------
- Make a better config system (YAML?)

Advancements:
-------------
- Make the ability to have agents build on top of their own research
- Run agent labs in parallel (asynch) 

"""







