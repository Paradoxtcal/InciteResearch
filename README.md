<div align="center"> <img src="assets/pig_avatar.png" width="140" alt="InciteResearch"/>

# InciteResearch

**The missing first step in AI-assisted research**

Turn a researcher's vague frustration into a breakable assumption, a falsifiable story, and annotated core-algorithm code ready to hand off to Cursor or Claude Code.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![arXiv](https://img.shields.io/badge/arXiv-2605.06345-b31b1b.svg)](https://arxiv.org/abs/2605.06345) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Quick Start](#quick-start)

English | [简体中文](README_zh.md)

</div>

---

Every existing AI research tool starts from a clear topic. [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) turns a direction into a full paper pipeline. [karpathy/autoresearch](https://github.com/karpathy/autoresearch) runs overnight experiment loops given a training script. [DeepInnovator](https://github.com/HKUDS/DeepInnovator) generates hypotheses from literature distributions. They all assume you already know what you want to work on.

Most real research starts somewhere messier: a method that feels wrong, a paper whose assumption seems shaky, an evaluation metric you've never trusted. InciteResearch works at that moment — before the topic is clear — and produces something concrete enough for the downstream tools to take over.

> 📄 **Paper:** [More Than Can Be Said: A Benchmark and Framework for Pre-Question Scientific Ideation](https://arxiv.org/abs/2605.06345) — Jie Yu, Song Qiu (arXiv:2605.06345 [cs.AI])

## Key Features

---

- Friction-first elicitation — starts from what bothers you, not what you've already decided
- Assumption Breaking over Gap Analysis — produces directions that challenge a field's premises rather than adding a module to an existing method
- Story arc validation — the method must be the logically necessary consequence of the insight, not merely a reasonable option
- AI-assistant-ready code output — `core_algorithm.py` is written for another AI to read, with `# WHY` and `# DIFF`annotations on every non-trivial line
- Direct IDE integration — `cursor_prompt.txt` is a ready-to-paste instruction for integrating the core code into a chosen baseline

## A New Paradigm

---

<div align="center"> <img src="assets/InciteResearch_p.png" width="800" alt="InciteResearch Paradigm"/> </div>

Most AI research tools position themselves as autonomous systems: given enough scaffolding, the model should be able to generate a paper end-to-end. InciteResearch is built on a different conviction. As illustrated above, the process begins with casual human conversation — the researcher steers the direction while AI broadens the thought space through assumption-breaking. The human then reflects and corrects; AI provides its own directional suggestions. This loop continues until the vague inspiration becomes an explicit, structured research proposal.

This is not a pipeline with a human in the loop. It is a cognitive collaboration where the human's tacit sense of "something is off" is the irreplaceable input, and the AI's role is to extend that intuition — giving it leverage, articulation, and reach — rather than to replace it.

Three operators make this concrete:

- **E (Elicitation)** — surfaces the researcher's implicit friction into a five-dimensional structured profile: friction points, motivation, constraints, taste, and refined topic. The friction point is the gravitational center; everything else orbits it.
- **V (Validity & Reframing)** — instead of finding gaps in the literature, identifies the hidden assumption that, if broken, maximizes the product of feasibility and novelty. The output is not an incremental rewrite of the original direction but a re-anchoring under a new conceptual coordinate system.
- **N (Necessity Checking)** — acts as a strict reviewer. The proposed method must be the logically necessary consequence of the insight, not merely one of several plausible options. Any component that cannot be derived from the causal chain is rejected.

This paradigm — **Elicit → Violate → Necessitate** — is not an arbitrary modularization of Socratic questioning. Ablation results show that removing V causes the sharpest drop across all metrics (novelty 4.250 → 3.500, impact 4.397 → 3.720), confirming that assumption violation carries structural weight, not merely stylistic variety. Removing E causes the second-largest drop in novelty and is the only ablation where feasibility _increases_, revealing that the E stage is responsible for anchoring generation on real friction rather than surface-level reformulation.

The output of InciteResearch — a structured proposal with an annotated `core_algorithm.py` and a `cursor_prompt.txt` — is designed specifically to serve as the input to downstream execution tools. InciteResearch covers Phase 0–2. It does not reimplement what other tools already do well.

## Quick Start

---

```bash
git clone https://github.com/Paradoxtcal/InciteResearch
cd InciteResearch
pip install -r requirements.txt

cp .env.example .env
# add your API_KEY to .env

python main.py
```

The session opens with a single question. Answer naturally — there is no form to fill in. Any language works.

## Scope of Application

---

**Applicable:** Empirical research involving code implementation, including: CS (CV / NLP / ML / Systems), Computational Biology, Data Science, and Computational Physics/Chemistry.

**Inapplicable:** Pure mathematical proofs, hermeneutics in the humanities and social sciences, and clinical trials requiring ethical review.

## Contributing

---

The one constraint: don't turn this into a form-filling pipeline. The core of the design lies in how AI gives wings to human thinking. The value is in discovering the research question through dialogue. Any change that skips or shortcuts elicitation defeats the purpose.

Most welcome: real usage cases, new Socratic trigger archetypes, domain-specific prompt improvements, evaluation methodology for idea quality.

## License

---

MIT