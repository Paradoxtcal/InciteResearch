<div align="center">
  <img src="assets/pig_avatar.png" width="140" alt="InciteResearch"/>

  # InciteResearch

  **The missing first step in AI-assisted research**

  Turn a researcher's vague frustration into a breakable assumption, a falsifiable story, and annotated core-algorithm code ready to hand off to Cursor or Claude Code.

  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-brightgreen.svg)](https://github.com/langchain-ai/langgraph) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

  [Quick Start](#quick-start) · [How It Works](#how-it-works) · [Roadmap](#roadmap)

  English | [简体中文](README.zh.md)

</div>

---

Every existing AI research tool starts from a clear topic. [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) turns a direction into a full paper pipeline. [karpathy/autoresearch](https://github.com/karpathy/autoresearch) runs overnight experiment loops given a training script. [DeepInnovator](https://github.com/HKUDS/DeepInnovator) generates hypotheses from literature distributions. They all assume you already know what you want to work on.

Most real research starts somewhere messier: a method that feels wrong, a paper whose assumption seems shaky, an evaluation metric you've never trusted. ResearchAgent works at that moment — before the topic is clear — and produces something concrete enough for the downstream tools to take over.

## How It Works
---

<div align="center">

<img src="assets/workflow.jpg" width="800" alt="InciteResearch Workflow Comparison"/>

</div>

The Socratic Elicitor uses four question archetypes to surface implicit knowledge:

- **Critique trigger** — "Which paper recently made you think something was off?"
- **Frustration trigger** — "As a reviewer, what type of submission would you most want to reject?"
- **Vision trigger** — "What would the most meaningful paper in this area look like in three years?"
- **Constraint trigger** — "What compute budget and timeline are you working with?" (fallback)

The Necessity Check enforces three tests before a method is accepted into the story:

1. Is there a simpler alternative? If yes, why isn't it enough?
2. Does every component have an explicit "because X, we must have Y" in the story?
3. Would scaling up the baseline achieve the same thing?

A method that fails any test is sent back for revision before the proposal is written.

## Key Features
---

- Friction-first elicitation — starts from what bothers you, not what you've already decided
- Assumption Breaking over Gap Analysis — produces directions that challenge a field's premises rather than adding a module to an existing method
- Story arc validation — the method must be the logically necessary consequence of the insight, not merely a reasonable option
- AI-assistant-ready code output — `core_algorithm.py` is written for another AI to read, with `# WHY` and `# DIFF` annotations on every non-trivial line
- Direct IDE integration — `cursor_prompt.txt` is a ready-to-paste instruction for integrating the core code into a chosen baseline
- PDF compression for bulk literature upload — compress and zip dozens of PDFs for ChatGPT's file upload, no API key required
- Session checkpointing — any node can be paused and resumed with `--resume SESSION_ID`

## Web UI (optional)

```bash
pip install streamlit

streamlit run app.py
```

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

```
==========================================================
  InciteResearch
  Your Phase-0 AI research partner: turn vague instincts into core algorithms
==========================================================

Let's start with what you're trying to do. It doesn't need to be clear yet.

What research direction are you thinking about?
> Hyperspectral single object tracking

AI: Got it — hyperspectral single object tracking.
My first question is: since many RGB single-object trackers already handle complex backgrounds and target appearance changes pretty well, if we simply feed hyperspectral data into an existing RGB tracking framework via basic preprocessing (e.g., PCA dimensionality reduction, or stacking multiple bands as deeper channels), what fundamental bottlenecks would we hit?
In other words, what is the kind of problem that you believe can only be properly solved in the hyperspectral domain?

You: I hadn't really thought about doing dimensionality reduction and then using an RGB tracker. My intuition is more that jointly handling spectral, spatial, and temporal dimensions in current hyperspectral trackers introduces a lot of issues. It seems necessary to consider them separately or design a new fusion strategy. An extreme case is what you said: not using spectral information at all.

AI: You mentioned that jointly considering spectral, spatial, and temporal dimensions causes many problems. Concretely, what is the most painful issue for you?
Is it poor generalization, excessive compute/resource cost, or lack of robustness in real applications?

✓ Researcher profile built | friction points: 0

You: Hyperspectral data usually has at least dozens of bands. Each wavelength can behave differently even for the same material — let alone different materials. And during tracking, the target’s appearance may change and the material properties may effectively change too. The possibility space feels huge. Can the model really converge?

✓ Researcher profile built | friction points: 5

Searching papers + mining assumptions...

[WARNING] Semantic Scholar search failed: 429 {"message": "Too Many Requests. Please wait and try again or apply for a key for higher rate limits. https://www.semanticscholar.org/product/api#api-key-form", "code": "429"}

✓ Papers: 57 | generated 3 candidate directions

--------------------------------------------------

Found 3 candidate directions (each breaks a different assumption):

1. Invariant Spectral Representation Learning under Dynamic Environmental Shifts

   Broken assumption: Spectral Feature Stability & Uniqueness (SFSU)

   One-liner: Develop a theoretical framework and model that explicitly models spectral instability rather than assuming uniqueness, learning spectral representations invariant to dynamic environment and material shifts for robust hyperspectral object tracking.

2. Intrinsic-Dimension-Aware Modeling with Sparsity Constraints for High-Dimensional Data

   Broken assumption: Complex Model High-Dim Data Processing (CMHDP)

   One-liner: Build a theory-driven, sparsity-constrained hyperspectral tracking model that explicitly extracts intrinsic low-dimensional manifold structure from high-dimensional observations, overcoming the curse of dimensionality while ensuring convergence and robustness.

3. Spectral-Native Self-Supervised Learning: Breaking Dependence on RGB Transfer

   Broken assumption: RGB Domain Knowledge Transferability (RGBDKT)

   One-liner: Develop a self-supervised learning framework tailored for hyperspectral data to learn robust, spectral-aware representations directly from spectral measurements, mitigating limitations of RGB-domain transfer and improving tracking performance.

Which direction do you pick? (1/2/3, or describe your own idea)
> 1

--------------------------------------------------

Candidate papers (title/abstract). Full text is skipped by default; pick only if needed:

1. [2025] MSITrack: A Challenging Benchmark for Multispectral Single Object Tracking (semantic_scholar)
2. [2025] Hyperspectral Object Tracking With Dual-Stream Prompt (semantic_scholar)
3. [2024] Incorporating Prompt Learning and Adaptive Dropping Hyperspectral Information Tracker for Hyperspectral Object Tracking (semantic_scholar)
4. [2025] A Multi-Stream Visual-Spectral-Spatial Adaptive Hyperspectral Object Tracking (semantic_scholar)
5. [2023] RawTrack: Toward Single Object Tracking on Mosaic Hyperspectral Raw Data (semantic_scholar)
6. [2026] HyperTrack: A Unified Network for Hyperspectral Video Object Tracking (semantic_scholar)
7. [2025] Hyperspectral Video Tracking With Spectral–Spatial Fusion and Memory Enhancement (semantic_scholar)
8. [2023] Learning a Deep Ensemble Network With Band Importance for Hyperspectral Object Tracking (semantic_scholar)
9. [2024] Domain Adaptation-Aware Transformer for Hyperspectral Object Tracking (semantic_scholar)
10. [2023] DSP-Net: A Dynamic Spectral–Spatial Joint Perception Network for Hyperspectral Target Tracking (semantic_scholar)

Which papers should we read full text for (PDF only)? e.g., 1,3 / all / Enter to skip
> 

✓ Selected direction: Invariant Spectral Representation Learning under Dynamic Environmental Shifts

Building the story arc...

✓ Mined 4 implicit assumptions | Novelty: 90% | Feasibility: 70%

✓ Story-arc closure checks passed

✓ Research proposal draft generated (11112 chars)

You: ok

--------------------------------------------------

A research proposal draft is ready.

OK. As an experiment design expert, I will translate your story arc into a detailed and executable research proposal draft. This proposal will strictly follow your core principles, highlight the essential algorithmic differences, choose appropriate baselines, design corresponding ablations, and include key risk analysis.

---

## Environment-Invariant Representation Learning for Hyperspectral Object Tracking: Treat Instability as Signal

### 1. Method Overview

Existing hyperspectral object tracking methods often rely on the Spectral Feature Stability & Uniqueness (SFSU) assumption. When dynamic environmental factors such as illumination, atmosphere, and target material state change drastically, robustness and generalization are fundamentally limited. Our key insight is: rather than treating spectral instability as “noise” to be filtered, we treat it as “signal” that reflects target–environment interaction, and explicitly model this dynamics. This forces us to design a framework that disentangles observed hyperspectral features into (i) an intrinsic, environment-invariant target representation (Z_target) and (ii) an environment-induced, dynamically changing representation (Z_env). By constructing an environment–spectrum causal model to understand how environment modulates spectra, combined with a disentangled representation learning architecture, auxiliary multimodal calibration, and adversarial adaptation, we aim to learn features invariant to environmental shifts for more robust and generalizable tracking.

### 2. Core Algorithm

Our core algorithm explicitly models the causal relationship between environment and spectrum and enforces learning disentangled, environment-invariant representations. The key difference from prior work is that instead of filtering environmental effects as noise, we actively exploit environmental variation as information to guide representation learning.

Pseudo-code: Environment-Invariant Hyperspectral Tracker (EI-HST)

```python
Input:
  H_t: hyperspectral cube at time t (W x H x B; B=bands)
  X_aux_t: auxiliary environmental/spatial data at time t (e.g., RGB, ...)
...
```


```
### PDF compression only (no API key needed)

```bash
python main.py --compress-pdfs --pdf-dir ./papers
# outputs papers_compressed.zip, ready to upload to ChatGPT
```

## Output Format
---

`core_algorithm.py` is not a runnable project. It is a structured specification for an AI coding assistant:
`cursor_prompt.txt` is a single paste-ready block:
```
I have a baseline implementation of [nnUNet / MMDetection / …] from [repo].
I need to modify it to implement [core idea].
The key change is in [file] → [function].
Please apply the following and keep everything else unchanged:
[core algorithm block]
```

## Relation to Other Tools
---

InciteResearch covers Phase 0–2. It deliberately does not reimplement what other tools already do well.

| Tool                                                               | What it does                                 | Relation                                                                 |
| ------------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------ |
| InciteResearch                                                     | Friction → assumption → story → core code    | You are here                                                             |
| [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) | Direction → full paper pipeline              | Downstream — our output is its input                                     |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch)  | Training script → autonomous experiment loop | Downstream — feeds off `core_algorithm.py`                               |
| [DeepInnovator](https://github.com/HKUDS/DeepInnovator)            | Literature → hypothesis via RL-trained model | Complementary — optimizes population novelty; we optimize researcher fit |
| [GPT-Researcher](https://github.com/assafelovic/gpt-researcher)    | Query → literature synthesis                 | Upstream — can replace Phase 1 retrieval                                 |
## Scope of Application
---

**Applicable:** Empirical research involving code implementation, including: CS (CV / NLP / ML / Systems), Computational Biology, Data Science, and Computational Physics/Chemistry.
**Inapplicable:** Pure mathematical proofs, hermeneutics in the humanities and social sciences, and clinical trials requiring ethical review.

## Roadmap
---

**v0.2**
- Local model support (DeepSeek-R1, Qwen2.5) for privacy-sensitive research
- Evaluation benchmark — measure idea execution rate (fraction of directions that enter actual experimentation)
- Implementation decision assistant — given the core algorithm, recommend concrete implementation choices: backbone selection, loss function, optimizer, key hyperparameter ranges, and any small but non-obvious components (e.g. whether to use LayerNorm vs BatchNorm at a specific position, warmup schedule, augmentation strategy) with a one-line justification for each tied back to the story arc

## Contributing
---

The one constraint: don't turn this into a form-filling pipeline. The core of the design lies in how AI gives wings to human thinking. The value is in discovering the research question through dialogue. Any change that skips or shortcuts elicitation defeats the purpose.

Most welcome: real usage cases, new Socratic trigger archetypes, domain-specific prompt improvements, evaluation methodology for idea quality.

## License
---

MIT
