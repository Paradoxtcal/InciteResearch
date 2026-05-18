<div align="center"> <img src="assets/pig_avatar.png" width="140" alt="InciteResearch"/>

# InciteResearch

**AI辅助科研中缺失的第一步**

把朦胧的「不对劲」推进为可审视的假设与故事线，以及尽量**落到方法层面**的提案表述。

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![arXiv](https://img.shields.io/badge/arXiv-2605.06345-b31b1b.svg)](https://arxiv.org/abs/2605.06345) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[快速上手](#%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B)

[English](README.md) | 简体中文

</div>

---

<div align="center">
  <div style="position:relative;display:inline-block;width:min(820px,100%);margin:auto;">
    <img src="assets/1.png" alt="" style="display:block;width:100%;height:auto;"/>
  </div>
</div>

现有的 AI 科研工具几乎都从一个已经清晰的具体**任务**出发：确定的问题设定、标准数据集或训练配方。真实研究往往更早：某个方法质感不对、某条假设经不起推敲、某个指标你从不买账。InciteResearch 面向的是**课题尚未被压缩成标题之前**的那一段：把朦胧的「不对劲」压成**可执行的方法叙述**（要构建什么、与谁对照、怎样算被证伪），细化到足以交给下游自动化工具。

> 📄 **论文:** [More Than Can Be Said: A Benchmark and Framework for Pre-Question Scientific Ideation](https://arxiv.org/abs/2605.06345) — Jie Yu, Song Qiu (arXiv:2605.06345 [cs.AI])

## 核心特性

---

- 摩擦优先的诱导——从困扰你的事情出发，而不是从已经写好的研究问题出发
- 假设打破而非缺口分析——产出攻击领域隐含前提的方向，而不是在 baseline 上叠加一个模块
- 故事线验证——**方法**必须是洞见在逻辑上的必然推论，而不仅仅是若干合理选项之一
- **结构化产出**——以 `research_proposal.md` 与会话状态为主；**`--refine-method`** 在同一故事线上把内容扩展为可追溯文献的长 **PLAN**

## 新范式

---

<div align="center"> <img src="assets/InciteResearch_p.png" width="800" alt="InciteResearch 范式图"/> </div>

大多数 AI 科研工具将自己定位为自主系统：只要脚手架足够完善，模型就应该能端到端地生成一篇论文。InciteResearch 建立在一种不同的信念之上。如上图所示，整个过程从随意的人机对话开始——研究者掌握方向，AI 通过打破假设来拓展思考空间。人类随后反思并修正；AI 也提供自己的方向性建议。这个循环持续进行，直到模糊的灵感最终以结构化研究提案的形式被显式化。

这不是一个"人在环路中"的流水线，而是一种认知协作：研究者对"某处有问题"的隐性感受是不可替代的输入，AI 的角色是延伸这种直觉——赋予它杠杆、清晰度和延伸力——而不是取代它。

三个算子使这一点变得具体：

- **E（诱导）** — 将隐性摩擦整理为结构化档案（摩擦点、动机、约束、品味及**可操作的方法切入点**），并以摩擦为引力中心组织其余材料。
- **V（有效性与问题重构）** — 不是在文献中寻找空白，而是识别那个一旦打破就能最大化可行性与新颖性之积的隐藏假设。输出不是对原始方向的增量改写，而是在一个新的概念坐标系下的重新锚定。
- **N（必要性检验）** — 扮演严格审稿人的角色。提出的方法必须是洞见在逻辑上的必然推论，而不仅仅是若干合理选项之一。任何无法从因果链中推导出来的组件都会被拒绝。

这一范式——**诱导 → 违反 → 必要化**——不是对苏格拉底式追问的任意模块化。消融实验结果显示，去掉 V 算子会导致所有指标出现最严重的下降（新颖性 4.250 → 3.500，影响力 4.397 → 3.720），证明假设违反承载的是结构性功能，而非仅仅是风格上的差异。去掉 E 算子会导致新颖性出现第二大幅度的下降，并且是唯一一个使可行性_上升_的消融实验，揭示出 E 阶段的作用在于将生成过程锚定在真实摩擦点上，而非表面层次的重新表述。

## 快速上手

---

### 安装

```bash
git clone https://github.com/Paradoxtcal/InciteResearch
cd InciteResearch
pip install -r requirements.txt

cp .env.example .env
```

按 `.env.example` 配置至少一种 LLM 及对应的 `*_MODEL`。

### 主流程（朦胧想法 → 结构化提案）

```bash
python main.py
```

> **提示（已有困惑时）：** 在第一次询问**研究方向**时，直接把困惑写清楚即可。后续若出现带编号的选项、想尽快往下走，**每次问答都输入 `1` 回车**（选第一项）即可跳过式前进。

会话以开放式问题开始，自然语言回答即可，无需表单，语言不限。

恢复会话：

```bash
python main.py --resume <research_state 中的 session_id>
```

### 方法精炼实验室（`--refine-method`）

**推荐顺序：**需要根目录 `research_proposal.md` 时，先 `python main.py`，再执行：

```bash
python main.py --refine-method
```

## 适用范围

---

**适用：** 涉及代码实现的实证研究，包括：CS（CV / NLP / ML / Systems）、计算生物学、数据科学、计算物理/化学。

**不适用：** 纯数学证明、人文社科的诠释学研究，以及需要伦理审查的临床试验。

## 参与贡献

---

唯一的约束：不要把这变成一个填表流水线。设计的核心在于 AI 如何给人类思维插上翅膀。价值在于通过对话发现研究问题。任何跳过或缩短诱导过程的改动都会违背初衷。

最欢迎：真实使用案例、新的苏格拉底式触发原型、领域专属的提示词改进、评估想法质量的方法论。

## 许可证

---

MIT