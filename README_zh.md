<div align="center"> <img src="assets/pig_avatar.png" width="140" alt="InciteResearch"/>

# InciteResearch

**AI辅助科研中缺失的第一步**

将研究者模糊的不满转化为可打破的假设、可证伪的故事主线，以及带注释的核心算法代码——随时可交付给 Cursor 或 Claude Code。

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![arXiv](https://img.shields.io/badge/arXiv-2605.06345-b31b1b.svg)](https://arxiv.org/abs/2605.06345) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[快速上手](#%E5%BF%AB%E9%80%9F%E4%B8%8A%E6%89%8B)

[English](README.md) | 简体中文

</div>

---

现有的 AI 科研工具都从一个清晰的课题出发：[AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) 将研究方向转化为完整的论文流水线；[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 给定训练脚本后可以跑通夜间实验循环；[DeepInnovator](https://github.com/HKUDS/DeepInnovator) 从文献分布中生成假设。它们都预设你已经知道自己要研究什么。

但真实的研究往往从更混乱的地方开始：一个感觉有问题的方法、一篇假设存疑的论文、一个你从未信任过的评估指标。InciteResearch 就工作在这个时刻——在课题还不清晰之前——并产出足够具体的内容，让下游工具得以接手。

> 📄 **论文:** [More Than Can Be Said: A Benchmark and Framework for Pre-Question Scientific Ideation](https://arxiv.org/abs/2605.06345) — Jie Yu, Song Qiu (arXiv:2605.06345 [cs.AI])

## 核心特性

---

- 摩擦优先的诱导方式——从困扰你的事情出发，而不是你已经决定好的方向
- 假设打破而非缺口分析——产出挑战领域前提假设的方向，而不是给现有方法加一个模块
- 故事线验证——方法必须是洞见在逻辑上的必然推论，而不仅仅是一个合理的选项
- AI 助手可读的代码输出——`core_algorithm.py` 专为另一个 AI 阅读而写，每一行非平凡代码都附有 `# WHY` 和 `# DIFF` 注释
- 直接集成 IDE——`cursor_prompt.txt` 是一段可直接粘贴的指令，用于将核心代码集成进选定的 baseline

## 新范式

---

<div align="center"> <img src="assets/InciteResearch_p.png" width="800" alt="InciteResearch 范式图"/> </div>

大多数 AI 科研工具将自己定位为自主系统：只要脚手架足够完善，模型就应该能端到端地生成一篇论文。InciteResearch 建立在一种不同的信念之上。如上图所示，整个过程从随意的人机对话开始——研究者掌握方向，AI 通过打破假设来拓展思考空间。人类随后反思并修正；AI 也提供自己的方向性建议。这个循环持续进行，直到模糊的灵感最终以结构化研究提案的形式被显式化。

这不是一个"人在环路中"的流水线，而是一种认知协作：研究者对"某处有问题"的隐性感受是不可替代的输入，AI 的角色是延伸这种直觉——赋予它杠杆、清晰度和延伸力——而不是取代它。

三个算子使这一点变得具体：

- **E（诱导）** — 将研究者的隐性摩擦显式化为一个五维结构化档案：摩擦点、动机、约束、品味、精炼课题。摩擦点是引力中心，其余一切都围绕它运转。
- **V（有效性与问题重构）** — 不是在文献中寻找空白，而是识别那个一旦打破就能最大化可行性与新颖性之积的隐藏假设。输出不是对原始方向的增量改写，而是在一个新的概念坐标系下的重新锚定。
- **N（必要性检验）** — 扮演严格审稿人的角色。提出的方法必须是洞见在逻辑上的必然推论，而不仅仅是若干合理选项之一。任何无法从因果链中推导出来的组件都会被拒绝。

这一范式——**诱导 → 违反 → 必要化**——不是对苏格拉底式追问的任意模块化。消融实验结果显示，去掉 V 算子会导致所有指标出现最严重的下降（新颖性 4.250 → 3.500，影响力 4.397 → 3.720），证明假设违反承载的是结构性功能，而非仅仅是风格上的差异。去掉 E 算子会导致新颖性出现第二大幅度的下降，并且是唯一一个使可行性_上升_的消融实验，揭示出 E 阶段的作用在于将生成过程锚定在真实摩擦点上，而非表面层次的重新表述。

InciteResearch 的输出——一份结构化提案、一个带注释的 `core_algorithm.py` 和一个 `cursor_prompt.txt`——专门设计为下游执行工具的输入。InciteResearch 覆盖第 0–2 阶段，不重新实现其他工具已经做得很好的部分。

## 快速上手

---

```bash
git clone https://github.com/Paradoxtcal/InciteResearch
cd InciteResearch
pip install -r requirements.txt

cp .env.example .env
# 在 .env 中填入你的 API_KEY

python main.py
```

会话以一个问题开始，自然地回答即可，无需填写任何表单，支持任意语言。

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