# 自进化Agent — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[self-evolving-agents-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/self_evolving_agents_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 Self-Evolving Agents** — 一页拿下 2024-2026 最前沿方向（详见 §1–§11 推导）。

1. **核心问题**：让 agent 在**长程任务**里持续提升能力，而不靠人类反复标注。形式化为 $\pi_t \to \pi_{t+1}$ 的更新算子 $\mathcal{T}$ 的收敛性 / 稳定性 / 渐进有效性。

2. **三大范式**：① Experience-Driven（人造任务 + reward，如 AgentTuning、Voyager）；② Adversarial Self-Play（Challenger-Solver，如 Absolute Zero、Ctx2Skill）；③ Meta-Learning / Reward-Free（无任务无奖励探索 + outcome-based reward，如 Native Evolution）。

3. **能力载体**：**自然语言 skill / world knowledge K 写成 markdown**——这是 2024-2026 年最重要的范式转移，绕开参数更新，所有内容都是 inference-time `system_prompt += K`。

4. **Ctx2Skill 5-角色 self-play**（arXiv 2604.27660）：Challenger / Reasoner / Judge / Proposer / Generator，**冻结 LM 但 skill set 在进化**。Cross-Time Replay 选 $\arg\max_i \rho^h_i \cdot \rho^e_i$ 防 adversarial collapse。

5. **Native Evolution 两阶段**（arXiv 2604.18131）：Evolution phase 无任务无 reward 探索 → 蒸馏 markdown K；Execution phase 用 K 当 system prompt。训练信号 $R_\text{evolve}(\mathcal{K}) = \text{Success}(\mathcal{T}_E\mid\mathcal{K}) - \text{Success}(\mathcal{T}_E\mid\varnothing)$。

6. **A²RD 三件套**（arXiv 2605.06924）：MVMem（textual states + frames + videos + dependency DAG）+ Adaptive Segment Gen + HITS（frame-level + video-level 自检）。直接迁移到任意长程 agent 当作 memory + audit 模板。

7. **理论上界**：[arXiv:2601.05280] 在**外源 grounding 信号缺失**时 closed-loop density matching 退化（**不是**所有 reward-free 训练必崩，是该 setting 下的特定结论）；[arXiv:2507.00075] 把 solver-verifier gap 建模 + 经验拟合成 capability dynamics。

8. **常见故障**：adversarial collapse（Challenger 极端化）、memory drift（K 内部矛盾累积）、reward hacking（self-rewarding 漂移）、bias amplification（agent 在自己输出上重训）、capability ceiling（外源 grounding 缺失时 self-improvement 退化）。

---

## §10 Skill / K 检索与排序（工程实践）

实际部署时，skill library 几十到几百条，必须**按需 load**——否则 token 爆炸。

### 10.1　Hybrid retrieval pipeline

```python
def hybrid_skill_retrieval(task: str, skills: list, k=3):
    """
    Stage A: 粗筛 (vector embedding, fast)
    Stage B: 精排 (LLM scoring on description, accurate)
    Stage C: 按 trigger 段精确匹配 (deterministic)
    """
    # ── Stage A: BM25 + dense embedding hybrid ──
    bm25_scores = bm25_search(task, [s.description for s in skills], topn=20)
    dense_scores = dense_search(task, [s.embedding for s in skills], topn=20)
    candidates = top_k(merge(bm25_scores, dense_scores), n=10)

    # ── Stage B: LLM rerank ──
    reranked = []
    for skill in candidates:
        prompt = f"task={task}\nskill trigger={skill.trigger}\n" \
                 f"Q: relevant? (yes / no / partial)"
        verdict = llm(prompt)
        score = {"yes": 1.0, "partial": 0.5, "no": 0.0}[verdict]
        reranked.append((score, skill))

    # ── Stage C: 关键词强匹配 ──
    keyword_hits = [s for s in skills
                    if any(kw in task.lower() for kw in s.exact_triggers)]

    # 合并去重 → 取 top k
    final = top_k(reranked + [(2.0, s) for s in keyword_hits], k=k)
    return [s for _, s in final]
```

### 10.2　Skill 排序公式

加权融合 3 个信号：

$$\text{score}(s, q) = \alpha_\text{sim}\, \cos(\mathbf{e}_s, \mathbf{e}_q) + \alpha_\text{prior}\, \log(1 + n_\text{used}(s)) + \alpha_\text{recent}\, \gamma^{\Delta t}$$

其中 $n_\text{used}$ 是历史调用次数（越常用越可靠），$\gamma^{\Delta t}$ 是 recency decay。

### 10.3　Skill 的更新（防陈旧）

每个 skill 维护：

- `success_count`, `fail_count`
- `last_updated`
- `version`

触发更新条件：

- `fail_count / total > τ_fail`（失败率过高）→ 修订
- `last_updated > T_stale`（陈旧）→ 重新探索
- environment 变化检测 → 触发 Native-Evolution-style 重新蒸馏
