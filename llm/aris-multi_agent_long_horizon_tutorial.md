# 多Agent长时程 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[multi-agent-long-horizon-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/multi_agent_long_horizon_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Multi-Agent & Long-Horizon Agent** — 2024-2026 LLM agent 第二次浪潮，一页拿下面试核心。

1. **范式定位**：单 LLM = "一次 forward 的 system 1"；agent = "把推理拆成 perceive → plan → act → reflect 的 system 2"。**multi-agent** = 多个 LLM role-play 不同身份协作；**long-horizon** = 同一 agent 跨多 turn / 多天保持目标。两者正交但常一起出现（如 ChatDev, MetaGPT）。

2. **multi-agent 三大原型**：
   - **role-play 对话**（CAMEL, Li et al. NeurIPS 2023, arXiv 2303.17760）：assistant-user 双角色 inception prompting；
   - **SOP / 流水线**（MetaGPT, Hong et al. ICLR 2024, arXiv 2308.00352）：把软件开发拆成 ProductManager / Architect / ProjectManager / Engineer / QAEngineer 五个固定角色；
   - **debate / aggregation**（Du et al. ICML 2024 arXiv 2305.14325 + Liang et al. EMNLP 2024 arXiv 2305.19118）：N agent 独立解答 → 互看 → 多轮迭代到一致。

3. **MoA (Mixture-of-Agents, Wang et al. ICLR 2025 Spotlight, arXiv 2406.04692)**：N 个 proposer 各出回答，aggregator 把 N 个回答 concat 进 prompt 再合成。开源 6.5B-70B 组合在 AlpacaEval 2.0 上**超过 GPT-4 Omni**。

4. **debate 收敛——经验观察 + 理论近似**：把每个 agent 的 update 视作对 peer 答案分布的混合算子，若该算子是 **doubly-stochastic averaging** 形（majority-vote softmax + temperature β），按 consensus dynamics 理论（线性 averaging 算子在 doubly-stochastic graph 上）会收敛到 **fixed-point set**（一致分布或离散一致 cluster），**而非唯一 fixed point**（Banach contraction 一般不成立——因为 averaging 算子有特征值 1）。经验上 N=3 与 N=5 在 GSM8K 上 accuracy gap ≈ 1-2pp（diminishing return，见 Du 2023 Fig 4 + Liang 2024 Table 3）。

5. **MemGPT (Packer et al., arXiv 2310.08560, 2023-10)**：把 LLM context 类比 OS RAM，把外部存储类比 disk。**page fault** = 模型 function-call "search archival memory" 触发的 retrieval。延迟代价：一次 page fault ≈ 一次额外 LLM call（数百 ms ~ 数秒），但避免了 context overflow。后续以 **Letta** 名义工程化（2024+）。

6. **GraphRAG (Microsoft 2024, arXiv 2404.16130)**：vector RAG 在 multi-hop 上崩；GraphRAG 先用 LLM 抽 entity-relation 三元组，建知识图谱 + community detection，按层级摘要后再做 retrieval-augmented generation。**global query**（"给整篇文档总结主题"）比 vector RAG 强 70-80%。

7. **tree-search agent**：
   - **ToT (Yao et al., NeurIPS 2023, arXiv 2305.10601)** = BFS/DFS + LLM evaluator；
   - **RAP (Hao et al., EMNLP 2023, arXiv 2305.14992)** = MCTS + world-model rollout；
   - **LATS (Zhou et al., ICML 2024, arXiv 2310.04406)** = MCTS + 自反思 (Reflexion) + value 估计；
   - **Agent-Q (Putta et al., 2024, arXiv 2408.07199)** = MCTS + DPO 离线训练 policy。**ToT**（BFS/DFS + LLM evaluator）没有 tree-search 置信区间公式；**RAP、LATS 原论文用的是标准 UCT**（UCB1 for Trees：$Q(s,a) + w\sqrt{\ln N(s)/N(s,a)}$，不含 AlphaZero 式已学习先验 $P(s,a)$）；本教程 §6.7 的实现示例额外引入 $\text{softmax}(\text{LLM\_logp})$ 作为先验，写成 **PUCT** 公式 $a^* = \arg\max_a Q(s,a) + c_\text{puct} P(s,a) \sqrt{N(s)} / (1+N(s,a))$——这是教程自身的工程改造，不是 RAP/LATS 原论文的统一做法。

8. **long-horizon benchmark 三件套**：**TAU-bench**（Yao et al. 2024 arXiv 2406.12045，customer service multi-turn）、**OSWorld** (Xie et al. NeurIPS 2024 arXiv 2404.07972，real OS GUI)、**SWE-bench** (Jimenez et al., ICLR 2024, arXiv 2310.06770，真实 GitHub issue 修复)。SOTA 2026-05 在 SWE-bench Verified ≈ 75%（Claude 4.6 Sonnet / o3 等），但 OSWorld 仍 < 60%。

9. **易踩坑**：multi-agent 不是越多越好（cost ∝ N, accuracy gain ∝ log N）；long-CoT context overflow 不是写更长 context window 能解（**lost-in-the-middle**, Liu et al. TACL 2024 arXiv 2307.03172）；memory stale 比 memory missing 更危险（错误 retrieval 会产生 confident wrong answer）；sub-agent blame-shifting（"那是 worker A 的错"）在 hierarchical orchestrator 模式中常见。

---

## §10 工程实践：subagent orchestration + budget tracking

```python
from dataclasses import dataclass
from typing import Callable, Dict
import time

@dataclass
class Budget:
    max_tokens: int = 200_000
    max_wall_seconds: float = 600.0
    max_subagent_calls: int = 20
    used_tokens: int = 0
    used_seconds: float = 0.0
    used_subagent_calls: int = 0

    def remaining(self):
        return dict(tokens=self.max_tokens - self.used_tokens,
                    seconds=self.max_wall_seconds - self.used_seconds,
                    subagents=self.max_subagent_calls - self.used_subagent_calls)

    def can_afford(self, est_tokens: int) -> bool:
        return (self.used_tokens + est_tokens <= self.max_tokens
                and self.used_subagent_calls + 1 <= self.max_subagent_calls)

class Orchestrator:
    """ Claude Code 风格的 orchestrator + N worker（带 budget tracking） """
    def __init__(self, llm_orch: Callable, llm_workers: Dict[str, Callable], budget: Budget):
        self.llm_orch, self.workers, self.budget, self.history = llm_orch, llm_workers, budget, []

    def call_subagent(self, name: str, prompt: str, est_tokens: int = 4000):
        if not self.budget.can_afford(est_tokens):
            return None
        t0 = time.time()
        out, used = self.workers[name](prompt)   # returns (output, tokens_used)
        self.budget.used_tokens += used
        self.budget.used_seconds += time.time() - t0
        self.budget.used_subagent_calls += 1
        return out

    def run(self, query: str, max_steps: int = 20):
        for step in range(max_steps):
            decision = self.llm_orch(query=query, history=self.history,
                                     budget=self.budget.remaining())
            if decision["action"] == "finalize":
                return decision["answer"]
            out = self.call_subagent(decision["subagent"], decision["prompt"],
                                     decision.get("est_tokens", 4000))
            if out is None:   # 预算不够 → forced finalize
                return self.llm_orch(query=query, history=self.history,
                                     budget=self.budget.remaining(),
                                     hint="budget_exhausted")["answer"]
            self.history.append({"step": step, "subagent": decision["subagent"], "result": out})
        # 步数耗尽: forced finalize
        return self.llm_orch(query=query, history=self.history,
                             budget=self.budget.remaining(),
                             hint="forced_finalize")["answer"]
```

> 💡 **budget tracking 是 production agent 的硬要求** — 没有 budget 的 agent 会**烧钱无上限**。Anthropic Claude Code / Cursor / OpenAI Agents SDK 都在 SDK 层加 budget hook——不只是 token，还有 wall time、sub-agent 调用次数、API rate-limit。生产环境**必须三轨同时控**。
