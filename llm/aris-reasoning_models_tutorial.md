# 推理模型Reasoning — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[reasoning-models-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/reasoning_models_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 Reasoning Model** — 2024-2026 LLM 最大范式转移，一页拿下面试核心。

1. **范式转移**：以前 scale **训练算力**（参数 + 数据），现在 scale **推理算力**（reasoning tokens / search / verification）。Snell et al. 2024 (arXiv 2408.03314) 给出 **compute-optimal test-time scaling** 配方：相同推理 FLOPs 下，best-of-N + PRM beam search + sequential revision 的混合策略比单一 best-of-N 高 **>4×** 效率；FLOPs-matched 设置下，小模型 + 优化 test-time compute 在某些任务上能匹配/超过 **14×** 更大的模型（PaLM-2 各档参数量官方未公开，"14×" 是 FLOPs-matched 推理算力对比口径，详见 §1.1）。

2. **o1 (OpenAI Sep 2024)**：用 RL 训练 hidden chain-of-thought，API 只返回 `reasoning_tokens` 计数而非内容。**o3 (Dec 2024)** 在 ARC-AGI 上 75.7% (低算力) / 87.5% (高算力 172× 预算)——首次在抽象推理 benchmark 上接近人类。

3. **DeepSeek-R1-Zero (arXiv 2501.12948, Jan 2025)**：**纯 RL 从 base model 直接训**，无 SFT cold-start，rule-based reward（答案对/错 + 格式），用 **GRPO**（无 critic），涌现出 "aha moment"——模型自己学会反思、回溯、验证。

4. **DeepSeek-R1**：四阶段 pipeline = SFT 冷启动（数千高质 CoT）→ reasoning-oriented RL → rejection sampling + 通用 SFT → 全场景 RL。在 MATH-500、AIME 等数学/代码 benchmark 上对齐 o1。

5. **GRPO（DeepSeekMath, arXiv 2402.03300）**：去掉 critic value network；对每个 prompt sample $G$ 个回答 $\{o_i\}$，用 **group-relative advantage** $A_i = (r_i - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$ 替代 GAE。显存降一半 + 训练更稳。

6. **PRM vs ORM**：ORM（outcome reward）只评 final answer；PRM（process reward，Lightman 2023 "Let's Verify Step by Step"）对每个推理 step 打分，best-of-N 选 trace 上更优。Math-Shepherd (Wang et al. 2023, arXiv 2312.08935) 从中间 step 采样 **Monte Carlo completion rollouts**（不是 MCTS tree search），用最终答案的 soft/hard estimation 自动标 step label，省去人标。

7. **s1 (Muennighoff Feb 2025, arXiv 2501.19393)**："Wait" budget forcing——在 `</think>` 处强行追加 "Wait" 让模型继续思考，1K 样本 SFT + 推理控制超 o1-preview 27% (AIME24)。

8. **易踩坑**：CoT ≠ 推理（模型可能 post-hoc 编故事）；self-consistency 在 distribution-shifted 题上崩；PRM 训练易过拟合 step pattern；GRPO 在 long-CoT 上 critic-free 反而是优势（critic 难学）。

---

## §10 25 高频面试题（L1 必会 / L2 进阶 / L3 顶级 lab）

按 gpt-5.5 xhigh 模拟的顶级 lab interviewer 视角排序。

### L1 必会题（任何 LLM 岗都会问）

<details>

<summary>Q1. Chain-of-Thought 是什么？什么时候用？</summary>

- few-shot prompt 给模型展示 "step-by-step" 推理 demonstration

- 在大模型（>62B）上 emergent 出推理能力（GSM8K +30+%）

- 小模型上 CoT 反而变差

- 后续 Kojima 2022 "Let's think step by step" 发现 zero-shot CoT 也行

把 CoT 当成"魔法 prompt"——它只是触发，能力来自 base model 自身

</details>

<details>

<summary>Q2. Self-Consistency 怎么工作？为什么比 greedy 好？</summary>

- temperature > 0 sample N 条 CoT

- 提取每条的 final answer，做 normalize（去单位、化分数）

- 多数票选 majority answer

- 直觉：正确答案是"吸引子"，多条 sampling 路径会收敛于此

- Wang 2022 报告 GSM8K +17.9%

不做 answer normalization；以为温度越高越好（实际 0.5-0.7 最佳）

</details>

<details>

<summary>Q3. Tree-of-Thought (ToT) 与 CoT 区别？</summary>

- CoT 是 autoregressive 单链路径

- ToT 是显式树搜索：每个节点是 thought，sample k 个 child，LLM 自评分

- ToT 可以回溯（backtrack）和剪枝

- 但 ToT 需要外部搜索框架 + 多次 LLM call，比 CoT 慢 5-50×

- Game of 24: CoT 4% → ToT 74%（GPT-4）

只说 ToT 是"多次 CoT"——它的核心是显式 evaluator + backtrack

</details>

<details>

<summary>Q4. ORM 和 PRM 区别？</summary>

- ORM (Outcome RM)：整条 trace 一个 reward（基于 final answer 对错）

- PRM (Process RM)：每个 reasoning step 一个 reward

- PRM 信号密集但标注贵（人标 PRM800K 80 万 step）

- Math-Shepherd (Wang et al. 2023, arXiv 2312.08935) 从中间 step 出发采样 **Monte Carlo completion rollouts**（不是 MCTS tree search），用「rollout 多少条最终对」的 soft/hard estimation 自动标 PRM step label

- Lightman 2023: N=1860 重排上 PRM 78.2% > ORM 72.4% > majority vote 69.6%

把 PRM 当成"训练 reward"——它主要用于推理时 verify，不一定参与 RL

</details>

<details>

<summary>Q5. Best-of-N 怎么工作？什么时候饱和？</summary>

- 同 prompt sample N 条 trace

- 用 verifier (ORM 或 PRM) 选最高分那条

- 简单题：N=4-8 饱和

- 难题：可以涨到 N=64-128 才饱和

- 理论上限 oracle pass@N = $1-(1-p_1)^N$，但 verifier 不完美时远低于此

以为 N 越大越好——verifier 误差会让 BoN 在某点之后开始下降（"verifier overfit to surface features"）

</details>

<details>

<summary>Q6. o1 / R1 的 reasoning_tokens 是什么？</summary>

- 模型生成的 hidden chain-of-thought tokens

- API 只返回 token 计数（如 OpenAI o1 返回 `reasoning_tokens` 字段），内容不公开

- 用户付费按 reasoning_tokens 计费（这就是 o1 贵的原因）

- R1 把 reasoning 包在 `<think>...</think>` 里，全文可见

- 推理算力 ≈ reasoning_tokens × model FLOPs/token

把 reasoning_tokens 当 "no-cost 优化"——它显著影响延迟和成本

</details>

<details>

<summary>Q7. GRPO 和 PPO 主要区别？</summary>

- PPO 需要 critic (value network)，估计每个 token 的 baseline

- GRPO 用 **group statistics 替代 critic**：同 prompt sample G 个 trace，advantage = $(r_i - \mu)/\sigma$

- GRPO 把 trace-level advantage 广播给该 trace 所有 token

- 显存降一半（无 critic），long-CoT 上 critic 本来就难学，所以 GRPO 反而稳

- 共享：PPO clipping、KL 正则 to reference policy

以为 GRPO 是"小改动"——它在 long-CoT 上的稳定性优势是质变

</details>

<details>

<summary>Q8. R1-Zero 的 "aha moment" 是什么？</summary>

- DeepSeek-R1 paper Fig 3 报告：纯 RL 训练几 K steps 后

- CoT 长度自发增长（数百 → 数千 token）

- 自发出现 "Wait, let me reconsider..." 等反思 pattern

- 性能跳变（AIME pass@1 15% → 70%）

- 直觉：rule-based reward + GRPO 让"思考更久 + 自验证"成为高 reward strategy

把 "aha moment" 当玄学——它是 reward shaping + 长 episode RL 的可预期涌现

</details>

<details>

<summary>Q9. R1-Distill 为什么用 SFT 不用 RL？</summary>

- R1 的 reasoning 能力靠大 base + 强 RL 涌现

- 直接对小模型 RL 难涌现（base 太弱，rollout 几乎全错，reward 信号过稀疏）

- 用 R1 生成的全部约 80 万条数据（60 万 reasoning + 20 万通用）做 SFT，等价 demonstration learning

- 论文报告：32B SFT-distill 在 AIME 上 72.6 vs 直接对 Qwen-32B RL 的 47.0

- Implication：小模型上**蒸馏 reasoning > 直接 RL**（在当前算法下）

以为 RL 总是比 SFT 好——前提是 base 够强

</details>

<details>

<summary>Q10. Snell 2024 的 "test-time compute > parameter scaling" 是什么意思？</summary>

- 固定 inference 算力预算，让 1B 模型多 sample + verify

- 在某些 MATH 子集上，可超过 14× 大的模型的 greedy 性能（PaLM-2 参数量官方未公开，"14×" 是 FLOPs-matched 推理算力对比口径，非精确参数量比值）

- 前提：base model 在该任务有 non-trivial pass@1（>30%）

- 不是普适：完全不会的任务，再多 test-time compute 也救不回来

- 实际工业部署常用 R1-Distill + BoN=8 + PRM 替代直接调 R1

把它当 "scaling law 终结" ——它只是"另一个维度的 scaling law"，不取代训练 scaling

</details>

### L2 进阶题（reasoning 方向 / research 岗）

<details>

<summary>Q11. 手推 GRPO advantage 公式，以及 std=0 的情况如何处理？</summary>

- 对每个 prompt sample G traces，得 rewards $\{r_1, \dots, r_G\}$

- $\mu = \frac{1}{G}\sum r_i$, $\sigma = \sqrt{\frac{1}{G}\sum(r_i - \mu)^2}$（biased std）

- $A_i = (r_i - \mu)/(\sigma + \epsilon)$

- 当 $G$ 个 rewards 全相同（全对或全错）→ $r_i - \mu = 0$ 且 $\sigma = 0$，所以
  - 加 $\epsilon$ 时 advantage $= 0/\epsilon = 0$ → **policy-gradient 项归零**；但 KL 正则项仍存在，**总 loss 仍可能有 KL 更新**（把 policy 拉回 reference）
  - 不加 $\epsilon$ 时是 $0/0 = $ NaN
  - 注意不是 $\pm\infty$（分子也是 0）

- 实践：要么 skip 该 prompt（GRPO_loss=0），要么 clamp $\sigma$ 到一个 floor（如 0.1）

- 这种情况说明该 prompt **过易或过难**，data filtering 应剔除

只写公式不说 std=0 的边界

</details>

<details>

<summary>Q12. R1 的 4 阶段 pipeline 每阶段目标是什么？</summary>

- **Stage 1 Cold-start SFT**：让 base 学会 human-readable reasoning format（不为 reasoning 能力本身）

- **Stage 2 Reasoning RL**：GRPO + 规则 reward 提升数学/代码推理

- **Stage 3 Rejection sampling + SFT**：扩展到非可验证领域 + 保留通用能力

- **Stage 4 All-scenario RL**：safety + helpfulness 收尾（类似 RLHF）

- 关键：reasoning 来自 Stage 2 RL；Stage 1 + 3 是 readability/generalization 注入；Stage 4 是对齐

把 4 阶段当"做菜步骤"——其实每阶段功能正交

</details>

<details>

<summary>Q13. 为什么 R1 的 reward 是 rule-based 而非学的 RM？</summary>

- 数学/代码可以**程序化验证**（答案正则匹配、单元测试）

- 学的 RM 容易被 hack（reward model overoptimization → policy 找到 trick 而非真正解题）

- 规则 reward 提供接近 ground-truth 的 signal，**避开了 learned-RM 漂出训练分布的主要 failure mode**（仍可能被正则漏洞、格式 trick、test-set 污染 hack，但远比 RM hacking 容易堵）

- 代价：只能用于可验证任务（math/code/format），不适合开放任务

- R1 Stage 4 又加了 RM 处理 helpfulness/harmlessness——可验证任务用 rule，开放任务用 RM

把 rule-based 当"简单"——它的关键是难以被 RM-hacking，而非简单

</details>

<details>

<summary>Q14. Budget forcing ("Wait" trick) 为什么有效？</summary>

- s1 在 1K reasoning trace 上 SFT，模型学会 `<think>...</think>` 格式 + 反思 pattern

- 推理时如果模型试图输出 `</think>` 但 token 数还没到 target budget

- 强行替换为 "Wait"，模型自然续接反思 token

- 等价 forcing 模型留在 thinking mode，多 sample 几条 internal reasoning path

- 失败模式：base 完全没见过反思 pattern → "Wait" 后接 garbage（s1 的 1K SFT 是必要前提）

以为 "Wait" 是 prompting trick——它依赖 SFT 注入的反思 pattern

</details>

<details>

<summary>Q15. PRM 怎么用在 best-of-N？aggregation 怎么选？</summary>

- 对 N 个 trace，每条用 PRM 给每个 step 打分 $p_1, \dots, p_T$

- 三种 aggregation：**min**, **prod (log-sum)**, **mean**

- Lightman 2023: **min ≈ prod >> mean**（mean 被高分 step 掩盖弱 step）

- min 直觉：trace 强度取决于最弱一环

- 代码细节：prod 用 log-sum 避免数值下溢

只用 mean——常见错误

</details>

<details>

<summary>Q16. PUCT 公式是什么？c_puct 怎么调？</summary>

- $U(s, a) = Q(s, a) + c_\text{puct} \cdot \pi(a \mid s) \cdot \sqrt{N(s)} / (1 + N(s, a))$

- $Q$ = exploit；$c_\text{puct} \cdot \pi \cdot \sqrt{N}/(1+N(s,a))$ = explore

- $c_\text{puct}$ 大：偏向 exploration（policy prior 和未访问 action 影响大）

- $c_\text{puct}$ 小：偏向 exploitation（已发现的高 Q action 主导）

- AlphaZero 用 $c_\text{puct} \approx 1.0$；MCTS-for-LLM 常用 1.5-2.0（因为 LLM policy prior 比围棋更准）

- 注意：分子是 $\sqrt{N(s)}$（parent visits），分母是 $1 + N(s, a)$（child visits）

把 $\sqrt{N}$ 错记为 child visits（错的）

</details>

<details>

<summary>Q17. Math-Shepherd 怎么自动标 step label？关键假设是什么？</summary>

- 对 trace 中每个 step $s_t$，从 $s_t$ 出发 rollout K 条 completion

- 数多少条最终 answer 正确，得 estimated step quality $\hat{q}_t = (正确数) / K$

- 用 $\hat{q}_t$ 当 BCE/MSE 标签训 PRM

- **关键假设**：base model 在该任务有 non-trivial success rate（否则 rollout 全错，$\hat{q}_t$ 全 0）

- Bootstrap 问题：弱 base → 没法用 MC rollout label；强 base → 不需要 PRM

- 实际：用中等强度 base（Mistral-7B post-SFT）作 rollout source

以为 MC rollout label 是免费——它需要 base 已有部分能力

</details>

<details>

<summary>Q18. CoT 是真的反映推理吗？怎么验证？</summary>

- 部分真实，部分 post-hoc rationalization（共识）

- Turpin 2023: 给 biased exemplar，模型 CoT 给出 plausible 但错误的解释（没 mention bias）

- Anthropic Sleeper Agents (Hubinger 2024): CoT 内容影响下游 action，不纯 post-hoc

- 验证方法：causal intervention（改 CoT 看 output 是否变）、faithfulness benchmark

- 面试 talking point：保持 balanced view，不要走极端

只说"CoT 是 explanation"或"CoT 全是 post-hoc"——两个极端都错

</details>

<details>

<summary>Q19. 怎么判断哪个 reasoning model 适合你的任务？</summary>

- 任务可验证 (math/code)：rule-based RL 路线（R1 / R1-Distill）

- 任务开放 (writing/dialogue)：hybrid RM 路线（Claude 3.7-extended / o1）

- 抽象推理 (ARC-AGI)：o3 高算力档（其他模型在 ARC-AGI 上仍很弱）

- 形式化证明：DeepSeek-Prover-V2（Lean 4 集成）

- 部署预算严：R1-Distill-7B/14B + best-of-N + PRM verifier（比直接 R1 便宜 10×+）

不看任务就推荐 o1——错配会贵且效果差

</details>

<details>

<summary>Q20. KL 正则在 GRPO 里的作用是什么？β 怎么调？</summary>

- $\mathcal{L}_\text{total} = \mathcal{L}_\text{PG} + \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_\text{ref})$

- $\pi_\text{ref}$ = 训练开始前的 policy（一般是 SFT 后或 base）

- $\beta$ 大：policy 不漂离 reference，但学不动新能力

- $\beta$ 小：policy 自由探索，但可能 collapse（生成 garbage）

- DeepSeek-R1 用 $\beta = 0.001$（很小，鼓励探索；社区复现/推测数值，R1 论文正文未逐一公布完整 RL 超参数表）；标准 RLHF 用 0.01-0.1

- 对 long-CoT，KL（k3 估计）是逐 token 量；用正确的"按 completion 长度归一化再对 G 条取平均"reduction 时，总 loss 不会随 CoT 长度机械放大——但 per-token KL 估计的噪声/方差会随轨迹变长而累积，所以 long-CoT 上 $\beta$ 仍常取比短 CoT 更小的值（这取决于 reduction 是 sum 还是 mean，不是"总量线性放大"这么简单）

照搬 RLHF 的 $\beta$ 到 long-CoT——会过度抑制探索

</details>

### L3 高级题（顶级 lab / 研究方向）

<details>

<summary>Q21. GRPO 比 PPO sample efficient 的 root cause 是什么？（不止"省 critic"）</summary>

- **Trace-level reward 与 trace-level credit 完美对齐**：当 reward 只来自 final answer，PPO 用 critic 做 per-token credit 反而引入噪声；GRPO 直接 broadcast trace-level advantage——advantage 估计本身仍带有 group baseline 引入的有限偏差（mean/std 都是有偏的样本统计），但比 critic 在长 episode 上的 high-variance 估计更稳，且对 trace-level reward 而言其方差更低

- **同 prompt 内 contrast 消除 prompt-level baseline noise**：advantage = $(r-\mu)/\sigma$ 等价于 paired comparison，比 critic 估的全局 baseline 准

- **Long-CoT 上 critic 难学**：reward 极度稀疏（episode 4K-32K token），$V_\psi$ 在中间 token 上几乎随机；GRPO 跳过这个学习问题

- **Rule-based reward 难以被 RM-hacking**：r1 用规则 reward，没有 learned RM 可被 over-optimization；policy 优化方向接近 ground-truth（仍可能被正则/格式漏洞 hack，但避开了 RM-distribution-shift 这条主路）

- **Group size G 控制 variance**：variance ∝ $1/G$；$G=16$ 给出足够低 variance 同时不爆显存

- 结论：GRPO 不是"小改动"，是在 long-CoT + rule reward 设定下的 algorithmically right answer

只说"省 critic"——表面原因

</details>

<details>

<summary>Q22. R1-Zero 的纯 RL 涌现 vs 历史 PPO RLHF (InstructGPT) 差别在哪？为什么前者突破后者不能？</summary>

- **Reward 来源**：R1-Zero 用 rule-based，InstructGPT 用学的 RM (preference model)

- **Reward density 与稀疏度**：rule reward 在 long-CoT 上是 response/trace-level sparse 但难以被 hacking 且 signal 接近 ground-truth；InstructGPT 的 learned RM 也输出 **response-level scalar preference reward**（不是逐 token 打分），token-level advantage 是 critic + GAE + KL penalty 共同构造的，RM 在 RLHF 全流程里仍易遭遇 reward overoptimization（policy 漂出 RM 训练分布 → 拿到不真实的高分）

- **Algorithm**：R1-Zero 用 GRPO；InstructGPT 用 PPO + critic + RM

- **Reward scope**：R1-Zero 训 reasoning（可验证）；InstructGPT 训 alignment（开放）——前者有 oracle reward，后者没有

- **Base model**：R1-Zero 用 V3-Base (671B MoE)，已有强 pretrained reasoning prior；InstructGPT 是 GPT-3 (175B dense)

- 历史原因：2022-2023 PPO+RM 范式被 RM overopt + critic 难学拖累；rule reward 在数学/代码上才走通

- 含义：**reasoning RL 突破 = rule reward + GRPO + 强 base + long-CoT 联合作用**，不是单点技术胜利

只说"DeepSeek 用了 GRPO"——错过整个 paradigm shift

</details>

<details>

<summary>Q23. 为什么 "Wait" 这么简单的 trick 能超 o1-preview？这告诉我们什么？</summary>

- s1 的核心：1K 精选 trace SFT (Qwen2.5-32B) + "Wait" budget forcing

- 第一层解释：1K SFT 已让模型学会 `<think>` 格式 + 反思 pattern 的"shape"

- 第二层：模型实际上**已经"知道怎么想"**（在 pretrain 中见过大量人类推理），SFT 只是激活 + 格式化

- 第三层："Wait" 强制模型在 thinking boundary 停下，重新采样——相当于强行做了一次 in-context self-revision

- 推论：**reasoning 能力的核心是激活而非注入**——base model 已有大量推理 prior

- Implication for research：
  - 不要假设 reasoning 必须靠大规模 RL 才能出
  - SFT data quality > quantity（s1K 1000 条 > 大量低质数据）
  - 推理控制 (budget forcing) 是 vastly underexplored 维度

- 反思：s1 不否定 R1 路线——R1-Distill 也是 distill 一种 SFT，s1 是这条思路的极端版本

只说 "s1 很简单很厉害"——错过 reasoning = activation 这个观察

</details>

<details>

<summary>Q24. 比较 sequential test-time compute (long CoT) 和 parallel test-time compute (BoN / MCTS) 的本质差异。什么时候选哪个？</summary>

- **Sequential（o1, R1, s1）**：单条 trace 拉长，模型自反思 + 自验证
  - 优势：单 KV cache（显存友好）；信息在 trace 内连续传递（后面 step 看得到前面所有 reasoning）
  - 劣势：早期错误传播到末端（无 backtrack）；难任务上需要极长 trace（10K-100K token）

- **Parallel（BoN, ToT, MCTS）**：多条 trace 由外部 aggregator/verifier 选，但"trace 间是否独立"因方法而异
  - 优势：可并行 → 延迟低
  - **BoN**：完全独立采样，trace 之间无信息交换，错误不传播
  - **ToT**：共享搜索树前缀，LLM evaluator 跨分支打分/剪枝，分支间有信息交换
  - **MCTS**：通过 visit-count backup 显式共享跨 trace 信息（见 §8.1 PUCT 公式）
  - 劣势：verifier/evaluator 必须精准，否则 aggregation 或 backup 会被误导

- **选型决策**：
  - 任务 sequential dependency 强（数学竞赛、定理证明）→ long CoT（错误信息后续可被反思修正）
  - 任务多解（codeforce、creative writing）→ BoN（多路径覆盖）
  - 任务有 well-defined intermediate verifier（math step）→ MCTS / PRM beam search
  - 延迟敏感 → parallel（可在 GPU 上并行）
  - 单 GPU 内存敏感 → sequential（单 KV cache）

- **未来方向**：sequential + parallel 混合——单条 long CoT 内嵌入 multi-path 探索（如 o1 内部可能就在做这个，但闭源不可知）

只说 "sequential 比 parallel 好"——任务依赖

</details>

<details>

<summary>Q25. 如果让你设计下一代 reasoning model，应该往哪几个方向走？（open-ended 顶级 lab interview）</summary>

可信回答框架（不需面面俱到，挑 2-3 个深入展开）：

- **方向 1 - 训练算法**：
  - GRPO 现在 trace-level；如何 token-level 又不引入 critic？（如学 PRM-as-critic）
  - Reward shaping：rule reward 太稀疏，能否 dense 化但保持 hard-to-hack（如形式化验证 intermediate steps）？
  - Continual RL：R1 训完就 freeze；能否 online RL during deployment？

- **方向 2 - Test-time compute scaling**：
  - Adaptive budget：根据题目难度动态分配 reasoning tokens（Snell 2024 起点）
  - Sequential + parallel 混合：long CoT 中嵌入 sub-tree exploration
  - Multi-agent debate：多个 LLM 互查、对抗

- **方向 3 - Verifier**：
  - Generative PRM 替代 scalar PRM（用 LLM 评 step quality 比 scalar head 更准）
  - Self-verifier：让模型自己 verify 自己（DeepSeek-Prover-V2 在 Lean 上是雏形）
  - Cross-domain transfer：math PRM 能否 transfer 到 code PRM？

- **方向 4 - 评测**：
  - 现有 reasoning benchmark (AIME, MATH) 接近饱和——下一代 evalution 标准？
  - Robustness：reasoning model 在 adversarial prompt 上是否 brittle？

- **方向 5 - 推理可解释性**：
  - CoT faithfulness（前 Q18）：让 CoT 真实反映 internal computation
  - Mechanistic 可解释性：能否定位到具体 attention head 负责"反思"？

- **方向 6 - Reasoning + agent**：
  - 现在 reasoning 主要在 single-turn；agentic setting 中 reasoning 怎么跨 turn 保持？
  - Tool use + reasoning 怎么 jointly optimize？

照搬现有方法 + 加一点——不展现 research taste

</details>

## §A 附录：核心 paper 时间线 + 一句话总结

按时间倒序：

| 日期 | Paper | arXiv | 一句话贡献 |
| --- | --- | --- | --- |
| 2025-04 | DeepSeek-Prover-V2 | 2504.21801 | subgoal decomposition + Lean 4 RL，MiniF2F 88.9% |
| 2025-02 | Claude 3.7 Sonnet | (no arXiv) | hybrid 模型，extended thinking budget 用户可控 |
| 2025-02 | s1: Simple Test-Time Scaling | 2501.19393 | 1K SFT + "Wait" budget forcing 超 o1-preview |
| 2025-01 | DeepSeek-R1 / R1-Zero | 2501.12948 | 纯 RL (GRPO + rule reward) 涌现推理；R1 = 4 阶段 pipeline |
| 2025-01 | rStar-Math | 2501.04519 | MCTS + PPM self-evolution, 7B 接近 o1-preview |
| 2024-12 | o3 (OpenAI) | (no arXiv) | ARC-AGI 75.7%-87.5%，首次抽象推理逼近人类 |
| 2024-12 | Gemini 2.0 Flash Thinking | (no arXiv) | Google 首个推理模型，thinking 显式可见 |
| 2024-09 | o1 (OpenAI) | (no arXiv) | 首个商用 reasoning model，hidden CoT + RL |
| 2024-08 | Snell et al. Test-Time Compute | 2408.03314 | 优化 test-time compute 可比拟 14×（FLOPs-matched 口径）模型 scaling |
| 2024-08 | rStar | 2408.06195 | MCTS + mutual reasoning，小 LM 大幅提升 |
| 2024-02 | DeepSeekMath / GRPO | 2402.03300 | GRPO 算法首次提出，去除 critic |
| 2023-12 | Math-Shepherd | 2312.08935 | MC completion rollout 自动标 PRM label（非 MCTS tree search） |
| 2023-05 | Tree of Thoughts | 2305.10601 | 显式 tree search + LLM evaluator |
| 2023-05 | Let's Verify Step by Step | 2305.20050 | PRM > ORM > majority vote；PRM800K 数据集 |
| 2022-03 | Self-Consistency | 2203.11171 | sample N + 多数票，GSM8K +17.9% |
| 2022-05 | Zero-shot CoT (Kojima) | 2205.11916 | "Let's think step by step"，zero-shot 触发 CoT |
| 2022-01 | Chain-of-Thought (Wei) | 2201.11903 | few-shot step-by-step demonstration，CoT 在大模型上 emergent |

> 💡 **建议精读 4 篇** — 准备面试时间有限时，按优先级读：

1. DeepSeek-R1 (2501.12948) —— 涵盖 GRPO + 4 阶段 + R1-Zero "aha moment"
2. DeepSeekMath (2402.03300) —— GRPO 算法原始论文
3. Let's Verify Step by Step (2305.20050) —— PRM 基础
4. Snell et al. (2408.03314) —— test-time compute scaling 范式

读完这 4 篇 + 本 cheat sheet，reasoning model 面试题目应能 80%+ 覆盖。

> ⚠️ **常考开放题准备** — 顶级 lab interview 经常问 open-ended 题（如 Q25），关键是展现 **research taste**：能列出 3-5 个具体方向（不是"我会做 reasoning model" 这种空话），每个方向能给一个 concrete proposal + 一个 expected failure mode。准备时不要死记，多读最近 6 个月 arXiv 上的 reasoning paper，构建自己的 taxonomy。
