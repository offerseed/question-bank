# Agentic RL — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[agentic-rl-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/agentic_rl_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Agentic RL** — RL for LLM agents 是 2024-2026 把 reasoning RL 推向真实工具使用、Web、代码与 GUI 的核心范式（详见 §1-§9 推导 + §10 25 高频题）。

1. **Agentic RL 与 RLHF 的本质区别**：RLHF 是 single-turn 偏好对齐，reward 来自 RM 对整段 response 的打分；**Agentic RL 是 multi-turn 决策，state 是 (obs, history)，action 是 (thought, tool_call)，更强调用可编程验证的客观结果（test-pass、task success、verifier）做主干，而非 RM 式的主观偏好打分**（但也允许用学出来的 reward model，如 WebRL 的 ORM，二者不是严格二分）。整条轨迹长度从 RLHF 的几百 token 涨到 agent 的数千乃至数万 token，credit assignment 难度上一个台阶。

2. **PPO/GRPO 在 agent 上的关键改造**（必背）：**token mask** 必须只对 agent 自己 generate 的 token 算 loss——observation token（tool 返回的 stdout / search snippet）属于环境，policy gradient 不能流到那里；否则 model 会试图"教 tool 怎么回答"，行为崩坏。GRPO 优势更明显：在 long-horizon trajectory 上，value model 几乎学不动 per-token V（中间几乎全 0 reward），组内归一化是更稳的 baseline。

3. **Reward 设计三层金字塔**：(a) **Outcome reward** 最便宜也最稀疏——final answer/task success 0/1；(b) **Process reward** 给每步打分，需要 PRM 或 step verifier；(c) **Hybrid / shaping**——tool-call shaping（鼓励调对工具）、length penalty（防 agent 拖太长）、format reward（强约束输出 schema）。R1 路线用 rule-based outcome reward（数学正确 + 格式），SWE-RL 用 test-pass（rule-based），WebRL 则训练一个 ORM 给 task success 打分（learned-RM 路线，见 §6.9）——**outcome-level 验证信号（rule-based 或 learned ORM）+ dense format shaping** 是 2025 工业实测最稳的组合。

4. **代表性早期 work**：**AgentTuning** (Zeng et al. 2023 arXiv 2310.12823 THU)——agent SFT 数据集 + 多任务训练；**Agent-FLAN** (Chen et al. 2024 ACL Findings arXiv 2403.12881)——把 agent corpus 拆成 multi-turn / formatted / negative example 三类；**ReFT** (Luong et al. 2024 ACL arXiv 2401.08967)——SFT warm-start + online RL on math reasoning，PPO 在 GSM8K 上 +9pp。这三篇是 Agentic RL 的"先 SFT 后 RL"标准三段式。

5. **Tool-augmented reasoning RL**：**ToolRL** (Qian et al. 2025 arXiv 2504.13958)——把 tool 调用嵌入 GRPO，reward 含 correctness + format + tool-use efficiency；**ReSearch** (Chen et al. 2025 arXiv 2503.19470)——把 search call 当 first-class action，rule-based reward 学 multi-hop search；**RAGEN / StarPO** (Wang et al. 2025 arXiv 2504.20073)——多回合 RL 训练 framework，state-action token level loss + critic-free GRPO 变种。共同点：**outcome-only reward + format shaping + token-mask loss + GRPO**。

6. **Web / GUI agent RL**：**WebRL** (Qi et al. 2024 ICLR-25 arXiv 2411.02337)——self-evolving curriculum + ORM + retrospective rollout，把 8B Llama 推到 WebArena 43%；**AgentQ** (Putta et al. 2024 arXiv 2408.07199)——MCTS 搜索 + AI critique + DPO offline 训练；**Computer-Use** (Anthropic Claude 3.5/3.7/4 Sonnet, 2024-10-22 起)——RLHF + RL 在屏幕截图 + 鼠标键盘 action space 上训练 GUI 控制（公开知识：训练细节未披露，但 system card 说明用了大量人工 + AI 反馈）。

7. **Code agent RL**：**CodeRL** (Le et al. 2022 NeurIPS arXiv 2207.01780) 首次把 unit test 当 reward 信号 + actor-critic；**PPOCoder** (Shojaee et al. 2023 arXiv 2301.13816) 加入 compilable + functional correctness 的 composite reward；**SWE-RL** (Wei et al. 2025 Meta FAIR arXiv 2502.18449) 用 rule-based reward（patch similarity + test-pass）在 GitHub PR 数据上做 RL，Llama-3.3-70B 把 SWE-bench Verified 推到 41%。

8. **Self-rewarding & exploration**：**Self-Rewarding LM** (Yuan et al. 2024 Meta arXiv 2401.10020) 让 policy 同时当 judge，iterative DPO with LLM-as-judge；但 self-rewarding 在 agent 上比单 turn alignment 更危险——judge 也是 agent 自己，**容易 reward drift / model collapse**。生产里多用 LLM-as-judge ensemble + rule-based grounding（test-pass、math checker）+ human spot check 三件套。

9. **长 horizon credit assignment 的"三种武器"**：(a) **GAE + γ < 1** 把信用沿轨迹回传，但在 sparse outcome reward 下退化为 MC return；(b) **Hindsight relabeling**（HER 思路在 agent 上的对应物）——失败轨迹按"中间状态当 goal"重新标 reward；(c) **subgoal decomposition + process reward**——把 50 步轨迹切成 5 个 subgoal × 10 步，PRM 给每个 subgoal 打分。L3 面试常问的"为什么 GRPO 在 long-horizon agent 上比 PPO sample efficient"——答案是 **trace-level reward 直接匹配 trace-level credit**，绕开 value model 在 long-CoT 上几乎学不动的痛点。

---

## §10 25 高频面试题（L1 必会 / L2 进阶 / L3 顶级 lab）

按难度分 3 档：L1 = 任何 agent / LLM RL 岗会问；L2 = research / alignment 团队会问；L3 = 顶级 lab 硬核题。每题点开看答案要点 + 易踩坑。

### L1 必会题（10 题）

<details>

<summary>Q1. Agentic RL 和 RLHF 的本质区别？</summary>

- **RLHF**: single-turn alignment，state = prompt，action = 整段 response，reward 来自 RM 对偏好的打分，horizon = 1
- **Agentic RL**: multi-turn decision-making，state = (obs, history)，action = (thought, tool_call)，reward 来自外部环境（test pass / task success），horizon = 10-200
- 算法层面都用 PPO/GRPO，但 Agentic 必须加 **action_mask**（只对 agent token 算 loss）
- reward 形式：RLHF 偏好打分（subjective），Agentic 客观 outcome（objective）

把它们当成同一件事；或不知道 action_mask 的必要性。

</details>

<details>

<summary>Q2. 为什么 Agentic RL 必须用 action_mask？</summary>

- agent trajectory 含两类 token：agent 自己 generate 的 + tool 返回的 observation
- 如果不 mask，PPO/GRPO 的 ratio 和 loss 会流到 observation token 上
- 后果：(a) policy 试图"教 tool 怎么回答"，无意义且会 reward hack；(b) gradient 被 low-information observation 稀释；(c) KL penalty 错误地把 environment text 当自己分布估计
- 实现：`action_mask: [B, L]`，1 = agent token，0 = prompt/obs/pad；loss 计算时除以 `mask.sum()` 标准化

说"只在 response 上算 loss"不够具体（agent task 没有"response"这个明确边界）；或漏了 mask 必须 cover all observation tokens。

</details>

<details>

<summary>Q3. GRPO 在 agent 上比 PPO 好的核心原因？</summary>

- **省 critic**：agent value 在长 horizon + sparse reward 下几乎学不动
- **trace-level reward 直接匹配 trace-level credit**：不需要 per-token V，避开 value 学不动的痛
- **组内归一化** 自动做 variance reduction，比 raw advantage 稳
- **省一份显存**：可以扩大 batch / group size
- 限制：长 trajectory 上 advantage 太粗（整段共享），credit dilution 仍存在

只说"省 critic"不全面；或不知道 trace-level reward 与 trace-level credit 的匹配关系。

</details>

<details>

<summary>Q4. Outcome reward 和 process reward 的区别？agent 上常用哪个？</summary>

- **Outcome reward**: trajectory 终点 1 个 reward（test pass / answer match），极稀疏，credit assignment 难，但 reward hacking 风险低
- **Process reward**: 每步打分，dense，credit assignment 易，但 reward hacking 风险高（PRM 可被 hack）
- **Agent 上主流: outcome reward**——agent task ground truth 明确（test/grader）；R1, SWE-RL, ReSearch, RAGEN 都用 outcome-only
- Process reward 主要用于 reasoning-heavy 任务（PRM 在数学 step 上），agent 上少用

以为 process reward 总是好（实际 agent 上 outcome 更稳）；或不知道 reward hacking 风险差异。

</details>

<details>

<summary>Q5. Verifier-based reward 比 learned RM 好在哪？</summary>

- **接近 ground truth**：避开 learned RM 漂出训练分布的主要 failure mode
- **可重复**：同一 trajectory reward 一致（learned RM 输出有噪声）
- **显存便宜**：执行 verifier 比一次 LLM forward 便宜数百倍
- **可解释**：reward 来自客观 rule，可 trace 失败原因
- **限制**：只能用于可验证任务（math/code/grader-able web）

说 verifier-based "完全没有 hacking"——错，仍可能被正则漏洞 / format trick / test 泄露 hack，但比 RM hacking 容易堵。

</details>

<details>

<summary>Q6. R1 / R1-Zero 的方法能直接用在 agent 上吗？</summary>

- 算法可以直接搬：GRPO + rule-based reward + per-step KL + token mask
- 但需要补：
  - **action_mask**: agent 有 observation token，R1 数学任务没有
  - **trajectory rollout infrastructure**: 含 tool I/O，比纯 generation 复杂
  - **format reward 调整**: agent task 的 format 是 JSON tool call，不只是 `<think>`
- 代表性工作 ReSearch / RAGEN / ToolRL / SWE-RL 都是 R1 算法 + 上述改造

说"R1 不能直接搬"——其实可以但要改 wrapper；或不知道 ReSearch / RAGEN / ToolRL 这些工作。

</details>

<details>

<summary>Q7. Agent RL 中为什么需要 SFT warm-start？</summary>

- base model 不知道怎么 emit 合法 tool call schema（JSON 格式 / argument 名称）
- 直接 from-scratch RL 几乎不可能 explore 到合法 tool call → reward 全 0 → 学不动
- SFT warm-start（AgentTuning / Agent-FLAN 数据）让 model "知道动作空间长什么样"
- 之后 RL 在合法 action 子空间内 optimize

只说"RL 慢，SFT 加速"——不够；核心是 action space exploration 困难，SFT 解决"知道动作空间"。

</details>

<details>

<summary>Q8. Length penalty 在 agent RL 中起什么作用？</summary>

- agent 容易学到"拖长 trajectory 拿对答案"的捷径（reward hacking）
- length penalty 给超出 budget 的 trajectory 减分：$r = r_\text{outcome} - \lambda \max(0, T - T_\text{target})$
- DAPO 的 "overlong reward shaping" 是工业级实现（进入 buffer 区间后分段线性 ramp 到 -1，而非指数衰减）
- 限制：penalty 过大会让 agent 不敢探索；要 cap

不知道 length-explosion 是 agent RL 常见 failure mode；或不会写 length penalty 公式。

</details>

<details>

<summary>Q9. Group std = 0 的情况怎么处理？</summary>

- 当 group 内 G 个 rollout reward 全相同（全 success / 全 fail）→ $\sigma = 0$
- 加 $\epsilon$ 时 advantage = 0，policy gradient 项归零，**但 KL 项仍存在**（policy 仍被拉回 reference）
- 不加 $\epsilon$ 时是 NaN
- 实践：
  - **Skip 该 prompt**（data filter）：常见做法，避免无信号更新
  - **Clamp σ 下限**（如 0.1）：只是防止 $0/0$ 产生 NaN 的数值稳定手段——全同组里分子 $r_i - \mu$ 已精确为 0，无论分母 clamp 到多少，advantage 恒为 0，**并不能恢复训练信号**
  - **DAPO dynamic sampling**：丢全对全错 group，是真正解决"无信号"的做法

说一定会 NaN（不对，看实现）；或不知道这种 prompt 暗示 task 过易/过难。

</details>

<details>

<summary>Q10. SWE-RL 是怎么训的？</summary>

- 数据：GitHub PR commit 历史，构造 ~76M context-issue-patch 三元组（Meta 公开数字，具体是否为最终三元组数还是原始 PR/commit 事件数，请核对原论文）
- Reward: rule-based = patch similarity (oracle ↔ pred) + test pass binary
- 算法: 纯 GRPO + format reward
- Model: Llama-3.3-70B
- Result: SWE-bench Verified 上 41%（无 scaffold），证明 rule-based RL 让 model 学到 emergent reasoning：file retrieval / root cause / test self-validation

不知道 SWE-RL 的 reward 设计；或以为是 SFT 而非 RL。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11. 推导 PPO loss 在 agent 上加 action_mask 后的形式。</summary>

1. 标准 PPO-Clip：$L = \mathbb{E}[\min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)]$
2. agent 有 $m_t \in \{0, 1\}$，1 = agent token，0 = obs/prompt
3. ratio 计算时 $\rho_t = \exp((\log\pi_\theta - \log\pi_\text{old}) \cdot m_t)$，observation 位置 $\rho = e^0 = 1$，不影响 surr1/surr2
4. loss 归一化：$L^\text{agent} = -\sum_t m_t \cdot \min(\rho_t A_t, \text{clip}) / \sum_t m_t$
5. KL 项也只在 agent token：$\text{KL}_\text{total} = \sum_t m_t \cdot \text{KL}_t / \sum_t m_t$

只写公式不解释 mask 的作用；或忘了 normalization 用 mask.sum() 而非 batch size。

</details>

<details>

<summary>Q12. RAGEN / StarPO 的关键贡献？</summary>

- **StarPO**（**S**tate-**T**hinking-**A**ctions-**R**eward Policy Optimization）: critic-free GRPO 变种，整段 trajectory 共享 advantage
- **严格 token mask**：observation 位置 mask = 0，loss 只在 agent token
- **rollout 多样性 = collapse 防火墙**：group_size $G = 16$ 比 $G = 4$ 显著更稳
- **trajectory length 信号**：失败 trajectory length 大时加 length penalty
- 适用：multi-turn agent task（与 single-turn alignment 区分）

只说"GRPO 变种"——不够；要说出 multi-turn 上的 stability 贡献。

</details>

<details>

<summary>Q13. WebRL 的 self-evolving curriculum 是怎么工作的？</summary>

- 初始 task set $\mathcal{T}_0$（small），训 policy 跑 trajectory
- 失败 trajectory 进入 buffer，加到下一轮 curriculum $\mathcal{T}_{k+1}$
- retrospective rollout：失败 trajectory 用 LLM 改造成"正确 trajectory"（hindsight relabel），再做 SFT
- ORM (Outcome Reward Model) 训自 task success → 在线给 RL reward
- Result: Llama-3.1-8B 在 WebArena 上 43%（vs GPT-4 14.4%）

不知道 self-evolving 的循环结构；或把 WebRL 当作纯 SFT。

</details>

<details>

<summary>Q14. 为什么 self-rewarding LM 在 agent 上比在 alignment 上风险更高？</summary>

- alignment 任务有"客观偏好分布"，LLM judge 与人评有较高相关
- agent 任务有**客观 ground truth**（test pass / task success）—— judge 自己可能错（认错对错）
- iterative drift：每轮把"自评对"的轨迹强化 → 远离 ground truth
- 探索退化：自评偏好已知 pattern → 抑制探索新 tool
- 主流做法：agent RL 优先 rule-based ground truth，self-rewarding 仅作开放式任务 fallback

说 self-rewarding 总是危险（alignment 上仍可用）；或不知道 agent 上 ground truth 客观性是关键

</details>

<details>

<summary>Q15. 推 outcome-reward sparsity 对 critic learning 的影响。</summary>

- value $V_\phi(s_t) = \mathbb{E}_\pi[\sum_{l \ge 0} \gamma^l r_{t+l} \mid s_t]$
- sparse terminal reward → $V(s_t) \approx \gamma^{T-t} \cdot P(\text{success} \mid s_t)$
- 中间状态 $s_t$ 的 value 几乎只取决于"未来是否成功"——这是隐含 long-horizon 预测
- **逐步 reward 稀疏（多数 step reward = 0）不等于训练 target 接近 0**：value MSE loss 的 target 是 bootstrapped return，在 $\gamma$ 接近 1（教程默认 $\gamma=0.99$）时中间状态的 target 近似等于终点结果 $P(\text{success} \mid s_t)$，可能长期维持在较高水平；若 $\gamma=1$，对一条已实现的成功轨迹，终点前每一步的 MC target 恒等于终点 reward，完全不趋于 0
- 真正导致 critic 难学的是**这个 target 本身方差大**、且需要长程预测最终成功与否，而不是 target 数值小、gradient 小
- 这也是 GRPO 省 critic 的理论基础：critic 要学的是高方差、长程依赖的量，省了反而省去 noise

只说"critic 学不动"；不会推 $V \approx \gamma^{T-t} P(\text{success})$；或把"逐步 reward 稀疏"和"value target 接近 0"混为一谈——二者是不同的量。

</details>

<details>

<summary>Q16. ToolRL 的 reward 设计有什么 nontrivial 之处？</summary>

- composite: $r = r_\text{correct} + \alpha r_\text{format} + \gamma r_\text{tool-eff}$
- $r_\text{tool-eff}$ 惩罚冗余 tool call（重复调同样工具 / 调用无效 tool）
- 这是典型的 shaping reward：缓解 length-explosion + tool-overuse 两个 failure mode
- weight 平衡：outcome ≫ format > tool-eff，避免 shaping override outcome
- BFCL benchmark 上 7B 接近 GPT-4

只说"加 tool call 奖励"；不知道 shaping reward weight 平衡是关键。

</details>

<details>

<summary>Q17. Hindsight relabeling 在 agent RL 中怎么用？</summary>

- 失败 trajectory 不丢，改造为 "alternative task" 的 successful trajectory
- 例：agent 想买 A 商品但停在 B → 改造为"找到 B 商品"，reward = 1
- 实现：`alt_task = describe(trajectory.final_state)`，relabel reward = 1
- 适用：开放 web 环境、navigation；不适用：math/code（错答案不能改成对答案）
- 起源：HER (Andrychowicz 2017 NeurIPS) for robot manipulation

不知道 HER 起源；或不知道适用边界（开放环境 vs 答题任务）。

</details>

<details>

<summary>Q18. Per-step KL penalty 和 trajectory KL penalty 的区别？</summary>

- **per-step KL**: 每个 agent token 算 KL($\pi_\theta(\cdot \mid s_t) \| \pi_\text{ref}(\cdot \mid s_t)$)；加进 reward 或 loss
- **trajectory KL**: 整段 trajectory 一个 KL；加进 loss
- 二者不是两种本质不同的散度——由链式法则，$\log\pi(\tau) = \sum_t \log\pi(a_t \mid s_t)$，"trajectory KL"在期望意义下就等于逐步 KL 之和；真正的实现差异在于：(a) 是把 KL 逐步加进 reward 做 shaping，还是只在整体 loss 里加一个标量项；(b) 用 sum 还是 mean 聚合；(c) 用什么数值稳定的估计器
- GRPO 用逐 token 的 K3 estimator 再求和（数值稳定）
- agent RL 上 per-step 更主流（trajectory 太长，直接对序列级 log-ratio 求 KL 数值不稳）

混淆两者本质不同（其实链式法则下是同一个量，差异在实现方式）；或不知道 K3 estimator 解决数值问题（K3: $\text{KL} \approx \exp(\Delta) - \Delta - 1$ 非负）。

</details>

<details>

<summary>Q19. Subgoal decomposition + process reward 怎么做？什么时候用？</summary>

- 长 trajectory 切 subgoal：100 步 trajectory → 5 个 subgoal × 20 步
- 每个 subgoal 终点给 process reward（subgoal 是否完成）
- 实现路径：
  - hand-crafted: 人写 subgoal 判据
  - LLM planner: planner LLM 拆 subgoal，verifier 判
  - PRM-style: PRM 评每步
- 适用：long-horizon agent + 可以 hand-craft subgoal 的任务
- 风险：subgoal boundary 错画 → agent 学到"刻意触发 subgoal reward 而不真正完成 task"

说 process reward 总是好——错；要说出风险与限制。

</details>

<details>

<summary>Q20. Online RL vs Offline RL 在 agent 上的 trade-off？</summary>

- **Online RL (PPO/GRPO)**: 数据效率低（每轮新 rollout），但持续学新分布
- **Offline RL (DPO/RFT)**: 数据效率高，但受限于 dataset 分布
- agent rollout 慢（含 tool I/O），online RL 训练吞吐低
- 实践：先 offline 起手（SFT + DPO），再 online refinement
- 代表：AgentQ 是 offline（MCTS + DPO）；WebRL 是 online；SWE-RL 是 online

只说"online 慢"；不知道 agent rollout 含 tool I/O 是主要瓶颈。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21. 推导 GRPO advantage 公式 + token mask 的完整 loss，并解释 agent token mask 的两种等价放置方式。</summary>

1. **Group-relative advantage**:
   - rollout group $\{r_1, ..., r_G\}$ per prompt
   - $\mu = \frac{1}{G}\sum r_i$, $\sigma = \sqrt{\frac{1}{G}\sum (r_i - \mu)^2}$
   - $\hat{A}_i = (r_i - \mu) / (\sigma + \epsilon)$

2. **Trajectory-level broadcast**: $\hat{A}_{i,t} = \hat{A}_i$（所有 agent token 共享）

3. **Token-masked ratio + loss**：
   - 朴素 $\rho_{i,t} = \exp(\log\pi_\theta(a_{i,t} \mid s_{i,t}) - \log\pi_\text{old}(a_{i,t} \mid s_{i,t}))$ —— 在 observation 位置也会有数值（model 估算 env text 的概率）
   - **要点**：只要最终 objective / gradient 只覆盖 agent token，mask 放 ratio 内还是 loss 外都**数学等价**：
     - **Inside-ratio**：$\rho_{i,t} = \exp((\log\pi_\theta - \log\pi_\text{old}) \cdot m_{i,t})$ → obs 位置 $\rho=1$，进 clip 后 $\min(\cdot)$ 项 = $A_{i,t}$ 但乘以 $m_{i,t}=0$ 后归零（在 loss 外的 sum 中）
     - **Outside-ratio (mask loss only)**：保留 obs 位 $\rho_{i,t}$ 数值；最终 loss = $-\sum_t m_t \cdot \min(...)$，obs 位贡献 $m_t = 0$ 直接归零
   - **两者梯度都只覆盖 agent token**（mask 是乘法，梯度对 obs 位都是 0）
   - 但实践上 **Inside-ratio 更安全**：避免 obs 位 $\rho$ 数值参与 clip 触发判断或被日志 / 监控（如 mean ratio）误读为异常。生产实现（verl / OpenRLHF）多用 inside-ratio

5. **Full loss**:
   $$L = -\frac{1}{G} \sum_i \frac{1}{\sum_t m_{i,t}} \sum_t m_{i,t} \cdot \Big(\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta \cdot \text{KL}_{i,t}\Big)$$

不会推 step 4 (mask 位置影响 ratio 数值)；或公式背得对但不解释 mask 设计哲学。

</details>

<details>

<summary>Q22. GRPO 在 long-horizon agent 上比 PPO sample efficient 的根本原因？</summary>

**Trace-level reward 与 trace-level credit 的自然对齐**（不止是"省 critic"）：

1. **Sparse terminal reward 下 critic 学不动**：$V(s_t) \approx \gamma^{T-t} P(\text{success})$，gradient 极小；PPO 的 GAE-advantage 受 noisy critic 拖累
2. **GRPO 用组内样本（含自身）的均值/标准差归一化后的 reward 当 advantage**：这是对 sequence-level advantage 的一个有偏但方差更低、且随 $G$ 增大渐近无偏（consistent）的估计——偏差来源包括：(a) 自身计入均值产生的 $(1-1/G)$ 缩放（相比留一法 baseline）；(b) std 归一化引入的与题目难度相关的非线性偏差；(c) PPO clip 本身引入的额外偏差
3. **Group baseline 比 critic baseline 更稳**：同 prompt G rollouts → group mean 自动反映该 prompt 的难度，variance reduction 更精准
4. **PPO clipping + group size 联合限制更新幅度**：避免单 outlier reward 推飞 policy
5. **省 value model 显存** 是 secondary benefit，不是 primary reason
6. **Rule-based outcome reward 难被 RM hacked**：在 agent 上比 learned RM 稳

只说"省 critic"——不够；要说出 sparse reward 下 critic learn 不动是根因。

</details>

<details>

<summary>Q23. 如何设计一个 RL framework 同时支持 reasoning RL（R1）和 Agentic RL（ReSearch / WebRL）？</summary>

抽象出五层：

1. **Data layer**:
   - reasoning: (prompt, ground_truth) tuples
   - agent: (task, env_spec, reward_fn) 三元组
   - 统一为 `Task(prompt, verifier)`，verifier 是 callable

2. **Rollout layer**:
   - reasoning: 直接 generate
   - agent: 含 tool I/O 的 multi-step rollout (vLLM + sandboxed tool executor)
   - 统一为 `Trajectory(tokens, action_mask, reward)` 接口

3. **Reward layer**:
   - reasoning: rule-based (answer match / test pass)
   - agent: composite (outcome + format + tool_eff + length penalty)
   - 统一为 `Reward(traj) -> float`

4. **Loss layer**:
   - PPO with action_mask
   - GRPO with group_id + action_mask
   - DPO with chosen_mask / rejected_mask
   - 通过 `loss_fn(batch, model, ref_model) -> loss` interface

5. **Infra layer**:
   - vLLM rollout pool
   - Sandboxed tool executor (Docker + gVisor)
   - Trajectory replay buffer (FIFO + priority)
   - Async trainer / rollout

代表实现: **verl** (字节) 已支持 reasoning + agent；**OpenRLHF** 部分支持。

只列 PPO 不考虑 agent rollout infra；或不知道 verl / OpenRLHF 的当前支持范围。

</details>

<details>

<summary>Q24. Anthropic Computer-Use 训练方法 — 已知 vs 推测的清晰边界</summary>

**官方公开（system card / blog）**：

- action space = 屏幕截图 (vision observation) + 鼠标 + 键盘 events
- 能力持续迭代 Claude 3.5 (new) 2024-10-22 → 3.7 / 4.0 / 4.5 / Opus 4.x
- 安全机制：constitutional AI 风格的护栏 + 红队 + prompt-injection 防御
- 训练涉及人工演示 + 合成数据（system card 一般性陈述）

**未公开 / 完全保密**：

- 具体 RL 算法（PPO? GRPO? Critic-free? 都没说）
- Reward signal 形式（task completion grader? Pair-wise preference? Safety classifier 权重?）
- Train data 规模 / 来源 / 演示 vs 合成比例
- 是否有专门的 screenshot RM / VLM-as-judge

**社区合理推测**（**仅推测，不要在面试中说成事实**）：

- 可能是 RLHF on screenshot trajectories（pair-wise 偏好 + task success outcome 混合）
- 可能 critic-free（呼应 DeepSeek-R1 GRPO 等开源趋势）
- 可能用 VLM-as-judge for screenshot 理解
- 可能 curriculum 简单 → 复杂

**面试时务必区分 "公开能力" vs "推测内部"**：说"Anthropic 用 GRPO + screenshot RM"是错的（无证据）；说"我推测可能用了 critic-free RL，因为 Anthropic 在其他场景倾向 GRPO/RLHF 风格"才是诚实的表述。这种区分能力是高级面试的加分项。

</details>

<details>

<summary>Q25. 如果让你设计 next-gen Agentic RL 算法，会怎么改进？</summary>

可能方向（任答 3-4 个，每个要有 trade-off 讨论）：

1. **Lightweight critic for long-horizon**: 不用 full-size value model，但用小型 step-level critic 缓解 trace-level credit dilution。VAPO 已尝试。Trade-off: 加显存 vs 缓解 long-trajectory 信号稀释

2. **Hierarchical reward**: subgoal-level reward + outcome reward 组合。Trade-off: 需要 subgoal definition（人工 or planner LLM），boundary 错画风险

3. **Off-policy correction with V-trace / Retrace**: rollout 慢，让 stale samples 也能用。Trade-off: IS bias vs sample efficiency

4. **Trajectory hindsight relabeling + RL**: 失败 trajectory 自动改造为 alternative task 的成功 trajectory，扩 data。Trade-off: 适用 open-ended task，不适用 closed-form answer

5. **Multi-task reward normalization**: 每个 task domain (math/code/web) 独立归一化，避免 reward scale 不平衡

6. **Reward model uncertainty**: 多 RM ensemble，min/mean-std 防 over-optimization。Trade-off: 算力

7. **Async distributed rollout**: rollout 与 train 完全异步，trajectory queue + worker pool。已是 industry default (verl, OpenRLHF v0.5+)

8. **Self-curriculum + adaptive difficulty**: WebRL 思路 + R-Zero 的 learnability reward 结合，自动找 model success rate ~50% 的任务

9. **Multi-objective Pareto optimization**: 不再单一 scalar reward，task success + safety + efficiency 同时优化，输出 Pareto front

只罗列 "加 attention / 加更多模型" 没 trade-off；或不知道 DAPO / VAPO / CISPO 等近期工作；或忽略 infra 层面 (async rollout) 的重要性。

</details>

## §A 附录：参考文献清单

按方向分组，论文经 web 检索 + arXiv 验证作者 / 年份 / 会议。少数 2025-2026 会议归属未定的论文以 arXiv 记。

**Agent SFT / 基础**

- Zeng et al. 2023 arXiv 2310.12823 *AgentTuning: Enabling Generalized Agent Abilities for LLMs* (THU)
- Chen et al. 2024 ACL Findings arXiv 2403.12881 *Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for LLMs*
- Luong et al. 2024 ACL arXiv 2401.08967 *ReFT: Reasoning with Reinforced Fine-Tuning*

**RL on agent / reasoning（基础算法）**

- Schulman et al. 2017 arXiv 1707.06347 *Proximal Policy Optimization Algorithms*
- Schulman et al. 2016 ICLR *High-Dimensional Continuous Control Using GAE*
- Shao et al. 2024 arXiv 2402.03300 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（GRPO 提出）
- DeepSeek-AI 2025 arXiv 2501.12948 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- Yu et al. 2025 ByteDance arXiv 2503.14476 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*

**Tool-augmented RL**

- Qian et al. 2025 arXiv 2504.13958 *ToolRL: Reward is All Tool Learning Needs* (arXiv preprint; no formal venue as of 2026-05)
- Chen et al. 2025 arXiv 2503.19470 *ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning* (accepted to **NeurIPS 2025**)
- Wang et al. 2025 arXiv 2504.20073 *Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning* (RAGEN; StarPO = **S**tate-**T**hinking-**A**ctions-**R**eward Policy Optimization)

**Web / GUI agent RL**

- Qi et al. 2024 ICLR-25 arXiv 2411.02337 *WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning*
- Putta et al. 2024 arXiv 2408.07199 *Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents*
- Furuta et al. 2024 ICLR arXiv 2305.11854 *Multimodal Web Navigation with Instruction-Finetuned Foundation Models* (WebGUM)

**Code agent RL**

- Le et al. 2022 NeurIPS arXiv 2207.01780 *CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning*
- Shojaee et al. 2023 arXiv 2301.13816 *Execution-Based Code Generation Using Deep Reinforcement Learning* (PPOCoder)
- Wei et al. 2025 Meta FAIR arXiv 2502.18449 *SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution*

**Embodied / robot agent**

- Baker et al. 2022 NeurIPS arXiv 2206.11795 *Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos*
- Kim et al. 2024 arXiv 2406.09246 *OpenVLA: An Open-Source Vision-Language-Action Model*

**Self-rewarding / exploration**

- Yuan et al. 2024 ICML arXiv 2401.10020 *Self-Rewarding Language Models*
- Andrychowicz et al. 2017 NeurIPS arXiv 1707.01495 *Hindsight Experience Replay*

**RLHF / DPO 基础（cross-reference）**

- Ouyang et al. 2022 NeurIPS *Training Language Models to Follow Instructions with Human Feedback*
- Rafailov et al. 2023 NeurIPS *Direct Preference Optimization*
- Bai et al. 2022 Anthropic arXiv 2212.08073 *Constitutional AI*
- Lee et al. 2023 Google arXiv 2309.00267 *RLAIF: Scaling RLHF with AI Feedback*

**Reward model / verification**

- Lightman et al. 2024 ICLR arXiv 2305.20050 (OpenAI 2023) *Let's Verify Step by Step* (PRM800K)
- Wang et al. 2024 ACL arXiv 2312.08935 *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations*
- Coste et al. 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*

**Infrastructure / framework**

- TRL (HuggingFace): https://github.com/huggingface/trl —— 标准 PPO / DPO / GRPO trainer
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF —— PPO / GRPO / RLOO 工业化实现
- verl (ByteDance): https://github.com/volcengine/verl —— GRPO / DAPO / agent RL 主流框架
- ReaLHF / AReaL (Ant Group + Tsinghua, async RL system, arXiv 2505.24298)

**SOTA benchmarks (2024-2026)**

- Jimenez et al. 2024 ICLR arXiv 2310.06770 *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- Yao et al. 2024 arXiv 2406.12045 *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*
- Xie et al. 2024 NeurIPS arXiv 2404.07972 *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*
- Zhou et al. 2024 ICLR arXiv 2307.13854 *WebArena: A Realistic Web Environment for Building Autonomous Agents*
- Mialon et al. 2024 ICLR *GAIA: A Benchmark for General AI Assistants*
- Chan et al. 2024 arXiv 2410.07095 *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering* (OpenAI)

**Anthropic Computer-Use（公开知识）**

- Claude 3.5 Sonnet (new) 2024-10-22 Computer Use beta launch (Anthropic blog + system card)
- Claude 3.7 / 4.0 / 4.5 / Opus 4.x system cards (Anthropic 公开)

代码框架建议：

- 起步用 TRL（HF）的 GRPOTrainer + 自写 verifier
- 工业化用 verl（GRPO/RLOO/DAPO 都支持，含 agent rollout）
- 自研 multi-turn 用 OpenRLHF v0.5+ 的 agent example + 加 tool sandbox
