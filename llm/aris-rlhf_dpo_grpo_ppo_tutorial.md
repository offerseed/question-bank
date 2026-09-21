# RLHF-DPO-GRPO-PPO — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[rlhf-dpo-grpo-ppo-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/rlhf_dpo_grpo_ppo_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **7 句话搞定 post-training alignment** — 一页拿下面试核心要点（详见后文 §2–§9 推导）。

1. **RLHF pipeline (Ouyang 2022 InstructGPT)**：SFT → RM (Bradley-Terry pairwise) → PPO + per-token KL；价值模型 (value head) 单独训练，policy 与 reference policy 通过 KL 约束。

2. **PPO 核心 (Schulman 2017)**：clipped surrogate $L^{\text{CLIP}}(\theta) = \mathbb{E}[\min(r_t A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$，重要性比 $r_t = \pi_\theta / \pi_{\theta_\text{old}}$；advantage 用 **GAE** $A_t^{\text{GAE}} = \sum_{l \ge 0} (\gamma\lambda)^l \delta_{t+l}$ 平衡偏差/方差。

3. **DPO 闭式 (Rafailov 2023 NeurIPS)**：KL-regularized RLHF 的最优策略 $\pi^*(y|x) \propto \pi_\text{ref}(y|x)\exp(r(x,y)/\beta)$，反解得 $r = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$；代入 Bradley-Terry，**$\log Z$ 在 pairwise 差中消掉**，留下纯 SFT-style 损失 $-\log\sigma(\beta\log\frac{\pi(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi(y_l)}{\pi_\text{ref}(y_l)})$。

4. **GRPO (DeepSeekMath 2024, R1 2025)**：对每个 prompt 采样一组 $G$ 个回答，advantage 用**组内归一化** $\hat{A}_i = (r_i - \text{mean}(\mathbf{r}))/\text{std}(\mathbf{r})$；**省掉 value model**，省一半显存，特别适合 LLM 数学/代码 RL。

5. **DPO 变体生态**：KTO（仅需 thumbs up/down，无需 pair）、IPO（防 reward overfit 的 $\ell_2$ 形式）、SimPO（去 reference，length-normalized）、ORPO（SFT + odds-ratio 一阶段融合）、RLOO（多采样 leave-one-out baseline，无需 value）、ReMax（greedy baseline，进一步省）。

6. **PRM vs ORM (Lightman 2023 arXiv / 2024 ICLR)**：Process-reward 监督每步 reasoning，Math-Shepherd (Wang 2024 ACL) 自动标 PRM；Outcome-reward 只看最终答案。**PRM 在数学推理上稳压 ORM**，但标注成本高。

7. **Reward hacking 是核心痛点**：模型 over-optimize proxy reward 导致答案变长、谄媚、风格异常；缓解手段 = KL penalty、reward clipping、length penalty、ensemble RM、Constitutional AI / RLAIF（Bai 2022, Lee 2023）。

---

## §10 25 高频面试题

按难度分 3 档：L1 = 任何 LLM 工程岗都会问；L2 = research/alignment 团队会问；L3 = 顶级 lab / DeepSeek 量级团队的硬核题。每题点开看答案要点 + 易踩坑。

### L1 必会题（10 题）

<details>

<summary>Q1.RLHF 的三个阶段是什么？</summary>

- Stage 1: SFT（监督微调）
- Stage 2: RM（用 Bradley-Terry 训 reward model，frozen）
- Stage 3: PPO + KL penalty（policy 优化 RM 分，受 reference policy KL 约束）

跳掉 SFT 直接说"RM + PPO"；或漏掉 KL penalty。

</details>

<details>

<summary>Q2.PPO clipped surrogate 的公式是什么？</summary>

- $r_t = \pi_\theta / \pi_{\theta_\text{old}}$ 是重要性比
- $L^\text{CLIP} = \mathbb{E}[\min(r_t A_t,\, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$
- 取 min 是悲观估计 (pessimistic bound)；advantage 为正时上限 $1+\epsilon$，为负时下限 $1-\epsilon$

说"clip 的是 reward"；或漏说 min 的作用。

</details>

<details>

<summary>Q3.GAE 是什么？$\lambda$ 怎么取？</summary>

- $A_t^\text{GAE} = \sum_l (\gamma\lambda)^l \delta_{t+l}$，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
- $\lambda = 0$ → TD(0)，偏差大方差小
- $\lambda = 1$ → Monte Carlo，偏差小方差大
- 典型 $\lambda = 0.95$，$\gamma = 0.99$

把 GAE 当成 advantage 本身（它是估计器）；或忘了 $\gamma\lambda$ 是联合衰减系数。

</details>

<details>

<summary>Q4.RM 的损失函数是什么？</summary>

- Bradley-Terry pairwise: $\mathcal{L} = -\mathbb{E}\log\sigma(r(x, y_w) - r(x, y_l))$
- 用 SFT model 初始化 backbone，最后接 scalar head
- 训完冻结，给 PPO 用

说 RM 用 BCE / MSE；或写成绝对 reward 监督。

</details>

<details>

<summary>Q5.DPO 损失公式？跟 RM 损失什么关系？</summary>

- $\mathcal{L}_\text{DPO} = -\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})$
- 形式跟 Bradley-Terry RM loss 一样，但**implicit reward** 是 $\beta\log(\pi/\pi_\text{ref})$，不是单独的 RM
- 起源：KL-regularized RLHF 最优解 $\pi^* \propto \pi_\text{ref}\exp(r/\beta)$，反解得到 $r = \beta\log(\pi/\pi_\text{ref}) + \beta\log Z$，$\log Z$ 在 pairwise 差里消掉

只背公式不知道闭式推导；或说"DPO 不需要 reference"（错，需要 $\pi_\text{ref}$ 算 log-ratio）。

</details>

<details>

<summary>Q6.DPO 训练时需要哪些数据？</summary>

- 偏好对 $(x, y_w, y_l)$：同一 prompt 下，人或 AI 评判 $y_w \succ y_l$
- 需要 $\pi_\text{ref}$（一般是 SFT model）
- **不需要** RM、value model、online sampling

说要 reward scalar 标注；或忘了 $\pi_\text{ref}$。

</details>

<details>

<summary>Q7.GRPO 比 PPO 省了什么？</summary>

- **省掉 value model**：advantage 用组内归一化 $\hat{A}_i = (r_i - \bar{r}) / \sigma_r$
- 显存少一份；调参少一组 ($c_v$、value lr 不需要)
- 适合 LLM RL，因为 LLM 的 per-token value 很难学

说省了 RM（错，GRPO 还要 RM 或 rule-based reward）；或说 GRPO 是 offline（错，它是 on-policy）。

</details>

<details>

<summary>Q8.RLHF 中 KL penalty 起什么作用？</summary>

- Reward 上加 $-\beta\log\pi/\pi_\text{ref}$（per-token KL）
- 防止 policy 偏离 SFT 太远，缓解 reward hacking
- $\beta$ 太小 → 漂移，$\beta$ 太大 → 学不到东西
- 在 DPO 中通过 $\pi_\text{ref}$ 隐式实现

说 KL 是 RM 的一部分（错，KL 是 policy 与 ref 之间的，不涉及 RM）。

</details>

<details>

<summary>Q9.为什么 SFT 之后还要 RL/DPO？</summary>

- SFT 只能模仿正例，**学不到对比信号**（A 比 B 好）
- RM 把"哪个好"显式建模，policy 优化"获得高 RM 分"
- DPO 跳过 RM 但保留了同样的对比信号
- 实测 RLHF/DPO 后 helpfulness、harmlessness、honesty 都有显著提升（InstructGPT 报告）

只说"RL 比 SFT 强"；或不知道对比信号这一点。

</details>

<details>

<summary>Q10.Reward hacking 是什么？怎么缓解？</summary>

- Policy over-optimize RM，找到 RM 盲点拿高分，但人类觉得变差
- 典型症状：答案变长、谄媚、重复套话、过度拒绝
- 缓解：KL penalty、RM ensemble、length penalty、composite reward (rule + RM)、early stopping by KL budget

只说"加 KL"，不知道其他缓解；或不知道 length bias。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11.推导 DPO loss（从 KL-regularized RLHF 起）。</summary>

1. 目标：$\max_\pi \mathbb{E}[r] - \beta \text{KL}(\pi || \pi_\text{ref})$
2. Lagrangian + 对 $\pi$ 求导 $\Rightarrow \pi^*(y|x) = \frac{1}{Z(x)}\pi_\text{ref}(y|x)\exp(r/\beta)$
3. 反解 $r(x, y) = \beta\log(\pi^*/\pi_\text{ref}) + \beta\log Z(x)$
4. 代入 Bradley-Terry $P(y_w \succ y_l) = \sigma(r_w - r_l)$，**$\log Z$ 消掉**
5. 把 $\pi^*$ 替换为可学 $\pi_\theta$，对偏好数据做 NLL → DPO loss

直接背公式，问"$\log Z$ 为什么消掉"答不上来（因为它不依赖 $y$，差掉）。

</details>

<details>

<summary>Q12.PPO 的重要性比 $r_t = \pi_\theta/\pi_\text{old}$ 为什么需要？</summary>

- 标准 PG 是 on-policy 的，但 PPO 在同一 batch 上做多次 update（K 个 epoch）
- 第 2 次开始 $\pi_\theta \ne \pi_\text{old}$（即采样分布），需要 importance sampling 校正
- $\nabla \mathbb{E}_{\pi_\theta}[A] = \mathbb{E}_{\pi_\text{old}}[(\pi_\theta/\pi_\text{old}) \nabla\log\pi_\theta A]$
- Clip 是为了防止 ratio 在多次 update 中飘太远

不知道 PPO 多次 update；或不清楚 IS 的角色。

</details>

<details>

<summary>Q13.DPO vs IPO 区别？什么时候用 IPO？</summary>

- DPO 用 sigmoid loss：偏好越极端，implicit reward 越大（无界）
- IPO 用 squared loss + 固定 margin $1/(2\beta)$：把每对偏好的 log-ratio 差拉向固定有限目标 $1/(2\beta)$，从而避免 DPO 在 deterministic 偏好下 sigmoid loss 被无限推向 $+\infty$（隐式 reward margin 发散）——这是防止边际发散，而非对整个 reward 函数给出严格的全局有界证明
- **当偏好数据 deterministic**（每对都 $y_w$ 总赢）时 DPO 容易过拟合，IPO 更稳
- Azar et al. 2024 AISTATS 给出统一框架（$\Psi$PO）

说 IPO 是改进 DPO 的 hyperparameter；或不知道 sigmoid 在 deterministic preference 下的 issue。

</details>

<details>

<summary>Q14.SimPO 相比 DPO 改了什么？</summary>

- **去 reference**：$r = (\beta / |y|) \log\pi(y)$，不需要 $\pi_\text{ref}$
- **Length normalize**：除以 $|y|$，缓解 DPO 的长度偏好
- **Reward margin** $\gamma$：要求 $r_w - r_l \ge \gamma$，超出后 loss/梯度渐近趋于 0（softplus 型，非 hinge 硬截断）
- 显存相对 DPO 节省约 11%（去掉 frozen $\pi_\text{ref}$ 的 14GB 权重，126→112 GB，并非"省一倍"）；但失去 KL anchor，需要更小心调 $\beta, \gamma$

只说"去 reference"，不说 length-norm 和 margin；或不知道 SimPO 也是 contrastive。

</details>

<details>

<summary>Q15.GRPO 的 advantage 公式？为什么这样设计？</summary>

- $\hat{A}_i = (r_i - \text{mean}_g(r)) / (\text{std}_g(r) + \epsilon)$，$g$ 是同 prompt 的 group
- 整段 response 内所有 token 共享同一 $\hat{A}_i$
- **设计理由**：LLM token-level value 难学；用 sequence-level reward + 组内统计直接做 variance reduction
- 哲学和 RLOO（leave-one-out 均值）、ReMax（greedy baseline）一致：用 sample baseline 替代 critic

写成 per-token advantage（在 outcome-supervision 版本下错，process-supervision 版本本就是 per-token）；或不知道 GRPO/RLOO/ReMax 的共性。

</details>

<details>

<summary>Q16.PRM vs ORM 差在哪？什么时候用 PRM？</summary>

- ORM 只在最终答案打分；PRM 每步 reasoning 打分
- PRM 在**多步推理任务**（数学、code）上明显更好（Lightman 2023, MATH 数据集 78% vs 72%）
- PRM 标注成本高 → Math-Shepherd 用 rollout-based 自动标
- PRM 既能做 search 重排，也能做 step-level RL（dense reward）

说 PRM 在所有任务都更好（不对，简单任务 ORM 足够）；或不知道 Math-Shepherd 的自动标注。

</details>

<details>

<summary>Q17.Constitutional AI / RLAIF 关键 idea？</summary>

- 用 AI（而非人）做偏好标注，省人工
- Constitutional AI (Bai 2022 Anthropic)：SL stage AI 自我改写 harmful → RL stage AI 偏好打分 → PPO
- RLAIF (Lee 2023 Google)：在 summarization 等任务系统验证 AI ≈ human
- **风险**：评委 LLM 有偏见会放大；通常和 rule-based / human RM 混合

说 RLAIF 就是 GPT-4 来标数据（不全对，CAI 强调"按宪法"自我评估）；或不知 CAI 早于 RLAIF 一年。

</details>

<details>

<summary>Q18.PPO 训练时为什么要在 reward 上加 KL，而不是在 loss 上？</summary>

- 加在 reward 上 → 通过 advantage 自然进入 PPO-clip surrogate，per-token 控制
- 加在 loss 上的 KL 仍可以是逐 token 的（如 GRPO 用 K3 estimator 逐 token 计算，见 §5.2），并不天然损失 token 分辨率；真正的差异在于加进 reward 的 KL 会通过 GAE/return 递归影响每个 token 的 advantage，而加进 loss 的 KL 是独立于 return 的附加惩罚项，不参与 value bootstrap
- GRPO 反而把 KL 放在 loss 上（用 K3 estimator），因为 GRPO 不展开 per-token advantage
- 两种放法本质都是 KL anchor，但实现细节不同

混淆两种放法；或不知道 GRPO 的 K3 estimator。

</details>

<details>

<summary>Q19.RLHF 中 value model 怎么初始化？为什么？</summary>

- 通常用 **RM 或 SFT model 初始化** value backbone，加新 value head
- 用 RM 初始化的好处：RM 已经"理解 reward"，value 收敛更快
- 用 SFT 初始化的好处：value 与 policy 共享底层表征
- DeepSpeed-Chat 默认用 RM 初始化；TRL 默认用 SFT
- **共享 trunk vs 独立模型是 trade-off**：A3C/A2C 等经典 actor-critic 共享 trunk 省显存，但 policy 与 value loss scale 不同容易相互干扰；LLM RLHF 主流（DeepSpeed-Chat / TRL / OpenRLHF）选独立模型 + separate optimizer，稳定性优先

随机初始化 value（实践中不可行，太慢）；或说"value 必须共享 trunk"/"绝对不能共享"，两个极端都不对——是 trade-off 不是定理。

</details>

<details>

<summary>Q20.DPO 训练崩了（margin 不涨 / loss 不降），怎么诊断？</summary>

- 看 `chosen_logp` 和 `rejected_logp` 是否同时下降（典型 DPO 退化）
- 看 reference policy 是否正确加载（forgot to load → log-ratio 变成 raw log-prob）
- 看 $\beta$：太小（< 0.01）loss 几乎 = sigmoid 常数，太大 ($> 1$) 容易 collapse
- 看数据：$y_w$ 和 $y_l$ 是否真的差异显著
- 看 length：DPO 偏好长 response，若 $y_w$ 系统性比 $y_l$ 短就有 bug

只说"调 lr"；或不知道 likelihood-decrease-for-both 问题。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21.DeepSeek-R1 vs R1-Zero 的差异？为什么需要 cold-start？</summary>

- **R1-Zero**：从 pretrain base 直接跑 GRPO + rule-based reward（数学正确 + 格式），**无 SFT**
  - 优点：emergent 长 CoT，证明 RL 能自激发 reasoning
  - 缺点：可读性差、混语言、格式不稳定
- **R1**：cold-start SFT (几千条高质量 reasoning) → reasoning RL → SFT → general RL（多阶段）
  - 修复 R1-Zero 的可读性问题
  - SOTA 数学 / 代码推理
- **R1-Zero 重要性**：证明 RL 单独能激发 reasoning，不必先 SFT；后续 self-play / pure-RL 路线的基础

只说 R1 是 R1-Zero 的改进；或不知道 cold-start 修的是什么（可读性，不是性能）。

</details>

<details>

<summary>Q22.GAE 中 $\gamma$ 和 $\lambda$ 谁更重要？LLM RLHF 中通常怎么取？</summary>

- $\gamma$ 是 reward 折扣（任务级），$\lambda$ 是 TD 估计的 trace-decay（算法级）
- **LLM RLHF 中 $\gamma = 1$**（不折扣，因为 reward 只在 terminal，折扣会让 early token 收到信号过小）
- $\lambda = 0.95$ 仍然有用：在 token 维度做 bias-variance 折中
- $\gamma\lambda$ 联合衰减是 GAE 的有效 trace-decay；$\gamma=1, \lambda=0.95$ 时实际 trace 约 20 token
- 若 $\gamma = 1, \lambda = 1$ → GAE 退化为 MC，与"reward-to-go - V baseline" 等价

不假思索照搬 game RL 的 $\gamma = 0.99$，不知道 LLM 中 $\gamma = 1$ 是更主流选择（在 terminal-reward + 短上下文 generation 下更合理）；或反过来认定 $\gamma$ 必须为 1——具体仍取决于 reward 是否 dense / 是否长 horizon，是工程选择不是定理。

</details>

<details>

<summary>Q23.如何设计一个 RL 框架同时支持 DPO / PPO / GRPO？</summary>

抽象出三层：

1. **Data layer**：preference pair (DPO) / prompt + group (GRPO) / prompt only (PPO) → 统一 batch interface
2. **Trajectory layer**：on-policy rollout (PPO/GRPO) vs offline (DPO)
   - PPO/GRPO 需要 vLLM/sgl 推理加速 + 异步 trajectory queue
   - DPO 完全 dataloader
3. **Loss layer**：插件化
   - PPO: clipped surrogate + value + entropy + GAE
   - DPO: log-ratio sigmoid
   - GRPO: clipped surrogate + group-normalized advantage + K3 KL
4. **Reference 管理**：DPO/PPO/GRPO 都需 $\pi_\text{ref}$，统一封装为 frozen + lazy load
5. **RM/rule reward**：插件化 reward backend（neural RM、unit-test、math checker）

参考：TRL、OpenRLHF、verl (字节)、SimplePO。**verl 是 GRPO/RLOO/DAPO 的主流实现框架**。

只列 PPO；或没考虑 trajectory queue 和异步 rollout。

</details>

<details>

<summary>Q24.如果 RM 和 policy 一起训会怎样？为什么主流 RLHF 不这么做？</summary>

- 直觉：让 RM 也在线学，给 RM 更多新分布的数据 → adversarial training
- **问题 1**：RM 训练目标和 policy 训练目标耦合，loss landscape 不稳定（GAN-like）
- **问题 2**：RM 训练需要 fresh human label，否则会被 policy 拖着 drift
- **问题 3**：RM 信号变化太快 → policy 学到的"什么是好" 不一致
- 折中方案：**iterative preference optimization** (Xu et al. 2023 arXiv 2312.16682 *Some things are more CRINGE than others: Iterative Preference Optimization with the Pairwise Cringe Loss*)、**self-rewarding LM** (Yuan et al. 2024 *Self-Rewarding Language Models*) — 每轮用最新 policy 生成响应 + 重新打分（同模型 / 人 / GPT-4），让 RM 隐式更新
- **OpenAI / Anthropic 主流：固定 RM，多轮 PPO**；可视化解释更稳定

只说"会不稳定"，不给具体原因；或不知道 iterative DPO / self-rewarding 这条 emerging path。

</details>

<details>

<summary>Q25.如果让你设计 next-gen 后训练算法，你会怎么改进 GRPO？</summary>

可能的方向（任答 2-3 个，且要有 trade-off 讨论）：

- **Dynamic sampling（DAPO）**：过滤掉组内 reward 全同（全对或全错，std=0，advantage 恒为 0→零梯度）的 prompt，并持续补充采样新 prompt 以维持 batch 中有效梯度样本数不变；组内采样数 $G$ 本身通常保持固定，并不随 reward 方差自适应调整
- **Token-level credit assignment**：GRPO 整段共享 advantage → 长 response 信号稀释。可以引入 lightweight critic（VAPO）或基于 step-level reward（PRM）的 partial credit
- **Off-policy correction**：GRPO 是 on-policy，rollout 慢；引入 V-trace / Retrace 让 stale samples 也能用
- **Multi-task reward**：rule + RM + style 合成 reward，每个维度独立归一化避免 reward scale 不平衡
- **Reward model uncertainty**：用 ensemble RM 的 min 或 mean - std 防止 over-optimization
- **Process reward integration**：PRM 给 step-level dense advantage，与 group baseline 联合（Math-Shepherd + GRPO 路线）
- **CISPO / 截断 IS** (Minimax 2025)：解决 GRPO 在 negative advantage 大 ratio 时的稳定性
- **DAPO** (ByteDance 2025)：clip higher + dynamic sampling + token-level loss + overlong shaping，已开源 verl 实现

只罗列"加 attention / 加更多模型"，没 trade-off；或不知道 DAPO / VAPO / CISPO 等 GRPO 后续工作。

</details>

## §A 附录：参考文献清单

按章节分组，全部经过 codex (gpt-5.5 xhigh) reviewer 验证作者-年份-会议正确：

**PPO / RL 基础**

- Schulman et al. 2017 arXiv 1707.06347 *Proximal Policy Optimization Algorithms*
- Schulman et al. 2016 ICLR *High-Dimensional Continuous Control Using GAE*
- Schulman 2020 blog *Approximating KL Divergence*（K3 estimator 来源）

**RLHF**

- Christiano et al. 2017 NeurIPS *Deep Reinforcement Learning from Human Preferences*（pairwise preference + RM 第一篇）
- Stiennon et al. 2020 NeurIPS *Learning to Summarize from Human Feedback*（OpenAI summarization 报告）
- Ouyang et al. 2022 NeurIPS *Training Language Models to Follow Instructions with Human Feedback*（InstructGPT）
- Bai et al. 2022 Anthropic arXiv 2204.05862 *Training a Helpful and Harmless Assistant with RLHF*
- Bai et al. 2022 Anthropic arXiv 2212.08073 *Constitutional AI*
- Lee et al. 2023 Google arXiv 2309.00267 *RLAIF: Scaling RLHF with AI Feedback*

**DPO 系**

- Rafailov et al. 2023 NeurIPS *Direct Preference Optimization*
- Azar et al. 2024 AISTATS *A General Theoretical Paradigm to Understand Learning from Human Preferences*（IPO）
- Ethayarajh et al. 2024 ICML *KTO: Model Alignment as Prospect Theoretic Optimization*
- Meng et al. 2024 NeurIPS *SimPO: Simple Preference Optimization with a Reference-Free Reward*
- Hong et al. 2024 EMNLP *ORPO: Monolithic Preference Optimization without Reference Model*
- Tang et al. 2024 ICML *Generalized Preference Optimization* (统一 DPO/IPO/SLiC 框架)

**Critic-free RL**

- Ahmadian et al. 2024 ACL *Back to Basics: Revisiting REINFORCE Style Optimization* (RLOO)
- Li et al. 2024 ICML *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method*
- Shao et al. 2024 arXiv 2402.03300 *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（GRPO 提出）
- DeepSeek-AI 2025 arXiv 2501.12948 *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*
- Yu et al. 2025 ByteDance arXiv 2503.14476 *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*

**Reward Modeling**

- Lightman et al. 2024 ICLR / OpenAI arXiv 2305.20050 (2023) *Let's Verify Step by Step*（PRM800K, PRM vs ORM）
- Wang et al. 2024 ACL *Math-Shepherd: Verify and Reinforce LLMs Step-by-Step without Human Annotations*
- Coste et al. 2024 ICLR *Reward Model Ensembles Help Mitigate Overoptimization*
- Eisenstein et al. 2024 COLM (arXiv 2312.09244, 2023) *Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking*
- Gao, Schulman, Hilton 2023 ICML *Scaling Laws for Reward Model Overoptimization*

**Iterative / Self-rewarding**

- Xu et al. 2023 arXiv 2312.16682 *Some things are more CRINGE than others: Iterative Preference Optimization with the Pairwise Cringe Loss*
- Yuan et al. 2024 ICML *Self-Rewarding Language Models*
- Pal et al. 2024 arXiv 2402.13228 *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*

代码框架：TRL (HuggingFace)、OpenRLHF、verl (ByteDance Seed)、Axolotl、LLaMA-Factory、SimplePO。**verl 是当前 GRPO/RLOO/DAPO 主流实现**，DeepSeek 系工作多基于 verl 复现。
