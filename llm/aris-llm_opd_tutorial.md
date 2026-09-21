# LLM在线蒸馏OPD — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[llm-opd-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/llm_opd_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 LLM OPD (On-Policy Distillation)** — 2025-2026 post-training 最热门的"便宜版 RL"范式，一页拿下面试核心要点（详见 §1-§9 推导 + §10 25 高频题）。

1. **OPD = On-Policy Distillation**：student 用自己当前 policy **采 trajectory**，teacher 在 student 自己访问到的状态上给 **per-token 监督信号**（KL / log-prob / soft label）。它**不是** DPO 的笔误，也**不是** Online Preference Distillation——这一术语在 LLM 上下文专指 "on-policy distillation"。代表 paper：MiniLLM (Gu 2024 ICLR, arXiv 2306.08543)、GKD (Agarwal 2024 ICLR, arXiv 2306.13649)、Thinking Machines blog (Lu 2025-10-27)、Qwen3 Technical Report (May 2025, arXiv 2505.09388)、Survey (Song & Zheng 2026, arXiv 2604.00626)。

2. **核心 loss（一行能写下来）**：sample $y \sim \pi_\theta(\cdot|x)$，对每个 token 算 reverse KL，定义为 $L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta}[\sum_t D_{\text{KL}}(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t}))]$。注意期望下标是 $y \sim \pi_\theta$（**student 自己采**），KL 方向是 $\pi_\theta \| \pi_T$（**reverse / mode-seeking**）——这两点是 OPD 与 vanilla KD 的本质区别。完整公式：
   $$L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t})\big)\right]$$

3. **三种 distillation 范式速记**：
   - **SFT / Hard distillation**：teacher 生成 $\hat y$，student 在 $\hat y$ 上做 cross-entropy（off-policy + hard label）
   - **Vanilla KD / Soft distillation (Hinton 2015)**：teacher 生成 $\hat y$，student match teacher 的 soft logits（off-policy + soft label，forward KL，mode-covering）
   - **OPD**：**student 自己生成 $y$**，teacher 在 $y$ 上算 logits 给 KL 信号（on-policy + soft label，reverse KL，mode-seeking）

4. **为什么 on-policy 关键**：off-policy distillation 有 exposure bias——training 时 teacher 的 prefix 都"完美"，但推理时 student 遇到自己的错误前缀**没见过**，错误 compound（误差随序列长度 $L$ 大约按 $O(L^2)$ 放大，见 §1.3）。OPD 把训练分布 = 推理分布对齐，把 compound 误差从 $O(L^2)$ 压到 $O(L)$。

5. **OPD vs RL（核心卖点）**：Thinking Machines blog 给出经验法则——RL 每条 trajectory 教 $O(1)$ bits（一个 outcome reward），OPD 教 $O(N)$ bits（每个 token 都有 teacher 的 soft label 监督）。在 Qwen3-8B-Base + Qwen3-32B-teacher 数学推理实验上，OPD **匹配 RL 在 AIME'24 的 gain，但 compute 降到约 1/9-1/30**。Qwen3 Tech Report 也独立报告 OPD ≈ RL 性能但 GPU 时长**只需 1/10**。

6. **与 DPO / GRPO 的关系**：(a) **DPO** 用 offline preference pair 做 closed-form RLHF，**没有 teacher logits，没有 student rollout**——与 OPD 几乎正交；(b) **GRPO** 用 group-relative advantage 做 on-policy RL，**有 student rollout 但 reward 是 sparse outcome**；(c) **OPD = GRPO 的"dense teacher KL 替代 sparse outcome reward"** 版本。Survey (Song 2026) 给出统一视角：OPD ≈ "KL-constrained RL with $\beta \to \infty$ 且 token-level reward 来自 teacher log-prob"。

7. **2025-2026 工业采用**：Qwen3（off-policy + on-policy 两阶段蒸馏 small models）、DeepSeek-R1 distillation series（off-policy SFT 为主，但后续 follow-up 用 OPD）、MiMo-V2、Kimi distill 系列（Gemma 2/3 的官方蒸馏方法是固定数据集上的 off-policy soft-target KD，不含 student self-rollout，不应归为 OPD 采用案例，见 §5.3）。Thinking Machines (Murati 团队 2025) 把 OPD 包装成"便宜 RL 替代品"路线。

8. **三个最常考的 footgun**：(a) **Length inflation / truncation collapse**——student rollout 越训越长，触发 truncation 后 gradient 偏置，validation 暴跌（Demystifying OPD 2026, arXiv 2604.08527）；(b) **Reverse KL mode collapse**——student 收敛到 teacher 单一模，生成多样性塌缩；(c) **Teacher / student gap 太大**：student 完全采不到 teacher 概率高的 region——注意这不是"KL 信号消失"，采到的极少数 teacher 低概率 token 反而会产生数值极端的惩罚（"prefix teach, suffix fade" 现象，arXiv 2605.13643，详见 §7.5）。

---

## §10 25 高频面试题

### L1 必会（10 题，post-training 工程师 / LLM RL 岗）

<details>
<summary><strong>L1-1：OPD 是什么？为什么叫 "On-Policy"？</strong></summary>

**答**：OPD = On-Policy Distillation。"On-policy" 指**训练数据（trajectory）来自 student 当前 policy $\pi_\theta$ 自己采样**，而非来自 teacher / dataset。teacher 只在 student 自己访问到的状态上提供 per-token 监督信号（通常是 reverse KL）。这与 off-policy KD（teacher 自己生成数据 → student 模仿）形成对比。
</details>

<details>
<summary><strong>L1-2：OPD 与 vanilla KD（Hinton）的核心区别是什么？</strong></summary>

**答**：三点：(1) **数据来源**——vanilla KD 用 dataset / teacher 生成数据，OPD 用 student 自己 rollout；(2) **KL 方向**——vanilla KD 用 forward KL（mode-covering），OPD 主用 reverse KL（mode-seeking）；(3) **解决的问题**——vanilla KD 解决"压缩 teacher 知识"，OPD 解决"压缩 + exposure bias"（autoregressive 生成的 train/test 分布不一致）。
</details>

<details>
<summary><strong>L1-3：写出 OPD 的 reverse KL 损失公式。</strong></summary>

**答**：
$$L_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim D,\, y \sim \pi_\theta(\cdot|x)}\!\left[\sum_{t=1}^{|y|} D_{\text{KL}}\!\big(\pi_\theta(\cdot|x, y_{<t})\,\|\,\pi_T(\cdot|x, y_{<t})\big)\right]$$
关键：期望下标 $y \sim \pi_\theta$（on-policy），KL 方向是 $\pi_\theta$ 在前（reverse / mode-seeking）。Sampled-token 近似 = $\log \pi_\theta(y_t) - \log \pi_T(y_t)$。
</details>

<details>
<summary><strong>L1-4：什么是 exposure bias？OPD 怎么解决？</strong></summary>

**答**：exposure bias = autoregressive 模型训练时 prefix 来自 ground-truth / teacher（"完美 prefix"），推理时 prefix 来自模型自己（含错误）—— 训练 / 推理分布不一致。理论上累积误差按 $O(L^2)$ 放大（Bagnell 2010）。OPD 通过在 student 自己 rollout 的 trajectory 上算 loss，把训练分布 = 推理分布，把累积误差压到 $O(L)$。
</details>

<details>
<summary><strong>L1-5：reverse KL 与 forward KL 的区别？为什么 OPD 倾向用 reverse？</strong></summary>

**答**：forward KL $D(\pi_T \,\|\, \pi_\theta)$ 期望在 teacher 下取，**强制 student 覆盖 teacher 所有 mode**（mode-covering），易得到模糊平均的输出；reverse KL $D(\pi_\theta \,\|\, \pi_T)$ 期望在 student 下取，**强制 student 避开 teacher 不可能的 token**（mode-seeking / zero-forcing），易得到 sharp 自信的输出。LLM 生成任务通常要"流畅 + 自信"，所以 reverse KL 是首选。
</details>

<details>
<summary><strong>L1-6：OPD 需要 teacher 的什么？只有 teacher API（无 logit）能做 OPD 吗？</strong></summary>

**答**：标准 OPD 需要 teacher 在 student 访问的每个 prefix 上算 **logits** 或 **log-probabilities**。如果 teacher 是 closed-source API（如 GPT-4）只能拿 sample 不能拿 logit，可以用 "Black-Box OPD" (arXiv 2511.10643) ：用 teacher samples 做 trajectory-level reward，组合 student log-prob 做近似——但失去 per-token dense 监督，回退到 trajectory-level，性能介于 OPD 与 RL 之间。
</details>

<details>
<summary><strong>L1-7：OPD 与 RL 的核心差异是什么？为什么 OPD 更 sample efficient？</strong></summary>

**答**：RL 每条 trajectory 提供一个 $O(1)$ scalar reward（outcome），OPD 提供 $O(N \log V)$ bits（每 token 整个 teacher 分布）。所以同样数量的 rollout，OPD 提供的监督信息**多一到两个数量级**。Thinking Machines 报告 OPD 在数学推理上以 1/9-1/30 的 compute 达到 RL 同等性能。
</details>

<details>
<summary><strong>L1-8：OPD 需要 critic / value model 吗？</strong></summary>

**答**：**不需要**。在教程采用的 semi-gradient（B1，对 state visitation stop-grad）近似下，$D_{\text{KL}}(\pi_\theta(\cdot|s_t) \,\|\, \pi_T(\cdot|s_t))$ 是该 step **即时（single-step）reward** 的条件期望闭式值，可以从已经算好的 student 与 teacher logits 直接读出，用作该 step REINFORCE 估计量的零偏差控制变量 / baseline（"KL for a KL", arXiv 2605.07865），不需要单独训 critic——这是 OPD 相对 PPO 的一个工程优势。但严格来说它不是完整轨迹目标的真正状态价值函数 $V(s_t) = \mathbb{E}[\sum_{t'\ge t} D_{\text{KL}}(s_{t'})]$（后者需要对未来所有 step 的 KL 求 return-to-go，即 §2.3 中"几乎没人做"的完整 B2 形式）；把单步条件期望直接称为 $V(s_t)$ 是概念上的简化说法。
</details>

<details>
<summary><strong>L1-9：OPD 与 DPO 的关系？两者能叠加吗？</strong></summary>

**答**：DPO 是 offline、binary preference、closed-form RLHF；OPD 是 online、per-token teacher logit、policy-gradient KD。两者**几乎正交**：DPO 不需要 teacher，OPD 不需要 preference pair。可以叠加：先 DPO 对齐人类偏好（学"什么是好回答"），再 OPD 蒸馏到小 student（学"怎么生成"）。
</details>

<details>
<summary><strong>L1-10：OPD 在生产中最常用的 failure mode 是什么？怎么 mitigate？</strong></summary>

**答**：**Length inflation / truncation collapse**——student rollout 越训越长，撞 max_length 后被 truncate 的轨迹主导梯度，val 暴跌。Mitigate：(1) loss per-token average 而非 sum；(2) length penalty；(3) max_length 比 sample max 大 2-4×；(4) 监控 val，spike 立即停。
</details>

### L2 进阶（10 题，资深 post-training / 论文复现）

<details>
<summary><strong>L2-1：推导 OPD 的 policy gradient 形式（含两条路线）。</strong></summary>

**答**：reverse KL 在 trajectory 期望下：
$$L(\theta) = \mathbb{E}_{s_t \sim \rho_{\pi_\theta}}\!\big[D_{\text{KL}}(\pi_\theta(\cdot|s_t)\,\|\,\pi_T(\cdot|s_t))\big].$$

$\theta$ 同时出现在 (a) **内层 KL**（$\pi_\theta(\cdot|s_t)$ 本身）和 (b) **state visitation** $\rho_{\pi_\theta}$。这导致 **两类不同的 gradient 计算**，文献常被混淆：

**(1) 内层 KL 的 REINFORCE 估计**（fixed-state $s_t$）：
$$\nabla_\theta D_{\text{KL}}(s_t) = \mathbb{E}_{y_t \sim \pi_\theta(\cdot|s_t)}\!\big[\nabla\log\pi_\theta(y_t|s_t)\cdot G_t^{\text{detach}}\big],\;\; G_t = \log\tfrac{\pi_\theta(y_t|s_t)}{\pi_T(y_t|s_t)}.$$
（推导：$\nabla\sum_v \pi_\theta(v)\log\tfrac{\pi_\theta(v)}{\pi_T(v)} = \sum_v\nabla\pi_\theta(v)\cdot(\log\tfrac{\pi_\theta}{\pi_T}+1) = \mathbb{E}_{y}[\nabla\log\pi_\theta(y)\cdot G + \nabla\log\pi_\theta(y)]$；第二项 $\mathbb{E}[\nabla\log\pi]=0$。）这是 §4.4 vOPD 用的 estimator。

**(2) Trajectory objective 的完整 policy gradient**（包含 state visitation）：
$$\nabla L = \mathbb{E}_\tau\!\Big[\underbrace{\sum_u \nabla\log\pi_\theta(a_u|s_u)\cdot {\textstyle\sum_{t\ge u}} D_{\text{KL}}(s_t)^{\text{detach}}}_{\text{return-to-go score term}} + \underbrace{\sum_t \nabla D_{\text{KL}}(s_t)}_{\text{inner grad}}\Big].$$
注意 score-function 权重是**未来所有 step 的 KL 之和**，不是同 token $G_t$。

**生产中的实践规则**：
- **Route A（教学清晰）**：rollout 时对 $\theta$ stop-grad（即 $\rho_{\pi_{\theta^-}}$ 固定），只反传内层 full-vocab KL（autograd 直接做），**不需要 REINFORCE**。这等价于"semi-gradient"近似。优势是零方差、一行 PyTorch；要求 teacher full logits 可用。（"零方差"特指给定固定 prefix 后内层 full-vocab KL 消除了 Route B1 的采样估计方差，训练流水线整体仍因 rollout 阶段 student 采样带有跨 batch 方差）
- **Route B（生产常见）**：sampled-token REINFORCE / importance-sampling estimator + control variate（§4.4）；与 PPO/GRPO 共享 sampled-token 接口、节约 teacher vocab memory、支持 logprob-only（灰盒）teacher（真正 sample-only 黑盒需 §6.5 Black-Box OPD）。MiniLLM (Gu 2024) 与 Thinking Machines Tinker 的开源实现都走这条，但两者近似程度不同：Tinker 走的是更贴近 (1) 的 fixed-state estimator；**MiniLLM 实际用带 future log-ratio return 的单步分解**（$R_t=\sum_{t'\ge t} r_{t'}$），是 (2) 的方差削减近似，不是纯粹的 semi-gradient。**完整、未削减的 (2) 才是几乎没人做的**——return-to-go 跨 step 累加，方差爆炸；生产中普遍在 (1) 附近或 MiniLLM 式的削减近似上折中。
- **不能**对 sampled-token $\log(\pi_\theta/\pi_T)$ 直接 `.backward()`：那是 pathwise + 固定 index，丢失 score-function 项，既不是 KL gradient 也不是 MLE。

**A 与 B 期望等价**，差异在方差 vs 工程接口 trade-off。

> "OPD 可以套 PPO/GRPO" 指的是 §4.5 中把 dense token KL **当作 reward** 喂进 GRPO 的 advantage（PPO ratio 仍是 student new vs old）；不是说"内层 KL gradient 形式与 PPO 等价"——这是常见的概念混淆。
</details>

<details>
<summary><strong>L2-2：GKD 与 MiniLLM 的关键差异？$\lambda$ 与 $\beta$ 分别控制什么？</strong></summary>

**答**：**MiniLLM** (Gu 2024)：纯 reverse KL on student rollout + REINFORCE 优化 + 一些稳定 trick（mixed policy $\pi_{\text{mix}} = (1-\alpha)\pi_\theta + \alpha\pi_T$，$\alpha = 0.2$ 防止 collapse；length penalty）。**GKD** (Agarwal 2024)：generalized 框架，两个 hyperparameter：(1) $\lambda \in [0, 1]$ 控制 **student-generated data fraction**（$\lambda=0$ 完全 off-policy on dataset $\hat y$，$\lambda=1$ 完全 on-policy on student $y$）；(2) $\beta \in [0, 1]$ 控制 **generalized JSD 的插值**——严格取极限时方向与直觉相反：$\beta \to 0$ 的单侧极限是 reverse KL（$D_\beta/\beta \to D_{\text{KL}}(\pi_\theta\|\pi_T)$），$\beta \to 1$ 的单侧极限是 forward KL（$D_\beta/(1-\beta) \to D_{\text{KL}}(\pi_T\|\pi_\theta)$），且 $\beta=0,1$ 处 $D_\beta$ 本身恒为 0（见 §2.1 表格端点说明）。GKD 是 MiniLLM 的 superset。
</details>

<details>
<summary><strong>L2-3：OPD 怎么集成进 GRPO？写出混合 reward 和正确的 ratio。</strong></summary>

**答**：核心两件事。
**(1) 混合 reward**（per-trajectory 标量，进 group-relative advantage）：
$$R(x,y) = \alpha\cdot R_{\text{outcome}}(x,y) + (1-\alpha)\cdot \sum_t \big(\log\pi_T(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t)\big),$$
$\alpha=0$ 退化纯 OPD，$\alpha=1$ 退化纯 GRPO，生产 $\alpha\in[0.3,0.7]$。

**(2) ratio 的写法（最常错的地方）**：GRPO/PPO 的 ratio 是
$$\rho_t = \exp\big(\log\pi_\theta^{\text{new}}(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t)\big),$$
即 **new student 与 old behavior student**——**teacher 绝不进入 ratio 分母**。Teacher 只在两个独立位置出现：(i) 作为 reward source（上面 $r_t^{\text{KL}}$）；(ii) 作为 reference distribution 出现在显式 KL penalty $D_{\text{KL}}(\pi_\theta\|\pi_T)$ 中（这一项闭式可微，§4.1 实现）。

最终 loss：
$$\mathcal{L} = -\mathbb{E}\!\left[\sum_t \min(\rho_t A_t, \text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)\right] + \beta\, D_{\text{KL}}(\pi_\theta\,\|\,\pi_T).$$

advantage $A$ 用 GRPO 组内 z-score，不需要 critic。代码见 §4.5。
</details>

<details>
<summary><strong>L2-4：vOPD 的 control variate 是什么？为什么"免费"？</strong></summary>

**答**：vOPD ("KL for a KL"，2026 Survey 中常被引用的 control-variate 思路) 用在 Route B（sampled-token REINFORCE estimator）上降方差。对单个 sampled $y_t \sim \pi_\theta$，token-level reward $\hat r_t = \log\pi_\theta(y_t|s_t) - \log\pi_T(y_t|s_t)$（**必须 detach**），baseline 取 $B(s_t) = D_{\text{KL}}(\pi_\theta\,\|\,\pi_T)(s_t)$（也 detach）：

$$\nabla_\theta L \approx \mathbb{E}\!\big[\nabla\log\pi_\theta(y_t|s_t)\cdot (\hat r_t - B(s_t))\big].$$

$B(s_t)$ 是 $\hat r_t$ 在 $y_t \sim \pi_\theta(\cdot|s_t)$ 下的条件期望，加上后**保持 unbiased**（因为 $\mathbb{E}_y[\nabla\log\pi_\theta(y)\cdot B(s_t)] = B(s_t)\cdot\mathbb{E}[\nabla\log\pi_\theta] = 0$）且**通常显著降低方差**。严格意义上的 minimum-variance baseline 是按 score-norm 加权 $\mathbb{E}[\|g\|^2 r]/\mathbb{E}[\|g\|^2]$（一般不等于条件均值），但条件均值在实践中已经足够好用。**"免费"**指 $B(s_t)$ 就是 student forward 同步算出的 full-vocab 闭式 KL（同一次 logits → softmax → 求和），无需额外 critic / inference。

> ⚠️ 实现时反复出错的两点：(a) `r_hat` 和 `baseline` 都必须 `.detach()`，否则 unbiasedness 失效、gradient 变形；(b) surrogate 的**符号是正号**：因为 $\nabla D_{\text{KL}} = \mathbb{E}[\nabla\log\pi_\theta\cdot\hat r]$，所以 `loss = +E[\log\pi_\theta\cdot(\hat r - B).detach()]`，`∇loss = +∇D_{\text{KL}}`，optimizer step `θ -= η∇L` 就在做 KL 下降。**写成负号会让 KL 上升**——代码见 §4.4。
</details>

<details>
<summary><strong>L2-5：Qwen3 的 OPD recipe 与 Thinking Machines blog 实现的差异是什么？</strong></summary>

**答**：**Qwen3** 用 off-policy SFT 冷启动 + on-policy distillation 两阶段，目标是端到端 build small model；**Thinking Machines blog** 强调把 OPD 当 RL 的便宜替代品，core 实验是"用 OPD 复现 RL 的 AIME'24 gain，FLOPs 节省 9-30×"。技术细节上：Qwen3 同时蒸馏 /think 与 /no_think 两种模式（dual-mode logits）；Thinking Machines 用 Tinker，advantage 形式上把 KL 当 negative reward 注入（OPD-RL 视角）。两者本质都是 reverse KL on student rollout，差异在多任务 / 多模式蒸馏的处理。
</details>

<details>
<summary><strong>L2-6：OPD 的 "sampled-token KL" 与 "full-vocab KL" trade-off？</strong></summary>

**答**：(a) **sampled-token (REINFORCE / importance-sampling，Route B)**：reward 用 $G_t = \log \pi_\theta(y_t) - \log \pi_T(y_t)$.detach()（只 query teacher 一个 log-prob），grad = $\nabla\log\pi_\theta(y_t)\cdot G_t$；便宜但方差大。**MiniLLM (Gu 2024) 与 Thinking Machines Tinker 的开源实现都是这种 sampled-token PG/IS 形式**（Tinker `train_on_policy.py` 用 `loss_fn="importance_sampling"` + `incorporate_kl_penalty`；MiniLLM 用 single-step decomposition + length norm + teacher-mixed sampling 等 trick 稳定方差）。(b) **full-vocab (Route A)**：loss = $\sum_v \pi_\theta(v)(\log \pi_\theta(v) - \log \pi_T(v))$，直接 autograd；零方差但要 teacher full logits + 整 vocab 显存；GKD 等概念推导常用这种形式当教学起点，也是工程上 teacher full logits 可用时的最简实现。**两者期望等价**，trade-off：Route A 零方差、一行 PyTorch，但要 full logits + vocab memory；Route B 适配 PPO/GRPO sampled-token 接口、支持 logprob-only（灰盒）teacher（真正 sample-only 黑盒需 §6.5 Black-Box OPD）、节约 vocab memory，需要 control variate（§4.4）降方差。
</details>

<details>
<summary><strong>L2-7：OPD 为什么比 off-policy KD 在 long-CoT 任务上更好？</strong></summary>

**答**：long-CoT 上 student 自身 rollout 与 teacher rollout 的状态分布差距更大（错误累积 $O(L^2)$）。off-policy KD 训练时 student 见的全是 teacher prefix（光滑、正确），推理时遇到自己的错误 prefix 完全没见过——错误按 $O(L^2)$ 复合累积（而非指数级，与 §1.3/L1-4 的结论一致）。OPD 训练时就在 student 自己的错误 prefix 上学，teacher 给"我会怎么 recover"的监督——直接训"错误修复能力"。
</details>

<details>
<summary><strong>L2-8：如果 student 与 teacher gap 极大（如 1.5B vs 671B），OPD 会失败吗？怎么 mitigate？</strong></summary>

**答**：**会失败**——student 的 sample 大概率落在 teacher 概率极低的 region。这里的失败机制不是"KL 信号趋零"：一旦采到这种 token，$\log\pi_T(y_t|s_t)\to -\infty$ 会让该 token 的 reverse KL 项**趋于 $+\infty$**（数值极端、梯度不稳），而 gap 过大时真正拖累训练的是 student 自身采样分布可能从未覆盖 teacher 高概率 token，导致有效学习信号方向性差（"student 看起来在被 teacher 认可，但其实就是在乱说话"）。Mitigate：(1) **off-policy SFT warm start**：先让 student 在 teacher trajectory 上 SFT 一段，状态分布拉近后再 OPD；(2) **intermediate teacher**：用中等大小的 teacher（如 70B）当桥梁；(3) **curriculum**：先短 trajectory，再逐渐放长；(4) **temperature scheduling**：student temperature 初期高扩大探索。
</details>

<details>
<summary><strong>L2-9：OPD 与 R1-Distill 是同一类方法吗？</strong></summary>

**答**：**不是**。R1-Distill 是 **off-policy** SFT distillation——R1 teacher 离线生成 800K trajectory，student 在这些数据上做 token-level cross-entropy，**没有 student rollout、没有 KL 信号**。OPD 是 on-policy + teacher KL。两者互补：R1-Distill 可以作为 OPD 的 cold-start init（先模仿 teacher style），再用 OPD 消 exposure bias。
</details>

<details>
<summary><strong>L2-10：怎么诊断 OPD 训练是否 healthy？关键监控量是什么？</strong></summary>

**答**：监控五件套：(1) **per-token reverse KL**——应单调下降，但不能贴 0（贴 0 = student 与 teacher 完全一致，可能 overfit）；(2) **rollout length / truncation rate**——truncation rate < 5%，否则 length collapse；(3) **token entropy**——下降但不应接近 0（mode collapse）；(4) **pass@1 vs pass@N (N>1)**——pass@1 涨而 pass@64 跌 = diversity 塌缩；(5) **val accuracy on held-out**——这是终极信号，spike 立即停。
</details>

### L3 顶级 lab（5 题，研究 / 算法 lead）

<details>
<summary><strong>L3-1：从 KL-constrained RL 视角统一 OPD 与 RLHF / GRPO。</strong></summary>

**答**：KL-constrained RL 目标：
$$\max_\theta\, \mathbb{E}_{y \sim \pi_\theta}[R(x, y)] - \beta\, \mathbb{E}_{y \sim \pi_\theta}\!\big[D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\big]$$
- **RLHF / PPO**：$R$ = RM scalar reward, $\pi_{\text{ref}}$ = SFT init, $\beta$ 小
- **GRPO**：同 RLHF，但 advantage 用组内归一化代替 GAE
- **OPD**：$R \equiv 0$（无外部 reward）, $\pi_{\text{ref}} = \pi_T$（reference = teacher），$\beta = 1$
- **OPD + GRPO**（生产标配）：$R$ = outcome verifier + dense teacher KL reward, $\pi_{\text{ref}} = \pi_T$

Survey (arXiv 2604.00626) 把这统一称为 "$f$-divergence minimization on student rollout"，OPD 是这类问题的 $R \equiv 0$ 特例，DPO 是 closed-form $R$ + offline 特例。**注意 RLHF 不是 $\beta \to 0$ 特例**：一般 KL-constrained RL 目标在 $\beta$ 取常见小正值（如 0.01-0.1）时本身就是 RLHF/PPO 的标准形式，并非某个极限特例；相反，$\beta \to 0$ 对应去掉 KL 正则、纯粹 reward 最大化的无约束 RL（易 reward hacking），这与 RLHF 保留有限非零 KL 惩罚以防止偏离参考策略的核心特征相悖。
</details>

<details>
<summary><strong>L3-2：OPD 的理论收敛性分析有哪些已知结果？</strong></summary>

**答**：核心已知结果（截至 2026-05）：(1) **不动点**：$L_{\text{OPD}} = 0$ iff $\pi_\theta = \pi_T$ on the support of $\pi_\theta$（reverse KL 性质，且只在 student 访问的支撑上对齐）。所以 OPD 不能让 student "超越" teacher——但能让 student 在自己的 capacity 内最大化模仿 teacher 的某个 mode；(2) **收敛性**：在凸 policy parametrization 假设下，policy gradient 收敛到 reverse KL 的局部最小（Geist & Pietquin 2014 风格），但 LLM 的非凸 parametrization 没有 global 保证；(3) **Rethinking OPD** (arXiv 2604.13016) 指出 OPD 可以在 reasoning 任务上"超越 teacher"——这看似矛盾，但解释是 teacher logits 中包含 dark knowledge（如 self-correction signal），student 通过 on-policy 训练激活了 teacher 也未必能稳定发挥的能力。这是 OPD 与传统 imitation learning 的关键区别。

**[needs-verify]** "超越 teacher" 现象在不同 paper 报告不一致，需查 Rethinking OPD 原文细节。
</details>

<details>
<summary><strong>L3-3：multi-teacher OPD 怎么做？DeepSeek-V4 报告的"OPD 替代 mixed RL"是什么意思？</strong></summary>

**答**：multi-teacher OPD 把多个 specialist teacher 的 logits 在每个 token 上加权 ensemble：
$$\pi_T(v|s_t) = \sum_k w_k(s_t) \cdot \pi_{T_k}(v|s_t)$$
weight $w_k$ 可以是固定权重（如 math task 上 math teacher 权重高）、context-dependent（routing-style）或可学习。DeepSeek-V4 在 model consolidation 阶段用 math / code / chat / reasoning 多个 specialist teacher 做 multi-teacher OPD，完全替代了之前 mixed RL（多 RM 加权）。优势是 **每 token 都有 dense supervision**，比 multi-RM 的 mixed RL sample efficient 显著。

**[needs-verify]** DeepSeek-V4 的具体 multi-teacher 实现细节（weight 选择、是否 token-level routing）需查原 tech report；目前公开材料主要是 secondary source 引用。
</details>

<details>
<summary><strong>L3-4：OPD 的 "process reward" 视角与 PRM 的关系？两者能融合吗？</strong></summary>

**答**：把 OPD 的 per-token teacher KL 看成 "process reward"：每个 token（在 sampled-token / behavior-policy rollout 上）都有一个 dense reward $r_t = \log\pi_T(y_t|s_t) - \log\pi_\theta^{\text{old}}(y_t|s_t) = -\log\!\frac{\pi_\theta^{\text{old}}(y_t|s_t)}{\pi_T(y_t|s_t)}$（与 §3 / §4.5 的 OPD-GRPO 一致；上标 old 表示 rollout 时记录的 behavior policy log-prob，避免误读 $-\log\pi_\theta/\pi_T$）。这与 PRM (process reward model, Lightman 2023 "Let's Verify Step by Step") 在思想上一致——都是 dense 而非 sparse supervision。差异：PRM 是 step-level（每个推理 step 一个 0/1），OPD 是 token-level（每 token 一个 KL 值）。**融合方案**：(1) step-level reward = $\sum_{t \in \text{step}_k} r_t^{\text{OPD}} + \lambda \cdot r_k^{\text{PRM}}$；(2) 用 PRM 当 trajectory filter（PRM 高分的 trajectory 才进 OPD 训练）；(3) 用 OPD 蒸馏一个 PRM（dense teacher signal 训 process-level verifier）。学术上这块是 2026 active research。

**[needs-verify]** "OPD + PRM 融合"具体 paper 与实验结果尚不完整；上述方案是综合多源材料后的合理推断。
</details>

<details>
<summary><strong>L3-5：从 information-theoretic 视角分析 OPD 为什么能 9-30× sample efficient 于 RL。</strong></summary>

**答**：考虑 trajectory $y$ 长度 $N$，vocab $V$。每条 trajectory 上：

- **RL outcome reward**：1 个 scalar，最多 $\log_2 V_R$ bits（$V_R$ = reward 离散度，二元 reward $\log_2 2 = 1$ bit；real-valued 约 $\log_2 1000 \approx 10$ bits）
- **OPD per-token teacher KL（sampled）**：每 token 一个 $\log \pi_T(y_t)$ 值，约 $\log_2 V \approx 17$ bits（典型 vocab 100K-128K）；trajectory 累计 $N \cdot 17$ bits
- **OPD per-token full KL**：每 token 整个 vocab 分布，理论上限 $\log_2 V$ bits per token（但实际信息量取决于 teacher 分布 entropy）

bit-rate 比：OPD / RL ≈ $N \cdot 17 / 10 = 1.7N$。在 long-CoT 任务（$N = 2K$-$8K$）上该比值约为 **3400-13600×** 量级——这从信息论角度解释了 Thinking Machines 报告的 9-30× compute efficiency 为什么有数量级的空间存在，但实际效率远没达到这个信息论上限。

**caveat**：这是 upper bound 论证，有两层不严谨之处需要注意。(1) 把连续标量 $\log \pi_T(y_t)$ 直接等同于携带 $\log_2 V \approx 17$ bits 信息是一种启发式类比——$\log_2 V$ 只是"从 $V$ 个离散候选中选 1 个"这一分类结果的信息熵上界，并不严格等于对数概率这个连续值本身携带的信息量；此 bit-rate 论证应视为启发式上界，而非严格的信息论推导。(2) 实际 compute efficiency 还受梯度噪声、teacher / student gap、optimizer 等影响，远达不到 3400-13600× 的理论上限。OPD 在简单任务上的 advantage 通常不到 10×，在 long-horizon 复杂任务上才能逼近报告中的 9-30×。
</details>

## §A 附录

### A.1　Sanity-check：用 OPD 收敛性检验你的实现

实现 OPD 后，做这三个 micro-test 确认 loss 正确：

```python
# Test 1: student == teacher → loss should be ~0
student.load_state_dict(teacher.state_dict())
loss, _ = per_token_reverse_kl_loss(student, teacher, ids, mask)
assert abs(loss.item()) < 1e-4, f"identical model should give zero KL, got {loss}"

# Test 2: student random init → loss > 0
student = init_random_model(...)
loss, _ = per_token_reverse_kl_loss(student, teacher, ids, mask)
assert loss.item() > 0.5, f"random student should give positive KL, got {loss}"

# Test 3: loss should decrease over training
losses = []
for step in range(100):
    loss = opd_train_step(student, teacher, ...)
    losses.append(loss)
assert losses[-1] < losses[0], "OPD should reduce KL over training"
```

### A.2　常见错误与正确做法

| 错误做法 | 现象 | 正确做法 |
|---|---|---|
| 在 teacher trajectory 上算 OPD loss | 退化为 off-policy KD，丢失 on-policy 价值 | 必须 student 自己 rollout |
| KL 方向写反（forward KL 当 reverse KL） | mode-covering，输出平庸 | reverse KL 是 $\pi_\theta$ 在前 |
| Mask 把 prompt token 也算进 loss | prompt token 上 KL 信号无意义 | `action_mask` 只标 student 生成的 token |
| 用 sum reduction 不 normalize 长度 | length inflation 训练崩盘 | per-token average（除以 mask.sum()） |
| Teacher 用 train mode（dropout） | logits 不稳，loss 噪声大 | teacher 必须 `.eval()` + `torch.no_grad()` |
| Student rollout 用 greedy decode | trajectory diversity 太低，OPD 学不到 robustness | sampling with temperature ≥ 1.0 |
| 不监控 truncation rate | 撞 max_length 后 silent failure | monitor; 超 5% 就调 max_new_tokens |

### A.3　核心 paper 与资源列表

**核心 OPD paper**：
- MiniLLM (Gu et al. 2023, ICLR 2024) — arXiv 2306.08543
- GKD: On-Policy Distillation of Language Models (Agarwal et al. 2023, ICLR 2024) — arXiv 2306.13649
- A Survey of On-Policy Distillation for LLMs (Song & Zheng 2026) — arXiv 2604.00626
- Rethinking On-Policy Distillation (2026) — arXiv 2604.13016
- KL for a KL (vOPD, 2026) — arXiv 2605.07865
- Black-Box On-Policy Distillation (2026) — arXiv 2511.10643
- Decoupling KL and Trajectories (2026) — arXiv 2605.16826

**工业 tech report**：
- Qwen3 Technical Report (Qwen Team 2025-05) — arXiv 2505.09388 (§3.2 OPD recipe)
- DeepSeek-R1 (DeepSeek 2025-01) — arXiv 2501.12948 (off-policy distillation series)

**Blog / 代码**：
- Thinking Machines Lab — "On-Policy Distillation" blog (Lu et al. 2025-10-27) — `thinkingmachines.ai/blog/on-policy-distillation/`
- Tinker Cookbook — `github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation`
- TRL GKD Trainer — `huggingface.co/docs/trl/gkd_trainer`
- Awesome OPD list — `github.com/thinkwee/AwesomeOPD`, `github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation`

**相关基础**：
- Hinton, Vinyals, Dean — "Distilling the Knowledge in a Neural Network" (2015) — arXiv 1503.02531
- Sanh et al. — "DistilBERT" (2019) — arXiv 1910.01108
- Kim & Rush — "Sequence-Level Knowledge Distillation" (EMNLP 2016) — arXiv 1606.07947
- Bagnell — "Reinforcement Learning and Imitation Learning" (theoretical foundation for exposure bias)
- DeepSeekMath GRPO (2024) — arXiv 2402.03300
- DPO (Rafailov et al. NeurIPS 2023) — arXiv 2305.18290

### A.4　[needs-verify] 标记一览

本 cheat sheet 中下列内容标记为 **[needs-verify]**，建议在面试前后查原始 paper / tech report 确认：

1. **L3-2 "OPD 超越 teacher"**：Rethinking OPD (arXiv 2604.13016) 报告的具体 setting 与 magnitude
2. **L3-3 DeepSeek-V4 multi-teacher OPD 实现细节**：weight 选择策略、是否 token-level routing
3. **L3-4 "OPD + PRM 融合"**：截至 2026-05 active research，尚无单一权威 paper 整合两者
4. **§6.2 Thinking Machines 数据**："9-30× FLOPs 节省" 来自 blog secondary source，原 blog 数字与具体 setting 应核对
5. **§5.2 timeline**：2025-2026 多篇 OPD-related arXiv paper 编号（如 2604.* 系列）来自 2026 Q1-Q2 投稿/预印本，部分 ID 可能在投稿后更新版本号或重排
6. **OPD 在 Qwen3 / MiMo 上的具体采用细节**：多数信息来自 Thinking Machines blog 与 Qwen3 paper §3.2，但 MiMo 的 distillation 部分细节需查其 tech report；**Gemma 2/3 已重新归类为 off-policy soft-target KD（非 OPD，见 §5.3 与 §0 TL;DR 第 7 条）**，这一归类基于对 Gemma 2 技术报告方法描述的回忆，未做逐字核对，建议再查原始报告确认
7. **§6.1 Qwen3 具体数值**：pass@64（vs pass@1）与 "+3-5pp" 的精确度量口径待核对原始 Qwen3 Technical Report §3.2（arXiv 2505.09388）

### A.5　术语速查表

| 中文 | 英文 | 含义 |
|---|---|---|
| 在线策略蒸馏 | On-Policy Distillation (OPD) | student 在自己 rollout 上做 KL 蒸馏 |
| 离线蒸馏 | Off-Policy Distillation | student 在 teacher / dataset 数据上蒸馏 |
| 暴露偏置 | Exposure Bias | autoregressive 模型 train/test 分布不一致 |
| 反向 KL | Reverse KL | $D(\pi_\theta \,\mid \, \pi_T)$，mode-seeking |
| 正向 KL | Forward KL | $D(\pi_T \,\mid \, \pi_\theta)$，mode-covering |
| 模式寻找 | Mode-Seeking | 锁定一个 mode，sharp 输出 |
| 模式覆盖 | Mode-Covering | 覆盖所有 mode，平均输出 |
| 教师强制 | Teacher Forcing | 训练时用 ground-truth prefix |
| 控制变量 | Control Variate | 降方差用的 baseline 项 |
| 截断坍塌 | Truncation Collapse | rollout 越训越长撞 max_length 导致崩盘 |
| 局部可教性塌缩 | Local Teachability Collapse | trajectory 后半 teacher 没东西可教 |

> ⚠️ **caveat** — OPD 作为 LLM post-training 的独立技术名词主要在 **2025 下半年**（Qwen3 + Thinking Machines blog）才广泛流行。在此之前同样的方法在 MiniLLM (2023) 与 GKD (2023) 中已经提出。所以"OPD 是新方法"在严格意义上不正确——它是被重新命名 + 大规模工业化的旧方法，受益于 reasoning model 时代对 dense supervision 的渴求。这一历史脉络在面试 L3 上常被问及，请注意区分"方法首次提出年份"与"术语流行年份"。
