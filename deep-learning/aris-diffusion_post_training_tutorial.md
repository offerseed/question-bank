# 扩散后训练 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[diffusion-post-training-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/diffusion_post_training_tutorial.md)

---

## §0 TL;DR

> 💡 **9 句话搞定 Diffusion Post-Training** — 一页拿下 RL/DPO/Flow-RL 全家桶（详见 §1–§10 推导）。

1. **为什么难**：diffusion 是多步 $T$ 步生成（典型 20–50 步），reward 只在终态 $x_0$ 给一次 —— **稀疏 terminal reward + 长 denoising trajectory + credit assignment** 三件事叠加，比 LLM RLHF 多一个"轨迹积分"维度。

2. **三条主线**：(i) RL on denoising MDP（DDPO / DPOK，把 $T$ 步 denoising 当 MDP）；(ii) Direct reward backprop（DRaFT / AlignProp / ReFL，把 reward 当 differentiable loss 沿 $T$ 步反传）；(iii) Preference optimization（Diffusion-DPO / D3PO / SPO / Diffusion-KTO / MaPO，把 LLM DPO 家族搬到 diffusion）。

3. **DDPO (Black et al. 2024 ICLR, arXiv 2305.13301)**：denoising 视为 $T$-步 MDP，state $= (x_t, t, c)$，action $= x_{t-1}$，per-trajectory reward $R(x_0, c)$；用 REINFORCE 或 PPO-clip 更新 $\log p_\theta(x_{t-1} \mid x_t, c)$。

4. **AlignProp (Prabhudesai et al. 2024 ICLR, arXiv 2310.03739)** & **DRaFT (Clark et al. 2024 ICLR, arXiv 2309.17400)**：reward $R$ 关于 $x_0$ 可导时，直接把 $R(x_0)$ 沿 $T$ 步 sampler **反传**到 $\theta$。**关键工程问题**：显存 $\mathcal{O}(T)$；DRaFT-K / AlignProp 只回传最后 $K$ 步（典型 $K \in \{1, 5\}$），配合 gradient checkpointing 把显存压到 $\mathcal{O}(K)$。

5. **Diffusion-DPO (Wallace et al. 2024 CVPR, arXiv 2311.12908)**：把 LLM 的 $\log\pi/\pi_\text{ref}$ 换成 diffusion 的 **per-step ELBO surrogate**——具体地，用 $-\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ 作为 $\log p_\theta(x_0)$ 的一个 lower bound 项，对 $(y_w, y_l)$ 拼成 DPO contrastive。

6. **D3PO (Yang et al. 2024 CVPR, arXiv 2311.13231)**：**完全免 RM**——直接把人类对生成图片的 thumbs up/down 信号代入 KL-regularized 最优解的 implicit reward；推导上与 DPO 平行，但放到 diffusion **per-step Markov chain** 上。

7. **SPO (Liang et al. 2024, arXiv 2406.04314)**：观察到不同 denoising step 偏好不同（高噪 step 学构图，低噪 step 学细节），把 DPO 推广为 **step-aware**——每个 $t$ 单独采 in-step pair $(x_{t-1}^w, x_{t-1}^l)$，loss 在 step 维度上加权。

8. **Flow-GRPO (Liu et al. 2025, arXiv 2505.05470)**：第一个把 GRPO 搬到 Flow Matching 的工作——ODE→SDE 同边缘转换 + denoising reduction，SD3.5-M GenEval 63% → 95%。它与 DGPO、DiffusionNFT 的完整推导和对比已独立成篇：[《现代 Diffusion 后训练》](modern_diffusion_post_training_tutorial.md)。

9. **Reward hacking is the real boss**：过饱和颜色、构图单调、风格收敛、PickScore 高但人眼丑 —— 缓解靠 reward ensemble (HPSv2 + PickScore + ImageReward + CLIP-Score)、KL anchor (Diffusion-DPO 的 $\beta$)、early stop on reward plateau。SD3 / FLUX **几乎不公开 post-training 细节**，但社区主流认为 SD3.5 Turbo 系列、FLUX.1 dev 走的是 DPO + 蒸馏混合路线。

> ✅ **vs LLM RLHF 一句话对比** — LLM RLHF 关心 "token-level credit assignment + KL anchor"；diffusion post-training 关心 "denoising-step credit assignment + 显存爆炸 (backprop) 或 sample 爆炸 (RL)"。本质相同问题——稀疏 reward + 长轨迹——只是轨迹的物理含义换了。

---

## §10 25 高频面试题

按难度分 3 档：L1 = 多模态/diffusion 岗常问；L2 = research/alignment 方向会问；L3 = 顶级 lab 的硬核题。

### L1 必会题（10 题）

<details>
<summary>Q1. 为什么 diffusion 模型需要 post-training？SFT 不够吗？</summary>

- SFT 只能模仿正例（"做得好的样子"），学不到**对比信号**（A 比 B 好）。
- Post-training 通过 reward / preference 提供对比信号，让模型在 alignment、aesthetic、prompt-faithful 维度都涨。
- 实测 Diffusion-DPO 在 PickScore 上 +5-10 点，远超继续 SFT。

只说"提升画质"是浅；要说清"对比信号 vs 模仿信号"的差异。
</details>

<details>
<summary>Q2. DDPO 把 diffusion 当成什么 MDP？state/action/reward 怎么定义？</summary>

- **State**: $s_t = (x_t, t, c)$
- **Action**: $a_t = x_{t-1}$（从 $p_\theta(\cdot \mid x_t, c)$ 采）
- **Transition**: 确定性 $s_{t-1} = (x_{t-1}, t-1, c)$
- **Reward**: $r_t = 0$ for $t > 1$，$r_1 = R(x_0, c)$（terminal-only）

说成"per-step reward"（错，只在终态）；或不知道 transition 是确定性的（noise 是 action 自带的）。
</details>

<details>
<summary>Q3. Diffusion-DPO 用什么代替 $\log \pi_\theta(y|x)$？</summary>

- 用 ELBO surrogate：$-\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$（DDPM 的 $L_\text{simple}$）作为 $\log p_\theta(x_0)$ 的代理。
- 这是 $\log p_\theta$ 的（negative）下界项，方向正确。
- 注意 $L_\text{simple}$ 把真正 VLB 的 timestep 权重 $\lambda_t$（$\omega(\lambda_t)$）设为 1，这是 Ho 2020 为了采样质量主动做的 reweighting，并非"log p_θ 的紧 ELBO"本身——把它当 log p_θ 的代理是近似，不是恒等。
- 配上 paired noise $\epsilon$（$y_w$ 和 $y_l$ 共享同一 $\epsilon$，方差缩减技巧）才更稳。

说用 $\log p(x_0)$ 解析式（错，diffusion 没闭式）；把 $L_\text{simple}$ 当成严格恒等的 ELBO（忽略了 $\omega(\lambda_t)=1$ 的 reweighting）；忘记 paired noise 能降方差。
</details>

<details>
<summary>Q4. AlignProp 和 DRaFT 的核心 idea？显存为什么是 $\mathcal{O}(K)$？</summary>

- **核心**：reward $R(x_0)$ 可导 → 直接对 $\theta$ 反传，跳过 RL。
- $T$ 步 sampler 是 differentiable computation graph，**vanilla 反传需存 $T$ 步 activation** → $\mathcal{O}(T)$。
- **DRaFT-K / AlignProp**：前 $T-K$ 步用 `no_grad`，只在最后 $K$ 步保留梯度，显存压到 $\mathcal{O}(K)$。
- 典型 $K=1$ 已能给强信号。

说 K=1 等于 REINFORCE（错，K=1 是 reparameterized gradient，方差远低于 REINFORCE）。
</details>

<details>
<summary>Q5. 为什么 Diffusion-DPO 的 $\beta$ 比 LLM DPO 大 1000 倍？</summary>

- LLM DPO: $\beta \in [0.05, 0.5]$
- Diffusion-DPO: $\beta \in [2000, 5000]$
- 原因（方向性直觉，非精确公式）：diffusion 的"trajectory log-likelihood"是 $T$ 个 Gaussian log-prob 之和，单步 $\epsilon$ 距离差吸收了 $T$ 倍系数，这解释了 $\beta$ 大几个数量级的方向。
- **但"$\beta T$ 就是真正的温度"不是一个干净的、implementation-independent 的事实**：它还依赖 MSE 是 mean-reduce 还是 sum-reduce（相差 $C \times H \times W$ 倍），以及 $T$ 具体指 DDPM 训练用的 $\sim$1000 步 schedule 还是推理用的 $\sim$20–50 步 schedule——本教程代码用 `.mean([1,2,3])` + 1000-step DDPM $t$-sampling，换一种 reduction/T 约定，"有效温度"就不再是同一个 $\beta T$。

说"diffusion 噪声大所以 β 大"（错，是 trajectory 长度的累计效应）；或把"$\beta T$ 是真温度"当成放之四海而皆准的精确等价（忽略了 mean/sum 归约和 T 约定的依赖）。
</details>

<details>
<summary>Q6. ImageReward / HPSv2 / PickScore 三者区别？</summary>

| | ImageReward | HPSv2 | PickScore |
| --- | --- | --- | --- |
| 数据规模 | 137K pair | 798K pair | 1M pair (Pick-a-Pic) |
| backbone | BLIP fine-tuned | CLIP fine-tuned | CLIP fine-tuned |
| scale | $[-1, 4]$ | $[0, 1]$ | logits |
| 偏好 | aesthetic + alignment | 高对比度 + alignment | SDXL 风格 |

工业上**取 ensemble**（最少两个）。

说三者都一样（错，scale 和偏好差异大）。
</details>

<details>
<summary>Q7. DDPO 用 DDPM 还是 DDIM 采样？为什么？</summary>

- **DDPM**（或 DDIM-eta=1）—— 需要 stochastic transition，REINFORCE/score-function 型策略梯度才有良定义的 log-density 可以求导。
- DDIM-eta=0 是 deterministic（Dirac delta transition），没有 well-defined 的 log-density——**REINFORCE 这类 score-function 估计在这里退化/失效**，但这不等于"policy gradient 恒为 0"：$x_0$ 仍是 $\theta$ 的确定性可微函数，pathwise/reparameterized gradient（§3 DRaFT/AlignProp 用的正是这个）依然存在且通常非零。DDPO 需要 stochastic transition，是因为它选用了 REINFORCE 这一特定梯度估计方式，不是因为确定性采样下"真梯度"消失。
- 类比：LLM RL 的 REINFORCE/PPO 必须用 sampling（temperature > 0），不能用 greedy——但 greedy 输出仍是参数的可微函数，只是 score-function 梯度用不了这条路径。

说 DDIM 也行（错，对 REINFORCE 而言要 eta > 0）；或以为确定性采样下 true gradient 恒为零（错，混淆了"REINFORCE 失效"和"真梯度为零"，见 §3 DRaFT/AlignProp 的 pathwise gradient）。
</details>

<details>
<summary>Q8. Reward hacking 在 diffusion 上典型症状有哪些？</summary>

- **Over-saturation**（颜色饱和度暴增）—— HPSv2/aesthetic 偏好鲜艳。
- **Center bias**（主体永远居中）—— RM 训练数据偏 centered。
- **Monotone composition**（不同 prompt 同一构图）—— mode collapse。
- **Watermark hallucination**（角落 fake 水印）—— RM 训练数据含水印。
- **Cartoon shift**（写实 prompt 输出 anime）—— RM 标注者偏好。

只说"过度优化"不具体；要能举至少 3 种具体视觉症状。
</details>

<details>
<summary>Q9. DPOK 比 DDPO 多了什么？它的 per-step KL 之和等于终态 KL 吗？</summary>

- DDPO 是纯 RL（REINFORCE / PPO-clip），KL 只通过 ratio clip 隐式存在；DPOK 在目标里加**显式** $\beta\,\text{KL}(p_\theta \Vert p_\text{ref})$，对应 LLM RLHF 的 "$\beta\log(\pi/\pi_\text{ref})$"。
- per-step Gaussian KL 之和是**联合轨迹 KL** $\text{KL}(p_\theta(x_{0:T}) \Vert p_\text{ref}(x_{0:T}))$ 的精确展开；由 data-processing inequality，它是**终态 marginal KL 的上界**，不是恒等式。DPOK 用这个可闭式算的上界当 surrogate。

把"上界"说成"等于"不得分。
</details>

<details>
<summary>Q10. Diffusion-DPO 训练时 $y_w$ 和 $y_l$ 的 noise 怎么处理？</summary>

- **paired noise**：$x_t^w$ 和 $x_t^l$ 用**同一个** $\epsilon$（即 $x_t^w = \sqrt{\bar\alpha_t}x_0^w + \sqrt{1-\bar\alpha_t}\epsilon$，$x_t^l$ 同理用同一 $\epsilon$）——这是**方差缩减**（common random numbers）技巧，不是数学正确性的前提。
- 不 paired（各自独立采样 $\epsilon^w,\epsilon^l$）时，loss 仍是同一 DPO 目标的无偏估计，只是方差更大、优化更不稳；$\beta$ 的尺度和可比性不因此改变。
- 这是 Diffusion-DPO 实践中最容易忽视、但会实打实影响训练稳定性的实现细节。

不知道 paired noise 能降方差（错失一个稳定性技巧）；或以为不 paired 会让 loss 数学上不正确（错，只是方差更大）；或以为 $\epsilon$ 是 $\epsilon_\theta$ 的预测（错，这里 $\epsilon$ 是 q-sample 的 noise）。
</details>

### L2 进阶题（10 题）

<details>
<summary>Q11. 推导 Diffusion-DPO loss（从 KL-regularized 最优解出发）。</summary>

1. KL-regularized 目标：$\max_p \mathbb{E}[R] - \beta\, \text{KL}(p \Vert p_\text{ref})$，最优解 $p^* \propto p_\text{ref} \exp(R/\beta)$。
2. 反解 implicit reward：$R(x_0, c) = \beta \log(p^*/p_\text{ref}) + \beta \log Z(c)$。
3. 代入 BT：$P(y_w \succ y_l) = \sigma(R_w - R_l)$；$\log Z$ 在差中消掉。
4. 替换 $p^* \to p_\theta$；$\log p_\theta$ 用 ELBO surrogate：$-L_\text{simple} = -\|\epsilon - \epsilon_\theta\|^2$（在 $x_t = q\text{-sample}(x_0, t, \epsilon)$ 处）。
5. 期望对 $t \sim U(1, T)$ 取，loss 变成 $-\log\sigma(\beta T [\Delta_w - \Delta_l])$，$\Delta_y = \|\epsilon - \epsilon_\text{ref}\|^2 - \|\epsilon - \epsilon_\theta\|^2$。

直接背公式答不上来 $\log Z$ 为什么消掉；或不知道 ELBO surrogate 的来源。
</details>

<details>
<summary>Q12. AlignProp 反传 $K$ 步显存 vs 性能怎么 trade-off？</summary>

- 显存：$\mathcal{O}(K \cdot M_\text{UNet})$。SDXL 单 forward $\sim$8 GB activation；$K=1 \to 24$ GB（含 weights + grad）；$K=5 \to 60$ GB；$K=10 \to 120$ GB。
- 性能：$K=1$ 实测已达 $K=5$ 的 95%；$K \ge 5$ 在大多数 reward 上无显著提升。
- **直觉**：最后一步 $x_1 \to x_0$ 对终态影响最大，前面 49 步的方差被压缩。
- 工业上 $K=1$ 是标准选择（24GB 单卡可训）。

只说"$K$ 越大越好"（错，性能曲线 saturate）；不知道显存量级。
</details>

<details>
<summary>Q13. DDPO 用 REINFORCE 和 PPO 区别？哪个 prod 常用？</summary>

- **DDPO-SF (REINFORCE)**：$\hat g = \sum_t \nabla \log p_\theta \cdot (R - b)$，简单但方差大。
- **DDPO-IS (PPO-clip)**：用 importance ratio $\rho_t = p_\theta/p_{\theta_\text{old}}$，多次 update 同 batch，clip $\rho_t$。
- **per-step ratio** 而非 trajectory ratio（避免 $T$ 个 ratio 累乘的方差爆炸）。
- Prod 常用 PPO 形式：稳一些，sample efficiency 更高。

说"trajectory-level ratio"（错，per-step）；或不知道两个都是 DDPO。
</details>

<details>
<summary>Q14. SPO 怎么得到 in-step preference pair？为什么需要 step-RM？</summary>

- **In-step**：给定 $x_t$，独立采两个 $x_{t-1}^a, x_{t-1}^b$（用 policy 的 stochastic transition $p_\theta(\cdot \mid x_t)$ 采两次）。
- 用**step-wise reward model** $R_\text{step}(x_{t-1}, x_t, c, t)$ 判 winner。
- step-RM 训练数据：base UNet rollout 多 trajectory，每步的 step-reward 由终态 reward 反推（类似 Math-Shepherd 的 rollout-based PRM）。
- 不用 step-RM 用终态 RM 也行，但要 rollout 到 $x_0$ 才能打分，贵 $T$ 倍。

不知道 in-step pair 怎么得（错，要采两次）；或不知道 step-RM 是 SPO 独有。
</details>

<details>
<summary>Q15. Diffusion-DPO vs D3PO 实质差异？</summary>

- **推导路径**：Diffusion-DPO 用 ELBO surrogate（单步 $\epsilon$-distance），D3PO 用完整 trajectory log-ratio。
- **数学等价性**：在 ELBO 下界 + 期望 over $t$ 下，D3PO 的 trajectory 形式退化为 Diffusion-DPO 的单步形式。
- **实践差异**：
  - Diffusion-DPO 每步只算一次 UNet forward（policy + ref），便宜。
  - D3PO 严格意义上要算完整 trajectory $T$ 次 forward。
- 工业部署主流用 Diffusion-DPO（便宜 + 稳）。

说"完全不同"（错，理论等价）；或不知道 D3PO 也是 DPO 家族。
</details>

<details>
<summary>Q16. ReFL 和 DRaFT / AlignProp 都是"reward 当 loss 反传"，实质差在哪？</summary>

- ReFL（ImageReward 原文）只在**一个随机中间步** $t'$ 上用 Tweedie 一步估计 $\hat x_0(x_{t'}, t')$ 算 reward，并和 $\mathcal L_\text{simple}$ **联合训练**：$\mathcal L = \mathcal L_\text{simple} - \lambda\,\mathbb E_{t'}[R(\hat x_0)]$。
- DRaFT / AlignProp 沿真实采样轨迹**多步反传**纯 reward loss（DRaFT-K 截断最后 $K$ 步，AlignProp 用 checkpointing）。
- 取舍：ReFL 更稳、reward 涨幅小（看的是一步估计不是真实轨迹）；DRaFT 涨幅大、显存 $\mathcal O(K)$、更容易 hack。

只答"ReFL 更早"不得分；要说出单步 Tweedie + 联训这两点。
</details>

<details>
<summary>Q17. MaPO 怎么去掉 reference model？loss 长什么样？</summary>

- 不用 $\log(p_\theta/p_\text{ref})$，直接用**绝对 likelihood margin**：
$$\mathcal{L}_\text{MaPO} = -\log\sigma\!\big(\beta(\hat\ell_w - \hat\ell_l) - \gamma\big) + \alpha \hat\ell_w$$
- $\hat\ell = -\|\epsilon - \epsilon_\theta\|^2$ 是 likelihood surrogate。
- $\gamma$ 是 margin（类似 SimPO），$\alpha\hat\ell_w$ 项防止"两边都降"。
- 显存省一半（无 ref UNet），训练快 15%，且解决 reference mismatch 问题（fine-tune 到风格差异大的目标时稳）。

说去 ref 就完事（错，要加 likelihood term 防 degenerate）；不知道 reference mismatch。
</details>

<details>
<summary>Q18. Reward ensemble 为什么用 min 比 mean 好？</summary>

- mean：被一个高分 RM 主导可能仍 hack。
- min：要所有 RM 都同意"好"才给高 reward → hacking 必须同时骗过所有 RM，难度指数级上升。
- 等价于 conservative aggregation (Coste 2024 ICLR for LLM)，diffusion 上同理。
- 代价：reward 偏保守，涨幅小。
- 工业上常用 `R = mean - k * std`（含 uncertainty penalty）作折中。

只说"防 hacking"不知道为啥 min；或不知道这是 LLM 也用的 ensemble 策略。
</details>

<details>
<summary>Q19. Diffusion-KTO 比 Diffusion-DPO 有什么独特优势？</summary>

- 只需 **per-image binary feedback**（thumbs up/down），**不需要 paired comparison**。
- 工业场景大量用户 reaction（喜欢/不喜欢）远多于 paired comparison → KTO 让这部分数据可用。
- prospect-theoretic 价值函数 $v(\cdot)$ 对正负 feedback 不对称（loss aversion）。
- 不需要"哪个更好"的标注成本。

不知道 KTO 是 unpaired（错，这是 KTO 全部 idea）；或不知道 prospect theory 来源。
</details>

<details>
<summary>Q20. 为什么 diffusion 没有 token-level KL anchor，而是 trajectory-level？</summary>

- LLM 的 KL 是 per-token：$\sum_t \log(\pi_\theta(y_t)/\pi_\text{ref}(y_t))$。
- Diffusion 的 KL 是 per-step（per-denoising-step），不是 per-pixel：$\sum_t \text{KL}(p_\theta(\cdot \mid x_t) \Vert p_\text{ref}(\cdot \mid x_t))$。
- 两个 Gaussian KL 有闭式：$\text{KL} = \frac{1}{2}\big[(\mu_\theta - \mu_\text{ref})^2/\sigma^2 + (\sigma_\theta/\sigma_\text{ref})^2 - 1 - 2\log(\sigma_\theta/\sigma_\text{ref})\big]$。
- 像素之间不独立（卷积/attention），所以 KL 是整张图 level 而非 per-pixel。

说"per-pixel KL"（错，per-step）；不知道 Gaussian KL 闭式。
</details>

### L3 顶级 lab 题（5 题）

<details>
<summary>Q21. 推 Diffusion-DPO loss 从 reverse ELBO 出发，说清 ELBO surrogate 为何有效。</summary>

1. DDPM ELBO：$\log p_\theta(x_0) \ge -\sum_{t=2}^T \text{KL}(q(x_{t-1}|x_t,x_0) \Vert p_\theta(x_{t-1}|x_t)) + \log p_\theta(x_0|x_1) - \text{KL}(q(x_T|x_0) \Vert p(x_T))$
2. 化简（Ho 2020）：$-\log p_\theta(x_0) \le L_\text{simple} + C$，$L_\text{simple} = \mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(x_t,t)\|^2$——**注意**：真正的 VLB 每个 $t$ 前有 SNR 权重 $\lambda_t$（即 $\omega(\lambda_t)$），$L_\text{simple}$ 是把 $\omega(\lambda_t)$ 设为 1 的 reweighted 版本（Ho 2020 为采样质量主动选择），并非对 VLB 直接化简得到的紧 bound。
3. KL-regularized 最优 $p^* \propto p_\text{ref}\exp(R/\beta)$，反解 $R = \beta\log(p^*/p_\text{ref}) + \beta\log Z$。
4. 代入 BT，$\log Z$ 消掉。
5. 用 ELBO surrogate 代 $\log p$：$\log p_\theta(x_0) \approx -L_\text{simple}$（**注意**：这是上界的取负，作为单 sample 估计 — 严格意义上是 lower bound 的一个项，而非 $\log p$ 本身，但作为 DPO 的 implicit reward proxy 数值有效）。
6. 期望对 $t$ 取，得到最终 loss。

**为什么 ELBO surrogate 有效的更深一层**：DPO 的 implicit reward 是 $\beta\log(p_\theta/p_\text{ref})$，它只依赖**相对** likelihood。ELBO surrogate 的常数项（$C$）在 $p_\theta$ 和 $p_\text{ref}$ 之间相消（两个模型用同一架构），只剩 $-\|\epsilon - \epsilon_\theta\|^2$ 的差。所以即使 ELBO 不是 $\log p$ 的紧 bound，**差异是可消的**。

只能写出最终公式背不出推导链；或不知道常数项相消是关键。
</details>

<details>
<summary>Q22. AlignProp 反传 $K$ 步显存 $\mathcal{O}(K)$ 是否真的无法绕过？</summary>

**理论上**可以，工程上很贵：

1. **Gradient checkpointing**：把 activation 的存储换成重算。每步 forward 不存 activation，反传时重新 forward 算 grad。
   - 显存：从 $\mathcal{O}(K \cdot M)$ 降到 $\mathcal{O}(\sqrt{K} \cdot M)$ + $\mathcal{O}(K \cdot \text{state})$。
   - 代价：反传速度慢 2-3x。
2. **Reversible ResNet**：如果 UNet 用 reversible 架构（i-RevNet 风格），反传时从 output 反推 input，不存 activation。
   - 但 Stable Diffusion / SDXL UNet 不是 reversible。
3. **Implicit gradient**：通过 fixed-point 假设把 $\nabla_\theta$ 写成 implicit function theorem。
   - 需要 sampler 收敛到 fixed point，diffusion 不满足。
4. **Truncated backprop with control variates**：DRaFT-K 已是这个方向；理论上加 control variates 可进一步降方差但不降内存。

**实践答案**：$K=1$ + gradient checkpointing + LoRA 是工程最优解。$\mathcal{O}(K)$ 不可绕过的本质是 — sampler 不是 reversible computation。

只说 gradient checkpointing 不到位；不知道 reversibility 假设。
</details>

<details>
<summary>Q23. DRaFT-1 只回传最后一步，这和 REINFORCE 等价吗？两种梯度估计的本质区别是什么？</summary>

不等价，是两类估计器：

1. **DRaFT-1 是 pathwise（reparameterized）梯度**：$\nabla_\theta R(x_0) = \nabla_x R \cdot \partial x_0/\partial\theta$，需要 reward 对图像可导，沿最后一步的计算图求 $\partial x_0/\partial\theta$。方差极小。
2. **REINFORCE 是 score-function 梯度**：$\mathbb E[R\,\nabla_\theta\log p_\theta(\tau)]$，不需要 reward 可导，只需要能算 $\log p_\theta$。方差大，靠 baseline / 组内归一化压。
3. 二者在期望上都是 $\nabla_\theta\,\mathbb E[R]$ 的无偏估计（各自前提下），但 DRaFT-1 只对最后一步的参数依赖求导——它是**截断**的 pathwise 估计，对更早步骤的依赖被丢掉了，所以是有偏的。
4. 工程含义：reward 可导（美学 / CLIP 类）优先 pathwise；reward 不可导（规则验证器、OCR）只能 score-function——这就是 DDPO 一系和 DRaFT 一系的分水岭。

把"截断"和"等价"混在一起说的，或者说 REINFORCE 也需要 reward 可导的，不得分。
</details>

<details>
<summary>Q24. SD3 / FLUX 是否真的用了 RL post-training？怎么判断？</summary>

**诚实回答**：公开论文 / 技术报告**都没明说**用 RL / DPO。但有以下线索：

1. **SD3 论文 (arXiv 2403.03206)**：只讨论 Rectified Flow + MM-DiT + reflow；没提 reward fine-tune。
2. **FLUX**：完全没发论文，model card 只提"trained on a large image-text dataset"。
3. **DALL-E 3 (OpenAI 2023)**：公开报告（Betker et al. 2023）核心是**synthetic recaptioning**——用 image captioner 给训练图像生成更详尽的合成描述并混合训练，以提升 prompt-following；并未描述 RLHF（reward model + PPO 等）后训练机制。
4. **业界共识**：闭源大模型（FLUX pro, DALL-E 3, Midjourney v6+）几乎确定有 reward-based fine-tune，但具体方法不公开。

**判断标准（black-box test）**：
- 给同一 prompt 让模型生成 100 张，FID-100 / multi-mode 多样性低 → 可能是 RL/DPO（mode collapse 信号）。
- prompt-image alignment 在 GenEval 高分但 portrait 风格单一 → reward over-optimization 信号。
- 同一 model 对 "vibrant"/"colorful" prompt 反应过强 → HPSv2/aesthetic RM 痕迹。

**结论**：FLUX 大概率有内部 DPO + distill 混合；SD3.5 推测有 SFT + 可能的 DPO。但**没有公开证据**——这道题的关键是答出"不公开但有间接证据"，避免胡编技术细节。

如果直接答"SD3 用了 Diffusion-DPO"是错的（论文没说）；要答"未公开但社区推测 + 列举证据"。
</details>

<details>
<summary>Q25. 如果让你设计一个 diffusion post-training pipeline，从 $0$ 开始，你怎么选？</summary>

**取决于约束**。给一个 generic 推荐：

**Phase 1: 偏好数据收集**
- 收集 paired preference (Pick-a-Pic 风格)：成本高但 DPO 直接可用。
- 收集 binary feedback (thumbs up/down)：成本低，用 Diffusion-KTO。
- 收集 rule-based ground truth (GenEval 类型 prompt + 自动 verifier)：成本低，用 Flow-GRPO。

**Phase 2: 算法选择**
- **首选 Diffusion-DPO**：offline、稳、便宜、社区代码成熟（HuggingFace `diffusers` 直接支持）。
- **如果 base 是 Flow Matching (SD3/FLUX)**：用 Flow-GRPO / DGPO / DiffusionNFT（选法见独立教程 §6、Q25），rule-based reward 优先。
- **如果 fine-tune 到新风格 / 显存紧**：用 MaPO（去 ref，省一半显存）。
- **如果 reward 可导且想榨干信号**：DRaFT-1 + LoRA，配 HPSv2 + PickScore ensemble。
- **NOT 首选 DDPO**：on-policy sampling 太贵，工程复杂度高，性能 vs DPO 无显著优势。

**Phase 3: Reward 设计**
- **Multi-RM ensemble**（min 或 mean - k·std）：HPSv2 + PickScore + ImageReward。
- **加 rule-based safety**：NSFW detector hard penalty。
- **加 rule-based alignment**：GenEval 自动 verifier（object count, OCR）。
- **每个 RM 独立 z-score 归一化**。

**Phase 4: 监控与 early stop**
- 每 N 步算 reward + FID-100k；reward 涨 + FID 涨 = hacking 信号。
- KL budget 监控：$\text{KL}(p_\theta \Vert p_\text{ref}) > K_\text{target}$ 时 stop。
- Human eval blind A/B (base vs RL) 每 1000 steps。

**Phase 5: distill 衔接**
- Post-training 完后做 ADD / LCM 蒸馏到 4-step / 1-step。
- 注意 distill 可能消除部分 RL 增益，需要 distill-aware 后训。

只答"用 Diffusion-DPO" 是浅；要答出"phase 分解 + 多 reward + 监控 + distill 衔接"才完整。
</details>

## §A 附录

### A.1 关键论文清单（含 arXiv ID）

| 论文 | 一句话 | arXiv | 发表 |
| --- | --- | --- | --- |
| **DDPO** | Diffusion 当 MDP，REINFORCE/PPO 训 | [2305.13301](https://arxiv.org/abs/2305.13301) | ICLR 2024 |
| **DPOK** | KL-regularized RL for diffusion | [2305.16381](https://arxiv.org/abs/2305.16381) | NeurIPS 2023 |
| **DRaFT** | 直接 reward 反传 $K$ 步 | [2309.17400](https://arxiv.org/abs/2309.17400) | ICLR 2024 |
| **AlignProp** | reward backprop with randomized truncation | [2310.03739](https://arxiv.org/abs/2310.03739) | ICLR 2024 |
| **ImageReward / ReFL** | 137K human pair RM + 单步 reward fine-tune | [2304.05977](https://arxiv.org/abs/2304.05977) | NeurIPS 2023 |
| **HPSv2** | 798K human pair RM | [2306.09341](https://arxiv.org/abs/2306.09341) | arXiv 2023 |
| **PickScore (Pick-a-Pic)** | 1M user pair, CLIP RM | [2305.01569](https://arxiv.org/abs/2305.01569) | NeurIPS 2023 |
| **Diffusion-DPO** | ELBO surrogate + DPO loss | [2311.12908](https://arxiv.org/abs/2311.12908) | CVPR 2024 |
| **D3PO** | trajectory-level DPO for diffusion | [2311.13231](https://arxiv.org/abs/2311.13231) | CVPR 2024 |
| **SPO** | step-aware preference + step-RM | [2406.04314](https://arxiv.org/abs/2406.04314) | arXiv 2024 |
| **Diffusion-KTO** | unpaired binary feedback (KTO for diffusion) | [2404.04465](https://arxiv.org/abs/2404.04465) | NeurIPS 2024 |
| **MaPO** | margin-aware, no ref | [2406.06424](https://arxiv.org/abs/2406.06424) | arXiv 2024 |
| **Flow-GRPO** | GRPO for Flow Matching via ODE→SDE | [2505.05470](https://arxiv.org/abs/2505.05470) | arXiv 2025 |
| **SD3 (Rectified Flow + MM-DiT)** | base model | [2403.03206](https://arxiv.org/abs/2403.03206) | ICML 2024 |
| **Constitutional AI (RLAIF 起源)** | AI feedback 替代 human | [2212.08073](https://arxiv.org/abs/2212.08073) | arXiv 2022 |
| **KTO (LLM)** | prospect theory alignment | [2402.01306](https://arxiv.org/abs/2402.01306) | arXiv 2024 |

### A.2 常用 reward model 资源

- **ImageReward**：https://github.com/THUDM/ImageReward
- **HPSv2**：https://github.com/tgxs002/HPSv2
- **PickScore**：https://github.com/yuvalkirstain/PickScore
- **CLIP**：OpenAI / OpenCLIP，多 backbone 可选

### A.3 开源训练代码

- **TRL (HuggingFace)**：`diffusers` + `DPO Trainer` for Diffusion-DPO（最成熟）
- **DDPO 原始仓库**：https://github.com/kvablack/ddpo-pytorch
- **AlignProp**：https://github.com/mihirp1998/AlignProp
- **DRaFT (Google research)**：https://github.com/clarkjkr/draft（Clark et al. 2024 ICLR）
- **MaPO**：https://github.com/mapo-t2i/mapo
- **Flow-GRPO**：通过论文 arXiv 2505.05470 找官方实现

### A.4 工程踩坑清单

| 坑 | 解 |
| --- | --- |
| Diffusion-DPO 没 paired noise | 建议 $\epsilon$ for $x_t^w$ 和 $x_t^l$ 共享（降方差，非数学强制） |
| DDPO 用 DDIM-eta=0 | REINFORCE 类 score-function 梯度失效，需 eta>0 或 DDPM（DRaFT/AlignProp 等 pathwise 方法反而需要确定性采样） |
| AlignProp 显存爆炸 | $K=1$ + gradient checkpoint + LoRA |
| Reward scale 不归一化 | 每个 RM 单独 z-score |
| RL 后 FID 暴跌 | 加 KL anchor 或 reward ensemble |
| $\beta$ 调不动 | Diffusion-DPO 用 $\beta \in [2000, 5000]$，不是 LLM 的 0.1 |
| Flow-GRPO 训练慢 | 用 denoising reduction ($T_\text{train} < T_\text{infer}$) |
| MaPO 训崩 | $\alpha\hat\ell_w$ 项必须够大防 likelihood 一起降 |
| Step-RM 训不起来 | 用 rollout-based 自动标注（类 Math-Shepherd） |
| reward hacking 检测不到 | 同时监控 reward + FID + human blind A/B |

### A.5 与 §0 TL;DR 的呼应

| TL;DR 条 | 详见章节 |
| --- | --- |
| 1. 为什么难 | §1 |
| 2. 三条主线 | §1.2 |
| 3. DDPO | §2.1–2.4 |
| 4. DRaFT / AlignProp | §3.2–3.3 |
| 5. Diffusion-DPO | §4.1 |
| 6. D3PO | §4.2 |
| 7. SPO | §4.3 |
| 8. Flow-GRPO | §5 |
| 9. Reward hacking | §3.6 + §7 |

> ✅ **学完 checkpoint** —

- 能口述 Diffusion-DPO loss 形式 + paired noise 细节
- 能解释 AlignProp 为什么 $K=1$ 够用 + 显存 $\mathcal{O}(K)$
- 能写 DDPO 的 state/action/reward + per-step ratio
- 能说出 Flow-GRPO 在版图里的位置（Line A、group-based advantage）；推导本身见独立教程
- 知道 SD3/FLUX 是否用 RL 的诚实答案（公开未明说）
