# 现代扩散后训练 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[modern-diffusion-post-training-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/modern_diffusion_post_training_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **一句话** — 本文的三种方法共享 GRPO 式的 **group sampling** 框架（同一 prompt 采 $G$ 张，组内归一化 reward），分岔只在一处：**reward 怎么进梯度**。

1. **Flow-GRPO**（Liu et al. 2025，arXiv 2505.05470）：把 ODE 改写成保边缘的 **SDE**，得到可算的 Gaussian 单步转移密度，然后做 **advantage 加权的 PPO-clip 策略梯度**。代价：训练 rollout 必须 SDE（推理仍可 ODE）、要存整条轨迹、每步算 likelihood、有离散化偏差。
2. **DGPO**（Luo, Hu, Tang 2025，arXiv 2510.08425，ICLR 2026）：**彻底不要策略梯度**，把 DPO 从"一对样本"推广到"一对子组"——正组 vs 负组进一个 sigmoid；likelihood 用 Diffusion-DPO 那套 **DSM 差**代替。于是可以用**确定性 ODE** 采样，只存干净图 + reward。报告比 Flow-GRPO 快约 20×。
3. **DiffusionNFT**（Zheng, Chen et al. 2025，arXiv 2509.16117，ICLR 2026 Oral）：在**前向加噪过程**上做 RL——一个 flow-matching 回归 loss，正例拉向 target、负例通过关于 old 的**反射参数化**推开。不需要 likelihood、不依赖 solver、推理 **CFG-free**。报告在 GenEval 上比 Flow-GRPO 快最多 25×。
4. 三者都**不要 critic**；但它们分别是 policy gradient / group preference logistic / reward-weighted 双支回归，不能笼统叫"三种 policy gradient"。
5. **old ≠ ref**：old 是采样时的行为策略（importance ratio 的分母、NFT 的反射中心），ref 是 KL 约束基准（Flow-GRPO 的 KL、DGPO 的 DSM 差）。NFT 的核心 loss 不需要固定 ref。
6. 三个数字的口径要分清：Flow-GRPO 自报 GenEval 0.63→0.95（CFG base）、PickScore 21.72→23.31（带 KL；无 KL 是 23.41）；DGPO 0.97；NFT 0.24→0.98（**0.24 是 CFG-free base**）。DGPO 的 3.74 vs 3.66 是 **UnifiedReward**，不是 PickScore。
7. 面试杀手题：**reverse SDE 的符号**（生成时间递减，drift 是 $v-\tfrac12 g^2 s$）、**DGPO 为什么能消掉 $\log Z$**（组内 advantage 和为零 ⇒ 正负权重总量相等）、**NFT 的负支为什么是 $(1+\beta)$**（关于 old 的反射）。

---

---

## §10 25 高频面试题

### L1 必会题（10 题）

<details>

<summary>Q1. Flow-GRPO / DGPO / DiffusionNFT 真正的区别是什么？</summary>

三者共享 group sampling + 组内归一化的采集框架。区别在 reward 怎么进梯度：Flow-GRPO 加权每步转移的策略梯度；DGPO 把正负组的 reference-relative DSM 差放进一个 sigmoid（偏好学习）；NFT 用 reward 加权正负两支 flow-matching 回归。三者都没有 critic，但不是"三种 policy gradient"。

只答"都是 GRPO 变体"不得分。

</details>

<details>

<summary>Q2. ODE 采样已经有随机 seed，为什么 Flow-GRPO 还要改成 SDE？</summary>

seed 提供的是样本级随机性；PPO 需要的是**单步转移密度** $p_\theta(x_{k+1}\mid x_k)$ 来算 importance ratio。ODE 的单步是 Dirac，没有密度。SDE 化后 Euler 离散的单步变成 Gaussian，并顺便增加中间步骤的探索。

说"ODE 不能探索"是错的。

</details>

<details>

<summary>Q3. old 和 ref 能混用吗？</summary>

不能。old 是采样时的行为策略——在 Flow-GRPO 里是本轮 rollout 的策略快照（ratio 的分母），在 NFT 里是反射中心、用 EMA 更新；ref 是固定的约束基准——Flow-GRPO 的 KL、DGPO 的 DSM 差。NFT 的核心 loss 用的是更新中的 old，不需要固定初始 ref。

</details>

<details>

<summary>Q4. NFT 里的 $r$ 是成功率吗？</summary>

不是。$r_i = \tfrac12 + \tfrac12\operatorname{clip}((R_i - \bar R_c)/Z_c, -1, 1)$ 是**每个样本**映射到 $[0,1]$ 的 reward，解释为 optimality probability。全局的量是 $\bar r_c = \mathbb E_{\pi_\text{old}}[r]$，噪声态上的是后验 $\alpha(x_t)$。

</details>

<details>

<summary>Q5. 写出 RF 的 $\hat x_0$、$\hat\epsilon$、score 换算。</summary>

$x_t = (1-t)x_0 + t\epsilon$，$v = \epsilon - x_0$；用网络预测 $v_\theta$：$\hat x_0 = x_t - tv_\theta$，$\hat\epsilon = x_t + (1-t)v_\theta$，$s_\theta := -\hat\epsilon/t$。score 用预测的条件均值，只在 $v_\theta = v^*$ 时等于真实 score。

</details>

<details>

<summary>Q6. DGPO 的正负组怎么分？权重是什么？</summary>

同一 prompt 的组内 $A_i = (R_i - \bar R)/s_R$；$A_i > 0$ 进 $\mathcal G^+$，$A_i \le 0$ 进 $\mathcal G^-$；$w_i = \lvert A_i\rvert$。因为 $\sum A_i = 0$，两组权重总量相等。

</details>

<details>

<summary>Q7. Flow-GRPO 的 denoising reduction 是什么？</summary>

训练时 SDE 用 10 步采样，评估用 40 步。论文报告"超过 4×"加速。理由是经验性的：连续边缘与步数无关，RL 学的方向修正对步数不敏感——不是严格等价。

</details>

<details>

<summary>Q8. DiffusionNFT 训练需要存什么？</summary>

只存干净图 $x_0$、它的 prompt $c$ 和 reward。训练时重新采 $t,\epsilon$ 构造 $x_t$。不存轨迹、不算 likelihood、不反传穿过 solver。

</details>

<details>

<summary>Q9. 为什么 DiffusionNFT 推理可以不用 CFG？</summary>

CFG 是 cond（正）和 uncond（负）两个模型之间的外推——离线的 reinforcement guidance。NFT 用在线的正负样本学出一个能替代 CFG 作用的 reward guidance（方向不保证和 CFG 相同），模型本身就带 guidance。从只有条件分支的模型出发，全程 CFG-free。

</details>

<details>

<summary>Q10. GenEval 0.63、0.24、0.95、0.97、0.98 各是什么口径？</summary>

0.63：SD3.5-M **带 CFG** 的 base；0.24：**CFG-free** base；0.95：Flow-GRPO；0.97：DGPO；0.98：NFT 单 reward 约 1k 步。NFT 多 reward 模型是 0.94。

把两者当作相同 CFG 设置下的 baseline 来比不得分（标明 CFG 口径后当然可以并列展示，论文自己就是这么做的）。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11. 推导保边缘 SDE，并说明生成方向的符号。</summary>

正向时钟 $u = 1-t$：$dX_u = [-v + \tfrac12 g^2 s]du + g\,dW_u$。Fokker–Planck 中 $-\tfrac12 g^2\nabla\cdot(ps)$ 与 $+\tfrac12 g^2\Delta p$ 抵消（因 $ps = \nabla p$），剩 ODE 的密度演化。换回递减的 $t$：drift 是 $v - \tfrac12 g^2 s$。离散噪声尺度用 $\sqrt{h}$，$h = t_k - t_{k+1} > 0$。

写成 $+\tfrac12 g^2 s$ 是把时间方向弄反了。

</details>

<details>

<summary>Q12. 为什么 Flow-GRPO 的 KL 有闭式？写出来。</summary>

policy 与 ref 的单步转移是**同协方差** $g^2 h I$ 的 Gaussian，KL 只剩均值差：$\lVert\mu_\theta - \mu_\text{ref}\rVert^2/(2g^2h)$。代入 drift 得 $\frac h2\big(\frac1g + \frac{g(1-t)}{2t}\big)^2\lVert v_\theta - v_\text{ref}\rVert^2$。是条件单步 KL，不是终态分布的 KL。

</details>

<details>

<summary>Q13. DGPO 为什么能消掉 $\log Z(c)$？</summary>

隐式 reward $r_\theta = \beta\log(p_\theta/p_\text{ref}) + \beta\log Z(c)$。组级 reward 是加权和，正负组相减后 $\log Z$ 的系数是 $\sum_{\mathcal G^+}w - \sum_{\mathcal G^-}w$；因为 $w = \lvert A\rvert$ 且 $\sum A = 0$，这个差恰为零。前提是平衡权重总量，而不是按人数平均。

</details>

<details>

<summary>Q14. DGPO 和 Diffusion-DPO 是什么关系？</summary>

同一套 reference-relative DSM surrogate。$G=2$ 时 DGPO 退化为 Diffusion-DPO（尺度吸进 $\beta$）。一般 DGPO 把单 pair 换成带 $\lvert A\rvert$ 权重的两个子组，组差进一个 sigmoid，不是逐对枚举。

</details>

<details>

<summary>Q15. DGPO 的 timestep clipping 解决什么？和 PPO clip 有关吗？</summary>

无关。少步采样的图有伪影，低噪声 $t$ 的回归会把伪影学走，所以只在 $[t_\min, 1]$ 采 $t$。这是训练 timestep 的截断；PPO clip 是 importance ratio 的截断。

</details>

<details>

<summary>Q16. 写出 Flow-GRPO 单步的 Gaussian log-density，说明"对维度求和"和"按维度平均"的区别。</summary>

$\log p = -\lVert x_{k+1} - \mu_\theta\rVert^2/(2g^2h) - \tfrac d2\log(2\pi g^2h)$，对 $d$ 维求和才是密度。官方代码按维度平均 log-prob，得到的是 $\rho^{1/d}$；论文公式和实现约定要分开说。

</details>

<details>

<summary>Q17. DiffusionNFT 的 $v_\theta^+$、$v_\theta^-$ 怎么定义？</summary>

关于 old 对称反射：$v_\theta^\pm = v_\text{old} \pm \beta(v_\theta - v_\text{old})$。展开负支即 $(1+\beta)v_\text{old} - \beta v_\theta$。$\beta$ 是 mixing 参数，$1/\beta$ 控制 guidance 强度。

</details>

<details>

<summary>Q18. 展开 NFT 的逐样本 loss，说明 $v_\theta = v_\text{old}$ 时梯度方向。</summary>

$d = v_\theta - v_\text{old}$，$e = v_\text{old} - v$：$\ell = \lVert e\rVert^2 + \beta^2\lVert d\rVert^2 + 2\beta(2r-1)e^\top d$。$d = 0$ 时梯度 $2\beta(2r-1)(v_\text{old} - v)$：正例朝 target 走，负例反向，$r = \tfrac12$ 只剩对 old 的二次约束。

</details>

<details>

<summary>Q19. Flow-GRPO 训练时哪些量固定、哪些重算？</summary>

固定：rollout 的状态序列、old log-prob、reward、advantage。重算：当前策略在这些状态上的 log-prob。ref 输出 stop-grad。存轨迹存的是状态，不是反传图。

</details>

<details>

<summary>Q20. 三种方法的 CFG 依赖分别是什么？</summary>

三种目标都不把 CFG 当数学前提。Flow-GRPO 的原 SD3 配置带 CFG，后续实现也支持 CFG-free；DGPO 论文未写，官方实现默认 rollout 带 CFG、DSM 更新用条件预测；NFT 主配方采集、训练、推理全程 CFG-free。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21. 证明 NFT 的最优解是 $v_\text{old} + \frac2\beta\Delta$，解释"2"从哪来。</summary>

逐样本 $d^* = (2r-1)(v - v_\text{old})/\beta$。取条件期望：$\mathbb E[(2r-1)(v - v_\text{old})\mid x_t] = 2\mathbb E[r(v - v_\text{old})\mid x_t]$（因 $\mathbb E[v\mid x_t] = v_\text{old}$），而 $\mathbb E[rv\mid x_t] = \alpha v^+$、$\mathbb E[r\mid x_t] = \alpha$，故为 $2\alpha(v^+ - v_\text{old}) = 2\Delta$。"2"来自正负两支各贡献一份同方向的位移。

</details>

<details>

<summary>Q22. 25× 能证明 NFT 的梯度估计本身快 25× 吗？</summary>

不能。它是特定 reward（GenEval）上、同为 10 步 rollout 的 **wall-clock 训练曲线**比较，包含收敛速度、CFG 有无、采样与实现差异。其它 reward 在 3–25× 之间。多 reward 模型的数字另算。

</details>

<details>

<summary>Q23. DGPO 里为什么"按人数平均"会破坏抵消？给个例子。</summary>

4 张图 reward $[0,1,2,7]$，正负 1:3。$w = \lvert A\rvert$ 时两边总量都是 $S$，$\log Z$ 的系数 $S - S = 0$。两种破坏方式：(a) 保留 $\lvert A\rvert$ 但把每一边按人数取平均，系数变成 $S - S/3 \ne 0$——抵消没了；(b) 权重全改成 1 直接求和，系数是 $1 - 3 = -2 \ne 0$，给所有隐式 reward 加同一个常数就会改变 logit。`code/diffusion_online_rl.py` 实验 C 验证的是 (b)。

</details>

<details>

<summary>Q24. Flow-GRPO 的"保边缘"在有限步、有学习误差、带 CFG 时还成立吗？</summary>

不严格成立。保边缘是连续时间性质，要求 $v$ 与 $s$ 对应同一族边缘。学习误差使 $s_\theta$ 与 $v_\theta$ 不再匹配；CFG 后的 velocity 与代入的 score **不保证对应同一前向加噪边缘族**，保边缘证明不能直接继承；有限步离散化引入额外偏差。所以 denoising reduction 只能是经验观察。

</details>

<details>

<summary>Q25. 如果让你选一个方法上线，怎么选？</summary>

看三件事：采样器（有 SDE 成本预算才考虑 Flow-GRPO）、存储（轨迹 $O(PGTd)$ vs 干净图 $O(PGd)$）、是否要 CFG-free（NFT 天然满足）。DGPO 和 NFT 都用 ODE、只存干净图；DGPO 保留 ref 约束，NFT 用 EMA old + 反射。三者的报告效率都是特定 reward 曲线上的比较，自己的 reward 要自己跑。

</details>

---

## §A 附录

### A.1　论文

| 方法 | arXiv | 会议 | 代码 |
| --- | --- | --- | --- |
| Flow-GRPO | [2505.05470](https://arxiv.org/abs/2505.05470) | — | github.com/yifan123/flow_grpo |
| DGPO | [2510.08425](https://arxiv.org/abs/2510.08425) | ICLR 2026 | github.com/Luo-Yihong/DGPO |
| DiffusionNFT | [2509.16117](https://arxiv.org/abs/2509.16117) | ICLR 2026 Oral | github.com/NVlabs/DiffusionNFT |
| NFT (LLM) | [2505.18116](https://arxiv.org/abs/2505.18116) | — | — |
| GRPO | [2402.03300](https://arxiv.org/abs/2402.03300) | — | — |

### A.2　Runnable toy

脚本：[`code/diffusion_online_rl.py`](code/diffusion_online_rl.py)，纯 PyTorch，CPU 几秒。

- **A** 保边缘：1-D $x_0\sim\mathcal N(1, 0.25)$，解析 $v^*$、$s^*$；从 $t=0.9$ 走到 $0.1$，比较 ODE / 正确 SDE / 只加噪不修正三组的均值方差。
- **B** NFT 梯度：固定配对 $t=0.5, x_0=0, \epsilon=1$，$\beta=0.5$；断言 $r=1,0,\tfrac12$ 的初始梯度为 $-1,+1,0$，正负例最优解 $\pm2$。
- **C** DGPO 抵消：4 样本组 $[0,1,2,7]$；断言正负权重总量相等、reward 整体平移不变、隐式 score 整体平移不变；单位权重的反例。

### A.3　工程踩坑

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| Flow-GRPO 从 $t=1$ 起步 NaN | $g(t) = a\sqrt{t/(1-t)}$ 分母为零 | 首步用相邻时间点 |
| Flow-GRPO clip 在某些 $t$ 失效 | ratio 分布随 timestep 漂移 | RatioNorm / 梯度重加权（GRPO-Guard） |
| DGPO 图变糊 | 低噪声 $t$ 学走了少步伪影 | timestep clipping $[t_\min, 1]$ |
| DGPO loss 不动 | 组内 reward 全同，标准差为零，advantage 按约定取零 | 换 prompt 或加大 $G$ |
| NFT 不稳 | $\beta$ 太小或 old EMA 追得太紧 | 调 $\beta$、放慢 EMA |
| 三者都「涨分不涨质」 | reward 和视觉质量脱钩（原始 reward 也会被 hack）；组归一化分数另有「不反映跨轮进步」的问题 | 看原始 reward + 图 + 多样性 |
