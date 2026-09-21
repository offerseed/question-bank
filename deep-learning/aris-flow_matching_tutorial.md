# Flow Matching — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[flow-matching-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/flow_matching_tutorial.md)

---

## §0 TL;DR

> 💡 **5 句话搞定 Flow Matching** — 一页拿下核心要点（详见后文 §1–§4 推导）。

1. **目标**：学一个 vector field $v_\theta(t, x)$，使 ODE $\dot{x}_t = v_\theta(t, x_t)$ 把 $x_0 \sim p_0$（噪声）演化到 $x_1 \sim p_1$（数据）。

2. **训练 (CFM)**：$\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t, z, x_t \sim p_t(\cdot|z)} \|v_\theta(t, x_t) - u_t(x_t|z)\|^2$，**simulation-free**（不用解 ODE 算 loss）。

3. **关键定理**：$\nabla_\theta \mathcal{L}_\text{FM} = \nabla_\theta \mathcal{L}_\text{CFM}$——所以学 conditional vector field 等价于学 marginal 的（Lipman et al. 2023）。

4. **最简版 (Rectified Flow)**：$x_t = (1-t)x_0 + tx_1$，target $u_t = x_1 - x_0$（$(x_0, x_1)$ 独立采样；只有配对本身来自 OT coupling 才严格称 "OT-CFM"）。SD3 / FLUX / Lumina 都用这个（独立采样版本）。

5. **采样**：从 $x_0 \sim p_0$ 出发，用 ODE solver (Euler / Heun / RK4) 积分到 $t=1$。

---

## §7 高级话题

### 7.1　Reflow（Liu et al., ICLR 2023）

Rectified Flow 之所以能少步数生成，关键是 **reflow 算法**：

1. 第一次训练得到 $v_\theta^{(1)}$（用独立配对 $(x_0, x_1) \sim p_0 \otimes p_1$）

2. 用 $v_\theta^{(1)}$ 跑 ODE 生成**coupled** pair $(x_0, x_1^{(1)})$，即 $x_1^{(1)} = \text{ODE}(x_0; v_\theta^{(1)})$

3. 用 coupled pair 重新训练得到 $v_\theta^{(2)}$，新的 trajectory 更**直**

4. 重复——Liu et al. 2022 证明在适当假设下，配对的 **convex transport cost** 非增（每次 reflow 不会让总传输成本变差）

"trajectory 变直"是直觉与经验观察；具体严格定理是 transport cost 单调性。实际 1-2 次 reflow 就能做到 4-step 媲美 50-step（如 InstaFlow）；SD3-Turbo 用的是 LADD（adversarial diffusion distillation）而非 reflow，机制不同，不应作为 reflow 的例子列出。极限：完全直线 → 1-step 生成（$x_1 = x_0 + v_\theta(0, x_0)$）。

### 7.2　Conditional Flow Matching (CFG)

对条件生成（如 text-to-image），模型接收额外条件 $c$：

$$v_\theta(t, x, c)$$

训练时以概率 $p_\text{drop}$（一般 0.1）把 $c$ 替换成空（如 null embedding），得到 **无条件 head**。

采样时用 **Classifier-Free Guidance**：

$$v_\text{CFG}(t, x, c) = v_\theta(t, x, \emptyset) + s \cdot \left[v_\theta(t, x, c) - v_\theta(t, x, \emptyset)\right]$$

$s$ 是 guidance scale（一般 1.5-7.5）。$s > 1$ 时放大 conditional 信号，提升文本对齐但损失多样性。

### 7.3　Logit-normal $t$（SD3 默认）

SD3 (Esser et al. 2024) 发现，**$t \sim \mathcal{U}[0, 1]$ 不是最优**。中间区域（$t \approx 0.5$）的 target 噪声-信号比最难学。改成：

$$t = \sigma(\tau), \quad \tau \sim \mathcal{N}(m, s^2)$$

即 $\tau$ 高斯采样后 sigmoid 映射回 $(0, 1)$，可调 $m, s$ 控制 $t$ 分布偏重哪段。默认 $m = 0, s = 1$ 时 $t$ 集中在 0.5 附近。这是 SD3 论文中 ablation 涨点的关键之一。

