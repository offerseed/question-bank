# 扩散模型基础 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[diffusion-foundations-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/diffusion_foundations_tutorial.md)

---

## §0 TL;DR

> 💡 **9 句话搞定 Diffusion 基础** — 一页拿下面试核心要点（详见 §1–§13 推导）。

1. **DDPM (Ho 2020)**：forward $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$ 闭式可采样；reverse $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta, \Sigma_\theta)$ 学反向 Gaussian；ELBO 化简到 $L_\text{simple} = \mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$（$\epsilon$-prediction）。

2. **三种视角等价**：DDPM 的 $\epsilon$、score-based 的 $s = \nabla \log p_t$、flow matching 的 $v$ 在 Gaussian path 下线性可逆 —— $s_\theta = -\epsilon_\theta / \sigma_t$，$v = \alpha'x_0 + \sigma'\epsilon$。

3. **Tweedie 公式**：$\mathbb{E}[x_0 | x_t] = x_t + \sigma_t^2 \nabla_{x_t} \log p_t(x_t)$ —— 一行式连接 denoiser 与 score。

4. **Score SDE (Song 2021)**：VP-SDE / VE-SDE 统一框架；**reverse-time SDE** 与 **probability flow ODE** 共享同一族边缘分布，ODE 形式直接给出 FM 的 vector field。

5. **DDIM (Song 2020 / ICLR 2021)**：non-Markovian forward 推出 deterministic sampler，**marginal 与 DDPM 相同**但采样路径可控（$\eta=0$ 确定性；$\eta=1$ + 走完整 $T$ 步退化为 DDPM ancestral，skip 步下则只是匹配 DDPM 方差，不严格等价）。

6. **EDM (Karras 2022)**：preconditioning 让网络输出方差恒为 1：$D_\theta(x;\sigma) = c_\text{skip}(\sigma) x + c_\text{out}(\sigma) F_\theta(c_\text{in}(\sigma) x, c_\text{noise}(\sigma))$；配合 $\sigma$-schedule + Heun 2nd-order，**FID SOTA 同时 NFE 降到 18-35**。

7. **CFG (Ho-Salimans 2022)**：训练时以概率 $p_\text{drop}$ drop 条件 → 同一 net 学 conditional/unconditional；推理 $\tilde\epsilon = (1+w)\epsilon_\theta(x,c) - w\epsilon_\theta(x,\emptyset)$，$w \in [3, 7]$ 是 text-to-image 主力。

8. **Production**：SD/SDXL 用 VAE latent + UNet；SD3 / FLUX.1 改用 **Rectified Flow + MM-DiT**；ControlNet 给 frozen UNet 加可训练 side branch；DiT 把 UNet 全换 Transformer。

9. **加速**：DPM-Solver++ 把 NFE 压到 10-20；Consistency Models 学 $f_\theta(x_t, t) \mapsto x_0$ 做到 1-4 步；LCM / LCM-LoRA / SDXL-Turbo (ADD) / SD3-Turbo (LADD) 让蒸馏在 Stable Diffusion 全家桶可用。

---

## §10 Conditioning：Classifier Guidance & CFG

### 10.1　Classifier Guidance (Dhariwal-Nichol 2021)

训练一个独立 classifier $p_\phi(c | x_t)$（在 noisy data 上），用 Bayes：

$$\nabla_{x_t} \log p(x_t | c) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p_\phi(c | x_t)$$

实践中给 classifier gradient 加 scale $w$（控制 guidance 强度）：

$$\tilde\epsilon = \epsilon_\theta(x_t, t) - w \sqrt{1-\bar\alpha_t}\, \nabla_{x_t} \log p_\phi(c | x_t)$$

> ⚠️ **Classifier guidance 的缺点** —— (a) 必须额外训 noisy classifier，工程负担；(b) classifier gradient 易"对抗"，在远离训练分布时退化；(c) 对 text-to-image 这种连续 condition 不友好。CFG 完全替代了它。

### 10.2　Classifier-Free Guidance (Ho-Salimans 2022)

**训练**：以概率 $p_\text{drop}$（一般 0.1）把 $c$ 替换为 $\emptyset$（null embedding），同一个 net 学 conditional 和 unconditional：

$$L_\text{CFG}(\theta) = \mathbb{E}\big[\|\epsilon - \epsilon_\theta(x_t, t, c \text{ or } \emptyset)\|^2\big]$$

**推理**：把 $w$ 称为 **guidance scale**：

$$\boxed{\; \tilde\epsilon = \epsilon_\theta(x_t, t, \emptyset) + (1 + w)\big[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\big] \;}$$

等价形式（Imagen / SD 实现常用）：

$$\tilde\epsilon = (1 + w)\, \epsilon_\theta(x_t, t, c) - w\, \epsilon_\theta(x_t, t, \emptyset)$$

> ⚠️ **CFG $w$ 的两种 convention** — 论文 Ho-Salimans 2022 原文 $\tilde\epsilon = \epsilon_\text{uncond} + (1+w)(\epsilon_\text{cond} - \epsilon_\text{uncond})$，即 $w = 0$ 是 unguided、$w > 0$ 增强。但 HuggingFace / SD UI 常用 $w' = w + 1$，即 $w' = 1$ 是 unguided、$w' = 7.5$ 是常用强度。**面试代码记得标明 convention**。

### 10.3　CFG 的几何意义

CFG 等价于把采样轨迹拉向"条件梯度"方向：

$$\nabla_{x_t} \log p(x_t | c) \approx \nabla_{x_t} \log p(x_t) + w \nabla_{x_t} \log \frac{p(x_t | c)}{p(x_t)}$$

第二项是"条件性 score 差"，把样本推向 conditional likelihood 高、unconditional likelihood 相对低的区域——直觉上"放大文本对齐"。

> ✅ **CFG 是 SD/SDXL/FLUX 文图对齐的核心** —— $w \in [3, 7.5]$ 是 Stable Diffusion 的实验 sweet spot；$w > 10$ 容易 over-saturated（颜色饱和、artifact）。FLUX 把 CFG 内化进 distillation（"guidance-distilled"），单 forward 就实现 CFG 效果——这是它推理速度的关键之一。
