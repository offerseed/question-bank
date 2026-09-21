# 扩散蒸馏 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[diffusion-distillation-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/diffusion_distillation_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Diffusion / Flow Distillation** — 把 50–1000 NFE 的 teacher 压到 1–4 NFE 的 student。一页拿下面试核心（详见后文 §1–§9 推导）。

1. **为什么**：diffusion 采样默认 50–1000 NFE，**网络前向占总延迟 >95%**；目标 ≤ 4 step 是 production 上线门槛（实时聊天 / 移动端 / 视频生成）。本文只讲 **few-step / one-step 蒸馏**，不涉 RL 后训练。

2. **Trade-off**：少 step 通常降质——naive uniform-skip DDIM 在 4 step 几乎不可用。蒸馏的本质是**用 teacher 的 50-step 轨迹/分布作为 supervision** 训练 student 一步直达。

3. **三大技术路线**：(a) **trajectory matching**（progressive distillation / CM / iCT / sCM / CTM / LCM / TCD）—— 让 student 复现 teacher ODE 解；(b) **distribution matching**（DMD / DMD2 / rCM）—— score gap 当 KL 梯度，匹配两个分布；(c) **adversarial**（ADD / LADD / SDXL-Lightning / FLUX-schnell）—— GAN loss + teacher 蒸馏。

4. **Consistency Models** (Song 2023 ICML)：学 $f_\theta(x_t, t) \to x_0$ 的 consistency function，**任一 $x_t$ 都映到同一 $x_0$**；boundary $f_\theta(x_{\sigma_\min}, \sigma_\min) = x_{\sigma_\min}$ 用 EDM-style precond 强制；CD（distillation，有 teacher）/ CT（training，无 teacher）。

5. **iCT** (Song-Dhariwal 2023)：**去 EMA target** + **pseudo-Huber loss**（替代 LPIPS）+ lognormal noise schedule + step-count curriculum，让 CT 接近 CD 质量。

6. **sCM / TrigFlow** (Lu-Song 2024 OpenAI)：连续时间 CM，**$x_t = \cos(t) x_0 + \sin(t) z$**（三角参数化让 EDM precond + PF-ODE + CM 同形式），1.5B ImageNet 512 2-step FID 1.88，**与最强 diffusion 差 <10%**。

7. **DMD** (Yin 2024 CVPR)：student 输出做"假分布"，**fake score** $s_\text{fake}$ 与 **real score** $s_\text{real}$ 之差当作 reverse-KL 梯度去推 student：$\nabla_\theta \text{KL}(p_\text{fake} \| p_\text{real}) = \mathbb{E}[(s_\text{fake} - s_\text{real}) \cdot \partial G_\theta / \partial \theta]$。**DMD2** (Yin 2024 NeurIPS) 去掉 regression loss、加 GAN、支持 multi-step student。

8. **ADD / LADD** (Sauer et al. 2023/2024 Stability)：teacher score distillation + **DINOv2 / VAE feature discriminator** 双重监督。**SDXL-Turbo** 1-step 1024、**SD3-Turbo** 4-step；**FLUX.1-schnell** 同样 LADD 系。

9. **LCM-LoRA** (Luo 2023)：把 Latent CM 训练成 **LoRA adapter**，~30 A100·h 就能让任意 SD 1.5 / SDXL fine-tune 用 4 step 出图，**不换 base model**。production 生态的关键启用器。

---

## §10 25 高频面试题（L1 必会 · L2 进阶 · L3 顶级 lab）

### L1 必会题（任何 ML / diffusion 岗位都可能问）

<details>
<summary>Q1. 为什么 diffusion 需要蒸馏？直接降 step 行不行？</summary>

- Diffusion sampling 50–1000 NFE，**网络前向占总延迟 >95%**，production 要 < 1 s 实时

- 直接减 step（如 50→4）会让 ODE 离散化误差爆炸：1-step Euler 误差 $O(\Delta t)$，4-step 时 $\Delta t$ 大 12.5×，图像高频细节崩塌

- 蒸馏的本质：**重新训一个 student**，让它学会"任意 $x_t$ 直接跳 $x_0$"（CM）或"输出分布匹配 teacher"（DMD）或"输出骗过 D"（ADD）

只说"diffusion 慢"不说网络前向是瓶颈；以为 DPM-Solver 就够了（10-NFE 是其物理极限）。

</details>

<details>
<summary>Q2. 写出 Consistency Models 的 consistency loss。</summary>

$$\mathcal{L}_\text{CD} = \mathbb{E}\big[d\big(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat x_{t_n}, t_n)\big)\big]$$

- $\theta^-$ 是 EMA target

- $\hat x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) v_\phi(x_{t_{n+1}}, t_{n+1})$，由 teacher 一步 ODE 得

- $d$ = L2 或 LPIPS

混淆 EMA target 与 stop-gradient（前者可学，后者纯停梯度）；忘 boundary 用 EDM precond。

</details>

<details>
<summary>Q3. CD vs CT 区别？</summary>

- **CD (Consistency Distillation)**：有 teacher diffusion，用它一步 ODE 算 $\hat x_{t_n}$

- **CT (Consistency Training)**：无 teacher，用 $\hat x_{t_n} = x_0 + t_n \epsilon$（同 epsilon 加不同 noise level）

- 原始 CT 质量远低于 CD（CIFAR FID 8.7 vs 3.55），iCT 通过 pseudo-Huber + lognormal sigma + curriculum 把 CT 提到 2.83，**反超 CD**

只说"CT 不用 teacher" 不说 iCT 的改进；以为 CD 一定比 CT 好（已被 iCT 反例）。

</details>

<details>
<summary>Q4. DMD 的核心思想是什么？</summary>

- 把 student $G_\theta(z) \to x$ 当作直接的 generator

- 优化 reverse-KL：$\text{KL}(p_\text{fake}^\theta \| p_\text{real})$，梯度 = score gap × ∂G/∂θ

  $$\nabla_\theta \text{KL} = -\mathbb{E}[(s_\text{real} - s_\text{fake}) \cdot \partial G_\theta]$$

- $s_\text{real}$ = teacher diffusion（frozen），$s_\text{fake}$ = 在 $G_\theta$ 输出上训的 fake diffusion

只说"DMD 是 distribution matching" 不会写梯度；不知道 $s_\text{fake}$ 也是个 diffusion model。

</details>

<details>
<summary>Q5. DMD 和 GAN 的本质区别？</summary>

- GAN 用 discriminator 给 **binary 信号**（real/fake），sample efficiency 低

- DMD 用 **score gap = ∇log(p_real/p_fake)** 给 **dense vector field 信号**，告诉 student 每个点该往哪移动

- 物理直觉：score gap 就是把 student 从 $p_\text{fake}$ 推向 $p_\text{real}$ 的"力"

- DMD2 实际把 GAN loss 也加上当 fidelity 辅助

不知道 score gap 的物理含义；以为 DMD 只是 GAN 的 variant。

</details>

<details>
<summary>Q6. ADD（SDXL-Turbo）为什么用 DINOv2 当 discriminator？</summary>

- 普通 GAN 训 D from-scratch，对 1-step generator **不稳定**（mode collapse / 训不动）

- DINOv2 提供"pretrained 高级 perceptual feature"，anchor 判别问题到强语义空间

- 多个 layer head + hinge loss 让训练稳定

- 同时省去 D 的训练成本（D 主体冻结，只训 1×1 conv heads）

只说"DINOv2 好用"不说为什么不能 from scratch；不知道 head 是多层的。

</details>

<details>
<summary>Q7. LCM 和 CM 的核心区别？</summary>

- **空间**：LCM 在 VAE latent 空间（节省 64×），CM 在 pixel space

- **CFG**：LCM 把 guidance scale $w$ 作为额外 condition $f_\theta(x_t, t, c, w)$ 喂进网络，**推理时无需双 forward**；CM 原文不处理 CFG

- **Skipping-step distillation**：LCM 用 $k$-step skip 的 teacher 加速收敛

混淆 LCM 和 LCM-LoRA（后者是把 LCM 写成 LoRA adapter）。

</details>

<details>
<summary>Q8. Rectified Flow 的 reflow 算法？</summary>

1. 用独立 pair $(x_0, x_1) \sim p_0 \otimes p_\text{data}$ 训 $v_\theta^{(1)}$

2. 用 $v_\theta^{(1)}$ 跑 ODE 生成 coupled pair $(x_0, x_1^{(1)})$

3. 用 coupled pair 重训 $v_\theta^{(2)}$，新轨迹更"直"

- **transport cost 非增定理**：每次 reflow 总传输成本不增

- 1-2 次 reflow 后能显著拉直轨迹、降低 NFE，但要 1-step Euler 媲美 50-step teacher，实践中（如 InstaFlow）通常还需叠加额外蒸馏 / 对抗微调，并非单靠 reflow 本身

只说"reflow 让轨迹变直" 不会推 transport cost 单调性；忘 InstaFlow 是 reflow 在 SD 上的应用。

</details>

<details>
<summary>Q9. SDXL-Turbo 和 SD3-Turbo / FLUX-schnell 的方法区别？</summary>

- **SDXL-Turbo (ADD)**：DINOv2 pixel-space discriminator + pixel-space score-distillation（把 student 输出加噪后送回 teacher 去噪重构再做回归，不是与 teacher 多步输出的 MSE）

- **SD3-Turbo / FLUX-schnell (LADD)**：把 D 搬到 latent space，用 teacher MM-DiT 中间层 feature 当 D backbone，**支持高分辨率 + 高参数量 base**

- ADD 受 DINOv2 原生训练分辨率（518）限制——超过可用位置编码插值（bicubic）或 patch / downsample 处理，并非架构上绝对不能超过，但有质量折损和额外计算成本；LADD 无此限制

- FLUX-schnell 是 LADD 的 RF 版本

不知道 LADD 是 ADD 的 latent 版；以为 FLUX-schnell 是普通 CM 蒸馏。

</details>

<details>
<summary>Q10. LCM-LoRA 为什么生态价值大？</summary>

- LCM 训练的"差异权重" $\Delta\theta = \theta_\text{LCM} - \theta_\text{SD}$ 可参数化为 LoRA（$r \in [8, 64]$）

- 用户原 SD 1.5 / SDXL fine-tune（DreamShaper / 角色 LoRA）**无需重训**，挂上 LCM-LoRA 就能 4-step 出图

- 生态侧：SD 一家有上万 fine-tune 模型，LCM-LoRA 是**唯一不破坏现有生态的加速方案**

- 训练成本低（~30 A100 hours / SDXL）

只说 LCM-LoRA 是"LCM 的 LoRA 版" 不说生态意义；忘 LCM-LoRA 训练 cost 远小于 LCM。

</details>

### L2 进阶题（research-oriented · 需熟悉 diffusion 训练细节）

<details>
<summary>Q11. iCT 比 CT 提升的四个改动是什么？为什么 EMA 可以去掉？</summary>

四改动：

1. **去 EMA**：直接用 stop_grad 做 target

2. **Pseudo-Huber loss**：$\sqrt{\|a-b\|^2 + c^2} - c$ 替代 LPIPS，自适应 robust

3. **Lognormal noise schedule**：$\log\sigma \sim \mathcal{N}(P_\text{mean}, P_\text{std}^2)$ 替代 uniform

4. **Step-count curriculum**：$N$ 从 10 渐增到 1280

**为什么可以去 EMA**：原 CT 的 EMA 防止"网络输出对自身求导收敛到 trivial $f \equiv 0$"。pseudo-Huber + lognormal sigma 让 loss surface 更"凸"（small-residual region 主导），stop_grad 就足够防 collapse。

只背改动名不知道原因；以为 EMA 必须有（仍是误区）。

</details>

<details>
<summary>Q12. sCM 的 TrigFlow 参数化为什么能同时简化 EDM precond / PF-ODE / CM？</summary>

$$x_t = \cos(t) x_0 + \sin(t) z,\; t \in [0, \pi/2]$$

- **EDM precond**：$D_\theta = \cos(t) x_t - \sin(t) F_\theta$，boundary 自动满足（$t=0$ 时 $D = x_0$）

- **PF-ODE**：$dx_t/dt = -\sin(t) x_0 + \cos(t) z$，干净

- **CM**：consistency function $f_\theta = \cos(t) x_t - \sin(t)(\sigma_d F_\theta)$，形式与 EDM 同构

- 关键：$\cos^2 + \sin^2 = 1$（variance preservation），且 $d\cos/dt = -\sin$ 给出"自然"的 ODE 项

只说"用 sin cos 简单"不说为什么"恰好"四件事都简化；不知道 $\sigma^2 + \alpha^2 = 1$ 是 VP 条件。

</details>

<details>
<summary>Q13. sCM 的 NCS warmup 是什么？为什么需要？</summary>

- NCS = Noise → Consistency → Score（warmup 顺序）

- 训练初期 $r \approx 0$，sCM loss 退化为标准 score matching（学 $F_\theta \approx \epsilon$）

- 渐增 $r$，consistency 项（JVP）接管

- **没有 warmup 直接 $r = 1$**：网络还没学到 score，JVP 是噪声方向，训练 NaN

- 类似 GAN 训练里"先训 D 再 alternate"，先建立 base representation 再加难

只说 warmup 是"训练 trick"不说背后是 score 先于 consistency；不知道 JVP 不收敛会 NaN。

</details>

<details>
<summary>Q14. DMD2 比 DMD 改了哪些？为什么这些改动重要？</summary>

三改动：

1. **去掉 regression loss**：DMD v1 需要预生成 teacher pair（贵 + mode 受限）；DMD2 完全靠 score gap + GAN

2. **加 GAN loss**：判别器看真实数据 + student 输出，提供 high-freq detail 监督

3. **Multi-step student**：训练时模拟 $K$-step inference trajectory，让同一权重支持 1/2/4-step

**重要性**：

- 去 regression → 数据量解锁（不再依赖 teacher pair）
- 加 GAN → 与 DMD score gap 互补（score 给 distribution-level signal，GAN 给 sample-level fidelity）
- multi-step → production 灵活性（同一模型 1-step / 4-step 切换）

不知道为什么需要 multi-step（"1-step 就够了吗"）；忘 GAN 在 DMD2 里是 auxiliary 而非主 loss。

</details>

<details>
<summary>Q15. CFG 蒸馏的两阶段流程？</summary>

**Stage 1 - Guidance distillation** (Meng 2023)：训 $\tilde\epsilon_\theta(x, c, w)$，把 $w$ 作 condition 喂进网络——

$$\mathcal{L}_\text{guide} = \|\tilde\epsilon_\theta(x_t, c, w) - \tilde\epsilon^*(x_t, c, w)\|^2$$

其中 $\tilde\epsilon^* = (1+w) \epsilon_\theta(x, c) - w \epsilon_\theta(x, \emptyset)$ 是 teacher 跑两次 forward 得到的 CFG 输出。Student 只跑一次。

**Stage 2 - Step distillation**：在 stage 1 基础上叠 progressive distillation，把 32 step 蒸到 4/2/1 step。LCM 直接同时做 stage 1 + stage 2。

只知道有 CFG 蒸馏不会写两阶段；不知道 LCM-LoRA 的 $w$-condition 来自这。

</details>

<details>
<summary>Q16. EDM preconditioning 在 CM 里的作用？</summary>

$$f_\theta(x, \sigma) = c_\text{skip}(\sigma) x + c_\text{out}(\sigma) F_\theta(c_\text{in} x, c_\text{noise})$$

具体取值（Song 2023）：

$c_\text{skip} = \sigma_d^2 / ((\sigma - \sigma_\min)^2 + \sigma_d^2)$, $c_\text{out} = \sigma_d (\sigma - \sigma_\min) / \sqrt{\sigma_d^2 + \sigma^2}$

**作用**：

1. **Boundary 自动满足**：$\sigma = \sigma_\min$ 时 $c_\text{skip} = 1, c_\text{out} = 0$，所以 $f(x, \sigma_\min) = x$（identity）

2. **Unit-variance**：让 $F_\theta$ 输入输出方差与 $\sigma$ 无关，训练稳定

不知道 $c_\text{skip}(\sigma_\min) = 1$ 是 boundary 的关键；混淆 EDM precond 与 score-based reparam。

</details>

<details>
<summary>Q17. ADD 的 distillation loss 和 DMD 的 score gap 本质区别是什么？为什么 ADD 仍然离不开 GAN？</summary>

- ADD 的 $\mathcal{L}_\text{distill}$ **不是**与 teacher 多步输出的 pixel MSE——而是把 student 1-step 输出重新加噪到随机噪声水平，送回冻结 teacher 去噪，用 student 输出与该 teacher 重构之间的距离做回归（score-distillation 风格，见 §4.1）

- 但它本质仍是**回归（regression）**：teacher 重构是该噪声水平下"合理输出"的一个（近似）均值估计，回归到它同样偏 **mode-covering** + **blurry**，会丢高频细节

- 加 GAN loss 才能补 high-freq → ADD 仍然离不开 GAN（不像 DMD 可以纯靠 score gap）

- 这就是为什么 ADD 的 distill loss 只是 "anchor"（防止蒸馏严重跑偏 / mode 塌缩），主战场仍是 GAN

- 对比 DMD：用 reverse-KL score gap → dense per-pixel gradient，不需要 GAN 也能出图（DMD2 额外加 GAN 是为了进一步提升质量，并非必需）

误以为 ADD 的 distill loss 是与 teacher 多步输出的直接 pixel MSE；以为 ADD 不需要 GAN 也能 work。

</details>

<details>
<summary>Q18. CTM 比 CM 多了什么能力？</summary>

- CM：只学 $f(x_t, t) \to x_0$（轨迹终点）

- CTM：学 $G(x_t, t, s)$，**任意 $s < t$ 都可跳**

- 实际收益：

  - inference step 数 runtime 可选（CM 固定）
  - 中间状态可控（适合做 image-to-image / inpainting）
  - 训练 + score matching auxiliary loss 防 trivial

- FID：CIFAR 1-step 1.73 / ImageNet 64 1.92（SOTA）

只说"CTM 是 CM 的 trajectory 版" 不说为什么"任意 s"有用；混淆 CTM 和 TCD（后者是 LCM 改进）。

</details>

<details>
<summary>Q19. Reflow 的 transport cost 单调性怎么证？</summary>

**setup**：考虑独立 pair $(x_0, x_1) \sim p_0 \otimes p_1$，初始 cost $C^{(0)} = \mathbb{E}\|x_1 - x_0\|^2$。

**reflow**：用 $v_\theta^{(1)}$ 跑 ODE 得 coupled $(x_0, x_1^{(1)})$，cost $C^{(1)} = \mathbb{E}\|x_1^{(1)} - x_0\|^2$。

**关键观察**：

- $x_1^{(1)} = x_0 + \int_0^1 v_\theta^{(1)}(x_t, t)\, dt$

- 在 $L^2$ 下 $\|x_1^{(1)} - x_0\| = \|\int v\, dt\| \le \int \|v\|\, dt$（Cauchy-Schwarz）

- 而 $v_\theta^{(1)}$ 训练目标是 $\mathbb{E}\|v - (x_1 - x_0)\|^2$ 最小化 → 期望意义下 $\|v\| \approx \|x_1 - x_0\|$

- 严格定理（Liu 2022 Theorem 3.6）：$C^{(k+1)} \le C^{(k)}$（OT 视角下 reflow 不增 transport cost）

直觉：reflow 单调降低 transport cost（不增，实践中往往严格减），轨迹因此越来越直；但这只保证"不增"，并不等价于证明收敛到全局最优传输（OT）解。

只说"轨迹变直"不会写 transport cost；不知道 Cauchy-Schwarz 直觉。

</details>

<details>
<summary>Q20. 蒸馏后的 student 怎么 evaluate？只看 FID 够吗？</summary>

**为什么 FID 不够**：

- FID 只算 Inception feature 的 mean + cov，对 **mode collapse 不敏感**（生成 50% mode 的 student FID 可能仍低）

- 对 high-freq detail 不敏感（Inception backbone 在 224×224 上 pool 严重）

**需要的辅助指标**：

- **Precision / Recall**（Kynkäänniemi 2019）：分别衡量"假图质量"和"覆盖多样性"

- **CLIP Score**：text-image alignment

- **HPSv2 / ImageReward / PickScore**：人类偏好

- **Step-wise FID**：1/2/4/8-step 都看，避免只优化 1-step

- **Mode count / coverage**：直接数生成图覆盖几个真实 cluster

只说"FID 就够"；忘人类偏好评估在 production 上线必看。

</details>

### L3 顶级 lab 题（research 深度 · 需会推导）

<details>
<summary>Q21. 从 PF-ODE 推 Consistency loss 的连续时间形式。</summary>

**PF-ODE**：$dx_t/dt = v_\phi(x_t, t)$（teacher）。

**Consistency 定义**：$f_\theta(x_{t+\Delta t}, t+\Delta t) = f_\theta(x_t, t)$ 沿同一 ODE 轨迹。

**一阶 Taylor**：

$$f_\theta(x_{t+\Delta t}, t+\Delta t) = f_\theta(x_t, t) + \Delta t \cdot \frac{d f_\theta}{dt} + O(\Delta t^2)$$

其中 $\frac{d f_\theta}{dt} = \partial_t f_\theta + (\nabla_x f_\theta)^\top \cdot \dot x_t = \partial_t f_\theta + (\nabla_x f_\theta)^\top v_\phi$（chain rule + PF-ODE 代入）。

**连续时间 consistency loss**：

$$\mathcal{L}_\text{cont}(\theta) = \mathbb{E}\!\left[\Big\|\partial_t f_\theta(x_t, t) + \nabla_x f_\theta(x_t, t) \cdot v_\phi(x_t, t)\Big\|^2\right]$$

**离散化**（CM 原版）：用 $\hat x_{t_n} = x_{t_{n+1}} + (t_n - t_{n+1}) v_\phi(\cdot)$ 当 teacher Euler，$f_{\theta^-}$ 当 target——

$$\mathcal{L}_\text{CD} \approx \mathbb{E}\|f_\theta(x_{t_{n+1}}, t_{n+1}) - f_{\theta^-}(\hat x_{t_n}, t_n)\|^2$$

只会写离散 loss 不会推连续形式；混淆 $\partial_t$ 和 $d/dt$（前者偏导后者全导）。

</details>

<details>
<summary>Q22. DMD 两个 score 的物理意义？为什么必须用 fake score 而不是 zero？</summary>

**物理意义**：

- $s_\text{real}(x, t) = \nabla_x \log p_\text{real}(x_t)$：把 $x_t$ 推向 real data 的"力"

- $s_\text{fake}(x, t) = \nabla_x \log p_\text{fake}(x_t)$：student 当前输出分布的 score

- 差 $s_\text{real} - s_\text{fake} = \nabla_x \log(p_\text{real}/p_\text{fake})$：reverse-KL 的梯度方向

**为什么 fake score 必要**：

- 如果只用 $s_\text{real}$（即 $s_\text{fake} \equiv 0$）：等价于把 student 推向 "$p_\text{real}$ 的 mode"——**mode collapse**

- $s_\text{fake}$ 提供"已经覆盖的位置不需要再推"的信号，类似 GAN 的 D 提供 contrastive feedback

- 数学：$\mathbb{E}_{p_\text{fake}}[s_\text{real} - s_\text{fake}]$ 是 Stein discrepancy，正确的 distribution matching 信号

**实现**：

- $s_\text{real}$ = teacher diffusion（frozen）
- $s_\text{fake}$ = 一个小 diffusion model，**在 $G_\theta$ 当前输出上做 DSM**，与 $G_\theta$ 联训

只说"DMD 用 score" 不说两个的角色区别；不知道 $s_\text{fake}$ 需要联训。

</details>

<details>
<summary>Q23. ADD vs LADD 的 scale 差异本质在哪？为什么 ADD 上不到 SD3 8B / FLUX 12B？</summary>

**ADD bottleneck**：

1. **DINOv2 input 分辨率**：ADD 用 DINOv2 base（518²）当 D backbone；超过此分辨率可用位置编码插值（bicubic）或 patch / downsample 处理，并非架构上绝对不能超过 518，但 1024² 输入会有质量折损与额外计算成本

2. **Pixel-space distill**：score-distillation 需要把 student 输出解码到 pixel space、加噪后再送回 teacher 去噪重构做回归（$d(G(z), \hat x_\phi(\text{noise}(G(z)), t))$，而不是与 teacher 独立多步输出 `teacher_ode(z)` 的 MSE），**back-prop 要穿过 VAE decoder，贵且不稳**

3. **Discriminator capacity**（工程直觉，非 ADD/LADD 原论文逐字结论）：DINOv2 ViT-L ~0.3-1B 参数远小于 SD3 8B / FLUX 12B 的 base，D 表达力可能不足以匹配大模型的生成分布

**LADD 解法**：

1. **Latent space**：D 直接在 VAE latent 上跑（128×128×16 for SD3），分辨率无关

2. **Teacher 自身的 MM-DiT block 当 D backbone**：把 SD3 自己的 transformer block 抽出来 fine-tune 成 D，**capacity 自动匹配 base 规模**

3. **Score distill in latent**：避开 VAE back-prop

**结果**：SDXL-Turbo（2.6B SDXL ADD）做到 1024² 已是 ADD 上限；SD3-Turbo（8B LADD）/ FLUX-schnell（12B LADD-style）需要 LADD 才能稳定训出。

只说"LADD 在 latent space" 不说为什么 ADD 上不到大模型；忘 DINOv2 分辨率限制是 hard cap。

</details>

<details>
<summary>Q24. Flow-OPD（2026 arXiv:2605.08063）与 DMD 在数学上有什么联系？</summary>

> 📍 **澄清**：Flow-OPD 主要是 multi-reward RL alignment paper，与本文 few-step inference distillation 主线略偏；这里出现是因为 name 包含 "Distillation"，详细讨论见 [diffusion_post_training_tutorial.md](diffusion_post_training_tutorial.md)。

**DMD**：reverse-KL 梯度（在 student 输出分布上），single teacher，single objective = match teacher distribution。

**Flow-OPD**：on-policy distillation with multiple **reward-specific** teachers（每个 reward GRPO fine-tuned 一个 specialist），是 **alignment paper**（多 reward 对齐）而非 inference distillation 论文。

**说"DMD 退化到 OPD"是错的**：DMD 的 reverse-KL 与 OPD 的 multi-teacher vector-field weighting 是**不同的数学目标**——一个是分布匹配，一个是 reward-aware policy supervision。两者目标侧重不同（**single-teacher distribution match vs multi-reward alignment**），没有 reduction 关系。面试中**不要**说"DMD 是 OPD 的特例"或反之，无可靠数学依据。

**实践意义**（safer 表述）：
- **DMD 更适合 few-step inference**（单一目标：match teacher 分布）
- **Flow-OPD 更适合 multi-reward alignment**（多 reward 对齐 + on-policy 训练）
- 两者解决不同问题，并非替代关系；详细 alignment 内容见 [`diffusion_post_training_tutorial.md`](diffusion_post_training_tutorial.md)

只把 Flow-OPD 当"另一种 inference distillation"是混淆 — 它的 multi-reward / RL 性质是核心；同样不要把"reduction to DMD"作为既定数学结论。

</details>

<details>
<summary>Q25. 设计一个能在 4 step 跑 1024² 视频 + 保 temporal coherence 的蒸馏方案。给出 loss 和 D 设计。</summary>

**Setup**：

- Teacher：50-step video diffusion (e.g. Wan 2.1 14B, Rectified Flow)
- Student：4-step video generator $G_\theta(z_{1:T}, c)$
- Target：1024² × 5 sec

**Loss 组合**（rCM-style + LADD-style）：

$$\mathcal{L}_\text{total} = \underbrace{\mathcal{L}_\text{sCM}^\text{trig}}_{\text{video CM, JVP-based}} + \lambda_1 \cdot \underbrace{\mathcal{L}_\text{score-reg}}_{\text{mode-seeking via score gap}} + \lambda_2 \cdot \underbrace{\mathcal{L}_\text{adv}^\text{video}}_{\text{temporal D}}$$

**Video Discriminator 设计**：

- **Backbone**：teacher 自己的 3D MM-DiT block（latent space，避开 VAE decode）

- **两个 head**：
  - **Spatial head**：单帧 latent → real/fake 信号（图像 quality）
  - **Temporal head**：连续 $k$-frame latent stack → real/fake（motion realism）

- **Optical flow consistency loss**（辅助）：
  $$\mathcal{L}_\text{flow} = \mathbb{E}\|f_\text{flow}(\hat x_{t}, \hat x_{t+1}) - f_\text{flow}(x_t^\text{real}, x_{t+1}^\text{real})\|$$

**训练 tricks**：

- **Multi-stage**：先在静态图（$T = 1$）上预训 → 再加 temporal D → 最后 fine-tune full video
- **Curriculum on T**：短片段先训（$T = 8$ frame）→ 长片段（$T = 80$ frame）
- **EMA on G**：避免 student 输出在不同 step 间 drift

**Evaluation**：

- VBench (静态质量 + 动态质量 16 维)
- FVD (Fréchet Video Distance)
- 人类对照（rCM-style）

**对比 baseline**：rCM 已在 Wan 2.1 14B 上做到接近——这是 production-grade direction，2026 还在快速发展。

需要把"图像蒸馏 + temporal 监督 + 大 base"三件事融合；只用单 D 看单帧会 motion 崩；只用 score-gap 没 GAN 会 detail 模糊。

</details>

## §A 附录：参考文献

**Consistency Models 家族**：

- Song et al. 2023, "Consistency Models", ICML 2023, [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Song & Dhariwal 2023, "Improved Techniques for Training Consistency Models" (iCT), [arXiv:2310.14189](https://arxiv.org/abs/2310.14189)
- Lu & Song 2024, "Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models" (sCM / TrigFlow), ICLR 2025, [arXiv:2410.11081](https://arxiv.org/abs/2410.11081)
- Kim et al. 2023, "Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion" (CTM), ICLR 2024, [arXiv:2310.02279](https://arxiv.org/abs/2310.02279)
- Luo et al. 2023, "Latent Consistency Models" (LCM), [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
- Luo et al. 2023, "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module", [arXiv:2311.05556](https://arxiv.org/abs/2311.05556)
- Zheng et al. 2024, "Trajectory Consistency Distillation" (TCD), [arXiv:2402.19159](https://arxiv.org/abs/2402.19159)
- "Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency" (rCM), [arXiv:2510.08431](https://arxiv.org/abs/2510.08431) (rCM acronym verified)

**Distribution Matching Distillation**：

- Yin et al. 2024, "One-step Diffusion with Distribution Matching Distillation" (DMD), CVPR 2024, [arXiv:2311.18828](https://arxiv.org/abs/2311.18828)
- Yin et al. 2024, "Improved Distribution Matching Distillation for Fast Image Synthesis" (DMD2), NeurIPS 2024, [arXiv:2405.14867](https://arxiv.org/abs/2405.14867)

**Adversarial Distillation**：

- Sauer et al. 2023, "Adversarial Diffusion Distillation" (ADD / SDXL-Turbo), [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)
- Sauer et al. 2024, "Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation" (LADD / SD3-Turbo), [arXiv:2403.12015](https://arxiv.org/abs/2403.12015)
- Lin et al. 2024, "SDXL-Lightning: Progressive Adversarial Diffusion Distillation", [arXiv:2402.13929](https://arxiv.org/abs/2402.13929)

**Flow / Rectified Flow**：

- Liu, Gong & Liu 2022, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023, [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Liu et al. 2023, "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation", ICLR 2024, [arXiv:2309.06380](https://arxiv.org/abs/2309.06380)
- "Flow-OPD: On-Policy Distillation for Flow Matching Models", [arXiv:2605.08063](https://arxiv.org/abs/2605.08063) (Flow-OPD 主要是 multi-reward RL alignment paper，与本文 few-step inference distillation 主线略偏；详见 `diffusion_post_training_tutorial.md`)

**CFG / Step Distillation**：

- Meng et al. 2023, "On Distillation of Guided Diffusion Models", CVPR 2023, [arXiv:2210.03142](https://arxiv.org/abs/2210.03142)
- Salimans & Ho 2022, "Progressive Distillation for Fast Sampling of Diffusion Models", ICLR 2022, [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)

**Foundations**：

- Ho, Jain & Abbeel 2020, "Denoising Diffusion Probabilistic Models", NeurIPS 2020 (DDPM)
- Song et al. 2021, "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021
- Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM), NeurIPS 2022, [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
- Lipman et al. 2023, "Flow Matching for Generative Modeling", ICLR 2023

**Production models**：

- Stable Diffusion XL: Podell et al. 2024 ICLR
- Stable Diffusion 3: Esser et al. 2024 ICML
- FLUX.1: Black Forest Labs 2024 (technical report)

**Diffusion / Flow Distillation Cheat Sheet** · 主要参考：Song 2023 (CM), Lu-Song 2024 (sCM), Yin 2024 (DMD/DMD2), Sauer 2023/2024 (ADD/LADD), Liu 2022 (RF)
