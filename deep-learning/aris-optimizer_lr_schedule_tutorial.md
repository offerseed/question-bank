# 优化器与学习率调度 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[optimizer-lr-schedule-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/optimizer_lr_schedule_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Optimizer / LR Schedule** — 一页拿下面试核心要点（详见后文 §1–§11 推导）。

1. **为什么不止用裸 SGD**：真实 loss 地形**病态**（Hessian 条件数 $\kappa=\lambda_{\max}/\lambda_{\min}$ 极大），裸梯度下降在陡方向来回**震荡**、在缓方向**爬不动**。两味药：**动量**（沿一致方向累积、震荡相消）+ **逐参数自适应步长**（对角预条件，拉平各坐标曲率失配）。优化器谱系就是"SGD → +动量 → +自适应 → Adam（两者都要）"。

2. **SGD / Momentum / Nesterov**：裸 SGD $\theta_t=\theta_{t-1}-\eta g_t$；heavy-ball 动量 $v_t=\mu v_{t-1}+g_t,\ \theta_t=\theta_{t-1}-\eta v_t$（抑制震荡、加速一致方向）；Nesterov 在**前瞻点** $\theta-\eta\mu v$ 求梯度（look-ahead，纠过冲）。动量 $\approx$ 梯度的**指数加权平均**，窗口 $\sim 1/(1-\mu)$。

3. **AdaGrad → RMSProp → Adam**：AdaGrad 逐坐标步长 $\propto 1/\sqrt{\sum g^2}$（**累加全部历史** → 只要梯度不会永远为 0（$\Sigma g^2\to\infty$，长训练几乎必然满足），LR 就单调趋于 0，长训练"学不动"）；RMSProp 把累加和换成 **EMA**（修好单调衰减）；Adam = RMSProp 的二阶 EMA + **一阶动量** + **偏差修正**。

4. **Adam 偏差修正**：$m,v$ 从 0 起步早期有偏（偏向 0），除以 $1-\beta^t$ 去偏。因 $\beta_2$（0.999）比 $\beta_1$（0.9）更接近 1，$v$ 偏得更重 → **不修正时早期步长偏大**（默认 $\beta$ 下首步 $\approx\frac{1-\beta_1}{\sqrt{1-\beta_2}}=3.16$ 倍 $\eta$），修正后首步回到 $\eta$。注意 $\epsilon$ 在根号**外**（PyTorch）。

5. **AdamW 解耦权重衰减**：**Adam 的 L2 正则 $\ne$ 权重衰减**。L2 把 $\lambda\theta$ 加进**梯度** → 经 $1/\sqrt{\hat v}$ 缩放 → 历史梯度大的参数被衰减得更少（解耦被破坏）。AdamW（Loshchilov & Hutter）**解耦**：Adam 步之外直接从权重减 $\eta\lambda\theta$。这是现代默认；**裸 SGD（无 momentum）**下两者等价、Adam 下不等价（带动量的 SGD 若用耦合 L2，衰减项经动量缓冲跨步累积，与解耦的 SGDW 不再严格等价）。

6. **前沿优化器**：**Muon**（把 2D 权重的动量经牛顿-舒尔茨迭代**正交化**，仅隐藏 2D 层，Kimi-K2 在用）、**Lion**（**符号动量**，单状态 → Adam 一半显存）、**Shampoo**（Kronecker 因子化的**全矩阵预条件**）、**SOAP**（在 Shampoo 特征基里跑 Adam）、**Adafactor**（因子化二阶矩 → 亚线性显存）、**LAMB**（逐层自适应，超大 batch）、**Sophia**（轻量二阶，对角 Hessian）。

7. **LR schedule**：**warmup**（早期 $\hat v$ 方差大 + 大 batch/Post-LN 不稳 → 线性拉起）；**cosine**（平滑降到 ~0，+ warm restart，**需预定总步数**）；**inverse-sqrt / Noam**（原始 Transformer 调度）；**WSD**（warmup-stable-decay，恒定段 + 末段短衰减 → **不用预定总步数**、支持续训与中途 checkpoint）；**one-cycle**（super-convergence）。

8. **权重衰减 + LR-batch 缩放 + no-decay 组**：wd 做正则（近来也有"wd $\approx$ 有效 LR 控制器"的视角）；**线性缩放规则**（batch ×$k$ → LR ×$k$，SGD；Adam 常用 $\sqrt k$）；**LayerNorm/RMSNorm gain、bias、embedding 不加 wd**（分 decay/no-decay 两组）。

9. **梯度裁剪 + LLM 超参**：按**全局范数**裁（保方向）vs 按值裁（改方向），防梯度爆炸/loss spike；LLM 预训练默认 $\beta_2=\mathbf{0.95}$（**不是** 0.999，长记忆对尖峰太迟钝）、$\beta_1=0.9$、$\epsilon=\text{1e-8}$、wd $=0.1$、grad-clip $=1.0$。

---

## §10 选优化器 + 显存账

### 10.1　优化器状态的显存

混合精度训练里，每个**可训练参数**的显存大致是：bf16 权重 2 B + bf16 梯度 2 B + fp32 master 4 B + **优化器状态**。优化器状态正是各方法的分水岭：

- **SGD+momentum**：1 个状态（动量 $v$）。
- **Adam / AdamW**：**2 个状态（$m, v$）**，fp32 下 $=8$ B/param——这往往是大头，是"省显存"动机的来源。
- **Lion**：1 个状态（动量）→ Adam 的一半。
- **Adafactor**：因子化 $v$ → 亚线性，近似只剩"参数级"开销。

由此催生的省显存路线：

- **8-bit Adam**（Dettmers et al., 2110.02861）：把 $m,v$ **分块量化到 8-bit** 存储（用时反量化），优化器状态显存降到约 $1/4$，质量几乎无损。
- **Adafactor**：因子化二阶矩，亚线性显存（T5）。
- **Lion**：少存一个状态，直接减半。

### 10.2　什么时候用哪个

> 🎯 **"用哪个优化器"决策树**
> - **Transformer / LLM 微调或预训练**：**AdamW**（安全默认，$\beta_2=0.95$、wd 0.1、grad-clip 1.0、配 warmup+cosine/WSD）。
> - **经典视觉（ResNet 等）**：**SGD+momentum** 常**泛化更好**、且省显存——CV 里仍有大量 SOTA 用它。
> - **想要额外训练加速 / 大规模**：**Muon**（2D 隐藏层）、**Shampoo/SOAP**（全矩阵预条件，肯花每步算力）。
> - **显存受限**：**8-bit Adam** / **Adafactor** / **Lion**。
> - **超大 batch**：**LAMB / LARS**（逐层 trust ratio）。
