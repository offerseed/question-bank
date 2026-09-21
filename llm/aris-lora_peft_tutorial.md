# LoRA与PEFT — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[lora-peft-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/lora_peft_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 LoRA / PEFT** — 一页拿下面试核心要点（详见后文 §2–§10 推导）。

1. **核心公式**：冻结预训练权重 $W_0$，只学一个低秩增量 $\Delta W = BA$，前向 $h = W_0 x + \frac{\alpha}{r} BA\,x$。其中 $B \in \mathbb{R}^{d\times r}$、$A \in \mathbb{R}^{r\times k}$、$r \ll \min(d,k)$。

2. **为什么低秩有效**：预训练模型有很低的"内在维度"（intrinsic dimension，Aghajanyan 2020）——低维子空间里优化即可逼近满参；Hu 2021 据此**假设**微调的权重更新 $\Delta W$ 近似低秩，用 $r=4\sim64$ 的低秩矩阵就能逼近（经验假设 + 实证，非严格定理）。

3. **初始化**：$A$ 随机（Kaiming）、$B = 0$，使得**训练起点 $\Delta W = 0$**（不扰动预训练），但梯度不为零、能学起来。两个都置零会永远学不动。

4. **缩放 $\alpha/r$**：解耦 $r$ 与学习率，换 $r$ 不用重调 lr。rsLoRA 指出高秩时应改用 $\alpha/\sqrt{r}$ 防梯度坍缩。

5. **零推理延迟**：训练后可把 $\frac{\alpha}{r}BA$ **合并**进 $W_0$ 得 $W' = W_0 + \frac{\alpha}{r}BA$，推理与原模型完全同构、无额外延迟（这是 LoRA 相对 Adapter / Prefix 的关键优势）。

6. **省的是什么显存**：主要省**优化器状态 + 梯度**（Adam 下满参微调要 16 bytes/param，LoRA 只对 ~0.1% 的可训练参数付这笔钱）；**激活显存不自动省**（仍要反传过冻结基座），需配合 gradient checkpointing。

7. **QLoRA**：基座用 **NF4**（4-bit NormalFloat，对正态权重信息论最优）量化 + **双重量化** + **paged optimizer**，LoRA adapter 走 bf16，单卡 48GB 即可微调 65B。

8. **家族**：DoRA（幅度-方向分解）、rsLoRA（$\sqrt{r}$ 缩放）、PiSSA（主成分初始化）、AdaLoRA（自适应秩预算）、(IA)³（缩放向量）、LoRA+（A/B 不同 lr）—— 都在补 LoRA 的某一块短板。

---

## §10 复杂度与资源

| 维度 | 满参微调 | LoRA | QLoRA |
| --- | --- | --- | --- |
| 可训练参数 | $\Psi$ | $\sum_i r_i(d_i+k_i)$（~0.1%~1%） | 同 LoRA |
| 基座存储 | bf16 $2\Psi$ B | bf16 $2\Psi$ B | NF4 4 bit/param $= 0.5\Psi$ B |
| 优化器+梯度显存 | $\approx 14\Psi$ B | $\approx 14\Psi_{\text{lora}}$ B | 同 LoRA |
| 激活显存 | 高 | **同满参**（需 checkpointing） | 同满参 |
| 训练前向计算 | $W_0 x$ | $W_0 x + \frac{\alpha}{r}B(Ax)$（多两次小 matmul） | + 反量化开销 |
| 推理延迟（合并后） | 基线 | **零额外** | 反量化或合并到 fp16 |
| 单卡可训最大模型（80GB·vanilla Adam 量级） | ~3B（7B 满参需 ZeRO / offload / 8-bit optimizer） | ~13B~33B | ~65B |

旁路前向开销：每个适配层多 $2 \cdot 2 L r d$ FLOPs（$Ax$ 与 $B(\cdot)$ 两次小 matmul），因 $r \ll d$，相对主路 $2Ld^2$ 可忽略。合并后旁路彻底消失。

> ⚠️ **"单卡最大模型"只是经验量级** — 真实上限强依赖序列长度、batch、gradient checkpointing、优化器选择（8-bit / offload）、target_modules 与是否 ZeRO 切分，上表仅为 vanilla Adam 假设下的粗略数量级，不是硬上限。
