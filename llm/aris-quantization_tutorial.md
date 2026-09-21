# 量化Quantization — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[quantization-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/quantization_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 LLM Quantization** — 一页拿下面试核心要点（详见后文 §2–§11 推导）。

1. **Affine quantization 公式**：$q = \mathrm{round}(x / s) + z$，反量化 $\hat{x} = s\,(q - z)$。对称量化 $z = 0$；非对称量化 $z$ 把 zero-point 对齐到一个整数。

2. **粒度三档**（scale 共享范围越小，精度越高、开销越大）：per-tensor → per-channel → per-group（一行/一列内每 $g$ 个元素共享一个 scale，$g = 32 / 64 / 128$ 主流）。

3. **LLM 量化痛点**：activation 存在 **per-channel 系统性 outlier**（少数 channel 量级是平均的 $50\text{-}100\times$），均匀量化必崩。Weight 分布相对平坦，weight-only 量化（GPTQ / AWQ）天生比 weight+act 容易。

4. **GPTQ (Frantar 2023 ICLR)**：基于 OBS 推导的最优 weight update 公式 $\delta_{\mathbf{w}} = -\dfrac{w_q - \mathrm{quant}(w_q)}{[H^{-1}]_{qq}}\,[H^{-1}]_{q,:}$，逐列量化 + Cholesky 加速 + 128 列 block，4-bit weight 几乎无损。

5. **AWQ (Lin 2024 MLSys)**：观察 "1% salient weights drive most loss"，按 activation 幅度选 salient channel，**per-channel scale $s_c$ 在 $w \to w\cdot s_c$ / $x \to x/s_c$ 下数学等价但量化误差降低**，grid search $s_c = \mathrm{mean}(|x_c|)^\alpha$。

6. **SmoothQuant (Xiao 2023 ICML)**：把 activation outlier **迁移到 weight**——$Y = (X \mathrm{diag}(s)^{-1})(\mathrm{diag}(s)\,W)$，数学完全等价但 $X / s$ 平滑得多，得以做 W8A8。

7. **低精度浮点格式族**：FP8 (E4M3/E5M2, Hopper)、MX (OCP MXFP8/MXFP6/MXFP4, 32-elem block + E8M0 shared exp)、NVFP4 (Blackwell B100/B200, FP4 E2M1 + per-16-elem FP8 E4M3 scale + per-tensor FP32 scale)。Blackwell tensor core 原生支持 FP4 matmul。

8. **KV cache quant**：K 用 **per-channel**（K 的 outlier 沿 channel 维稳定），V 用 **per-token**（V outlier 沿 token 维变化）——KIVI / KVQuant 的基本设计。QServe 进一步把 W4A8KV4 整套量化做 SM89/SM90 kernel-level co-design。

9. **PTQ / QAT 的 2026 取舍**：≥4-bit weight-only 通常先考虑 PTQ（GPTQ / AWQ 没被取代）。QAT 值得考虑的条件是：亚 4-bit；W4A4 而 PTQ 的平滑 / 旋转变换达不到目标；端侧内存硬约束；以及你本来就掌握训练管线。它已是发布路径——Gemma 3/4 官方 QAT 权重、Apple 端侧 2 bits/weight、Llama 3.2 的 QLoRA-QAT 胜过 SpinQuant PTQ。代价因配方而异：ParetoQ 的 **~10%**（125M / 100B token）是训练预算占比，不能换算成相对 PTQ 的倍率；要成对的倍率就看 Llama 3.2 model card 的发布成本——QLoRA-QAT 1,300 GPU-h vs SpinQuant 1.7 GPU-h（1B）。

---

## §10 QAT 与低精度训练

### 10.0 一句话定位

PTQ 回答"已有 checkpoint 怎么压"，QAT 回答"在 ≤4-bit / W4A4 / 端侧内存硬约束下怎么不掉点"。2025 年起这条线不再是研究玩具：Google 为 Gemma 3 / Gemma 4 发官方 QAT 权重，Apple 的端侧模型以 2 bits/weight 发布，Meta 在自家 Llama 3.2 1B/3B 的 model card 上把 QLoRA-QAT 和 SpinQuant PTQ 并排列出。

> ✅ **2026 年怎么选** — ≥4-bit weight-only，PTQ 仍是默认，2025–26 没有方法在这个档位取代 GPTQ / AWQ。QAT 值得考虑的条件：亚 4-bit；W4A4 而 SmoothQuant / 旋转类 PTQ 达不到目标；端侧内存硬约束；你本来就掌握训练管线。它已是发布路径而非研究玩具。代价因配方而异：ParetoQ 的 ~10%（125M / 100B token，§10.4）是训练预算占比，不能换算成相对 PTQ 的倍率；Llama 3.2 model card 上成对的发布成本则是 QLoRA-QAT 1,300 GPU-h vs SpinQuant 1.7 GPU-h（1B）、1,600 vs 2.4（3B，§10.5）。两个数回答的是两个问题。

### 10.1 可微化：STE 及其偏差

Round / clamp 在数学上是不可导（round 的导数几乎处处为 0），反向传播无信号。**STE** 把量化-反量化函数 $\mathrm{QDQ}(x) = s\,(\mathrm{clamp}(\mathrm{round}(x/s), Q_\min, Q_\max))$（对称量化为例）的梯度近似为：

$$\frac{\partial \mathrm{QDQ}(x)}{\partial x} \;\overset{\text{STE}}{:=}\; \mathbf{1}\!\left[\,s\,Q_\min \le x \le s\,Q_\max\,\right]$$

即"前向用量化值，反向在 clipping 范围内 pass-through 梯度（饱和区梯度置 0）"。这是 LSQ / DoReFa / PACT 等 QAT 方法的基础。

工程上这套东西叫 **fake quant**：forward 里权重走一遍 $W \to \mathrm{QDQ}(W) \to$ GEMM，GEMM 本身仍是 BF16；backward 里梯度绕过 round 落到 FP32 的 latent weight 上，优化器更新的也是这份 latent weight；只有导出推理权重时才真正量化并打包成 INT4 存下来。

STE 从一开始就是**启发式**，通常不是量化目标的真梯度（Bengio et al. 2013, arXiv 1308.3432 提出时就是这么讲的）。失配点很具体：loss 在**量化后**的权重上算，更新却打在**量化前**的 FP 权重上，中间差一个 rounding，所以这个梯度是有偏的。2606.09012 分析了偏差的方向，结论是它并非随机噪声：在该文的局部 river–valley–basin 几何模型下，这个偏差系统性地把 latent weight 推向量化后 loss 更低的 basin，这解释了 STE 为什么实践上远好于它的理论地位。CAGE (2510.18784) 顺着这条线做显式修正：用曲率信息（curvature-aware）改写 STE 的更新方向，在 Llama 式预训练、对照 QuEST 的设置下把 W3A3 训到与 W4A4 相当的质量。

### 10.2 可学习量化参数：PACT → LSQ → LSQ+

STE 解决了"梯度传不传得下去"，但 clipping 范围和 step size 仍是手调超参。这条线把它们变成可学习参数。

**PACT (1805.06085)** 学 activation 的 clipping 上界 $\alpha$。先做一个三段式的 clipped ReLU：

$$y = 0.5\left(\lvert x \rvert - \lvert x - \alpha \rvert + \alpha\right)$$

展开就是 $x \lt 0$ 时 $y = 0$，$0 \le x \lt \alpha$ 时 $y = x$，$x \ge \alpha$ 时 $y = \alpha$。再均匀量化到 $k$ bit：

$$y_q = \mathrm{round}\!\left(y \cdot \frac{2^k - 1}{\alpha}\right) \cdot \frac{\alpha}{2^k - 1}$$

论文对 $\alpha$ 的求导（Eq. 3）是一个**近似**：把 QDQ 对 $y$ 的梯度当成恒等（round 用 STE 穿过），同时忽略量化步长 $\alpha/(2^k-1)$ 本身对 $\alpha$ 的显式依赖。在这个近似下：

$$\frac{\partial y_q}{\partial \alpha} = \begin{cases} 0, & x \lt \alpha \\ 1, & x \ge \alpha \end{cases}$$

保留 round 的 STE、完整展开 $\alpha$ 的显式依赖后，还会多出一项 $(\mathrm{round}(m y/\alpha) - m y/\alpha)/m$（$m = 2^k - 1$），即范围内元素的 rounding 残差；PACT 把它丢掉了，LSQ 保留的正是这一项（见下）。

读法：在这个近似下，落在 clipping 范围内的元素完全不关心 $\alpha$ 取多少，只有**被截断的元素**才给 $\alpha$ 传梯度。上游 loss 传来的梯度正负都有，所以这不是"$\alpha$ 只会被推大"；真正的麻烦是 $\alpha$ 一旦大到几乎没有元素被截断，梯度就变得稀疏，$\alpha$ 卡在平台上不动，而量化范围已经被撑得过宽。PACT 因此在 $\alpha$ 上加 L2 正则往回拉。

**LSQ (1902.08153)** 直接学 step size $s$（weight 和 activation 都学）：

$$\bar v = \left\lfloor \mathrm{clip}(v / s,\, -Q_N,\, Q_P) \right\rceil, \qquad \hat v = \bar v \cdot s$$

对 $s$ 的梯度是：

$$\frac{\partial \hat v}{\partial s} = \begin{cases} -v/s + \lfloor v/s \rceil, & -Q_N \lt v/s \lt Q_P \\ -Q_N, & v/s \le -Q_N \\ Q_P, & v/s \ge Q_P \end{cases}$$

和 PACT 的关键差别在第一行：范围**内**的元素也给 $s$ 梯度，大小正好是 rounding error $\lfloor v/s \rceil - v/s$。所以 $s$ 被整层的 rounding 误差驱动，而不是只被截断项驱动，这是 LSQ 在低 bit 上稳定胜过 PACT 的原因。

★必考的一步★ 是 **gradient scale**：反向时把 $\partial L / \partial s$ 乘上

$$g = \frac{1}{\sqrt{N_W Q_P}}\ \text{(weight)}, \qquad g = \frac{1}{\sqrt{N_F Q_P}}\ \text{(activation)}$$

$N_W$ 是该层权重元素数，$N_F$ 是该层 activation 元素数。为什么需要：LSQ 要平衡的是**相对更新量**——单个 $w_i$ 的梯度只来自它自己一个元素，而 $s$ 被整层共享，它的梯度是 $N$ 个元素贡献的和。不校正的话，$s$ 的"更新量 / 自身量级"比权重大约 $\sqrt{N Q_P}$ 倍（$Q_P$ 来自 $s$ 自身量级比权重小这一侧），同一个学习率下要么 $s$ 直接跑飞、要么训练震荡。乘上 $g$ 就是把这个比值拉回和权重同一量级。

```python
import torch, torch.nn as nn

def grad_scale(x, g):                       # 前向恒等，反向梯度乘 g
    return (x - x * g).detach() + x * g

def round_ste(x):                           # 前向 round，反向恒等
    return (x.round() - x).detach() + x

class LSQWeightFakeQuant(nn.Module):
    """LSQ (1902.08153) 的 weight 量化器：step size 可学 + gradient scale。"""
    def __init__(self, num_elements, bits=4):
        super().__init__()
        self.Qn, self.Qp = 2 ** (bits - 1), 2 ** (bits - 1) - 1   # INT4: -8 / 7
        self.g = 1.0 / (num_elements * self.Qp) ** 0.5            # 1/sqrt(N_W * Q_P)
        self.s = nn.Parameter(torch.tensor(1.0))

    def forward(self, w):
        s = grad_scale(self.s, self.g)
        v = round_ste((w / s).clamp(-self.Qn, self.Qp))
        return v * s                        # fake quant：值域已离散，dtype 仍是 FP

w = torch.randn(256, 512, requires_grad=True)
q = LSQWeightFakeQuant(w.numel(), bits=4)
q.s.data.fill_(2 * w.abs().mean().item() / q.Qp ** 0.5)   # LSQ 论文给的初始化
q(w).square().sum().backward()
print(q.s.grad, w.grad.abs().mean())               # 去掉 g，s.grad 会再放大约 958 倍（本例两打印值之比约变成 8,400）
```

**LSQ+ (2004.09576)** 补两件事：一是给量化器加**可学习的 offset**，让它能表达非对称范围——ReLU 时代 activation 恒非负，对称量化够用，但 GELU / Swish 有一段负值，没有 offset 就要浪费掉一整个符号位或者直接把负半边截掉；二是用 **MSE 最小化初始化** $s$ 和 offset 而不是从 min/max 起步，低 bit 下 LSQ 对初值敏感到换个种子能差好几个点。

**学什么、不学什么**：$s$、offset、$\alpha$ 是学的；bit width、粒度、哪几层不量化（embedding / lm_head / norm）是人定的。粒度上 weight 在 4-bit 用 per-channel 就够；低于 4-bit 常见做法是退到 per-group（$g = 32/64/128$），但这是工程选择不是硬性要求——ParetoQ 在 channel 粒度上也给了可用的亚 4-bit 结果。

2505.14302 用 268 次 QAT 实验拟了一条 scaling law，量的是 QAT 相对 BF16 的**训练 loss gap**，两个结论值得记：gap 随 group 变粗而增大，也**随训练 token 数增多而增大**——同一配置训得越久，QAT 之后的 gap 反而更大（常被解释成训练把权重推得更"满"、留给量化的余量更小，但这只是解释，论文没有验证这个机制）；另外在他们的实验里 W4A4 的瓶颈不在权重，而在 **FC2 的 activation outlier**（FFN 第二个 Linear 的输入）——这是观察到的主导项，不是普适定律。

> ⚠️ **BN folding 不要往 LLM 上套** — "先把 BatchNorm 折进 conv 权重再做 fake quant"是 CNN QAT 的标准动作，因为 BN 的 running mean/var 在推理时是常数，不折就会让训练和推理量化的对象对不上。LLM 用 RMSNorm，没有 running statistics，这一步整个不存在。LLM QAT 里形似的操作是把 RMSNorm 的 $\gamma$ 吸收进下一层权重（§7.2），目的是让旋转可交换，和 BN folding 不是一回事。

### 10.3 QAT 的三种用法

(a) **PTQ 初始化 + STE finetune**：先用 GPTQ / AWQ 把 checkpoint 压到目标 bit，拿它初始化量化参数和权重，再 STE finetune 几千步。好处是起点已经接近目标量化形式；风险是被 PTQ 找到的那个解锁住。

(b) **从 FP checkpoint 直接 QAT**：跳过 PTQ 初始化，从 BF16 权重开始 fake-quant 训练。Gemma 的 QAT 权重和 ParetoQ 走这条路；bit 越低这条路越有吸引力——2-bit 的 PTQ 起点本身质量就低，未必值得当锚点。

(c) **from scratch 原生低 bit**：**BitNet b1.58 2B4T (2504.12285)** 是目前最完整的公开例子——2B 参数、4T token 从零训练，weight ternary $\{-1, 0, +1\}$。**BitNet v2 (2504.18415)** 把 activation 也拿下：H-BitLinear 在矩阵乘之前插一个 online Hadamard 变换把 activation 分布打平，使原生 INT4 activation 可行（不是 INT8 退让）。

**EfficientQAT (2407.11062)** 单独记，它把"QAT 很贵"这个前提直接拆掉：两阶段——Block-AP 逐 block 训练该 block 的全部参数（显存只需装下一个 block）+ E2E-QP 只训量化参数做端到端对齐。2-bit Llama-2-70B 在**单张 A100-80GB 上 41 小时**训完，下游平均 69.48 vs FP16 的 72.41。

### 10.4 预算怎么分

**ParetoQ (2502.02631)** Finding-1 是这条线最该背的数：固定 100B token 总预算、MobileLLM-125M，把预算切成"FP 预训练 + QAT finetune"两段，最优点在 **~90% FP 预训练 + ~10% QAT**；**FP 预训练占比再往上超过 ~90%** 就开始掉点——留给 QAT 的预算不够了。

Finding-2 解释了 bit 越低为什么越要多给 QAT：≥3-bit 时 QAT 做的是**补偿**（compensation），权重相对 FP 起点的 **L1 相对变化**只有 10–20%；≤2-bit 时做的是**重构**（reconstruction），相对变化约 40%。这是权重移动幅度的度量，论文并没有据此论证"换了一个 basin"。

**Compute-Optimal QAT (2509.22935, Apple + EPFL, ICLR 2026)** 把这件事外推到更大规模：最优 QAT 占比不是常数，随总算力增大而上升，并且可以用 **tokens-per-parameter-byte** $D / [N \cdot (B/8)]$（训练 token 数 ÷ 量化后**全部**参数字节数，$N$ 是参数量、$B$ 是位宽）预测——同样的 token 数，模型压得越狠，该分给 QAT 的比例越高。

落到具体工程：**Gemma 3 QAT 公布的数字是约 5,000 步**，teacher 用的是未量化 checkpoint 的输出概率。5,000 步，对着一个已经预训练完的模型——这是 Gemma 3 这一份配方的量级，不是通用数字。

### 10.5 发布实践：谁在发 QAT 权重

- **Gemma 3 QAT**（Google 博客 2025-04-18）：int4 覆盖 1B / 4B / 12B / 27B。博客给的数字是 perplexity 掉点相对 llama.cpp Q4_0 PTQ **减少 54%**；weights-only 显存 27B 54 → 14.1 GB、12B 24 → 6.6 GB、4B 8 → 2.6 GB、1B 2 → 0.5 GB。

- **Gemma 4 QAT**（博客 2026-06-05）：QAT 覆盖 E2B / E4B / 12B / 26B-MoE，除 Q4_0 外另发一个移动端格式，并对**生成 token 的那几层**做定向 2-bit，把 E2B 压到 1 GB 级。Google 只说 QAT 优于自家 PTQ baseline，没有公布百分比——面试里别替他们编一个数。

- **Llama 3.2 1B / 3B**（Meta 博客 2024-10-24）：把 QLoRA-QAT 和 SpinQuant PTQ 并排发。1B 平均分 BF16 36.1 / QLoRA-QAT 35.7 / SpinQuant 33.1——QAT 几乎接平 BF16，同规模 PTQ 掉 3 分。配置是 weight 4-bit group-32、activation 8-bit per-token 动态；端侧实测 2–4$\times$ 速度，体积 −56%、内存 −41% 是 Android OnePlus 12 上的平均值。这份 card 也同时给出了两条路线的发布成本，是目前少见的成对数字：QLoRA-QAT 1,300 GPU-h vs SpinQuant 1.7 GPU-h（1B）、1,600 vs 2.4（3B）——约 765$\times$ / 667$\times$。

- **Apple 端侧模型 (2507.13575, 2025-07)**：3B decoder 压到 **2 bits/weight**，手段是 QAT + learnable weight clipping（正是 §10.2 那条线）；embedding 走 4-bit QAT，KV cache 8-bit，再挂 LoRA quality-recovery adapter 把质量拉回来；论文 Table 3 给的端侧质量是 MMLU **67.8 → 64.4**。同一篇里 server 侧用 ASTC **PTQ**（MMLU **80.0 → 79.2**）——端侧上 QAT、数据中心上 PTQ，这个分野比任何论证都直白。ASTC 那个 3.56 bpw 是编码载荷（128 bit / 36 个权重），算上 FP16 的 block 最小开销后是 **4 bpw**，而且还没算 adapter——别把 3.56 说成 server 模型整体的存储率。

- **NVIDIA QAD (2601.20088, 2026-01)**：quantization-aware distillation，用 FP teacher 的分布对 NVFP4 student 做 KL 蒸馏。重点不在压缩率：论文验证的是把 QAD 用在**已经走完 SFT / RL / model merging 的模型**上，把量化掉的精度重新训回来。

### 10.6 亚 4-bit 的真相：ParetoQ

ParetoQ 的另一半贡献是**统一比较框架**：1 / 1.58 / 2 / 3 / 4-bit 在同一套流程里训，按 bit 选择量化函数与训练预算（binary / ternary / 2-bit 的学习率和训练长度和 3 / 4-bit 不同，典型饱和预算约 30B vs 10B token），不是所有 bit 共用一套超参。bit 之间的比较由此才成立（此前各 bit 的 SoTA 来自不同论文、不同配方，横着比没有意义）。在这个前提下：

- **ternary / 2-bit / 3-bit 普遍优于 4-bit 和 binary**——不是"bit 越低越差"的单调曲线，binary 是断崖，4-bit 则在同等模型体积下浪费了容量。

- ternary 600M 超过此前 SoTA 的 ternary 3B。

- 综合硬件，**2-bit 往往最实用**：论文提到**有些实现**把 ternary 的 1.58 bit 按 2 bit 存，那种情况下 2-bit 吃到同样的带宽收益。它在体积—精度上的结论是 ternary / 2-bit / 3-bit 大体相当，不是 2-bit 必然更好。

### 10.7 QAT 的兄弟：低精度训练

QAT 是"低精度推理 + 高精度训练"；低精度训练是把 GEMM 本身降到 FP8 / FP4。两者共用 STE、scaling、outlier 这三个问题，所以面试常连着问。

- **DeepSeek-V3 FP8 (2412.19437 §3.3)**：第一个公开的大规模 FP8 训练配方。关键是**细粒度 scaling**——activation 按 $1 \times 128$ tile、weight 按 $128 \times 128$ block 各算各的 scale，而不是 per-tensor；以及**周期性 FP32 promotion**：tensor core 的 FP8 累加精度不够，每累 $N_C = 128$ 个元素（约 4 个 WGMMA）就把部分和搬到 CUDA core 上按 FP32 加。embedding、output head、MoE gating、norm、attention 仍是 BF16 / FP32。结果是相对 loss 误差 < 0.25%——这个数来自两次约 1T token 的验证规模对照，不是 14.8T 全程和 BF16 对跑；正式训练则是 14.8T token 全程 FP8。

- **MXFP8 (2506.08027)**：两个和 OCP MX v1 规范不同的选择——所有 tensor 统一用 **E4M3**（规范允许梯度用 E5M2），以及 block scale 的 round **向上取整**（规范是向下）。向下会让 block 内最大值溢出，向上只损失一点分辨率。8B 模型 15T token，ppl 与 BF16 差在 0.5% 以内。

- **NVFP4 预训练 (2509.25149, NVIDIA)**：E2M1、16 元素一个 block、每 block 一个 E4M3 scale，外面再套 per-tensor FP32 scale；编码 scale 取 $s_{enc} = 6 \cdot 448 / \mathrm{amax}$（6 是 E2M1 的最大值，448 是 E4M3 的最大值）。四个工程决定值得背：weight 用 **2D $16 \times 16$ block**，这样 forward 和用 $W^\top$ 的 backward 看到的是同一套量化；**random Hadamard 只加在 Wgrad 的输入上**，不是全局；**stochastic rounding 只用在梯度上**，用在 forward 上反而有害（forward 要的是确定性最优，不是无偏）；**前 2 个和后 8 个 block 保持 BF16**，占线性层的 16%。12B 模型 10T token，MMLU-Pro 62.58 vs FP8 的 62.62。另有一组 8B 的对照：MXFP4 要 1.36T token 才追平 NVFP4 1T token 的 loss（+36%）——那是 loss 对齐实验，不是上面这组 12B / 10T 的 MMLU-Pro 比较。

- **Quartet (2505.14669) / Quartet II (2601.22813)**：给 FP4 训练拟 scaling law，并提出 MS-EDEN 这个无偏量化器——回答的是"FP4 训练在什么规模上才划算"，不是"能不能跑"。

> ⚠️ **别把 FP8 和 FP4 训练说成一回事** — FP8 预训练是 production（DeepSeek-V3 已用 14.8T token 证明）；FP4 预训练目前只到研究验证的 12B / 10T 规模，**没有任何前沿模型是 FP4 预训练出来的**。DeepSeek-V4 里提到的 FP4 走的是存储和 indexer 路径，不是 pretraining 的 GEMM——这是最常见的读错。

### 10.8 易踩坑

- **gpt-oss 的量化别替它下结论**：2508.10925 §2.1 的原话是 MoE 权重"post-trained with quantization"到 MXFP4、4.25 bits/param，覆盖 90+% 的参数。card 没有说明这次量化是不是 training-aware——"在 post-training 阶段量化"和 QAT 并不互斥，QAT 本来就可以发生在 post-training。面试里照原话讲。

- **别用"~10% 预算"去换算"比 PTQ 慢几倍"**：ParetoQ 的 ~10% 是 125M / 100B token 下训练预算怎么切（§10.4），Gemma 3 的 ~5,000 步蒸馏、EfficientQAT 单张 A100 41 小时训完 2-bit 70B 也都是没有配对 PTQ 分母的绝对量——三个数都答不了"相对 PTQ 多少倍"。要倍率就得找同一份 card 上成对的两条路线：Llama 3.2 的 QLoRA-QAT 1,300 GPU-h vs SpinQuant 1.7 GPU-h（1B）、1,600 vs 2.4（3B），约 765$\times$ / 667$\times$。也就是说"100–1000$\times$"在这个量级上并不离谱；错的是把它当成普适常数，或当成要不要上 QAT 的唯一依据。

- **BN folding 是 CNN 习惯**：LLM 没有 BatchNorm，见 §10.2 的 callout。

- **QAT 的粒度要对齐部署 kernel**：用 group-64 训出来的 QAT 权重，落到只支持 group-128 的推理 kernel 上就得重新量化，训练时换来的精度可能大半还回去（还多少取决于模型和 bit 宽，不是必然全丢）。先确认目标后端支持的 group size，再定 QAT 配置。
