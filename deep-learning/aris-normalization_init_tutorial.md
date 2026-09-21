# 归一化与初始化 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[normalization-init-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/normalization_init_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 Normalization / Residual / Init** — 一页拿下面试核心要点（详见后文 §1–§11 推导）。

1. **为什么要归一化**：深网络逐层放大 / 缩小激活，方差以 $g^L$ 指数发散或塌缩；归一化把每层激活拉回受控尺度，关键收益是**平滑了 loss landscape**（更小的梯度 Lipschitz / β-smoothness，Santurkar 2018），从而能用更大学习率、堆更深——**不是**"减少 internal covariate shift"那套旧说法。

2. **BatchNorm**：沿 **batch（+空间）维**对每个 channel 归一化，$\hat x=\frac{x-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}$；**训练用 batch 统计 + 维护 running mean/var，推理用 running stats** → train$\ne$eval，且强依赖 batch 大小（小 batch / 变长序列 / RL / online 全踩雷）。

3. **LayerNorm**：沿**特征维 per-token** 归一化（与 batch 完全解耦），$y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$；batch=1 也能用、变长序列也能用、train$=$eval——所以 Transformer / RNN 用它而不用 BN。

4. **RMSNorm**：丢掉 re-centering，只按 RMS 缩放 $\bar x=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot\gamma$（无均值、无 $\beta$）；论点是 **re-scaling 不变性比 re-centering 更重要**，省一次 reduction，LLaMA 之后的事实默认。

5. **Pre-LN vs Post-LN**：Post-LN（原始 Transformer，LN 在残差相加**之后**）质量上限略高但**需 LR warmup、深了不稳**；Pre-LN（LN 在残差分支**内部**）有干净恒等梯度路径 → 稳、免 warmup，但**残差流幅度随深度按 $\sqrt L$ 增长**、深层贡献被稀释，必须补一个 final LN。

6. **放置变体**：DeepNorm（放大残差 $\alpha x_l$ + 缩小 init → 训 1000 层）、Sandwich / double LN（Gemma2 分支前后各一个）、**QK-Norm**（点积前归一化 Q、K，压住 attention logits 防爆）。

7. **残差 + 缩放**：$y=x+F(x)$ 的雅可比 $I+\partial F/\partial x$ 给出**梯度高速路**（恒等项保证梯度不消失）；缩放技巧把分支起点压向恒等——$1/\sqrt N$ 深度缩放、LayerScale（learnable per-channel $\lambda$）、ReZero（learnable 标量 init 0）、SkipInit。

8. **初始化**：核心目标是**方差保持**——Xavier（tanh，$\text{Var}(W)=\frac{2}{n_\text{in}+n_\text{out}}$）、Kaiming（ReLU，$\text{Var}(W)=\frac{2}{n_\text{in}}$，那个 2 补偿 ReLU 砍掉的一半方差）；残差网再按深度下调，**GPT-2 把残差投影权重 $\times\frac{1}{\sqrt{2N}}$**；Fixup 以 init 为主（+ 少量可学 scalar）就免归一化训深残差网。

9. **μP / 归一化-free / 工程**：μP 让**最优超参（尤其 LR）宽度不变** → 小宽度调好 zero-shot 迁移到大模型；Fixup / NFNets / DyT 探索去归一化（前沿方向，非定论）；工程上盯紧 final LN、$\epsilon$ 位置、fp32 reduction、fused kernel。

---

## §10 归一化-free 与前沿

归一化层带来 train/eval 差异（BN）、跨设备同步（多卡 BN）、额外 reduction 等麻烦，于是一直有人问：**能不能不要归一化？** 这是研究方向，**非定论**，但思路很有启发。

### 10.1　Fixup / NFNets：用 init + 显式方差控制替代归一化

- **Fixup**（§8.5）：纯靠初始化为主（分支末层置 0 + 深度下调）+ 少量可学标量乘子 / 偏置，训深残差网、无任何归一化层，ImageNet 上逼近 BN-ResNet。证明 norm 非必需（但仍需极少量可学 scalar 补偿被砍掉的仿射自由度，非严格"零额外可学参数"）。
- **NFNets**（Brock et al., 2021, arXiv 2102.06171）：Normalizer-Free Networks，系统性地去掉 BN。三件套——**Scaled Weight Standardization**（标准化权重而非激活）+ **解析设计的缩放残差块** $x_{l+1}=x_l+\alpha\,F(x_l/\beta_l)$（用解析的 $\alpha,\beta_l$ 精确追踪 / 控制每层方差）+ **Adaptive Gradient Clipping（AGC）**（按参数范数自适应裁剪梯度，找回 BN 在大 batch 下的稳定性）。结果在 ImageNet 上**超过 EfficientNet 且无任何归一化层**，说明 BN 的好处（控尺度 + 正则 + 大 batch 稳定）可以被显式手段分别补回。

### 10.2　DyT（Dynamic Tanh）：用可学 tanh 替掉 LN

DyT（*Transformers without Normalization*, 2025, arXiv 2503.10622，arXiv id 以引用核查为准）来自一个观察：**训练好的 LayerNorm 的输入-输出曲线，长得像一条被压扁的 $\tanh$**（对中间值近线性、对离群值 S 形饱和）。既然 LN 的效果近似一个逐元素 squashing，那干脆**省掉求均值 / 方差的 reduction，直接学一个 tanh**：

$$\text{DyT}(x) = \gamma\odot\tanh(\alpha\,x) + \beta,$$

其中 $\alpha$ 是一个**可学标量**（控制输入尺度、对应 LN 里"除以 std"的作用），$\gamma,\beta$ 是逐通道仿射。论文报告在 ViT / LLM / diffusion 等多处用 DyT 替换 LN/RMSNorm，效果相当，且**去掉了归一化的统计量计算**（不再需要逐 token 的 reduction）。

> 🎯 **前沿，诚实定位**
> Fixup / NFNets / DyT 共同传递一个信息：**归一化在理论上不是训练深网络的必要条件**——它的核心作用（控方差、平滑地形、压离群值）可被"好 init / 显式方差控制 / 可学 squashing"等手段替代。但 **LN/RMSNorm 仍是当前生产系统的稳妥默认**（鲁棒、即插即用、生态成熟）。面试谈这些要点明"研究方向 / 有前景"，别说成"已取代归一化"。
