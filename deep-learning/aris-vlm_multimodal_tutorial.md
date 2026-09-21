# VLM多模态 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[vlm-multimodal-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/vlm_multimodal_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 VLM** — 一页拿下视觉-语言模型面试核心要点（详见后文 §1–§13 推导与代码）。

1. **视觉 encoder = ViT 主导**：Dosovitskiy et al. 2021 (ICLR) 把图像切 $P\times P$ patch（一般 $P=14$ 或 $16$）做线性投影 + 可学习 positional embedding + 可选 `[CLS]` token，输入 Transformer encoder。**CLIP / SigLIP / LLaVA 的视觉端都是 ViT 变体**。

2. **CLIP 对称 InfoNCE（必推）**：Radford et al. 2021 (ICML) 让 image embedding $\mathbf{u}_i$ 和 text embedding $\mathbf{v}_i$ 在共享空间里做对比学习，loss 为 **行 softmax + 列 softmax 平均**：$\mathcal{L} = \tfrac{1}{2}(\mathcal{L}_{i\to t} + \mathcal{L}_{t\to i})$。温度 $\tau$ 可学习（log-parameterize 为 `logit_scale=log(1/τ)`；被 clip 的是 $\exp(\text{logit\_scale})=1/\tau$ 的上界 100，等价 $\tau$ 下界 0.01，而非 $\tau$ 本身被 clip 到 100）。

3. **SigLIP 用 sigmoid 替 softmax**：Zhai et al. 2023 (ICCV) 把 N×N 相似度矩阵的每一项独立做 binary CE，**摆脱 batch-wise softmax 归一化**，因此对 batch size 不再线性敏感，单机能用 32k+ batch 训；引入 learnable bias $b$ 修正初期 negative dominance。**SigLIP-2 (Google 2025)** 加入 caption + self-distillation + dense local objectives 并扩展多语言。

4. **LLaVA = projector + 2-stage train**：Liu et al. 2023 (NeurIPS) 用一个 **单层线性投影矩阵 $W$** 把 frozen CLIP 视觉特征投到 LLM token 空间（**2 层 MLP projector 是 LLaVA-1.5 的升级**）。**Stage 1** 只训 projector 做 feature alignment（caption 数据），**Stage 2** 解冻 LLM 做 visual instruction tuning（GPT-4 生成的 158K instructions）。

5. **Q-Former vs Projector 是 BLIP-2 的核心 trade-off**：Li et al. 2023 (ICML) 用 32 个 learnable query token 在 frozen image encoder 上做 **cross-attention**，把任意分辨率/数量的 patch 压成固定 32 token——计算预算稳定但**信息有损 + 训练复杂**。LLaVA 的 MLP 简单但 token 数随分辨率二次增长。

6. **Flamingo / Llama-3.2-Vision = gated cross-attn**：Alayrac et al. 2022 (NeurIPS) 用 **Perceiver Resampler**（64 latent query）把 visual feature 压成固定数 token，再在 LLM 每隔几层插入 **gated cross-attention** 层（$\tanh$ 门控初始化为 0，保留 frozen LLM 的 text-only 能力）。

7. **Qwen2-VL 的 M-RoPE 必考**：Wang et al. 2024 把 RoPE 沿 head_dim 分成 6 段，按 axis 序列 $(t, h, w, t, h, w)$ 分配 (t / h / w 三组位置 id)；典型配置 `mrope_section=[16,24,24]`（单位是半 head_dim 的对数，$\sum \times 2 = $ head_dim=128，**全部 128 维都旋转**）。这样每个 token 同时携带 (t, h, w) 三维位置而不需要扁平化。**配合 native dynamic resolution**（不再 padding 到固定 224×224）。

8. **训练三段式 + 偏好优化**：(1) **alignment** 训 projector / Q-Former；(2) **visual instruction tune** 解冻部分 LLM；(3) **preference**（LLaVA-RLHF, RLAIF-V, VLM-R1, DPO/PPO）治理幻觉、对齐 long-tail。**VLM-R1 (2025)** 用 GRPO + verifiable reward 把推理能力迁到视觉-语言任务。

---

## §10 Qwen2-VL / DeepSeek-VL：动态分辨率 + M-RoPE

### 10.1　Native Dynamic Resolution

Qwen2-VL (Wang et al. 2024)、DeepSeek-VL (Lu et al. 2024)、InternVL-2 都抛弃了"resize 到固定 224²"的传统：

- **保留原始 aspect ratio**：把图按 patch_size 的整数倍 resize 到接近原尺寸的最大值

- **patch 数动态**：Qwen2-VL 的 `smart_resize` 先把每条边 round 到 28（$=2P$，匹配后续 2×2 spatial merge）的倍数，例如 $1024 \times 768$ 图先被 resize 到约 $1036 \times 756$；按 $P=14$ 切得 $74 \times 54 = 3996$ 个 patch，经 2×2 spatial merge 后得到 999 个视觉 token

- **不再使用固定 pos embed 表**：必须用 **可扩展的位置编码**（RoPE 或 2D ALiBi-like）

### 10.2　M-RoPE（Multimodal RoPE）

Qwen2-VL 的核心创新。**回顾普通 1D RoPE**：把 query / key 的每对维度 $(2k, 2k+1)$ 看作复数，乘上位置相关的旋转：

$$\mathbf{R}_{m,k} = \begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix}, \quad \theta_k = 10000^{-2k/d}$$

应用到 $\mathbf{q}_m$ 后，$\mathbf{q}_m^\top \mathbf{k}_n$ 只依赖 $m - n$（相对位置）。

**M-RoPE 的扩展**：一个视觉 token 有 (t, h, w) 三个位置维度。**所有 head_dim 都旋转**——但每对维度 $(2k, 2k+1)$ 根据所在区段，用 t / h / w 三个位置 id 之一参与旋转角度：

$$(\cos(m_\text{axis}\,\theta_k),\ \sin(m_\text{axis}\,\theta_k)), \quad \text{axis} \in \{t, h, w\}$$

具体地，Qwen2-VL 的 `mrope_section`（**单位是半 head_dim 对**，即每个数代表多少对 $(2k, 2k+1)$）。一对 = 2 个实数维度，所以"section sum × 2 = head_dim"。

> 💡 **Qwen2-VL 默认 `mrope_section = [16, 24, 24]`** — 即三个 axis 各占 16 / 24 / 24 对维度；总 $(16+24+24) \times 2 = 128 = $ head_dim。实现上把 section 翻倍成 $[16, 24, 24, 16, 24, 24]$ 沿 head_dim 切，分别用 (t, h, w, t, h, w) 的位置 id 旋转——**全部 128 维都参与旋转**，没有"不旋转 dim"。空间维（h, w）占 48 对 > 时间维（t）的 16 对；这是一种符合直觉的可能解释（时间变化通常慢于空间内容变化），但**并非 Qwen2-VL 论文明确证实的设计动机**——具体 16/24/24 切分比例也可能只是经验调参结果。

文本 token 没有显式 (h, w)：Qwen2-VL 让 $m_t = m_h = m_w$ 等于该 text token 的 1D 位置 id，三个 axis 给出**完全相同的旋转角**，等价于普通 1D RoPE。

### 10.3　Qwen2.5-VL 升级

Qwen2.5-VL（Bai et al. 2025）在 Qwen2-VL 基础上：
- **绝对时间编码**：M-RoPE 的 t 维改用真实时间戳（秒），不是帧 index，**支持任意 FPS 视频**
- **动态视觉 token budget**：根据任务复杂度调 token 数
- **agent / GUI 能力**：训练数据加入 web screenshot / mobile UI 操作 trace

### 10.4　DeepSeek-VL / VL2：高分辨率 tiling + Hybrid encoder

DeepSeek-VL (Lu et al. 2024) 用**双 vision encoder**：
- **SigLIP**：处理全局语义（低分辨率）
- **SAM-B**（Segment Anything backbone）：处理高分辨率细节

两路特征 concat 喂给 projector + LLM。**DeepSeek-VL2 (2024.12)** 进一步换成 MoE LLM + 动态分辨率，单 image 视觉 token 可达 1700+。

### 10.5　Code: M-RoPE 三维位置嵌入（核心 50 行，对齐 Qwen2-VL HF 实现）

```python
import torch

def build_mrope_cos_sin(positions, head_dim, mrope_section=(16, 24, 24), base=1000000.0):
    """
    Build cos/sin tensors for Qwen2-VL style M-RoPE.

    positions: LongTensor [3, B, L]   (axis 0: t / h / w; B batch; L seq len)
    head_dim:  per-head dim (must equal 2 * sum(mrope_section))
    mrope_section: tuple of 3 ints; each = number of (half-dim) entries per axis
    Returns: cos, sin both [B, L, head_dim], ready for LLaMA-style rotate_half.
    """
    assert 2 * sum(mrope_section) == head_dim, "2 * sum(mrope_section) must = head_dim"
    half = head_dim // 2                                                # = sum(mrope_section)

    # 标准 RoPE 频率: θ_k = base^{-2k/head_dim}, k = 0..half-1
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() * 2 / head_dim))   # [half]
    inv_freq = inv_freq.to(positions.device)

    # 对每个 axis 算 [B, L, half] 的 angle / cos / sin
    cos_axes, sin_axes = [], []
    for a in range(3):
        ang = positions[a].float().unsqueeze(-1) * inv_freq                     # [B, L, half]
        cos_axes.append(ang.cos())
        sin_axes.append(ang.sin())

    # 把 half-dim 按 mrope_section 切成 3 段，分别取 t/h/w 的 cos/sin
    cos_chunks, sin_chunks = [], []
    offset = 0
    for axis, s in enumerate(mrope_section):
        cos_chunks.append(cos_axes[axis][..., offset:offset+s])                 # [B, L, s]
        sin_chunks.append(sin_axes[axis][..., offset:offset+s])
        offset += s
    cos_half = torch.cat(cos_chunks, dim=-1)                                    # [B, L, half]
    sin_half = torch.cat(sin_chunks, dim=-1)

    # LLaMA-RoPE 风格 duplicate 到 full head_dim
    cos = torch.cat([cos_half, cos_half], dim=-1)                               # [B, L, head_dim]
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin

def rotate_half(x):
    """(x1, x2) -> (-x2, x1), LLaMA convention."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_mrope(q, k, cos, sin):
    """
    q, k:    [B, num_heads, L, head_dim]
    cos, sin:[B, L, head_dim]
    """
    cos = cos.unsqueeze(1)                                                       # broadcast over heads
    sin = sin.unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

> ⚠️ **M-RoPE 三个常见误读** — 容易踩的坑。

- **"head_dim 切成 t/h/w 三段独立 1D RoPE"**：错。Qwen2-VL 实际是 **6 段 alternating** `[s_t, s_h, s_w, s_t, s_h, s_w]`，全 head_dim 都旋转

- **"section 单位是 dim"**：错。`mrope_section=[16,24,24]` 单位是 **对数**（每对 = 2 个 head_dim 元素），$\sum \times 2 = $ head_dim = 128

- **"text token 没有 (h, w) 怎么办？"**：让 $m_t = m_h = m_w$ 等于 text 的 1D 位置 id，三个 axis 旋转角相同，退化回 1D RoPE
