# Attention机制 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[attention-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/attention_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **7 句话搞定 attention** — 一页拿下面试核心要点（详见后文 §2–§9 推导）。

1. **公式**：$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V$。

2. **为什么除 √d_k**：若 $q_i, k_i \sim \mathcal{N}(0,1)$ 独立，$q\cdot k$ 方差 $= d_k$；除 $\sqrt{d_k}$ 把方差拉回 1，避免 softmax 饱和。

3. **Multi-Head**：把 $D$ 拆成 $H$ 个 head，每个 head 在不同 subspace 独立做 attention，concat 后 $W_o$ 投影。**固定 $D$ 且 $d_k=D/H$ 时，标准 MHA 参数量 $\approx 4D^2$（不随 $H$ 变）；MQA/GQA 下 K/V 投影变小**。

4. **Self vs Cross**：Self 的 Q/K/V 同源；Cross 的 Q 来自 query stream，K/V 来自 context stream（encoder output / image tokens / text embedding）。

5. **Causal mask vs Padding mask**：前者用下三角阻断未来；后者用 `[B,1,1,L_k]` 屏蔽 padding 列。

6. **复杂度**：$O(B H L^2 d_k)$ 时间，$O(B H L^2)$ score 显存——长序列瓶颈在二次项。

7. **易踩坑**：全 masked row → softmax NaN；FP16 下 $QK^\top$ 可能 overflow；attention weight ≠ 因果解释。

---

## §10 25 高频面试题

codex (gpt-5.5 xhigh) 作为顶级 lab 面试官视角列的，按难度分 3 档。每题点开看答案要点 + 易踩坑。

### L1必会题（任何 ML 工程岗都会问）

<details>

<summary>Q1.Attention 公式是什么？</summary>

- $\text{softmax}(QK^\top / \sqrt{d_k}) V$

- Softmax over keys 维度

- 输出是 value 的加权和

把 softmax 维度写到 query 维。

</details>

<details>

<summary>Q2.为什么除以 √d_k？</summary>

- 若 $q_i, k_i$ 独立零均值单位方差

- Dot product 方差约 $d_k$

- 缩放后方差回到 1，避免 softmax 饱和

只说"防止数值太大"，不给方差推导。

</details>

<details>

<summary>Q3.Q/K/V 分别代表什么？</summary>

- Q 是检索请求

- K 是匹配索引

- V 是被聚合内容

说 Q/K/V 是三份不同输入；self-attn 中它们同源但投影不同。

</details>

<details>

<summary>Q4.Multi-head 为什么有用？</summary>

- 不同子空间建模不同关系

- 多种位置/语义模式并行

- Concat 后再融合

说 head 越多一定越好。实际 $d_k$ 太小会限制表达力。

</details>

<details>

<summary>Q5.MHA 参数量随 head 数怎么变？</summary>

- 固定 $D$ 且 $d_k = D/H$（标准 MHA）

- $W_q + W_k + W_v + W_o$ 共 $4D^2$，**不随 $H$ 变**

- 但若用 MQA/GQA，K/V 投影矩阵会变小（$H_\text{kv} < H$ 个 head）

- 这就是为什么"head 数是免费的"在标准 MHA 下成立，但在 MQA/GQA 下有显存收益

误以为 head 多参数也线性多 H 倍；或忘了 MQA/GQA 改变了 K/V 投影维度。

</details>

<details>

<summary>Q6.Self-attention 和 cross-attention 区别？</summary>

- Self: Q/K/V 同源

- Cross: Q 来自 target，K/V 来自 context

- Cross 常用于 encoder-decoder、diffusion text conditioning

只说"cross 有两个输入"，不说明 Q 与 KV 来源。

</details>

<details>

<summary>Q7.写 causal mask 怎么做？</summary>

- `torch.tril(torch.ones(L, L, dtype=torch.bool))`

- 明确说 True=keep 还是 True=mask（API 间不一致）

- Broadcast 到 `[B, H, L, L]` 或让框架隐式 broadcast

上下三角写反；忘记 broadcast 维度对齐。

</details>

<details>

<summary>Q8.Padding mask mask 的是哪一维？</summary>

- 通常 mask key/value 列（让 padding 位置概率为 0）

- Shape 可为 `[B, 1, 1, L_k]` 对齐 head + query 维

- 注意：mask key 列**不足以**让 padded query 输出为 0；padded query 行通常用 loss ignore / output zeroing / packed sequence 等手段单独处理

以为 padding mask 一手包办——它只防止"看到 padding"，但 padded query 自己的输出还需要外部处理。

</details>

<details>

<summary>Q9.Attention 复杂度？</summary>

- 时间 $O(B H L_q L_k d_k) = O(B L^2 D)$

- Score memory $O(B H L_q L_k)$

- 长序列瓶颈是二次项

只说 $O(n^2)$，漏 head 和 hidden 维。

</details>

<details>

<summary>Q10.Attention dropout 放在哪里？</summary>

- 放在 softmax weights 之后、与 V matmul 之前

- Training 才启用，eval 时关闭

- Dropout 后权重行和不一定是 1（期望意义上为 1）

Sanity check 时还要求 dropout 后 row-sum = 1（错的）。

</details>

### L2进阶题（research-oriented 岗位）

<details>

<summary>Q11.手推 softmax 的 Jacobian。</summary>

- $y_i = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$

- $\dfrac{\partial y_i}{\partial x_j} = y_i (\delta_{ij} - y_j)$

- 矩阵形式：$J = \text{diag}(y) - yy^\top$

只写对角项，漏交叉项 $-y_i y_j$。

</details>

<details>

<summary>Q12.用 -∞ 做 mask 有什么坑？</summary>

- 正常情况 masked 位置 softmax 概率为 0 ✓

- **全 masked row → softmax 输出 NaN**（$0/0$）

- 修复路径：先避免 all-`-inf` 行（临时放开），softmax 后把该行 weights 与 output 强制清 0，并确保该 query 不进入 loss / 残差累积

- Fused kernel / API 对 sentinel 数值有约束；fp16 下用一个 dtype-safe 大负数（如 `finfo(dtype).min`）更稳

以为 -inf 永远安全；或只在 softmax 后清 0 而不防 NaN。

</details>

<details>

<summary>Q13.Log-sum-exp trick 是什么？</summary>

- softmax 前先减 max(logits)，等价不改变概率

- 防止 $e^{x_i}$ overflow（fp32 max ≈ 3.4e38，但 $e^{100}$ 已经溢出）

- $\log \sum_j e^{x_j} = m + \log \sum_j e^{x_j - m}$ 其中 $m = \max_j x_j$

忘了 $QK^\top$ overflow 可能发生在 softmax 之前（matmul 累加阶段）。

</details>

<details>

<summary>Q14.PyTorch nn.MultiheadAttention 的 in_proj_weight 顺序？</summary>

- Shape `[3D, D]`

- 顺序：**Q, K, V**（cat dim=0）

- Linear weight 是 `[out, in]`，所以 `cat([W_q.weight, W_k.weight, W_v.weight], dim=0)`

拼成 K/Q/V 或转置 weight。

</details>

<details>

<summary>Q15.attn_mask 和 key_padding_mask 区别？</summary>

- `attn_mask` 控制 query-key pair 级别（一般是 causal）

- `key_padding_mask` 控制 key token 整体可见性（一般是 padding）

- 两者 bool 语义：`nn.MultiheadAttention` 是 **True = mask out**；`F.scaled_dot_product_attention` 的 bool mask 是 **True = keep**（相反！）

- 同时用时，在 mask-out 语义下合并是 **OR**（任一为 True 就屏蔽）；在 keep 语义下是 AND（两者都 True 才保留）

不查 API 文档直接套用 True/False；或把 AND/OR 搞反。

</details>

<details>

<summary>Q16.Cross-attention 中 L_q 和 L_k 能否不同？</summary>

- 可以——这正是 cross-attention 的常态

- Scores shape 是 $[L_q, L_k]$

- Mask 必须对齐 key 维度

默认 cross-attn 必须等长。

</details>

<details>

<summary>Q17.为什么需要 output projection W_o？</summary>

- 融合不同 head 的输出

- 映射回 $d_\text{model}$ 与残差相加

- 给模型学习 head 间组合（不是简单 concat）

以为 concat 后已经结束。

</details>

<details>

<summary>Q18.Pre-norm vs post-norm 对 attention block 的影响？</summary>

- Pre-norm：`x + Attn(LN(x))`，深层训练更稳定，gradient 沿残差路径相对保持

- Post-norm：`LN(x + Attn(x))`，Vaswani 原论文用，超深时需 warmup / careful init

- **多数 decoder-only LLM 用 pre-norm（常配 RMSNorm 变体）**，但具体架构有例外

把 norm 位置当纯工程细节，或说"现代 LLM 都用 pre-norm" 太绝对。

</details>

<details>

<summary>Q19.Attention weight 等于"模型解释"吗？</summary>

- 可视化有参考价值（注意聚焦位置）

- 但 **不等于因果解释**

- Value 路径和后续层都会改变实际贡献

- Jain & Wallace "Attention is not Explanation" (2019)

直接把高 attention 权重当成"模型理由"。

</details>

<details>

<summary>Q20.Mixed-precision attention 注意什么？</summary>

- **fp32 accumulation**：matmul 累加 / softmax 关键步骤在 fp32 完成，再 cast 回低精度

- **Softmax max-subtraction**（log-sum-exp）防 exp overflow，PyTorch `F.softmax` 内部已做

- **Mask sentinel**：fp16 下用 `torch.finfo(dtype).min` 而非字面 -inf

- **BF16 vs FP16**：BF16 动态范围与 fp32 相近，更适合 attention；fp16 表数范围窄，QK^T 易 overflow

- **Fused kernels**（FlashAttention, `F.scaled_dot_product_attention`）内置 kernel-level 稳定化，比手写 naive 安全

FP16 下直接手写 naive attention 不做 fp32 accumulation。

</details>

### L3高级变体（顶级 lab / diffusion 方向）

<details>

<summary>Q21.KV cache 如何优化自回归解码？</summary>

- 解码第 $t+1$ 步时，只为新 token 算 $Q$（1×D）

- 复用历史 $K, V$（已经在 cache 里），append 新 $k_{t+1}, v_{t+1}$

- 每步 attention 从 $O(t^2)$ 变 $O(t)$，整段生成从 $O(L^3)$ 变 $O(L^2)$

- Per-sample 显存：$L_\text{ctx} \cdot n_\text{layers} \cdot 2 \cdot H_\text{kv} \cdot d_\text{head} \cdot \text{bytes}$（MQA/GQA 下 $H_\text{kv} \ll H$）

说 KV cache 减少训练成本——错。它只用于 autoregressive inference。另：cache 是 KV heads 数量，不是 Q heads。

</details>

<details>

<summary>Q22.MQA 和 GQA 解决什么？</summary>

- MQA：多个 Q head 共享一组 K/V（K/V 只有 1 个 head）

- GQA：折中，K/V 有 $G$ 组（$1 < G < H$）

- 主要收益：**decode 时 KV cache 显存 + 显存带宽**（大幅降低）

- 同时也减少 K/V projection 的参数和计算（K/V 投影矩阵变小），**但不减少 Q / O projection**

- 质量影响：通常 **GQA 质量损失小于 MQA**，具体取决于模型规模和训练方式（LLaMA-2 70B / LLaMA-3 / Mistral / Qwen-2 都用 GQA）

以为减少了 Q projection；或说"GQA 质量基本不掉"过于绝对。

</details>

<details>

<summary>Q23.FlashAttention 核心 trick？</summary>

- **Block tiling**：把 $Q, K, V$ 切成 SRAM-sized block，分批 load

- **Online softmax**：增量维护 running max $m$ 与 running sum $\ell$，**避免 materialize** 完整 $L \times L$ scores / probs 矩阵到 HBM

- **Recompute on backward**：反向时根据 saved $m, \ell$ 重算 scores，不存中间结果

- 关键：**IO-aware exact attention**（数学等价，不是近似）

- HBM IO 复杂度约 $O(L^2 d^2 / M + Ld)$，对比标准 attention 的 $O(L^2 + Ld)$ HBM traffic——长序列下显著减少 IO（不是 FLOPs）

说它是近似 attention（如 Performer / Linformer）——错，FlashAttn 是 exact；或把 IO 复杂度和 FLOPs 复杂度混淆。

</details>

<details>

<summary>Q24.RoPE, ALiBi, absolute position 的区别？什么是 attention sink？</summary>

- **Absolute**：位置向量加到 input embedding 上（Vaswani sinusoidal / GPT-2 learned）

- **RoPE**：对 $Q, K$ 做位置相关的旋转，保留**相对位置**信息（位置项只经由 $m-n$ 进入 $q_m^\top k_n$，但内容向量仍决定具体数值——不是只依赖位置）

- **ALiBi**：在 score 上加距离 bias $-m |i-j|$，自然外推

- **Attention sink**：训练好的 LLM 会让前 1-4 个 token（特别是 [BOS]）获得异常高的 attention，即使内容无关——softmax 强制和为 1，模型需要"垃圾位"。StreamingLLM 利用此现象做长序列推理。

把 attention sink 当成 padding / CLS token 的正常 attention 行为。

</details>

<details>

<summary>Q25.Attention 在 diffusion / latent diffusion 里怎么用？</summary>

- **U-Net latent tokens 作 Q**，text embedding 作 K/V，做 **cross-attention** 注入文本条件

- Self-attention 在每个 spatial resolution 内做（image patches × image patches）

- **CFG (Classifier-Free Guidance)**：两次 forward，差值放大 conditional 信号

- DiT (Diffusion Transformer)：把 U-Net 换成 pure Transformer，conditioning 通过 AdaLN / cross-attn / token-concat

- Video diffusion：空间 attn + 时间 attn + 时空 attn 的组合（长 video 是开放问题，$L \sim 10^5$）

说 diffusion 只靠卷积；或者只在 DiT 里才有 attention（错，U-Net 里也有大量）。

</details>

## §A 附录：完整 from-scratch 代码骨架

参考 from-scratch 实现（`code/mha.py`）包含：

- `MultiHeadAttention`——标准多头 self-attention（单个 fused `qkv` Linear 一次产出 Q/K/V，支持 `[N,N]` additive mask；无 NaN 全 mask 防护，无 cross-attention wrapper）；`__init__` 里 `assert embed_dim % num_heads == 0` 校验维度，不满足时抛 `AssertionError`（注意 `assert` 在 `python -O` / `PYTHONOPTIMIZE` 下会被整体移除，生产代码应改用显式 `raise ValueError(...)`）

- `make_causal_mask()`—— 生成上三角填 `-inf` 的因果 mask

- 3 个 sanity check：`sanity_check()`（对齐 `nn.MultiheadAttention`，diff < 1e-5）、`shape_check()`、`causal_check()`

实跑 sanity check 输出（PyTorch 2.x，CPU 几秒跑完）：

```
== Multi-Head Attention sanity ==
[sanity] max diff vs nn.MultiheadAttention = 5.96e-08
[sanity] PASS
[shape] input  x: (2, 1024, 768)
[shape] output y: (2, 1024, 768)
[causal] applied causal mask, output shape (1, 8, 32)

All checks passed.
```

代码经独立 reviewer 静态检查 + PyTorch 实跑 sanity check，与 `nn.MultiheadAttention` diff < 1e-5。

---

## 📜 Runnable Code

本 tutorial 的核心概念在 [`docs/tutorials/code/`](code/) 里有最小可跑的 PyTorch 实现：

- [`mha.py`](code/mha.py) — 标准 Multi-Head Self-Attention + causal mask + 跟 `nn.MultiheadAttention` 数值对齐验证
- [`axial_attention.py`](code/axial_attention.py) — H/W 轴向 attention + 复杂度对比表 + 感受野隔离测试

每个脚本默认 CPU 几秒跑完，自带 `assert` sanity check。完整说明见 [`code/README.md`](code/README.md)。
