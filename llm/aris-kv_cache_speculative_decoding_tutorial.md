# KV-Cache与投机解码 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[kv-cache-speculative-decoding-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/kv_cache_speculative_decoding_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 KV cache + Speculative Decoding** — 一页拿下面试核心要点（详见后文 §2–§9 推导）。

1. **KV cache 公式**：单 sample 显存 $= 2 \cdot L_\text{ctx} \cdot N_\text{layers} \cdot N_\text{kv\_heads} \cdot d_\text{head} \cdot \text{bytes}$，"2" 来自 K+V。LLaMA-3-70B（GQA, $H_\text{kv}=8$）4K context fp16 ≈ **1.25 GB/sample**——这就是为什么不用 MHA。

2. **Prefill vs Decode 不对称**：prefill 处理整段 prompt（$O(L^2)$ FLOPs，compute-bound）；decode 一次生成 1 token（每 step $O(L)$ FLOPs，但要读全部 KV，**memory-bandwidth-bound**）。这条不对称解释了一切现代 inference 系统设计。

3. **PagedAttention**（Kwon et al., SOSP 2023, vLLM）：把 KV cache 切成 page，用 block table 解 fragmentation；显存利用率从 ~70% 提升到 ~96%。

4. **Continuous batching**（Orca, Yu et al., OSDI 2022）：iteration-level scheduling，请求完成不等待整 batch，配合 PagedAttention 是 vLLM 的两根支柱。

5. **MQA → GQA → MLA**：MQA（Shazeer 2019）极端共享 K/V，质量略掉；GQA（Ainslie et al., EMNLP 2023）$G$ 组折中；MLA（DeepSeek-V2, May 2024）low-rank latent $c_t^{KV}$ + **decoupled RoPE**——RoPE 不能直接吸进 latent compression，必须留一个独立小维度 $d_\text{head}^R$ 携带位置。

6. **Speculative Decoding 核心**（Leviathan et al., ICML 2023; Chen et al., 2023）：用小 draft 模型 $q$ 提议 $K$ 个 token，target 模型 $p$ **一次 forward 并行验证**；rejection sampling 保证输出分布与 $p$ 完全等价（exact，不是近似）。

7. **接受概率公式**：单 token 接受率 $\alpha = \mathbb{E}_{x \sim q}[\min(1, p(x)/q(x))]$；期望生成 token 数 $E[\tau] = \dfrac{1-\alpha^{K+1}}{1-\alpha}$（$K$ 是 draft 长度，含最后 bonus token）。

8. **Medusa / EAGLE / Lookahead**：Medusa（Cai et al., ICML 2024）多头 + 静态 tree attention，默认走 typical acceptance（不严格保证 exact，可换标准 rejection sampling 换取 exact）；EAGLE/2/3（Li et al., 2024-2025）特征级 draft + 动态 tree，走标准 rejection sampling（保 exact）；Lookahead Decoding（Fu et al., ICML 2024）是 Jacobi iteration + n-gram pool 验证，机制上不是同一套 rejection-sampling $\alpha$ 框架——**三者的 draft 生成和 verification 机制均不同**。

---

## §10 25 高频面试题

按难度分 L1（必会）/ L2（进阶）/ L3（顶级 lab）。所有题点开看答案 + 易踩坑。

### L1 必会题（任何 inference / serving 岗位都会问）

<details>

<summary>Q1.KV cache 公式是什么？</summary>

- 单 sample：$2 \cdot L_\text{ctx} \cdot N_\text{layers} \cdot N_\text{kv\_heads} \cdot d_\text{head} \cdot \text{bytes}$
- "2" 来自 K + V
- $N_\text{kv\_heads}$：MHA = $H$, MQA = 1, GQA = $G$
- LLaMA-3-70B（GQA, $H_\text{kv}=8$）@4K fp16 ≈ 1.25 GB/sample

写 $H$（Q heads）；忘乘 2；忘 $L_\text{ctx}$ 是当前长度不是 max length。

</details>

<details>

<summary>Q2.为什么训练时不用 KV cache？</summary>

- 训练时所有位置同时算（teacher forcing，已知 ground truth）
- 没有"先有部分序列、再 append 新位置"这个时序
- KV cache 是**推理专属**优化

把 KV cache 当作通用优化用在训练。

</details>

<details>

<summary>Q3.Prefill 和 decode 阶段的瓶颈分别是什么？</summary>

- Prefill：$O(L^2)$ attention FLOPs，**compute-bound**
- Decode：每步只算 1 个 token，但要读完整 cache + weights，**memory-bandwidth-bound**
- Arithmetic intensity 极低 → GPU FLOPs 利用率往往 < 10%

说"decode 也是 compute-bound"——错。Decode batch 太小时 GPU 大部分时间在等内存。

</details>

<details>

<summary>Q4.MQA / GQA / MHA 区别？</summary>

- MHA：$H$ 个 K/V head（同 Q）
- MQA：所有 Q head 共享 **1** 组 K/V
- GQA：$H$ 个 Q head 分 $G$ 组（$1<G<H$），每组共享 K/V
- 主要省的是 **KV cache 显存 + 显存带宽**，不省 Q projection

以为 MQA 省的是 Q 计算；说 GQA 质量"基本不掉"过于绝对。

</details>

<details>

<summary>Q5.Speculative decoding 公式？</summary>

- Draft $q$ 提议 $K$ 个 token，target $p$ 一次 forward verify
- 每位置接受概率 $r = \min(1, p(\tilde x)/q(\tilde x))$
- 整体接受率 $\alpha = \sum_x \min(p(x), q(x))$
- 期望生成 $E[\tau] = \dfrac{1 - \alpha^{K+1}}{1 - \alpha}$

说 "spec decoding 是近似采样"——错，是 **exact**（rejection sampling 保证）。

</details>

<details>

<summary>Q6.PagedAttention 解决什么？</summary>

- 朴素 KV cache 必须连续大 tensor，预分配 max length → internal fragmentation
- 不同 request 长度不一 → external fragmentation
- 显存利用率仅 ~70%
- PagedAttention：切 page + block table，利用率提升到 ~96%
- 支持 prefix sharing（COW）

说 PagedAttention 减少 attention FLOPs——错，FLOPs 不变；它优化的是**显存利用率 + 并发请求数**。

</details>

<details>

<summary>Q7.Continuous batching 是什么？</summary>

- 调度粒度从 request 改成 iteration（每 forward 都重检查 batch）
- 完成的请求立即踢出，腾出 slot 给新请求
- 提出：Orca (Yu et al., OSDI 2022)
- 缩短平均等待时间，提高 GPU 利用率
- vLLM = Orca continuous batching + PagedAttention

以为 continuous batching 是把不同长度 sequence 都 pad 到最长——那是 static batching 的老做法。

</details>

<details>

<summary>Q8.Draft 模型怎么选？</summary>

- 大小：典型 target / 30 - target / 10（如 70B target + 7B draft）
- 同 tokenizer、同 vocab（否则 rejection sampling 算不出 $p/q$）
- 同 prompt format / 同 RLHF 后训练（否则 distribution gap 大，$\alpha$ 低）
- 经验：$\alpha \in [0.5, 0.8]$，太低就别上 SD

选 draft 太大（如 target / 3）；或选不同 tokenizer。

</details>

<details>

<summary>Q9.KV cache 量化最常见做法？</summary>

- FP8（H100 原生支持）几乎无损
- INT8 per-token quant 也可以接受
- INT4 / INT2（KIVI, KVQuant）需要精细 outlier 处理
- KIVI 的关键：**K per-channel, V per-token** 不对称量化

把 K 和 V 用同一个 quant 方案——容易掉点；K 和 V 的 outlier 分布不一样。

</details>

<details>

<summary>Q10.Prefix caching 是什么？</summary>

- 多个请求共享同一段 prompt 前缀（system prompt、few-shot）
- 用 hash(prefix) 索引 page 池，命中跳过 prefill
- 配合 COW 处理后续分叉
- ChatGPT 这种 system-prompt heavy 服务命中率 90%+

以为 prefix caching = prompt 全部缓存——只 cache prefix；用户特定部分还要 prefill。

</details>

### L2 进阶题（research-oriented / inference 系统岗）

<details>

<summary>Q11.推 spec decoding 的接受概率 $\alpha$，并解释它为什么保证 exact sampling。</summary>

- 设 draft $\tilde x \sim q$，接受规则 $r = \min(1, p/q)$
- $\Pr[\text{accept} \land X=x] = q(x) \cdot \min(1, p(x)/q(x)) = \min(p(x), q(x))$
- $\alpha = \sum_x \min(p, q) = 1 - \tfrac{1}{2} \|p - q\|_1$
- 被拒后从残差 $p'(x) = \max(0, p-q) / (1-\alpha)$ 重采
- 总输出概率 $= \min(p, q) + \max(0, p-q) = p(x)$ ∀x
- 所以每位置等价于直接从 $p$ 单 step 采样

只写 accept 部分，漏 reject 残差分布；忘证 $\min + \max = p$；说 spec 是近似。

</details>

<details>

<summary>Q12.MLA 为什么必须 decoupled RoPE？详细推导。</summary>

- 朴素 MLA absorb trick：$q^\top k = c_q^\top (W^{UQ\top} W^{UK}) c_{kv}$，中间是常数矩阵 $\tilde W^{QK}$，预乘即可
- 加 RoPE 后：$q^{R\top} k^R = c_q^\top W^{UQ\top} R_{s-t} W^{UK} c_{kv}$
- 中间块 $W^{UQ\top} R_{s-t} W^{UK}$ **依赖相对位置 $(s-t)$**，不能预乘
- absorb 失效 → cache 是省了，compute 退化回 MHA
- 解：拆出独立 RoPE 通道 $k^R \in \mathbb{R}^{d_r}$（所有 head 共享），content 通道走 absorb，RoPE 通道走标准 dot product
- 总 cache：$d_c + d_r$ per token

只说"加 RoPE 出问题"不展开；不知道 RoPE 的 $R_t^\top R_s = R_{s-t}$ 性质；不知道 $k^R$ 是所有 head 共享。

</details>

<details>

<summary>Q13.Continuous batching 在 prefill + decode 混跑时怎么处理？</summary>

- Prefill 一次性算长段，FLOPs 大；decode 单 token，FLOPs 小
- 直接混 batch 会让 decode 等 prefill 长时间（HOL blocking）
- Sarathi-Serve 的 **chunked prefill**：把长 prefill 切等大小 chunk
- 每 iteration coalesce 一个 prefill chunk + 多个 decode token
- stall-free schedule：保证 decode 永远跟着跑

以为 prefill 必须一次跑完；忘记 Sarathi-Serve 是 OSDI 2024。

</details>

<details>

<summary>Q14.Tree attention（Medusa / EAGLE 用）的 mask 怎么写？</summary>

- 把 tree 节点按 BFS 拍平成线性序列 $[t_0, \dots, t_M]$
- $\mathcal A(i)$ = 节点 $i$ 的祖先（含自身）
- attention mask $M[i, j] = 1 \iff j \in \mathcal A(i)$
- 即"causal 在树上的推广"
- 用于一次 forward 同时 verify tree 里所有路径

写成 lower-triangular causal mask（只适用 chain，不适用 tree）；忘记把 mask 的形状从 $[L,L]$ 推广。

</details>

<details>

<summary>Q15.spec decoding 的实际加速公式？为什么 draft 太大会反效果？</summary>

- $\text{speedup} = E[\tau] / (1 + Kc)$，$c = T_q/T_p$
- $E[\tau] = (1-\alpha^{K+1})/(1-\alpha)$
- 分母里 $Kc$ 是 $K$ 次 draft forward 的开销
- 若 $c$ 太大（draft 太大），即使 $\alpha$ 高也会被分母吃光
- 极端：$c=1$ 时 speedup ≤ 1（draft 跟 target 一样慢）

只写 $E[\tau]$ 不算 draft 开销；漏 bonus token 那一项。

</details>

<details>

<summary>Q16.Self-speculative decoding 和普通 spec decoding 区别？</summary>

- Self-spec：draft 是 target 自己跳过一个精选层子集（见 §8.6，非固定深度的 early exit）
- 不需要独立 draft model，零额外训练
- 但 draft 与 target 高度相关，$\alpha$ 通常较高
- 加速一般 1.5-2×（不如 EAGLE 但更省事）
- 论文：Zhang et al. 2024（"Draft & Verify"）

说必须有额外训练；混 self-spec 和 layer skipping inference（后者不是 exact）。

</details>

<details>

<summary>Q17.KV cache eviction 和 sparse attention 怎么影响 spec decoding？</summary>

- 长 context 下 KV cache 才是带宽瓶颈，weights 已经被 prefill 摊销
- 这时 draft 用 sparse / sliding window KV（StreamingLLM 风格）能跑得快
- target 用完整 cache 做 verify 保证 exact
- 代表：MagicDec、TriForce（hierarchical：小 draft → sparse target → full target）
- 收益：长 context 下 vanilla SD 失效（1.1×），MagicDec 能保 2×+

把 sparse KV 当成 lossy 近似（实际只用于 draft，verify 时全 cache 仍 exact）。

</details>

<details>

<summary>Q18.Medusa 用 typical acceptance 替代 rejection sampling，损失了什么？</summary>

- 严格意义上**丢了 exact sampling**——不再保证输出分布等于 target
- 但 typical acceptance 用 target 自身的 typical set 阈值约束，质量基本不掉（论文实测和 base 模型 score 接近）
- 如要严格 exact，可把 verification 换成标准 rejection sampling（Leviathan/Chen 公式）
- **Medusa-1 vs Medusa-2 区分点是训练范式**：Medusa-1 冻 backbone 只训 head；Medusa-2 联合训 backbone + head；二者默认都用 typical acceptance

把 Medusa-1 / Medusa-2 的区别说成 "exact vs 非 exact"（错——它们是训练范式不同）；说 Medusa 完全等价于 target sampling。

</details>

<details>

<summary>Q19.EAGLE 和 Medusa 的核心差异？</summary>

- Medusa：多 head **直接预测未来 token**，independent (不 autoregressive)
- EAGLE：draft 在 **feature space autoregressive**（前一步 hidden + 前 token → 下一步 hidden + token）
- EAGLE 更准（feature 信息丰富），但需要训 draft（含 transformer 层）
- EAGLE-3 进一步抛 feature regression，直接 token + 多层 fusion + training-time test
- 实测 EAGLE > Medusa 接受率，但 Medusa 部署更简单（参数更少）

把 EAGLE 当 Medusa 的小改进；说"EAGLE 也是多 head"——错，EAGLE 是 1 个 mini-transformer。

</details>

<details>

<summary>Q20.PagedAttention 和 FlashAttention 关系？</summary>

- FlashAttention：attention kernel 内部 SRAM tiling + online softmax，**单 kernel** 内优化（避免 materialize $L^2$ scores）
- PagedAttention：把 KV cache 切 page，按 page table 间接寻址；**memory layout** 优化
- 二者正交，可以叠加：vLLM 用 paged + flash 思路写 paged attention kernel
- 区分点：FlashAttention 减 HBM IO；PagedAttention 减显存碎片

混淆二者；以为 PagedAttention 是 attention 算法变体（实际只是内存管理 + 配套 kernel）。

</details>

### L3 顶级 lab 题（最严苛级别）

<details>

<summary>Q21.推 spec decoding 的 acceptance $\alpha$ 完整证明，并解释 sampling 等价性如何推广到 temperature / top-p。</summary>

- 单 token：$\Pr[X=x] = q(x) \min(1, p(x)/q(x)) + (1-\alpha) p'(x)$，代入 $p'$ 得 $\min(p,q) + \max(0, p-q) = p$
- $\alpha = \sum_x \min(p, q) = 1 - \tfrac{1}{2}\|p-q\|_1$
- 等价于 TV distance 的连接公式
- 关键原则：rejection sampling 的等价性只依赖于 "draft proposal 分布 $\tilde q$" 和 "target 目标分布 $\tilde p$" 各自有效。**只要把 $p, q$ 在公式里替换成 sampler 处理后的 $\tilde p, \tilde q$，整套等价性照旧**
- Temperature $T$：常见做法是 $\tilde p_T(x) \propto p(x)^{1/T}$ 和 $\tilde q_T(x) \propto q(x)^{1/T}$；把它们代进 $\alpha, p'$ 公式即可
- Top-p：把 $p$ truncate + renorm 到 $p$ 自己的 top-p 集合得到 $\tilde p$，**draft proposal 分布** $\tilde q$ 是 draft 实际采样的那个分布；只要二者都是合法分布，rejection 都 exact
- 实践中 draft 用与 target 相同的 sampler 是惯例（让 $\tilde q$ 接近 $\tilde p$ 提高 $\alpha$），但不是数学必需——draft 完全 greedy 也合法，只是 $\alpha$ 会暴跌
- 多 token：每位置 $\alpha_i$ 用对应的 $\tilde p_i, \tilde q_i$；bonus token 用 $K+1$ 位置的修正 logits（经 sampler 处理后）直接采

只写单 token 等价；把"draft 必须用同 sampler"误说成数学必需（实际只是高 $\alpha$ 的策略）；忽略 bonus token。

</details>

<details>

<summary>Q22.MLA 的 absorb trick 完整数学推导：为什么 inference 时不用还原 K/V？</summary>

- KV cache：$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}$
- K, V 升投影：$k_t^{(i)} = W^{UK,(i)} c_t^{KV}, v_t^{(i)} = W^{UV,(i)} c_t^{KV}$
- Q 同理：$q_t^{(i)} = W^{UQ,(i)} c_t^Q$
- attention 分数（无 RoPE）：$(q_t^{(i)})^\top k_s^{(i)} = (c_t^Q)^\top \underbrace{W^{UQ,(i)\top} W^{UK,(i)}}_{\tilde W^{QK,(i)}} c_s^{KV}$
- $\tilde W^{QK,(i)}$ 形状 $d_c' \times d_c$，**与 (t, s) 无关**，加载模型时预乘
- inference 时直接 $(c_t^Q)^\top \tilde W^{QK,(i)} c_s^{KV}$，**完全不算 $k_s$**
- 类似地 attention output：$\text{out}^{(i)} = \sum_s w_s v_s^{(i)} = (\sum_s w_s c_s^{KV})^\top W^{UV,(i)\top}$
- 把 $W^{UV,(i)}$ 吸进 $W^O$：$W^O_\text{absorbed} = W^O (\text{blockdiag}(W^{UV,(i)}))$
- 结论：cache 只 latent，compute 在 latent 空间；absorb 省的是 HBM 读取/物化 K/V 的带宽，FLOPs 因 $d_c$（512）常大于 $d_\text{head}$（128）反而可能略增，但换来的带宽节省对 decode 这种 bandwidth-bound 阶段是划算的（**不是"省 cache 不增 compute"**）

朴素地说"还原 K/V 不就行了"——还原后 compute 退化到 MHA；以为 absorb 只能用在 inference 是因为"不能 backprop"——错，absorb 后的形式完全可微，训练时同样能用；真正原因是训练时权重每步都在更新，若用 absorb 形式就得每步重新计算组合矩阵 $\tilde W^{QK}$，这笔额外开销并不经济，不如直接用未吸收的两段投影做 forward/backward——absorb 的收益来自推理时权重固定，组合矩阵只需算一次并长期复用。

</details>

<details>

<summary>Q23.解释为什么 MLA 在加 RoPE 时必须分离一个独立通道，能不能用别的方式保住 absorb？</summary>

- 核心：RoPE 把 $R_{s-t}$ 塞进 $\tilde W^{QK,(i)}$，破坏"常数矩阵"性质
- 替代方案 1：把 RoPE 直接放在 latent $c^{KV}$ 上——但 latent 维度小，旋转语义不对（RoPE 设计在 head dim 上配对 sin/cos）
- 替代方案 2：用 ALiBi（直接加 bias 不旋转）——但破坏 LLaMA-3 兼容预训练
- 替代方案 3：放弃 absorb，每 step 还原 K/V——compute 退化到 MHA
- DeepSeek-V2 的选择：**decoupled RoPE 通道 $d_r=64$ 所有 head 共享**，cache 增量约 **12.5%**（$d_r/d_c = 64/512$；按总量算约 11.1%，$d_r/(d_c+d_r) = 64/576$），并非约 5%，content 通道保持 absorb
- 妙处：这个独立通道在所有 head 间共享 $k_t^R$，是"省 cache 的最后一公里"

说"加 RoPE 不影响 MLA"——错；不知道 decoupled 通道是 head-shared。

</details>

<details>

<summary>Q24.长 context（128K+）下，为什么 vanilla speculative decoding 收益坍塌？怎么救？</summary>

- Vanilla SD 的收益假设：target 单次 verify forward 能摊销 weight + KV cache 的 HBM 读取（$K+1$ 个候选位置共享同一次批量矩阵乘法，cache 只读一次）——这个摊销假设本身在长 context 下依然成立，**不是**它收益坍塌的原因
- 真正的问题在 **draft 侧**：若 draft 模型维护和 target 同样长度的完整 KV cache，draft 生成 $K$ 个 token 是**串行自回归**过程，每一步都要单独重新读一遍自己的 $O(L_\text{ctx})$ cache，总开销正比 $K \cdot L_\text{ctx}$——这才是长 context 下拖累加速比的主因
- 另外，"HBM 带宽（cache）取代 weight 加载成为瓶颈"对 GQA 模型不该在 128K 就成立：LLaMA-2/3-70B（GQA, $H_\text{kv}=8$）在 128K context 下 KV cache 约 **40 GiB**，仍明显小于 **140 GB** 权重，要到远超 128K（约 **450K+ token**）才会反超
- 救法 1：**MagicDec** — draft 用 sparse KV（StreamingLLM），把 draft 侧的串行 $O(L_\text{ctx})$ 读取压成常数窗口
- 救法 2：**TriForce** — 三层：小 LM → target+sparse cache → target+full cache
- 救法 3：合并 KV cache 压缩（H2O eviction）+ SD：cache 小了 vanilla SD 也救活

只说"长 context spec decoding 不 work"，不知道为什么；不知道 MagicDec/TriForce 是 2024 长 context SD 的 SOTA。

</details>

<details>

<summary>Q25.设计 LLM serving 系统时，决定上什么优化的 mental model 是什么？</summary>

- **Step 1 测 workload**：prompt 长度分布、生成长度分布、QPS
- **Step 2 按瓶颈选优化**：(a) 显存不够装 batch → PagedAttention + prefix caching + KV 量化；(b) 长 prefill 卡 decode → Sarathi-Serve chunked prefill；(c) 短 batch decode 带宽 bound → spec decoding（小 batch 收益最大）；(d) 长 context 带宽 bound → MagicDec / TriForce；(e) 跨请求 prompt 重复 → prefix caching + COW
- **Step 3 注意互动**：SD + large batch 收益降（large batch 已经 compute-bound）；PagedAttention + SD cache rollback 用 page table 改指针；KV 量化 + SD **不要求** draft/target 用一致量化方案（如 target FP8、draft INT4 都合法，exactness 只依赖各自真实计算出的 $p, q$，与 §7.5 sampler 不必一致同理）——量化方案不同只会拉大 $p, q$ 差距、降低接受率 $\alpha$，不影响输出分布的 exactness
- **Step 4 监控 metrics**：tokens/sec, p95 TTFT, p95 TPOT, GPU utilization
- 关键 trade-off：throughput vs latency，SD 偏 latency 改善，continuous batching 偏 throughput

只罗列技术名词不讲触发条件；不知道 SD 在 large batch 下收益降；忽略真实 workload 测量。

</details>

## §A 附录：参考实现 + Sanity Check

### A.1　组件汇总

参考 from-scratch 实现包含：

- `NaiveCachedAttention` —— 单层 MHA + KV cache append
- `PagedKVCache` —— page table + COW 共享 sketch
- `MQA_GQA_Attention` —— 三合一通用版本
- `MLAAttention` —— 含 decoupled RoPE 通道
- `speculative_decode` —— exact 数学等价的 spec loop（含 rejection + bonus token）

### A.2　Sanity check 期望输出

```
[a] naive cache append    prefill (1,16,128) → decode 8 token  ✓
[b] MQA/GQA/MHA shape + cache 大小一致                          ✓
[c] MLA cache = d_c + d_r 元素                                  ✓
[d] spec decode rejection: 100k 样本估 α 与理论值差 < 1%        ✓
[e] spec decode 输出 vs target 直接采样: TV < 0.01              ✓
[f] paged cache COW: ref_count + share 正确                    ✓
```

### A.3　主要参考文献

- **KV / Serving 系统**
  - Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023.
  - Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022.
  - Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024.

- **Attention 变体**
  - Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", arXiv:1911.02150, 2019 (MQA).
  - Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023.
  - DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", arXiv:2405.04434, May 2024 (MLA).

- **KV cache 量化**
  - Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024.
  - Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024.

- **Speculative decoding**
  - Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023.
  - Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling", arXiv:2302.01318, 2023 (DeepMind).
  - Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", ICML 2024.
  - Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024.
  - Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees", EMNLP 2024 (arXiv:2406.16858).
  - Li et al., "EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test", arXiv:2503.01840, 2025.
  - Fu et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding", ICML 2024.
  - Miao et al., "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification", ASPLOS 2024.
  - Sun et al., "TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding", arXiv:2404.11912, 2024.
  - Sadhukhan et al., "MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding", arXiv:2408.11049, 2024 (ICLR 2025).
  - Zhang et al., "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding", ACL 2024.

代码与公式均经独立 reviewer 静态检查（gpt-5.5 xhigh，跨模型），数学等价性论证通过。
