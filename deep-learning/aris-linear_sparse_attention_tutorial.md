# 线性与稀疏注意力 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[linear-sparse-attention-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/linear_sparse_attention_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定高效注意力 / SSM / 稀疏注意力** — 一页拿下面试核心要点（详见后文 §1–§11 推导）。

1. **二次瓶颈**：softmax attention 训练 $O(L^2 d)$ 时间，decode 时 KV cache 随上下文线性增长（$O(L)$ 显存、每步 $O(L)$ 计算）。逃逸有三条路：**线性注意力**、**SSM / Mamba**、**稀疏注意力**。

2. **线性注意力**：用 kernel feature map $\phi$ 把 softmax 换成 $\phi(q)^\top\phi(k)$，再靠**矩阵乘结合律**把 $\phi(Q)\big(\phi(K)^\top V\big)$ 从 $O(L^2)$ 降到 $O(L)$。等价于一个**隐藏状态是矩阵** $S_t\in\mathbb{R}^{d_k\times d_v}$ 的 RNN：$S_t = S_{t-1} + \phi(k_t) v_t^\top$。

3. **SSM / Mamba**：连续状态空间 $h_t = A h_{t-1} + B x_t,\; y_t = C h_t$，经离散化（$\Delta$ + 零阶保持）得递推。**Selective SSM（S6）**让 $B, C, \Delta$ 随输入变化（数据相关门控），打破 LTI 换来内容选择能力；配硬件感知 selective scan。

4. **Mamba-2 / SSD**：State Space Duality —— 一个 selective SSM **等价于一种结构化掩码线性注意力**（通过 N-半可分矩阵 / N-semiseparable matrix，$N$ 为 SSM 状态维；秩 $\le 1$ 的"1-半可分"只是 $N=1$ 的退化特例）。这让 SSM 能用 matmul 密集的 **chunkwise** 算法在 tensor core 上高效训练。

5. **Delta rule（DeltaNet）**：更新是**改写**而非纯累加——$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$，那个 $(I-\beta k k^\top)$ 项先**擦掉旧关联**再写新值（在线最小二乘 / 误差修正）。Gated DeltaNet 再加一个标量/对角衰减门 $\alpha_t$ 做遗忘。

6. **Chunkwise parallel**：把序列切块，**块内**用二次的并行 attention，**块间**用 $O(1)$ 的递推状态跨块传递。这是让线性注意力 / SSM / DeltaNet 既能 $O(L)$ 推理、又能 matmul 并行训练的**核心桥梁**。

7. **稀疏注意力（可训练）**：保留 softmax 但只 attend 一个子集。**NSA**（DeepSeek，原生可训练 + 硬件对齐，三分支：compress + select + sliding window）；**MoBA**（Moonshot，MoE 式把 query 路由到 top-k key block）；**Lightning Attention**（MiniMax-01，IO-aware 线性）；**DeepSeek-V3.2 DSA**（轻量 lightning indexer 打分选 token）。

8. **推理杀手锏**：线性 / SSM 保持**固定大小**递推状态（序列长度上 $O(1)$）→ 常数显存 + 每 token $O(1)$ 解码；softmax 的 KV cache 随长度线性涨；稀疏减少 KV 访问/计算但仍存 KV。

9. **混合架构（hybrid）**：纯线性/SSM 的 **recall（联想回忆 / in-context copy）偏弱** —— recall–throughput 权衡。修法是在大量线性/SSM 层里**交错少数注意力层**（full / sliding / 门控 attention，具体类型和比例各模型不同）：Jamba、Hymba、Qwen3-Next、Kimi-Linear、MiniMax-01 都走这条路，兼顾吞吐与 recall。

---

## §10 工程实践与常见误区

- **线性注意力不是免费午餐**：它的 recall 弱是**固定大小状态的容量本质**，不是 bug。需要精确长程检索的任务（多跳 QA、长代码补全、大海捞针）上，纯线性/SSM 会掉点——这也是 hybrid 存在的根本理由。
- **"线性 = $O(N)$ 所以一定更快更好"是错的**：(1) 渐进复杂度低 ≠ 实际更快——常数和访存模式很重要，FlashAttention 高度优化后，短到中等序列上 softmax 往往**更快**；线性/SSM 的优势要到很长序列才显现。(2) 更快 ≠ 更好——recall 短板独立于速度。
- **chunk size 要调**：$C$ 太小→并行度低、块间递推占主导；太大→块内 $C^2$ 项变贵 + 显存涨。典型 $C\in[64,256]$，按序列长度 / 硬件调。
- **没有 softmax 归一化 → 数值稳定性要单独管**：纯累加状态会无界增长 / 量级漂移。实践靠**衰减门（$\gamma,\alpha_t$）**、**对状态或输出做 normalization**（RMSNorm / L2）、**$\beta,\Delta$ 的有界激活（sigmoid/softplus）**来稳住。直接搬 softmax attention 的实现习惯过来容易炸。
- **稀疏的硬件对齐很关键**：理论稀疏 ≠ 实际加速。随机 gather、非连续访问会让稀疏 kernel 比 dense 还慢——这就是 NSA / MoBA 强调"块连续、tensor-core 友好"的原因。评估稀疏方法**一定看真实墙钟 / 端到端吞吐**，别只看 FLOPs。
- **和 RoPE 的关系**：线性注意力 / SSM 不像 softmax attention 那样直接套 RoPE——它们的"位置信息"主要来自递推结构本身（衰减 $\gamma^{t-s}$、$\Delta$、门 $\alpha$ 提供了隐式的相对位置 / 时序衰减）。**例外**：RetNet 原始公式里 $Q,K$ 除了乘标量衰减 $\gamma$，还先各乘一个复数旋转因子 $\Theta_n=e^{in\theta}$（xPos/RoPE 式相对位置旋转），使 $Q_nK_m$ 的相位差为 $e^{i(n-m)\theta}$——完整的衰减/位置项其实是 $\gamma^{n-m}e^{i(n-m)\theta}$，并非只有标量幅度衰减（§2.4 为简化只写了幅度部分）；纯累加、不带任何衰减/旋转的线性注意力状态，其求和顺序本身才是（忽略因果截断意义下）不敏感于历史顺序的。混合架构里**full attention 层用 RoPE，线性/SSM 层用各自的衰减机制**；硬把 RoPE 塞进线性注意力的 $\phi(q),\phi(k)$ 要小心（会和衰减项相互作用）。
- **区分"linear attention"和"sub-quadratic 但仍是 attention"**：FlashAttention 是**精确 softmax** 的 IO 优化（仍 $O(L^2)$ FLOPs，只省显存），**不是**线性注意力；Performer/Linformer 才是次二次近似。线性注意力 / SSM 是"**用固定状态递推替代 attention**"，根本上换了计算范式。这三类常被面试者混为一谈。

> ⚠️ **最大 footgun #1：拿渐进复杂度当实际性能**
> "我的方法 $O(L)$，所以一定比 $O(L^2)$ 的 Transformer 快/好"——错两次：实际速度看常数和访存（FlashAttention 很强），质量看 recall（线性有短板）。**任何高效注意力 claim 都要给真实墙钟 + 长上下文 recall benchmark**，否则站不住。

> ❌ **最大 footgun #2：以为 SSD / SSM "就是" attention 或线性注意力"约等于" softmax**
> SSD 是"SSM ≡ **结构化掩码（半可分）线性注意力**"的精确等价，**不是** softmax attention；线性注意力是否"近似 softmax"要看 feature map——**只有 Performer 类（FAVOR+ 随机特征）显式构造为 $\exp(q^\top k)$ 的无偏有限维近似**，ELU+1（Katharopoulos）、RetNet/GLA 等则是换用另一个核函数 / 相似度度量，并未试图逼近 softmax 的 exp 核（见 §2.1）。二者共同的局限是：任何有限维内积（无论是否逼近 softmax）都**表达不出 softmax 的任意尖锐 one-hot 选择**，这是 recall 变弱的根源——但"近似 softmax"这个说法只适用于 Performer 类方法。把这些等价 / 近似过度宣称成"等同 / 无损替代 softmax"，或把"近似 softmax"当成所有线性注意力变体的共性，都是经典错误，会被追问到底。
