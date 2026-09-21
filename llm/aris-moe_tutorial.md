# MoE混合专家 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[moe-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/moe_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 MoE** — 一页拿下 2026 秋招核心要点（详见后文 §1–§9 推导）。

1. **核心思想**：把单个 FFN 换成 $N$ 个 expert + 一个 router，每个 token 只走 $k \ll N$ 个 expert，**总参数 ↑、激活参数不变**（sparse activation）。计算量约等于 $k$ × 单个 expert-FFN（等价于 $k$ 个等宽 dense FFN，与 $N$ 无关，详见 §7.1），但内存 / 显存按总参数计。

2. **路由公式**（Token-Choice top-k）：$g_i(x) = \text{softmax}(W_g x)_i$，选 $\mathcal{T}_k(x) = \text{TopK}_i\, g_i(x)$，输出 $y = \sum_{i \in \mathcal{T}_k(x)} g_i(x) \cdot E_i(x)$。Gate 概率作为 **soft 权重**乘到 expert 输出上，反传时让 router 可微。

3. **历史脉络（必背 5 篇）**：Shazeer 2017（首个深度学习里可工作的 MoE 层）→ GShard 2020（top-2 + capacity factor + all-to-all）→ Switch Transformer 2021（top-1 极简，aux loss + load balance）→ Mixtral 8x7B/8x22B 2024（首个开源主流 MoE，top-2）→ DeepSeek-V3 2024（671B/37B，**aux-loss-free**，fine-grained + 1 shared）。

4. **DeepSeek 路线（2026 面试热点）**：DeepSeekMoE 2024 提出 **fine-grained experts**（拆细，$mN$ 个小 expert 选 $mK$ 个）+ **shared experts**（少数 expert 给所有 token，吸收通用知识）。V2 用 MLA + DeepSeekMoE，V3 把路由专家做到 256 + 1 shared，改用 expert bias 在线更新做主要负载均衡（仍保留极小权重的 sequence-wise aux loss 兜底，非完全取消 aux loss，详见 §6.2）。

5. **Aux-loss-free balance**：在 router score 上加每 expert 的偏置 $b_i$，**只用于 top-k 选择**，不进梯度（也不进最终 gate 权重）；每个 step 按"实际 load - 期望 load"反向更新 $b_i$（多负载 expert 减偏置）。**不破坏 sparse gradient，不引入干扰梯度**。

6. **容量与丢 token**：每个 expert 的容量 $C = \lceil \alpha \cdot T \cdot k / N \rceil$（$\alpha$ 是 capacity factor，常用 1.0-1.25）。超过容量的 token 走 **residual bypass**（直接跳过 expert，原始残差通过），或被 drop。Switch Transformer 论文里这是核心工程细节。

7. **并行**：MoE 几乎必须用 **Expert Parallelism (EP)**——把不同 expert 放到不同 GPU，token 路由后做 **all-to-all**（dispatch）→ expert 算 → all-to-all（combine）。DeepEP 与 DualPipe 是 DeepSeek-V3 在 H800 上把 EP 通信和计算 overlap 的两大工程武器。

8. **常见 bug**：①routing collapse（某些 expert 被全部 token 选、其他 expert 饿死）；②fp16 下 router logit overflow；③推理时 **all experts 必须全部加载到显存**——只是激活参数少，**显存依然按总参数算**；④小 batch 时 EP 通信占满，吞吐反而比 dense 差。

---

## §10 25 高频面试题

按难度分 L1（10 必会）/ L2（10 进阶）/ L3（5 顶级 lab）。每题点开看答案要点 + 易踩坑。

### L1必会题（任何 ML 工程岗都会问）

<details>

<summary>Q1.MoE 在做什么？为什么用 sparse activation？</summary>

- 把 dense FFN 换成 $N$ 个独立 expert + router，每 token 只走 $k \ll N$ 个
- **总参数 ↑（容量）但激活参数不变（FLOPs）**——用参数换计算效率
- 训练存得下，推理算力可控
- 关键：MoE 减少 FLOPs，但**不减少推理显存**（all experts 必须常驻）

把 MoE 想成 ensemble。MoE 是**单一前向通路上选 expert**，与"训多个模型 vote"完全不同。

</details>

<details>

<summary>Q2.Top-k token-choice MoE 的 routing 公式是什么？</summary>

- Router: $s(x) = W_g x \in \mathbb{R}^N$，$g(x) = \text{softmax}(s(x))$
- 选 $\mathcal{T}_k(x) = \text{TopK}_i\, g_i$，输出 $y = \sum_{i \in \mathcal{T}_k} \tilde{g}_i E_i(x)$
- Mixtral/DeepSeek：top-k 上 **renormalize**（$\tilde{g}_i = g_i / \sum_{j \in \mathcal{T}_k} g_j$）让权重和 = 1
- Router 没 bias（标准做法）

只说"选 top 几个 expert"不写公式；忘了 renormalize；说 Q/K/V 那套（attention 公式，搞混了）。

</details>

<details>

<summary>Q3.Switch Transformer 的 aux load-balance loss 是什么？</summary>

- $\mathcal{L}_\text{aux} = \alpha \cdot N \sum_i f_i P_i$
- $f_i$ = expert $i$ 接到的 token 比例（不可微，但有数值）
- $P_i$ = expert $i$ 的平均 gate 概率（可微）
- 乘积越小说明分布越均匀（均匀时 $\sum_i f_i P_i = 1/N$）

只说"鼓励均匀"但写不出公式；只写 $\sum P_i^2$（这是 entropy regularizer，不是 Switch 公式）。

</details>

<details>

<summary>Q4.Capacity factor 是什么？为什么需要？</summary>

- $C = \lceil \alpha \cdot Tk/N \rceil$，每个 expert 的容量上限
- $\alpha = 1.25$ 是 Switch / GShard 默认（留 25% 缓冲）
- 超 capacity 的 token → **residual bypass**（expert 输出 0，残差通过；Token Drop 与 Residual Bypass 本质是同一机制，见 §4.2）
- 传统实现里没 capacity 限制会让静态 buffer / EP 通信调度变复杂；但 grouped GEMM / block-sparse kernel（如 DeepSeek-V3 的 dropless 实践）可以支持变长 token 数，capacity 不是绝对必需项

只答"防止 OOM"；不知道 residual bypass，以为 drop 就是直接归零；或以为 capacity factor 是唯一可行方案（忽略 dropless 已是主流实践）。

</details>

<details>

<summary>Q5.MoE 推理时显存按什么算？</summary>

- **按总参数算，不是激活参数**——所有 expert 必须常驻 GPU
- 因为下一个 token 可能路由到任意 expert，按需 load 延迟无法接受
- 671B MoE FP8 ≈ 671 GB，**8× H100 80G = 640 GB 还差一点**；实际部署常用 16× H100 80G（2 节点）或 8× H200 141G
- 真正节省的是 memory bandwidth（每 token 只读 active 部分权重）

以为 MoE 显存按 active 算，所以"Mixtral 13B 单卡能跑"——错，Mixtral 47B 总参，单 24G 卡跑不下。

</details>

<details>

<summary>Q6.DeepSeek-V3 总参数 / 激活参数 / expert 数？</summary>

- **总参数 671B，激活 37B / token**
- **256 routed experts + 1 shared expert**
- Top-8 routed + 1 shared = 9 个 expert per token
- Attention 用 **MLA**，FFN 用 DeepSeekMoE
- arXiv: 2412.19437, 2024.12

把 V3 跟 V2 (236B/21B) 数字搞混；说 V3 有"多少 expert"但记错 256。

</details>

<details>

<summary>Q7.Mixtral 8x7B 真的有 7B × 8 = 56B 参数吗？</summary>

- **不是**。8x7B 的命名只表示"8 expert, 每 expert 7B 量级"
- **实际总参 46.7B**——因为 attention / norm / embedding 在所有 expert 间共享，不是每个 expert 复制一份
- 激活参数 12.9B（不是 14B，因为 router gate 是 sparse 加权）

被名字误导，直接 $8 \times 7 = 56$。

</details>

<details>

<summary>Q8.MoE 训练为什么常 routing collapse？怎么治？</summary>

- 没 balance 机制时，强者愈强：被多选 → 训得多 → 更可能被选
- 治疗：**aux loss (Switch)** / Expert Choice / Aux-loss-free bias (V3) / Z-loss / expert dropout / 适当增大 capacity factor
- 监控：每 expert 的 load + aux loss 数值 + validation loss

只说"加 aux loss"但不给具体公式；不知道还有 expert-choice 这类硬约束方法。

</details>

<details>

<summary>Q9.MoE 训练需要哪些并行？</summary>

- **DP（数据）+ TP（张量）+ PP（流水）+ EP（expert）**
- EP 是新增维度：把 $N$ 个 expert 分到不同 GPU
- 每个 MoE layer 需要 **2 次 all-to-all**（dispatch + combine）
- DeepSeek-V3：16-way PP × 64-way EP × ZeRO-1 DP，跨 2048 卡 H800

忘了 EP；以为 dense 的 DP+TP+PP 就够用。

</details>

<details>

<summary>Q10.MoE 在 inference 端的好处是什么？</summary>

- **不是**"同 active 参数下比 dense 有 bandwidth 优势"——同 active 参数时，dense 和 MoE 每 token 读取的权重量级本来就接近
- 真正的优势场景是**同 total 参数**（即同等知识容量）的对比：要撑起 MoE 那么大的总参数（知识容量），全激活 dense 模型每 token 都要读完整权重；MoE 只需读 active 的那一小部分，bandwidth 大幅更小
- 换句话说：MoE 能用接近小 dense 模型的带宽预算，撑起远大于它的知识容量——这才是 sparse activation 的杠杆点，不是"同预算比较出的优势"
- 671B / 37B 的 V3 每 token 只读 ~5.5% 权重 → bandwidth 与 ~37B dense 相近，但知识容量对齐的是全激活 671B dense（成本高得多）

只说"算力少"——但显存上不去（全 expert 加载），所以单卡 deployment 还是难；也不要说成"同 active 参数下 MoE 天然带宽更省"。

</details>

### L2进阶题（research-oriented 岗位）

<details>

<summary>Q11.Expert Choice routing 是什么？为什么 autoregressive decoder 不能用？</summary>

- 反向：每个 expert 选 top-$M$ 个 token（$M = Tk/N$）
- **天然 balance**：每 expert 严格选 $M$ 个 token，不需 aux loss
- 不 drop token（capacity 是 hard 给定）
- **不能 autoregressive**：第 $t$ 个 token 的路由依赖整个 batch（含未来 token）的 score → 不能 token-by-token 生成
- 用在 encoder / vision / BERT-style；decoder 用 token-choice
- Zhou et al. 2022 NeurIPS, arXiv 2202.09368

以为它能直接替代 Mixtral / DeepSeek 的 token-choice；忘了 causal 这一限制。

</details>

<details>

<summary>Q12.Top-k 的离散操作怎么反传梯度？</summary>

- Top-k 本身**不可微，不参与梯度**
- 但被选中的 $k$ 个 expert 的 gate 权重 $g_i$（softmax 后）**进入 output 加权求和**，对 $g_i$ 可微
- 未被选中的 expert 的 router weight 这次 step 拿不到梯度（chicken-and-egg → 需 aux loss / bias 打破）
- 类似 hard attention，但因为加权求和，梯度信号还是能到 router

以为"top-k 必须用 Gumbel-softmax 才能反传"——其实标准做法直接用 softmax + top-k 选择，gate weight 就是梯度通路。

</details>

<details>

<summary>Q13.DeepSeekMoE 的 fine-grained expert 是什么意思？</summary>

- 把 $N$ 个 baseline expert 拆成 $mN$ 个更小的 expert（每 expert $d_\text{ff}$ 缩到 $d_\text{ff}/m$）
- 路由数从 $K$ 提到 $mK$
- **总计算 / 总参数不变**，但组合数 $\binom{mN}{mK} \gg \binom{N}{K}$
- 每 token 的"专家组合身份"指数级丰富 → 专精度提升
- Dai et al. 2024, arXiv 2401.06066

以为 fine-grained = 增加 expert 数量（部分对）但漏掉"每个 expert 缩小、路由数同比例放大"这个保持算力的关键。

</details>

<details>

<summary>Q14.Shared expert 是什么？为什么需要？</summary>

- 一种**永远激活、不经 router**的 expert（每 token 都走它）
- 吸收"通用知识"（语言 / 常识），让 routed expert 专心做 specialization
- DeepSeekMoE / Llama 4 都有；Mixtral / Switch 没有
- DeepSeek-V3: 1 shared expert + 256 routed (8 selected)

误以为 shared expert 是 ensemble；或不知道它不通过 router（router 不为 shared expert 投票）。

</details>

<details>

<summary>Q15.DeepSeek-V3 的 aux-loss-free balance 怎么做的？</summary>

- 每 expert 一个偏置 $b_i$，top-k 选择用 **biased score** $s_i + b_i$（$s_i$ 是 sigmoid affinity，bias 加在 sigmoid 之后，详见 §2.4）
- 但 **gating weight 用原始 $s_i$**（保持主梯度路径干净）
- 每 step 按 sign 更新：$b_i \leftarrow b_i - u \cdot \text{sign}(c_i - \bar{c}_i)$
- $b_i$ 是 buffer 不进梯度图，不被 optimizer 更新
- 兜底还有极小 sequence-level aux loss（防单序列内部极端不均）
- Wang et al. arXiv 2408.15664；V3 在 2412.19437 中应用

说"V3 完全没有 aux loss"——不准确（还有 sequence-wise 兜底）；或把 biased score 用到 gating weight 上（会污染主梯度，违背设计意图）。

</details>

<details>

<summary>Q16.MoE 训练的 all-to-all 是什么？通信量怎么算？</summary>

- 每 MoE layer 做 2 次 all-to-all：dispatch（按 expert 发 token）+ combine（按 token 收回）
- 通信量 per layer ≈ $2 T k D (G-1)/G$（$G$ 是 EP group size）
- 通常占训练总时间 30-50%，DeepEP / DualPipe 是用来 overlap 通信与计算的工程武器
- $G$ 大 → 更稀疏 → 通信比例反而 ↑

只答"用 NCCL 通信"；不清楚通信量随 $T \cdot k \cdot D$ 线性增长；不知 DualPipe / DeepEP。

</details>

<details>

<summary>Q17.MoE 上做 LoRA 微调有什么问题？ESFT 怎么解？</summary>

- LoRA 加到 attention 层 → 不动 expert；加到所有 expert → $N$ 倍 LoRA 参数，且部分 expert 训不到
- **ESFT** (arXiv 2407.01906)：先 forward 统计 router score → 选任务最相关的 top-$k$ expert → 只训这些
- 性能 ≈ 全参 fine-tune，显存 ↓ 90%、时间 ↓ 30%

不知道 ESFT；以为 "MoE = 大模型，LoRA 一招通吃"。

</details>

<details>

<summary>Q18.MoE 推理时显存全 load，那 sparse activation 的好处去哪了？</summary>

- 好处转移到 **memory bandwidth**——每 token 只读 active 比例权重
- LLM decode 是 memory-bound（不是 compute-bound），所以 bandwidth 减小直接转化为 throughput
- 671B/37B 模型：每 token 读 ~5% 权重 → throughput ≈ 30B dense
- 但显存仍按总参数预算：671B FP8 裸权重 ≈671GB > 8×80GB(H100)=640GB，单纯 8 张 H100-80G **放不下**（不是"8 卡就够用"）；按裸权重算至少需要 9 张 80G 卡，加上 KV cache/激活值等开销，实际部署常见 **16×H100(2 节点)** 或 **8×H200 141G**（与本文 §7.2 表格一致）

以为 sparse 推理可以"按需 load 1 个 expert"——硬件延迟禁止；或以为 throughput 收益来自 FLOPs 节省（实际更多来自 bandwidth）。

</details>

<details>

<summary>Q19.MoE 和 MoD (Mixture-of-Depths) 区别？</summary>

- MoE 沿 **width** 稀疏：每 token 选部分 expert
- MoD 沿 **depth** 稀疏：每层选 top-$k$ 个 token 走整层，其他 token skip
- 每 token 计算量：MoE 固定（top-k expert），MoD 动态（取决于多少层接受它）
- 不互斥，可以叠加

以为 MoD = MoE 的别名；或不知道沿 depth 也能稀疏化。

</details>

<details>

<summary>Q20.Mixtral 8x7B 实际总参数为什么不是 56B 而是 46.7B？</summary>

- 8 个 expert 只在 **FFN 层** —— attention / norm / embedding / unembedding **全部共享，只算一份**
- Mixtral 用 hidden_size=4096、intermediate_size=14336、SwiGLU（3 个矩阵）FFN、32 层：单个 expert 全模型累计参数 $\approx 3 \times 4096 \times 14336 \times 32 \approx 5.637\text{B}$；shared 部分（attention + embed + norm，全模型累计）$\approx 1.60\text{B}$
- Mixtral 8x7B 总参 = **shared + 8 × 单 expert** = $1.60 + 8 \times 5.637 \approx 46.70\text{B}$——与官方 **46.7B** 精确吻合，不需要"router weight"之类的模糊修正项
- 激活参数（per token，top-2，shared 只算一次）= $1.60 + 2 \times 5.637 \approx 12.87\text{B}$，与论文给出的 **12.9B** 一致

直接 $8 \times 7 = 56$ 当总参（把 attention/embed 也乘 8）；或激活算成 $2 \times 7 = 14$ 忘掉 shared 已经只算一份。

</details>

### L3顶级 lab / Research 方向

<details>

<summary>Q21.DeepSeek-V3 aux-loss-free 为何不破坏 sparse routing semantics？</summary>

- **关键设计**：bias $b_i$ **只用于 top-k 选择**，不进入 gating weight，更不进入梯度图
- 这意味着：router 的梯度信号**只来自语言建模 loss**，与传统 aux loss "把平衡目标作为额外梯度" 完全不同
- 类比控制系统：bias 是 outer-loop controller（按 load 反馈调节），主梯度是 inner-loop optimizer（按 task loss 优化表达力）——两个 loop **解耦**
- 对比 aux loss：aux loss 的梯度会拉 router 权重往"均匀分布"方向，可能与"任务最优 routing"方向冲突
- Loss-free 让 router **在主任务下仍然 sparse + specialize**，bias 只是后处理纠偏
- 一个细节：bias **不进 gating weight** 这一点至关重要——否则相当于通过反向通路偷偷影响梯度，loss-free 名义就名不副实

把 bias 当成 learnable parameter 进 optimizer；以为 loss-free 就是"啥都不加"（实际有 sign 更新和 sequence-aux 兜底）。

</details>

<details>

<summary>Q22.Fine-grained + shared 比同总参 dense 优势在哪？理论 / 实证？</summary>

- **理论容量论**：$mN$ 个小 expert 的 routed combination 数 $\binom{mN}{mK}$ 远超 $\binom{N}{K}$，每 token 可表达的"专家组合身份"指数级丰富，更接近"每 token 独立小子网"的极限
- **专精论**：shared expert 吸收通用知识，让 routed expert 摆脱"既要又要"的负担，更可能 specialize
- **算力论**：同 active FLOPs 下，dense 必须把容量塞进激活参数；MoE 把容量塞进 dormant expert，活时只取一小撮——可以在不增加 FLOPs 的前提下把总参数推到 10x+
- **实证**：DeepSeekMoE 16B vs LLaMA-2 7B：FLOPs 相近，benchmark 上接近或超过（论文 Table 3）；DeepSeekMoE 145B vs DeepSeek 67B dense：性能持平，但 145B MoE 训练 FLOPs 只用 ~28%
- **失败模式**：若 routing 不够好（collapse），fine-grained 反而退化（少数小 expert 通吃，等效 dense 但效率更低）
- **2025-2026 实证趋势**：V3 把这条路线推到极致（256 routed），证明在 1T 级 token 训练下 fine-grained 优势显著

只说"参数多就行"——但没拆开"组合容量 ≠ 单参数容量"；忽略 routing 质量是这条路线 work 的前提。

</details>

<details>

<summary>Q23.MoE inference latency 真的能与 dense active 同台吗？瓶颈在哪？</summary>

- **理论**：每 token 只读 active 比例权重，bandwidth-bound 下 throughput 应正比于 active 参数
- **实际几个 caveat**：
  1. **Routing overhead**：router 计算 + top-k 选择 + scatter/gather 本身有 latency，长 context 下也分摊不少
  2. **Expert load 不均**：单 batch 内 routing 不均 → 部分 expert 排队，wave 浪费
  3. **EP 通信**（多卡推理）：dispatch / combine 的 all-to-all 占很大比例
  4. **Memory bandwidth ≠ memory size**：每 token 5% 权重，但 5% 落到哪些 expert 是 token-dependent，cache 命中差 → 实际带宽利用率不如理论上限
- **Mixtral 8x7B 在单卡 H100 80G**：bandwidth 利用率 ~70-80%（vs dense 70B 利用率类似），latency 与 13B dense 接近
- **V3 671B**：8 卡部署下 inference 延迟略高于 30B dense（多卡通信不可忽略）
- **结论**：理论可以，工程上还有 5-30% gap 取决于 batch size / 上下文长度 / 多卡拓扑

只说"理论上一样"，不知道实际有 routing / EP 通信 / cache 命中等几个隐藏 cost；或反过来说"MoE 推理一定慢"，也不全对（small batch / single GPU 下 Mixtral 与 dense 13B 接近）。

</details>

<details>

<summary>Q24.MoE 训练的 stability 问题有哪些？怎么 mitigate？</summary>

- **Router logit 爆炸**：训练初期 router 权重大幅波动，fp16 下 logit 可 overflow → **Z-loss** $\mathcal{L}_z = \beta \sum_t (\log \sum_i e^{s_i(x_t)})^2$ 惩罚 logit 量级（ST-MoE 2022 提出）
- **Routing collapse**：解 §4.3 提到的方案（aux loss / bias / capacity factor / expert dropout）
- **Expert representation drift**：训练中 expert 间表示渐变（被 router 协同 shape），可能让 fine-tune 时小数据 overfit 某些 expert
- **Aux loss 干扰**：$\alpha$ 大 → loss 拉 router 往均匀，损害性能；$\alpha$ 小 → balance 不够。**这是 loss-free 的根本动机**
- **bf16 vs fp16**：bf16 动态范围与 fp32 接近，更适合 router；fp16 下 router logit 容易 overflow，必须 z-loss + softmax 稳定化
- **Small batch 下 router 噪声大**：每 step 的 $f_i$ 估计噪声大，aux loss 不稳。生产里通常用 micro-batch grad accumulation 增大 effective batch

只答"加 aux loss"——但忽略 z-loss、router 数值稳定、bf16 重要性、small-batch 的统计估计问题。

</details>

<details>

<summary>Q25.如果让你设计下一代 MoE，会改什么？</summary>

这是开放题，没标准答案。给一些 2026 前沿 lab 关心的方向 + 你的判断角度：

- **更细的 expert**：DeepSeek 把 expert 推到 256，进一步推到 1024 / 4096？瓶颈在 routing collapse + EP 通信，需要新的 balance 方法
- **动态 top-k**：每 token 选不同数量 expert（easy token 走 1，hard token 走 4）。和 MoD 思想结合，"既稀疏 width 又稀疏 depth"
- **Differentiable routing**：Sinkhorn / hash-based / learned permutation，让 top-k 部分可微，router 训练更快收敛
- **Expert sharing across layers**：跨层共享一部分 expert pool（类似 ALBERT 的 cross-layer sharing），进一步压总参数
- **更好的 MoE post-training**：ESFT 之后是什么？对 SFT/RLHF 阶段 router 不一定保持 pretrain 时的 specialization，需要专门的 alignment 方案
- **MoE × long context**：1M context 下每层 routing 决策乘以 1M，router 本身可能变成瓶颈 → routing reuse / hierarchical routing
- **MoE 服务化**：单卡跑不动 671B，怎么让中小公司也能部署？expert offload / streaming weights / disaggregated expert servers
- **理论分析**：fine-grained vs dense 容量的严格量化？routing 的 implicit regularization？aux-loss-free 的收敛保证？目前都还很少

回答时**先说一个具体方向**，给清楚 motivation + technical sketch + 可能的 failure mode，比泛泛而谈"我会让 expert 更多"加分多。

</details>

## §A 附录：参考文献清单（按时间）

按出现先后整理。注：**所有 arXiv ID 已交叉验证**。

1. **Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.** (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR. arXiv:1701.06538.
2. **Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., Chen, Z.** (2020). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. arXiv:2006.16668.
3. **Fedus, W., Zoph, B., Shazeer, N.** (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR 23(120):1-40. arXiv:2101.03961（预印本发布于 2021 年，正式发表于 JMLR 2022 年）。
4. **Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A., Chen, Z., Le, Q., Laudon, J.** (2022). Mixture-of-Experts with Expert Choice Routing. NeurIPS. arXiv:2202.09368.
5. **Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., Fedus, W.** (2022). ST-MoE: Designing Stable and Transferable Sparse Expert Models. arXiv:2202.08906.（Z-loss 来源）
6. **Dai, D., Deng, C., Zhao, C., Xu, R., et al.** (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. ACL. arXiv:2401.06066.
7. **Jiang, A. Q., Sablayrolles, A., et al.** (2024). Mixtral of Experts. arXiv:2401.04088.
8. **DeepSeek-AI** (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434.
9. **Wang, Z., Chen, D., Dai, D., Xu, R., Li, Z., et al.** (2024). Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning (ESFT). arXiv:2407.01906.
10. **Wang, L., Gao, H., Zhao, C., Sun, X., Dai, D.** (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts. arXiv:2408.15664.
11. **DeepSeek-AI** (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.
12. **Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P., Santoro, A.** (2024). Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models. arXiv:2404.02258.
13. **Meta AI** (2025). Llama 4: Scout, Maverick, Behemoth (blog announcement, April 5, 2025).
14. **Qwen Team** (2025). Qwen3 Technical Report. arXiv:2505.09388.
15. **DeepSeek-AI / DualPipe / DeepEP** (2025). Open-source releases at github.com/deepseek-ai/{DualPipe,DeepEP}.

> 💡 **2026 秋招高频提问 paper top-5**（按面试出现频率）：DeepSeek-V3 (2412.19437) > Mixtral (2401.04088) > Switch Transformer (2101.03961) > DeepSeekMoE (2401.06066) > Loss-Free Balance (2408.15664)。准备这 5 篇基本覆盖 95% MoE 问题。
