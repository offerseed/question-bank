# RAG与文本嵌入检索 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[rag-embedding-retrieval-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/rag_embedding_retrieval_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **9 句话搞定 RAG + 嵌入** — 一页拿下面试核心要点（详见后文 §2–§11 推导）。

1. **RAG 是什么**：检索增强生成 = 先用 query 从外部知识库**检索**相关片段，再把片段拼进 prompt 让 LLM **生成**答案。应对三件事：知识过期、幻觉、私有数据（Lewis et al. 2020）。

2. **两半互补**：左半「**训嵌入**」是对比学习（双塔 + InfoNCE + 难负例），数学密；右半「**用嵌入**」是 RAG 管线（chunk → 检索 → 重排 → 拼 prompt），系统密。

3. **嵌入主公式 InfoNCE**：$\mathcal{L} = -\log \frac{\exp(\text{sim}(q,d^+)/\tau)}{\sum_i \exp(\text{sim}(q,d_i)/\tau)}$ —— 本质是「正样本 vs 一堆负样本」的 softmax 交叉熵，温度 $\tau$ 调锐度。

4. **双塔 vs 交叉编码器**：双塔（bi-encoder）query/doc 分开编码，**可预计算**、快、用于召回；交叉编码器（cross-encoder）拼一起编码，**不可预计算**、慢但准，用于**重排 top-k**。

5. **难负例（hard negative）**：从检索 top-k 里挑「像但不相关」的当负例，比随机负例信息量大；但有 **false negative 陷阱**——挑到的「难负例」可能其实相关，反成噪声。

6. **稀疏 vs 稠密 vs 混合**：BM25（词项精确匹配，强在罕见词/实体）+ 稠密向量（语义匹配）互补，用 **RRF**（Reciprocal Rank Fusion）按排名融合：$\text{RRF}(d)=\sum_r \frac{1}{k+\text{rank}_r(d)}$，$k\approx60$。

7. **Matryoshka 表征**：一次训练让嵌入的**前 $m$ 维**也是好嵌入，部署时按需截断维度（省存储/加速召回），靠多粒度损失 $\sum_m \mathcal{L}(\text{emb}[:m])$。

8. **RAG vs 长上下文 vs 微调**：RAG 注入**可更新的外部事实**、可溯源；长上下文塞全文但贵且有「lost in the middle」；微调改**能力/风格**不擅长注入海量新事实。三者常组合。

9. **评测**：别只看生成，要分层评 **检索质量**（recall@k / nDCG）+ **生成忠实度**（faithfulness / context relevance / answer relevance，如 RAGAS）。

---

## §10 工程实践与常见 bug

- **chunk 大小是头号旋钮**：太大检索粗、太小断上下文；先试 256/512 token + 10–20% overlap，按评测调。
- **比调 chunk size 更高价值的改进**：**Contextual Retrieval**（Anthropic 2024，嵌入前给每个 chunk 补一段 LLM 生成的全文上下文）与 **late chunking**（Jina 2024，先整文过长上下文 encoder、再切块池化）都针对「普通切块丢失文档级上下文」这一根因，往往比单纯调块大小收益更大。
- **嵌入模型要对域**：通用嵌入在专业领域（医疗/法律/代码）可能弱，考虑领域微调或选对口模型（BGE-M3 / E5 / 领域模型）。
- **别纯稠密**：实体/罕见词/精确匹配场景务必加 BM25 混合，否则「检索不到明明有的那条」。
- **难负例 false negative**（§2.3）：挖负例要去重、过滤，否则训歪。
- **lost in the middle**（§7）：关键片段放 context 头尾，别堆中间。
- **重排别省**：召回 recall 够但精度不足时，cross-encoder 重排 top-k 往往是高 ROI 的一步（是否最高取决于召回质量 / 延迟预算 / 候选规模）。
- **索引会过期**：库更新后要增量重嵌入 / 重建索引；陈旧索引 = 陈旧答案。
- **评测要分层**（§8.3）：检索和生成分开归因，否则定位不了问题在哪。
- **温度 $\tau$ 与 batch**：对比学习里大 batch（更多 in-batch 负例）+ 合适 $\tau$ 很关键，小 batch 效果差。

> ⚠️ **「加了 RAG 还是幻觉」** — 多半不是 LLM 的锅：检索召回了无关/错误片段，LLM「忠实地」据此作答。先查检索（recall@k、片段对不对），再怀疑生成。
