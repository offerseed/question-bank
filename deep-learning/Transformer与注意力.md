# Transformer 与注意力机制题库

> Transformer 是现代深度学习的核心架构，面试高频考点。

---

## 03 Transformer

**Q：详细介绍 Transformer 的整体架构。**

**参考答案：**

Transformer（Vaswani et al., 2017）由 Encoder 和 Decoder 组成：

**Encoder（N层堆叠）：**
1. Input Embedding + Positional Encoding
2. Multi-Head Self-Attention
3. Add & Norm（残差连接 + LayerNorm）
4. Feed-Forward Network（两层线性 + ReLU）
5. Add & Norm

**Decoder（N层堆叠）：**
1. Output Embedding + Positional Encoding
2. Masked Multi-Head Self-Attention（防止看到未来）
3. Cross-Attention（关注 Encoder 输出）
4. Feed-Forward + Add & Norm

**关键设计：**
- 并行计算（非序列化）
- Multi-Head Attention 捕获多视角信息
- Positional Encoding 注入位置信息
- 残差连接解决深层网络退化

**延伸思考：**
- Encoder 和 Decoder 的 Attention 有什么本质区别？
- 为什么 Decoder 需要 Mask？
- Post-Norm vs Pre-Norm 的区别？

---

## 04 注意力机制

**Q：Self-Attention、Cross-Attention、Multi-Head Attention 的区别？**

**参考答案：**

| 类型 | Q来源 | K,V来源 | 用途 |
| :--- | :--- | :--- | :--- |
| Self-Attention | 同一序列 | 同一序列 | 序列内建模 |
| Cross-Attention | Decoder | Encoder | 跨序列交互 |
| Masked Self-Attention | 同一序列 | 同一序列 | Decoder因果约束 |

**注意力公式：**
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

**为什么除以 √d_k？**
当 d_k 较大时，QK^T 的值会变大，softmax 会推向梯度极小的区域（接近 one-hot）。除以 √d_k 缩放，使梯度更稳定。

**Multi-Head 的意义：**
将 Q, K, V 分成 h 组分别做 attention，再拼接。不同 head 可以关注不同子空间的信息（如语法关系、语义关系等）。

**延伸思考：**
- 为什么是 softmax 而不是其他归一化？
- Attention 复杂度是 O(n²)，有哪些优化方法？（Linear Attention, Flash Attention 等）
