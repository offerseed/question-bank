# 长上下文RoPE-YaRN-MLA — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[long-context-rope-yarn-mla-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/long_context_rope_yarn_mla_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 Long Context** — 一页拿下面试核心要点（详见后文 §2–§9 推导）。

1. **RoPE**：对每对维度 $(2i, 2i+1)$ 做位置 $m$ 相关的 2D 旋转，$\theta_i = 10000^{-2i/d}$。$q_m^\top k_n$ 仅依赖**相对位置** $m-n$（不依赖绝对 $m, n$ 各自），且无需训练参数。

2. **PI (Position Interpolation, Chen 2023)**：把所有 $\theta_i$ 同除以 $s = L_\text{new}/L_\text{train}$（等价于把绝对位置 $m$ 缩到 $m/s$）。**伤害高频**（早期维度的相位分辨率被压缩），但实现简单。

3. **NTK-aware (bloc97 2023)**：换底，新底 $b' = b \cdot s^{d/(d-2)}$。**低频维度被强压缩、高频维度几乎不变**，零样本外推优于 PI。

4. **YaRN (Peng 2023)**：NTK-by-parts（分段处理频率）+ temperature scaling（拟合公式 $\sqrt{1/t} \approx 0.1\ln s + 1$，即 $t \approx 1/(0.1\ln s + 1)^2$，工程上以 attention scale 的方式乘进 cos/sin cache 实现，非独立第三组件）。两组件分别解决：高频/低频分别处理、补偿外推后注意力熵增（稀释 softmax）。

5. **LongRoPE (Ding 2024 ICML)**：演化搜索每维独立的缩放因子 $\lambda_i$，加上 short-context "rescue"，把上下文推到 2M tokens。

6. **MLA (DeepSeek-V2)**：$\mathbf{c}_t^{KV} = W_\text{DKV}\mathbf{h}_t$ 把 K/V 压成 $d_c \ll N_h d_h$ 的 latent；RoPE 必须**解耦**——单独留一份 $d_h^R$ 维 RoPE key（共享 across heads），否则旋转矩阵不能"吸进" $W_\text{UK}$ 里。

7. **Streaming + Sink (Xiao 2024 ICLR)**：保留前 4 个 token（attention sink，softmax 的"垃圾桶"）+ 滑动窗口；window 外的 token 直接丢，但 sink 不能丢，否则 PPL 爆炸。

8. **System**：Ring Attention / Context Parallel 跨设备 chunk K/V；FlashAttention 2/3 块化 + online softmax；Mistral SWA 把每层感受野从 $L$ 降到 $W$（多层叠加仍可看远）。

---

## §10 滑动窗口与流式注意力

### 10.1 Sliding Window Attention (Mistral 2023)

每个 query 只看前 $W$ 个 key（$W$ = window size，Mistral-7B 用 $W = 4096$）。

- **复杂度**：从 $O(L^2 d)$ 降到 $O(L W d)$，长序列线性。
- **多层叠加感受野扩展**：第 1 层看 $W$，第 2 层每个位置看前 $W$（其中每个 token 又看了它的前 $W$），有效感受野 $2W$；$\ell$ 层后感受野 $\ell W$。**所以 32 层 × 4096 window ≈ 131K 有效感受野**。

```python
def sliding_window_mask(L: int, W: int, device=None) -> torch.Tensor:
    """
    L: sequence length; W: window size (含 self)
    返回 [L, L] bool mask, True=可见.
    位置 i 可看 j ∈ [max(0, i-W+1), i] (causal + sliding window)
    """
    idx_q = torch.arange(L, device=device).unsqueeze(1)   # [L, 1]
    idx_k = torch.arange(L, device=device).unsqueeze(0)   # [1, L]
    causal     = idx_k <= idx_q
    in_window  = idx_k > (idx_q - W)
    return causal & in_window

# 示例 L=8, W=4:
# row 0: [T F F F F F F F]
# row 1: [T T F F F F F F]
# row 2: [T T T F F F F F]
# row 3: [T T T T F F F F]
# row 4: [F T T T T F F F]    ← 0 号被推出 window
# row 5: [F F T T T T F F]
# row 6: [F F F T T T T F]
# row 7: [F F F F T T T T]
```

> 💡 **SWA 的实战意义** — Mistral-7B 训练长度 8K，推理时配合 SWA 可处理 32K+ 上下文（每层只看本地 4K，多层叠出全局），同时显存/计算线性。但纯 SWA 无法保留窗口外的远距 token（自然没有**远距精确检索**能力，如 needle-in-haystack 远端针）。StreamingLLM 加 attention sink 的动机是**防止**朴素窗口驱逐早期 token 时 softmax 失去"垃圾桶"而导致 PPL 崩溃，从而支持**无限流式生成**——sink 并不能恢复被淘汰 token 的信息，远距检索能力仍然缺失（见 §10.3）。

### 10.2 StreamingLLM — Attention Sink + Sliding Window

Xiao et al. ICLR 2024 ("Efficient Streaming Language Models with Attention Sinks") 提出推理时一个关键发现：

**LLM decode 时，softmax 强制 attention 权重和为 1，但 query 实际可能"什么也不想 attend"。模型于是把权重大量倒给前 1-4 个 token（特别是 `<bos>`），形成 attention sink。** 这些 token 内容上没什么信息，但它们的 KV cache **不能丢弃**——丢了之后 softmax 没有"垃圾桶"，剩下 token 的注意力分布被强行重整，PPL 爆炸。

StreamingLLM 推理策略：

1. **永远保留** 前 $S$ 个 token 的 KV cache（$S = 4$ 经验值）作为 sink。
2. **滑动窗口** 保留最近 $W$ 个 token 的 KV cache。
3. window 之外、sink 之外的 token，**直接丢弃 KV**。

总 KV cache 大小为 $S + W$，与序列长度 $L$ 解耦，达到**真正的流式**生成。

### 10.3 StreamingLLM 推理循环代码

下面是 **教学示意版**，重点展示控制流。生产实现（HuggingFace `streaming_llm` / 原作者 `streaming-llm` repo）有两个关键细节，下面注释里说明。

```python
@torch.no_grad()
def streaming_decode(model, input_ids, max_new_tokens,
                     sink_size=4, window_size=2044):
    """
    教学版 streaming inference: sink 区 + sliding window.
    总 cache = sink_size + window_size, 与生成长度无关.

    关键细节 (生产代码必须做):
    (a) Cache 存的是 *RoPE 之前* 的 K (即 W_K @ h, 未旋转), 同时记录该 token 的"逻辑位置".
        每次 forward 时, 根据当前 cache 中各 token 的*新* 逻辑位置, 对 sink / recent K 重新施加 RoPE.
        否则裁剪 + 位置漂移会让 cache 中的旋转角对不上新的逻辑位置.
    (b) Sink 区位置永远固定在 [0, S), recent window 位置永远固定在 [S, S+W),
        新 token 用 S+W (即 cache 容量上限) 作为它的逻辑位置.
        这样模型见到的"最大相对位置"始终 ≤ S+W, 永远不会触碰 RoPE 训练上限.
    """
    device = input_ids.device
    B = input_ids.size(0)
    total = sink_size + window_size

    # ----- 1) Prefill -----
    # past_kv_pre[i] = (k_pre, v) 其中 k_pre = W_K @ h, 未做 RoPE
    past_kv_pre = _prefill_unrotated(model, input_ids)            # 实现细节略

    # 若 prompt 已超 sink+window, 裁剪 (sink 段 + 最近 window 段)
    def trim_unrotated(past_kv_pre):
        new_past = []
        for (k_pre, v) in past_kv_pre:
            if k_pre.size(-2) <= total:
                new_past.append((k_pre, v));  continue
            sink = (k_pre[..., :sink_size, :], v[..., :sink_size, :])
            recent = (k_pre[..., -window_size:, :], v[..., -window_size:, :])
            new_past.append(
                (torch.cat([sink[0], recent[0]], dim=-2),
                 torch.cat([sink[1], recent[1]], dim=-2))
            )
        return new_past

    past_kv_pre = trim_unrotated(past_kv_pre)
    # logits 来自 prefill 最后一步
    next_token = _last_logits(model, past_kv_pre).argmax(-1, keepdim=True)
    generated = [next_token]

    # ----- 2) Autoregressive decode -----
    for step in range(max_new_tokens - 1):
        cur_len = past_kv_pre[0][0].size(-2)                       # 当前 cache 中 token 数
        # 给 cache 中每个 token 分配"逻辑位置"; 注意 prompt 极短时 cur_len < sink_size,
        # 此时所有 token 都视为 sink (没有 recent window).
        if cur_len <= sink_size:
            cache_pos = torch.arange(cur_len, device=device)        # [cur_len]
        else:
            cache_pos = torch.cat([
                torch.arange(sink_size, device=device),             # sink 段: [0..S)
                torch.arange(sink_size, cur_len, device=device),    # window 段: [S..cur_len)
            ])                                                       # 长度 = cur_len
        new_pos = torch.tensor([cur_len], device=device)             # 新 token 逻辑位置

        # 对 cache 中的 K_pre 重新施加 RoPE (按 cache_pos), 对新 token 按 new_pos.
        out = model(next_token, past_kv_pre=past_kv_pre,
                    cache_pos=cache_pos, new_pos=new_pos, use_cache=True)
        past_kv_pre = trim_unrotated(out.past_kv_pre)
        next_token = out.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(next_token)

    return torch.cat(generated, dim=-1)
```

> ⚠️ **直接裁剪 RoPE 之后的 K cache 是错的** — 一个常见 bug：把 HF 默认的 K cache（已 RoPE）直接按上面的方式裁剪 + 用逻辑位置 id 喂新 token，会得到自相矛盾的相对位置（cache 中的 K 用原始绝对位置旋转过，但新 query 用逻辑位置旋转）。**正确做法**：保留未旋转的 K（`W_K @ h`，未乘 cos/sin），每步根据当前逻辑位置重新做 RoPE；或者用作者 repo 提供的 `enable_streaming_llm()` patch，它修改了 attention layer 以接受 "position-shift" 形式的旋转。

> ⚠️ **StreamingLLM 不增加模型有效上下文** — 它让模型可以**永久流式生成**而不爆显存，但实际能看到的还是 sink + window 范围内的 token。中间被丢弃的内容**真的看不到了**。要长上下文检索能力还是得依赖 YaRN / LongRoPE / SSM 等真正的上下文扩展。

### 10.4 Lost-in-the-Middle (Liu 2023)

Liu et al. 2023 ("Lost in the Middle: How Language Models Use Long Contexts") 经验观察：**长上下文模型对 prompt 头部和尾部的关注度远高于中间**，造成"中间内容更难被检索到"。

- **U-shaped curve**：把 key info 放在 prompt 不同位置，检索准确率随位置呈 U 形（首尾高，中间低）。
- **原因**：causal LM 训练分布中，第一个 token 影响最广（attention sink 同源问题）；最后一个 token 是 next-token 预测的直接前驱。中间内容被两端"挤压"。
- **缓解**：(a) 把重要信息放在 prompt 开头或结尾；(b) Recurrent retrieval (chain prompts)；(c) 训练时增加中段权重 (位置感知 loss weighting)。

> 💡 **面试要点** — 这不是"位置编码外推失败"——是模型**有效**学到了长上下文，但 attention 分布存在偏好。和 RoPE/YaRN 解决的问题不同。
