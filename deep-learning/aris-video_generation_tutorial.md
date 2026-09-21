# 视频生成 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[video-generation-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/video_generation_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **7 句话搞定 Video Generation** — 2024-2025 视频生成大爆发。一页吃下面试高频要点（详见后文 §1–§11 推导）。

1. **范式**：主流视频生成 = **3D Causal VAE 压缩 + Latent DiT 扩散 / Flow Matching**。Sora（2024-02）首次把 Transformer 推到 60s + 高分辨率；Hunyuan-Video (Tencent 2024-12) / Wan 2.x (Alibaba 2024-2025) / Mochi-1 / CogVideoX / Movie Gen / Kling / Veo 2 都是这一架构家族的变体。

2. **3D Causal VAE**：把 $H \times W \times T$ 视频压成 $h \times w \times t$ latent（典型空间下采样 $8\times$、时间下采样 $4\times$）。**因果 (causal)** 关键：当前帧 latent **不能看未来帧**——使训练好的 VAE 同时支持图像（$t{=}1$）与视频（$t{>}1$）、并允许后续帧流式 / 自回归生成。

3. **Spacetime Patches (Sora)**：把视频 latent 切成 $p_t \times p_h \times p_w$ 的 3D patch 当 token；像 ViT 但增加了时间维。支持**变分辨率 / 变时长 / 变长宽比**：直接把不同 shape 的 patch 序列打包到一个 batch（用 mask 区分），无需 resize 到固定尺寸。

4. **时空 Attention 三大变体**：(a) **Factorized 2+1D**（Latte / OpenSora / AnimateDiff）：先 spatial-only，再 temporal-only，复杂度 $O(T \cdot S^2 + S \cdot T^2)$；(b) **Full 3D**（Sora / Hunyuan-Video / Mochi）：所有 token 互相 attend，复杂度 $O((ST)^2)$，最贵但效果最好；(c) **Window / 稀疏 ST**（Wan 2.x / CogVideoX 部分块）：滑窗 3D，折中。

5. **MM-DiT for Video (Hunyuan-Video / Mochi)**：text token 与 video token **同序列**做 self-attention（不再用 cross-attn），两个 stream 各自的 QKV 投影 + AdaLN 调制；条件信息在 token-level 直接交互——Hunyuan-Video/Mochi 把 SD3 image MM-DiT 思路扩展到视频；Wan 2.x 采用的是对 UMT5 文本 embedding 的传统 cross-attention 注入，非 joint-stream MM-DiT。

6. **Image-to-Video (I2V)**：主流三招——(i) **First-frame concat**：把 ref image 编码后沿 channel 拼到 latent；(ii) **Cross-attention 注入**：ref image token 作 K/V；(iii) **AnimateDiff** 风格：冻结 T2I 主干，只插入 temporal module。SVD / DynamiCrafter / I2VGen-XL / Wan-I2V 是代表。

7. **长视频** = keyframe + interpolation / hierarchical / autoregressive chunks。**评测**：VBench (Huang CVPR 2024) 16 维细粒度评分，是当下事实标准；FVD（Unterthiner 2018）仍作辅助；CLIPSim-V 评估 text-video 对齐。

> ⚠️ **Caveat** — 本文中 model-specific 数字（参数量、压缩比、attention 类型）均依据各模型公开 paper / tech report；具体训练 hyperparam 与最终架构以原文为准。

---

## §10 复杂度与显存

### 10.1 Token 数与 attention cost

| 视频规格 | VAE 下采样 | VAE 后 latent shape | Patchify (1,2,2) 后 N | Full-3D Attn $N^2$ |
| --- | --- | --- | --- | --- |
| $256{\times}256{\times}16$ (2s, 8fps) | $4{\times}8{\times}8$ | $4 \times 32 \times 32$ | $4 \cdot 16 \cdot 16 = 1024$ | $\approx 10^6$ |
| $720p{\times}48$ (2s, 24fps, latent $12{\times}90{\times}160$) | $4{\times}8{\times}8$ | $12 \times 90 \times 160$ | $12 \cdot 45 \cdot 80 = 43200$ | $\approx 1.9 \times 10^9$ |
| $1080p{\times}120$ (5s, 24fps, latent $30{\times}136{\times}240$) | $4{\times}8{\times}8$ | $30 \times 136 \times 240$ | $30 \cdot 68 \cdot 120 = 244800$ | $\approx 6.0 \times 10^{10}$ |

很快爆。面试常问 "长视频/高分辨率 attention 瓶颈"——答：$O(N^2)$ 二次成本 + FlashAttention 也减不了 token 数本质；解法是 factorized / window / cascaded 多阶段。

### 10.2 显存与 FLOPs 要点

- **Score 矩阵显存**：vanilla 是 $O(L^2)$；FlashAttention 降到 $O(L)$ activation
- **KV cache**：纯 diffusion 无 AR step，训练时无 KV cache 概念；activation checkpointing + ZeRO-3 是必备
- **3D Causal VAE 推理**：可分 chunk，显存 $O(\text{chunk}_T \cdot h \cdot w \cdot C)$
- **训练 FLOPs**：13B 模型 per-batch attention FLOPs ≈ $4 B N^2 d$ × layers（B = batch size）；按 §10.1 同一公式与最大规格数字（$N \approx 2.45 \times 10^5$, $N^2 \approx 6 \times 10^{10}$）粗算，attention 部分 FLOPs 量级在数十 PFLOP/sample（具体随 hidden dim、layer 数体量浮动），而非几百到上千 PFLOP，用数千 H100 数月级别训出
