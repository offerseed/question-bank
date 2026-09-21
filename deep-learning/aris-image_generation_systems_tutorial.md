# 图像生成系统 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[image-generation-systems-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/image_generation_systems_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 Image Generation 系统** — 一页拿下 production text-to-image 栈核心（详见 §1–§10 推导）。

1. **LDM 关键**：VAE encode 把 $H\times W\times 3$ 压成 $h\times w\times c$（SD 1.x: $8\times$ 下采样、$c=4$），扩散在 latent 上做，token 数降到 $1/8^2=1/64$（**数量级估计**，假设每 token 计算量不变；实际 FLOPs 节省还受 latent 通道更宽、self-attention FLOPs 按 token 数平方 scale 等因素影响，非精确的 $64\times$ 等价），最后再 VAE decode 出像素（Rombach et al. 2022 CVPR）。

2. **SD 1.x → SDXL → SD3 → FLUX 主线**：1.x 用 CLIP-L 文本编码 + U-Net；SDXL 双编码器（OpenCLIP-G + CLIP-L）+ 2.6B U-Net + size/crop conditioning + Refiner（Podell et al. 2024 ICLR）；SD3 换 **MM-DiT** + Rectified Flow（Esser et al. 2024 ICML）；FLUX.1 12B MM-DiT 加 parallel attention（Black Forest Labs 2024）。

3. **CFG（必考）**：训练时按概率 $p_\text{drop}\approx 0.1$ 把条件 $c$ 替换成 $\emptyset$；推理时输出 $\hat\epsilon_\text{cfg} = \hat\epsilon_\emptyset + s\,(\hat\epsilon_c - \hat\epsilon_\emptyset)$，$s \in [1.5, 12]$（Ho & Salimans 2022）。

4. **ControlNet 零卷积**（Zhang et al. 2023 ICCV）：把 U-Net encoder 整段 **trainable copy**，每个连接到主干的卷积 $W=0,b=0$ 初始化——前向恒等、**梯度非零**（$\partial L/\partial W = \delta \cdot x \neq 0$），训练时从干净恒等映射出发逐步注入条件信号，从而保护预训练能力。

5. **IP-Adapter**（Ye et al. 2023）：**Decoupled Cross-Attention**——为图像条件**新增一组** $W_K', W_V'$，与原文本 cross-attn 并行，输出相加：$\text{out} = \text{Attn}(Q, K_\text{txt}, V_\text{txt}) + \lambda\,\text{Attn}(Q, K_\text{img}, V_\text{img})$。仅训新增 K/V + projector，~22M 参数。

6. **LoRA**（Hu et al. 2022 ICLR）：$\Delta W = B A,\ B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \ll \min(d,k)$；只训 $A, B$，原 $W$ 冻结。SD 上典型 $r \in \{4,8,16,32\}$，比全量小 50–200×；推理可 merge $W' = W + \alpha B A$。

7. **DreamBooth**（Ruiz et al. 2023 CVPR）：rare-token（如 `sks dog`）+ **prior preservation loss** $L = \|\epsilon - \hat\epsilon(x_t, t, \text{"a sks dog"})\|^2 + \lambda \|\epsilon' - \hat\epsilon(x_t', t, \text{"a dog"})\|^2$，第二项防 language drift / 过拟合。

8. **DiT vs MM-DiT**：DiT 用 **AdaLN-Zero**（条件 $\to$ MLP $\to$ scale/shift/gate，最后一层 $W_\text{gate}=0$ 实现恒等启动，Peebles & Xie 2023 ICCV）；MM-DiT 把 text token 和 image token **拼成一个序列做联合 self-attention**（每个模态独立 QKV 投影，但 attention 是全局的），信息流双向（Esser et al. 2024）。

---

## §10 评测：FID / CLIP-Score / ImageReward / HPSv2 / PickScore

| 指标 | 计算 | 评什么 |
|---|---|---|
| **FID** (Heusel et al. 2017 NeurIPS) | Inception-V3 pool3 特征的 Fréchet distance（real vs gen） | 整体分布相似度（多样性 + 真实度） |
| **CLIP-Score** | CLIP 算 (text, image) cosine similarity，平均 | 文本对齐 |
| **ImageReward** (Xu et al. 2023 NeurIPS) | 拿人类偏好数据训的 reward model（ViT + CLIP backbone） | 人类整体偏好（含审美 / 文本对齐 / 真实度） |
| **HPSv2** (Wu et al. 2023) | Human Preference Score V2，类似 ImageReward 但更多数据 | 人类偏好（更细分类别） |
| **PickScore** (Kirstain et al. 2023 NeurIPS) | Pick-a-Pic 数据集训练，CLIP-based | 用户偏好 |

> ⚠️ **FID 局限** — (i) 对 mode collapse 不敏感（生成方差小 FID 反而可能涨）；(ii) Inception-V3 是 ImageNet 训的，对人脸 / 艺术 / 非自然图偏置严重；(iii) **生成数应至少 10K**，<5K 时方差极大，paper 间不可比；(iv) FID-30K vs FID-10K 不能直接比。**面试要主动指出 FID 不是 final word**，必须配合 human preference 指标（IR / HPSv2 / PickScore）。
