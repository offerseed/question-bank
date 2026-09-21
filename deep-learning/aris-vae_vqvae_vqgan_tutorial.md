# VAE-VQVAE-VQGAN — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[vae-vqvae-vqgan-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/vae_vqvae_vqgan_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **8 句话搞定 VAE / VQ-VAE / VQ-GAN / FSQ** — 一页拿下面试核心要点（详见后文 §2–§9 推导）。

1. **连续 VAE 目标**：最大化 ELBO，$\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_\text{KL}(q_\phi(z|x)\,\|\,p(z))$；reparameterization $z = \mu + \sigma \odot \epsilon$ 让梯度穿过随机采样。

2. **KL 闭式（必考）**：$D_\text{KL}(\mathcal{N}(\mu,\sigma^2 I)\,\|\,\mathcal{N}(0,I)) = \tfrac{1}{2}\sum_i (\mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1)$。

3. **Posterior collapse**：KL → 0 → 解码器忽略 $z$；缓解：KL annealing、free bits、$\beta$ schedule、自回归先验。

4. **VQ-VAE**：把 encoder 输出 $z_e(x)$ 映射到最近邻 codebook 向量 $e_k$，loss = recon + $\|\text{sg}[z_e] - e\|^2$（codebook）$+ \beta \|z_e - \text{sg}[e]\|^2$（commitment）。

5. **Straight-Through Estimator (STE)**：argmin / quantize 不可导，前向用量化值，反向直通梯度 $\partial \mathcal{L}/\partial z_q \to \partial \mathcal{L}/\partial z_e$。

6. **VQ-GAN**：VQ-VAE + perceptual (LPIPS) + adversarial (PatchGAN) + 后训 Transformer prior；为 LDM / Parti / Muse 等离散 token 模型奠基。

7. **FSQ（2024）**：每维标量量化到 $L_i$ 个固定 level（经 tanh 缩放 + round 得到，具体公式见 §9.2 Eq.4），隐式 codebook 大小 $\prod_i L_i$（如 $L=8, d=6 \Rightarrow 8^6 = 262{144}$），**没有可学 codebook 参数、不会 codebook collapse**，但 round 本身不可导，仍需一行 STE（前向 round、反向 identity），loss 只剩 reconstruction。

8. **生态对比**：连续 latent（VAE / KL）适合 LDM 续 diffusion；离散 token（VQ-VAE / VQ-GAN / FSQ / LFQ）适合 AR / MaskGIT Transformer prior，是 Parti / Muse / Cosmos 等的核心组件。

---

## §10 复杂度与资源对比

| 模型 | latent 类型 | 训练参数 (encoder+decoder) | 主要 loss | Codebook collapse | STE 依赖 |
| --- | --- | --- | --- | --- | --- |
| **VAE** | continuous Gaussian | $\sim$10-100M | recon + KL (closed form) | N/A | 否 (reparameterization) |
| **$\beta$-VAE** | continuous Gaussian | 同 VAE | recon + $\beta$·KL | N/A | 否 |
| **NVAE** | hierarchical continuous | 80M-200M | recon + multi-layer KL | N/A | 否 |
| **VQ-VAE** | discrete via codebook | 50-200M | recon + codebook + $\beta$·commitment | **常发生** | 是 |
| **VQ-VAE-2** | hierarchical discrete | 100-500M | 同 VQ-VAE × 2 层 | 同上 | 是 |
| **VQ-GAN** | discrete + adversarial | 50-300M (+ D) | recon + LPIPS + GAN + codebook + commitment | 同上 | 是 |
| **dVAE** | categorical (logits) | 50-200M | recon + KL to uniform | 较少（categorical 分布学习） | 否（Gumbel-softmax 反传 soft） |
| **FSQ** | scalar quantize per dim | 30-150M | recon (+ perceptual) | **几乎不发生** | 是（但极简） |
| **LFQ** | binary scalar quantize | 30-150M | recon (+ entropy reg) | **几乎不发生** | 是 |

> 💡 **生态位定位** —

- 想做 **diffusion / FM**：用 KL-VAE / SD VAE（连续 latent）
- 想做 **AR 生成（GPT-style 图像 token）**：用 VQ-GAN / FSQ / LFQ
- 想做 **MaskGIT / Muse / 并行 decode**：用 VQ-GAN / FSQ
- 想做 **video / 长 sequence**：用 FSQ / LFQ（codebook usage 高、无 collapse）
