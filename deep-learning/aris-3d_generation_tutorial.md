# 3D生成 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[3d-generation-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/3d_generation_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **11 句话搞定 3D Generation** — Embodied AI / AR / VR 面试核心要点（详见后文 §1–§12 推导）。

1. **三大表示**：**NeRF**（隐式神经场 + 体渲染）、**3DGS**（显式 Gaussian 点云 + 光栅化）、**Mesh / SDF**（显式表面 / 隐式距离场）。重建质量与速度的 sweet spot：3DGS（Kerbl 2023 SIGGRAPH Best Paper）。

2. **NeRF 核心公式**：$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\sigma(\mathbf{r}(t))\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt$，其中 $T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)$ 是 transmittance。离散化得到 $\alpha$-compositing：$C \approx \sum_i T_i (1-e^{-\sigma_i\delta_i})\mathbf{c}_i$。

3. **Instant-NGP** (Müller 2022 SIGGRAPH)：**多分辨率 hash 网格** + tiny MLP，约 4+ 个数量级加速；hash collision 由 MLP 在含碰撞表项上自动学习消歧（被 loss + 多尺度冗余共同压制）。

4. **3DGS 核心**：场景表示为一组 3D Gaussian $\{\mu_i, \Sigma_i, \alpha_i, c_i(\mathbf{d})\}$，**可微光栅化**通过将 3D 协方差用 Jacobian $J$ 投影到 2D：$\Sigma' = J W \Sigma W^\top J^\top$，按深度排序后做 front-to-back alpha-blending。

5. **DreamFusion SDS** (Poole et al. 2022 arXiv → ICLR 2023 Outstanding)：用 pretrained 2D diffusion 监督 3D 表示：$\nabla_\theta \mathcal{L}_\text{SDS} = \mathbb{E}_{t,\epsilon}[w(t)(\epsilon_\phi(x_t;y,t)-\epsilon)\,\partial x/\partial \theta]$，**故意去掉 U-Net 对 $x_t$ 求导的 Jacobian 项**，使得训练 simulation-free。代价：mode-seeking → over-saturation / Janus。

6. **VSD** (Wang 2023 NeurIPS, ProlificDreamer)：把 3D 参数 $\theta$ 视为 random variable $\mu(\theta)$，**最小化的是渲染加噪图像分布**之间的 KL：$\mathbb{E}_t\big[D_\text{KL}\big(q_\mu^t(x_t|y)\,\|\,p_\phi^t(x_t|y)\big)\big]$。梯度形式为 **relative score** $\nabla_\theta \approx (\epsilon_\phi(x_t;y,t) - \epsilon_\psi(x_t;y,t,\pi))\,\partial x/\partial\theta$，其中 $\epsilon_\psi$ 是 LoRA 微调的辅助 score。CFG 可从 100 降至 7.5。

7. **Single-image / Few-view 3D**：Zero-1-to-3 (Liu 2023 ICCV) 用 viewpoint conditioned diffusion；SyncDreamer / MVDream 学多视图联合一致性；TripoSR / InstantMesh / Stable Fast 3D 把 image-to-mesh 推到秒级（TripoSR ~0.5 秒、InstantMesh ~10 秒）。

8. **Feed-forward 重建**：**DUSt3R** (arXiv:2312.14132) 把 SfM 换成一次前向——直接回归 pointmap，两张图的 3D 点都落在第一张图的相机系里，**位姿成了输出而不是输入**；损失除以有效点到原点的平均距离，所以默认 up-to-scale。**VGGT** (2503.11651, CVPR 2025 Best Paper) 一个 backbone 引出四个 head（相机 / depth / point map / track）。COLMAP 由此从"必经"降级为"精度基准"（§7、§12.1）。

9. **3D Foundation Models (2024-26 开源)**：**TRELLIS** 的 **SLAT** = 稀疏体素坐标 + per-voxel latent（$N=64$，$L\approx 20\text{K}$ active voxel），两阶段 rectified flow，一份 latent 解码成 3DGS / 场 / mesh；**Hunyuan3D** 沿 2.0 → 2.1 → 2.5 → Omni / Studio / Buffalo 迭代（**没有 3.0**），shape→texture 两阶段；**CLAY** (arXiv:2406.13897) 多分辨率 VAE + latent DiT。

10. **原生 mesh 与 latent 表示这条轴**：无论是从场里提等值面（marching cubes、FlexiCubes）还是像 TRELLIS.2 那样从 O-Voxel 直接转 mesh，**这些输出都不保证具有适合编辑、绑骨和形变的边流**——这才是美术不收货的原因。**MeshGPT → MeshAnything V2 (AMT) → BPT → TreeMeshGPT** 这条线在压 token（基线是朴素序列的 9 token / 面），**Meshtron 则换成用架构扛长序列**（hourglass + 滑动窗口）。latent 表示（SLAT / 稀疏体素 / VecSet / triplane）都能按坐标查询解码，它决定的不是"能出几种格式"，而是**计算与显存摆在哪里**。

11. **Embodied AI 关键应用**：Sim2Real 资产生成、NeRF/3DGS 作为可微 simulator、language-conditioned 3D affordance。**面试常见交叉**：NeRF SLAM、Gaussian-Splat scene editing、3D 物理一致性。

---

## §10 复杂度 / 资源对比

| 方法 | 训练 | 推理耗时 | 运行显存（注明阶段） | 模型 / 表示规模 |
| --- | --- | --- | --- | --- |
| NeRF vanilla | 1-2 天 | 数秒 | 8 GB | <10 MB MLP |
| Instant-NGP | 5 秒 - 5 分钟 | 30 fps+ | 4-12 GB | 100-500 MB hash |
| 3DGS | 10-30 分钟 | 100 fps+ | 6-24 GB | 100 MB - 1 GB Gaussian |
| 2DGS | 与 3DGS 接近 | 与 3DGS 接近 | 类似 | 类似 |
| DreamFusion (NeRF+SDS) | 2 hr / 物体 | — | 12 GB | NeRF 本身 |
| DreamGaussian (3DGS+SDS) | 2 分钟 / 物体 | — | 8-16 GB | — |
| ProlificDreamer (VSD) | 3-6 hr / 物体 | — | 24 GB | — |
| TripoSR feedforward | 训练 50 GPU 天 | 0.5 秒 (A100) | inference 6 GB | 1.5 GB |
| GS-LRM / Long-LRM（前馈 3DGS） | 大规模多视图训练 | 0.23 秒 / 约 1 秒（32 视图 960×540） | — | — |
| VGGT（前馈重建） | 大规模多视图训练 | 秒级 / N 张图一次前向 | — | ~1B 参数 |
| TRELLIS | 训练 100+ GPU 天 | 数秒 | inference 16 GB | 数 GB |
| Hunyuan3D 2.0 / 2.1 | 训练大集群 | 数十秒 | inference 24+ GB | 多模型组合 |
