# 世界模型 — 高频面试题

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）
> 教程原文：[world-models-tutorial.md](https://github.com/wanshuiyin/ARIS-in-AI-Offer/blob/main/docs/tutorials/world_models_tutorial.md)

---

## §0 TL;DR Cheat Sheet

> 💡 **一句话** — world model 的面试题从来不是"谁家的世界模型更强"，而是三个机制问题：**模型保留什么信息、怎样预测动作后果、这些预测如何进入决策**。三条技术谱系给出三套答案，而且它们互相重叠。

1. **三条谱系**：(1) latent dynamics / model-based RL——学一个能在里面 rollout 的紧凑状态空间；(2) predict-in-representation-space（JEPA）——只预测编码器的输出，不重建像素；(3) 生成式视频与交互式模拟器——直接生成可交互的未来观测。
2. **谱系不是阵营**：Dreamer 4 同时是生成式动态模型和 imagination RL；Cosmos 3 横跨理解、生成与动作；视频扩散模型同样在 VAE latent 里工作——"latent" 不是 RSSM/JEPA 的专有词。
3. **状态**：环境真实状态 $x_t$ 不可观测，模型状态 $s_t=(h_t,z_t)$ 是它的替身。RSSM 递推全篇固定写成 $h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})$：$h_t$ 提供跨时刻记忆，$z_t$ 承载随机性与新信息。
4. **prior vs posterior**：$p_\theta(z_t\mid h_t)$ 用于预测，$q_\phi(z_t\mid h_t,o_t)$ 吸收当前观测。**imagination 里只能用 prior**——调用后验等于偷看还没发生的观测。
5. **KL balancing**：$\mathcal L_{KL}=\beta_{dyn}D_{KL}(\mathrm{sg}\,q\Vert p)+\beta_{rep}D_{KL}(q\Vert\mathrm{sg}\,p)$。两项都是 $q\Vert p$ 方向，分开的是**谁被停梯度**，不是 KL 方向，更不是"把系数调小"。
6. **决策接口四种**：PlaNet 在潜空间在线规划；Dreamer 在 imagination 里训 actor-critic；MuZero 用 MCTS 且完全不重建观测；TD-MPC2 decoder-free 地学动态+奖励+价值供 MPC 使用。
7. **JEPA 公共结构**：$P_\theta(E_\phi(x),m)\to\mathrm{sg}(E_{\bar\phi}(y))$，非对称 predictor + EMA 目标分支 + 停梯度。这套组合**不是不塌缩定理**：常数 encoder 配常数 predictor 依然是退化解。
8. **生成式路线的关键不是画质而是动作接口**：latent action model 的小 codebook、相机位姿、文本条件、执行器动作，语义完全不同；无动作视频只提供先验，要用于动作选择还得补上接口和证据。
9. **数字带口径**：Genie 3 的 720p / 24 FPS / 数分钟一致性全部出自 2025-08-05 的官方博客，无论文、无公开参数量（11B 是 Genie 1 的）；GameNGen 的速度按版本报——v1 摘要写 ">20 FPS"，v2 正文是**单块 TPU-v5、4 步 DDIM 下的 20 FPS**，29.4 dB 是**下一帧预测**的 PSNR，不是长程 rollout 的保真度，更不是物理正确性证据。
10. **面试杀手题**：KL balancing 为什么不等于调小系数、V-JEPA 2-AC 的 zero-shot 到底 zero 了什么、以及"视频逼真能否证明懂物理"——Physics-IQ 给的是实验性答案（跨模型 $r=-0.46$、$p=0.249$，未检出显著相关），不是"两者无关"的定论。

---

---

## §10 25 高频面试题

### L1 必会题（10 题）

<details>

<summary>Q1. 什么是 world model？</summary>

智能体内部学到的环境结构与动态模型：给定至今的观测与动作，预测"这样做之后会怎样"。**必要的输出取决于用途**——要在模型里做 RL，就得能在想象里算回报并截断 episode（Dreamer 这类实现为此学奖励头与 continuation 头；奖励函数或终止规则已知时用已知的那份也行）；要做视觉目标导航只需能比较目标表征的距离；要当可玩模拟器就必须生成人能看的观测。回答时把三个机制问题摆出来：保留什么信息、预测哪个量、这个量怎么进决策。

只说"能预测未来的模型"，或直接把它等同于视频生成器，都不得分。

</details>

<details>

<summary>Q2. RSSM 的 prior 和 posterior 分别是什么？imagination 时用哪个？</summary>

递推是 $h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})$。prior $p_\theta(z_t\mid h_t)$ 看不到当前观测，posterior $q_\phi(z_t\mid h_t,o_t)$ 吸收观测。训练时两者都在：重建走 posterior，KL 把 prior 拉向 posterior。**imagination 里没有未来观测，只能用 prior**——调用后验等于把还没发生的观测偷看进来，想象出的回报就不可信了。

</details>

<details>

<summary>Q3. PlaNet 和 Dreamer 怎样选动作？</summary>

PlaNet 不训练策略网络：每一步在潜空间用 **CEM** 在线规划——从一个高斯分布采一批动作序列、用 prior 向前 rollout、用奖励头打分，取 elite 序列重新拟合分布，迭代若干轮后执行**末轮分布均值**的第一个动作，下一步重来，全程不解码图像。说成"采一次样、执行得分最高的那条样本"就漏掉了 CEM 的迭代重拟合。Dreamer 在 imagination 里训练 actor-critic：用 posterior 取真实起点，只用 prior 展开 $H$ 步，在想象状态上更新策略与价值，部署时策略一次前向出动作。

一句话差别：**PlaNet 把算力花在决策时，Dreamer 花在训练时。**

</details>

<details>

<summary>Q4. 视频看起来逼真，能证明模型理解物理吗？</summary>

Physics-IQ（arXiv 2501.09038）给的是实验性答案：真实拍摄 66 个物理场景、396 段视频，给模型条件帧或条件视频（视被测模型是图生视频还是视频续写而定），让它生成五秒后续，指标是空间 IoU、时空 IoU、加权空间 IoU 与 MSE。它同时用"MLLM 能否区分真假视频"衡量视觉真实感，跨模型相关为 $r=-0.46$、$p=0.249$——**在这组模型与样本量下未检出显著相关**。

说成"研究证明视觉质量与物理理解无关"就错了：未检出显著相关不是证明独立。正确说法是"在 Physics-IQ 的这组测量里，逼真没有换来物理指标上的优势"。

</details>

<details>

<summary>Q5. $h_t$ 和 $z_t$ 各自负责什么？</summary>

$h_t$ 是确定性的，提供**不经过随机重采样的递推记忆通道**；$z_t$ 是随机的，表达环境随机性与当步的新信息。别把确定性说成"无损"——固定宽度的 GRU 状态一样会压缩和遗忘，它只是不被每步采样噪声冲刷。两个方向都别推广过头：**不是只有递推向量才能建模记忆**（transformer 用 attention 直接回看历史），也**不是任何确定性 world model 都不能工作**（确定性环境里它可以很好）。

</details>

<details>

<summary>Q6. Dreamer 一类模型有哪几个预测头？纯视频模型也需要吗？</summary>

观测、奖励、continuation 三个。后两个是为了"在模型里做 RL"：**在 Dreamer 这套实现里**，想象里的回报由学到的奖励头给出、episode 的截断由学到的 continuation 头给出，缺哪个都算不下去。**纯视频生成模型未必需要它们**——它的用途是生成可看、可交互的未来。

但这条必要性只到这套实现为止。奖励函数或终止规则已知时（大量仿真与游戏环境就是），直接用已知的那份同样能在想象里做 RL；观测重建更不是必要条件——MuZero 与 TD-MPC2 都不重建观测照样规划。把"有没有奖励头"当成 world model 的定义线，会在第一个追问里塌掉。

</details>

<details>

<summary>Q7. Genie 3 有多大？和 Genie 1 的 11B 是什么关系？</summary>

Genie 3 **没有论文、也没有公开参数量**；720p、24 FPS、"数分钟"级一致性、约一分钟视觉记忆，全部来自 2025-08-05 的官方博客。**11B 属于 Genie 1**（arXiv 2402.15391），它的三组件（视频 tokenizer、8 个码的 latent action model、动态模型）同样不能拿来推断 Genie 3 的内部结构。Project Genie（2026-01-29 博客）的 60 秒是**消费者原型的限制**，不是 Genie 3 的研究上限。

给 Genie 3 报 11B 是硬伤。

</details>

<details>

<summary>Q8. 三条谱系里说的 "latent" 是同一个东西吗？</summary>

不是。RSSM 的 latent 是带 prior/posterior 的状态变量，服务于在里面 rollout；JEPA 的 latent 是编码器输出的表征，没有生成式先验；视频扩散模型的 latent 是 VAE 压缩后的生成空间。共同点只有一条：**不在像素上算**。用"谁在 latent 里工作"来区分谱系，会把视频模型误分到 JEPA 那边。

</details>

<details>

<summary>Q9. 无动作视频预训练的模型能直接拿来选动作吗？</summary>

不能直接。无动作视频提供的是"世界通常怎么演化"的先验，很有价值。要用于**动作选择**，必须补上动作接口，并给出该接口下的决策证据——文本条件、相机运动、执行器动作、LAM 码的语义完全不同，不能互相顶替。V-JEPA 2 的做法就是分两段：预训练 action-free，第二阶段用带动作的数据训 action-conditioned predictor。

</details>

<details>

<summary>Q10. DreamerV3 的"一套超参数"是什么意思？它的钻石怎么来的？</summary>

指**同一套超参数**覆盖 150 多个任务，即不必逐环境调参；**不是**一套权重通吃所有环境——每个任务仍各自训练。它在 Minecraft 里从零收集到钻石，是**通过与环境交互**做到的。Dreamer 4（arXiv 2509.24527）的"离线拿钻石"是另一篇、另一套设定，两者不能混为一谈。

</details>

### L2 进阶题（10 题）

<details>

<summary>Q11. KL balancing 是不是就是把 KL 系数调小？</summary>

不是。$\mathcal L_{KL}=\beta_{dyn}D_{KL}(\mathrm{sg}\,q\Vert p)+\beta_{rep}D_{KL}(q\Vert\mathrm{sg}\,p)$：**两项都是 $q\Vert p$ 方向**，分开的是谁被停梯度。$\beta_{dyn}\gg\beta_{rep}$ 的含义是"先让 prior 去追 posterior（学动态），再让 posterior 为了好预测让步（正则表征）"。调小单一系数只是整体放松一致项，得到的是另一种失衡：要么 posterior 退化、重建全靠 $h_t$，要么 prior 永远追不上 posterior，imagination 一展开就漂。系数按版本报：DreamerV3 v1 的式 4 / 表 W.1 是 $\beta_{dyn}=0.5,\beta_{rep}=0.1$，v2 正文是 $1$ 与 $0.1$。

写成 $D_{KL}(p\Vert q)$、或说 balancing 改变了 KL 方向，都不得分。

</details>

<details>

<summary>Q12. JEPA 怎样避免 collapse？</summary>

靠一组机制的组合：非对称结构（predictor 只在一侧）、EMA 的慢速目标分支、目标分支停梯度、掩码任务本身的难度。关键是下一句——**这不是不塌缩定理**：常数 encoder 配常数 predictor 依然是损失极低的退化解，效果依赖初始化、EMA 系数、掩码难度与优化器设置，实践中要监控表征方差或秩。

§9.2 对应的 toy（见 §A.3 实验 2）把这件事跑出来：共享权重且不停梯度时，上下文表征的方差按几何级数衰减；换成停梯度 + EMA 教师后，对称点处梯度为零，塌缩不再是吸引子——但从塌缩点出发它照样停在那里。

</details>

<details>

<summary>Q13. V-JEPA 2-AC 的 zero-shot 到底指什么？</summary>

指**新部署环境不再采数据**：在 Franka 机械臂上给定视觉目标做 reaching / grasp / pick-and-place，不在部署实验室重新采集。结构上，预训练仍是 action-free 的 masked prediction（$L_1$、EMA、停梯度）；2-AC 冻结 encoder，只训 action-conditioned predictor，输入含机器人动作与末端状态。论文说的 "<62 h unlabeled robot video" 是**没有任务语义标签**，不是没有动作记录。规划是表征空间的 MPC：预测动作序列执行后的表征，最小化与目标图像表征的 $L_1$ 距离，CEM 搜索，执行首个动作后重规划——**目标表征不是奖励标签**。

它不证明任意机器人、任意任务、任意长程分解都免采数据。

</details>

<details>

<summary>Q14. 把 VQ codebook 调得很小，就能自动发现真实动作吗？</summary>

不能保证。小 codebook 的论据是**容量**：一个码至多携带 $\log_2 K$ bits（$K=8$ 即 3 bits），逼模型丢掉帧间的大部分差异。但容量小只保证"留下的信息少"，**不保证留下的是执行器动作**，也不保证同一个码跨场景含义一致。arXiv 2506.15691 给出了机制：动作引起的变化与外生变化（光照、相机抖动、背景运动）**在竞争同一个瓶颈容量**，方差大的无关变化可能被优先编码；线性情形下这与 PCA 保留最大方差方向有直接联系。

</details>

<details>

<summary>Q15. 写出 imagination 里的 λ-return，并说明起点与长度约定。</summary>

按 §2 的时间约定：

$$G_t^\lambda=\hat r_{t+1}+\gamma\hat c_{t+1}\big[(1-\lambda)v(s_{t+1})+\lambda G_{t+1}^\lambda\big],\qquad G_H^\lambda=v(s_H)$$

critic 的回归目标要停梯度。DreamerV3 v1 的表 W.1 取 $H=15$、正文的预测序列长度 $T=16$：**以 $H$ 计动作转移数，状态序列含起点因而有 $H+1$ 个状态**。被追问"到底是 15 还是 16"时，答"数转移是 15、数状态是 16"。

</details>

<details>

<summary>Q16. free bits 和分类潜变量的 1% 均匀混合是同一件事吗？</summary>

不是，两件独立机制。**free bits** 是 $\max(\tau, D_{KL})$，对**联合 KL**（先对所有 latent 组求和）截下限，防止 KL 被压过头把表征容量耗光。**1% 均匀混合**是 **DreamerV3**（v1 正文与附录 C）给分类概率混入均匀分布，防止某一类概率被压到 0 导致 $\log$ 爆炸与梯度病态；它不是 DreamerV2 的改动，DreamerV2 带来的是分类潜变量本身。一个管目标函数的形状，一个管分布的数值健康。

另外，分类潜变量 + straight-through 本身是**有偏**的梯度近似——它把不可导的采样当恒等映射来传梯度，这是第三件事。

</details>

<details>

<summary>Q17. symlog 和 twohot 的分工是什么？</summary>

**symlog** 是尺度压缩：$\mathrm{symlog}(x)=\mathrm{sign}(x)\log(1+\lvert x\rvert)$，逆变换 symexp，对称且在 0 附近近似恒等，用来把跨数量级的信号压进可训练范围。**twohot** 是编码方式：把连续标量按距离分给相邻两个桶，于是回归变成分类，期望值可精确恢复。

DreamerV3 v1（式 10 之后的正文与附录 C）里 **reward 与 critic 都用 symlog twohot**——**不要把它推广到所有预测头**，观测与 continuation 头是另一回事。

</details>

<details>

<summary>Q18. DreamerV3 的 return normalization 和普通 advantage 标准化差在哪？</summary>

它取一个批次回报的**第 95 与第 5 分位之差** $S$ 做 EMA，再除以 $\max(1,S)$。三点不同：只做尺度、不减均值；**有下限 1**；跨批次平滑。下限是要害——稀疏奖励任务里回报几乎全为 0，$S$ 很小，普通的"减均值除标准差"正好会把近零回报的噪声放大成巨大梯度，而 $\max(1,S)$ 把这条路堵死。

</details>

<details>

<summary>Q19. PETS 和 MBPO 解决的是同一个问题吗？</summary>

不是。**PETS**（arXiv 1805.12114）解决**不确定性表示**：概率模型的 ensemble 同时表达环境噪声与数据不足带来的不确定性，规划时用轨迹采样把它传播下去。**MBPO**（arXiv 1906.08253）解决**误差累积**：从真实数据里的状态分支出很短的模型 rollout 再做 off-policy 更新，把模型偏差的复合控制住。

一句话：PETS 说"我不确定"，MBPO 说"我只敢往前看几步"。

</details>

<details>

<summary>Q20. Dreamer 4 为什么不能套用 DreamerV3 的描述？它的离线结论限定条件是什么？</summary>

骨架不同：Dreamer 4 是 **tokenizer + transformer 动态 + shortcut forcing**，没有 RSSM、KL balancing、分类潜变量那一套，硬套等于答错架构。它的钻石是在论文的**离线 Minecraft 设定**下取得的，训练流程包含行为克隆、奖励建模和在模型内部做 RL；论文报告的"单 GPU 实时"指的是**交互推理**，不是训练。DreamerV3 的钻石则来自与环境交互——两个结论不能互相替代。

</details>

### L3 顶级 lab 题（5 题）

<details>

<summary>Q21. 什么时候该预测像素，什么时候该预测表征？</summary>

判据是**决策依赖哪些信息**，不是哪条路线更先进。

预测像素的理由：细节可能直接决定接触、碰撞与奖励（DIAMOND 的实验就支持这一点——被压掉的小物体和细边缘往往正是触发条件）；需要人看、需要当可交互模拟器、需要一个跨任务通用的观测接口。**扩散生成不是"用 MSE 预测一个平均未来"**，它建模条件分布，采出的是一个具体清晰的未来。

预测表征的理由：忽略不可预测的细节省算力，预测集中在对下游有用的量上；下游本来就是表征空间的距离或 MPC（DINO-WM 在冻结 DINOv2 特征上学动态、V-JEPA 2-AC 用目标表征的 $L_1$）。

两条各有硬边界：像素路线**画得像不等于动得对**；表征路线**表征损失小不等于动作后果正确**——表征空间的距离是模型自己定义的，与"这个动作会不会把杯子打翻"之间没有免费的等价关系。

</details>

<details>

<summary>Q22. MuZero 不重建观测，为什么还能规划？</summary>

因为规划只用得到搜索树里的那几个量：奖励、价值、策略先验。潜状态不必能恢复环境真实状态 $x_t$，只需对这些量**预测充分**。代价有两个：没有解码器就没有"模型在想什么"的可视化通道，调试只能靠奖励预测误差、规划收益这类间接指标；而且"预测充分"是设计原则加实证结果，**不是严格的 value-equivalence 定理**。

TD-MPC2 是同一思路在连续控制上的版本：decoder-free 地学潜空间动态、奖励与价值供 MPC 用，其规模实验里 317M 的单一 agent 覆盖 80 个任务。

</details>

<details>

<summary>Q23. 离线学到的 world model，为什么不能靠"无限 imagination"解决探索？</summary>

因为模型只在数据覆盖的状态—动作分布上可信。想象再多次也不会凭空产生数据里没有的信息；更糟的是策略优化会**主动找到并利用模型高估回报的轨迹**——在模型里回报极高，放回真实环境不成立。这正是 MBPO 用短分支 rollout、PETS 用不确定性传播要压住的东西。

两件事要分开说：**扩大覆盖只能靠补充数据或新的环境交互**；**悲观/不确定性惩罚不扩大覆盖**，它做的是把决策约束回模型有证据的区域，让策略别去赌那些数据没覆盖到的高回报幻觉。把惩罚项说成"扩大覆盖的第二条路"是答反了方向。

Dreamer 4 的离线结果不是反例：它有为那个设定准备的离线数据、行为克隆与奖励建模，结论的作用域是那套设定。

</details>

<details>

<summary>Q24. 怎样判断一个 world model 对机器人是真的有用？</summary>

问四件事。**一、动作接口是什么**——执行器动作、相机位姿、LAM 码还是文本条件？只有第一种直接对应可执行控制。**二、决策证据是什么**——闭环成功率、哪个平台、多少任务、有没有换规划设置？V-JEPA 2.1 表 6 就是现成例子：同一规划设置下 grasp 从 60% 到 70%（+10 个百分点）是表征比较，改了规划设置并放长 horizon 后的 80%（+20 个百分点）已经不是纯表征比较。**三、误差在哪累积**——长 horizon 一致性、记忆保持、接触动力学。**四、成本口径**——速度在什么硬件、什么部署（Matrix-Game 3.0 的最高 40 FPS @ 720p 是 8 卡跑 DiT + 1 卡做 VAE 解码的异步部署，不是单卡）。

视频质量分数、表征 probe 准确率都不能替代前三问——V-JEPA 2 常被引用的 77.3 是 SSv2 的 attentive-probe top-1，不是机器人成功率。

</details>

<details>

<summary>Q25. 给你一份刚发布的 world model 报告，怎样快速定位它？</summary>

先按三问读：**预测哪个量**（像素 / 表征 / 规划量）、**动作接口是什么**、**预测如何进决策**。这三问定谱系，比看它自称什么可靠——Dreamer 4 同时是生成式动态与 imagination RL，Cosmos 3 横跨理解、生成与动作。

再查证据层级：有论文还是只有博客；代码、权重、数据分别开放到哪一步（**计划发布不算已开放**）；每个数字有没有给全任务、指标、模型版本、设置与硬件。

最后放进评测坐标：WorldModelBench 测指令遵循与物理/常识违规，WorldScore 测可控性、质量与动态性，WorldRoamBench 测交互过程中的动作跟随、视觉一致、物理合理与记忆保持，WorldArena 2.0 把视觉触觉、模型内策略优化与真实机器人平台纳进来。它们问的不是同一件事，某一项好不能替另一项背书；这些基准里"没有模型全面满足"的结论，也只限定在它们测过的模型与协议之内。

分级要小心：L0–L7（arXiv 2606.15032）是**多个交叉轴**上的提案；Predictor / Simulator / Evolver 出自 **Agentic World Modeling（arXiv 2604.22748）**，别记到 Definition & Roadmap 头上——后者（arXiv 2607.06401）是又一份提案，用的是另一套角色划分（Renderer / Simulator / Planner 等）。**这几套都不是领域共识**；GLP critique（arXiv 2507.05169）代表的是另一种立场。§10 的 L1/L2/L3 只表示面试题难度。

</details>

---

## §A 附录

### A.1　论文与出处

证据层级：**✅ 论文**（arXiv 公开，引用的细节取自论文）／**⚠️ 仅博客**（只有官方博客或新闻稿，无论文）。第三种情况单独写明：**有论文但只提方向**（未逐项核验其数字），不并进"仅博客"。

**谱系一：latent dynamics / MBRL**

| 工作 | 出处 | 一句话 |
| --- | --- | --- |
| World Models | ✅ 1803.10122 | VAE + MDN-RNN + 线性 controller；**VizDoom 实验**里策略完全在 dream 里训练，CarRacing 的 controller 在真实环境训练 |
| PlaNet | ✅ 1811.04551 | RSSM；latent overshooting；潜空间在线规划 |
| Dreamer | ✅ 1912.01603 | imagination 里训 actor-critic（路径梯度） |
| DreamerV2 | ✅ 2010.02193 | 分类潜变量；KL balancing；从想象中达到 Atari 人类水平 |
| DreamerV3 | ✅ 2301.04104 | symlog / twohot / return normalization；一套超参数覆盖 150+ 任务；Minecraft 钻石来自环境交互 |
| Dreamer 4 | ✅ 2509.24527 | tokenizer + transformer 动态 + shortcut forcing；离线 Minecraft 设定下的钻石；单 GPU 实时**推理** |
| MuZero | ✅ 1911.08265 | 只学奖励/价值/策略量，无重建；MCTS |
| TD-MPC2 | ✅ 2310.16828 | decoder-free 潜动态 + 奖励 + 价值供 MPC；317M 单 agent / 80 任务 |
| DayDreamer | ✅ 2206.14176 | 该实验中四足约 1 小时真实交互学会行走 |
| PETS | ✅ 1805.12114 | 概率 ensemble + 轨迹采样 = 不确定性表示 |
| MBPO | ✅ 1906.08253 | 从真实数据分支的短 rollout = 控制模型偏差 |
| IRIS | ✅ 2209.00588 | 离散 tokenizer + transformer 想象 |
| DIAMOND | ✅ 2405.12399 | 扩散 world model；视觉细节影响控制回报 |

**谱系二：predict-in-representation-space**

| 工作 | 出处 | 一句话 |
| --- | --- | --- |
| LeCun 立场文章 | OpenReview v0.9.2（2022-06-27） | 在抽象表征空间预测；**无 arXiv 编号** |
| I-JEPA | ✅ 2301.08243 | 非对称 predictor + EMA 目标；patch 表征平方 $L_2$；多块 masking |
| V-JEPA | ✅ 2404.08471 | 视频 masked feature prediction；$L_1$；无负样本、无像素重建 |
| V-JEPA 2 / 2-AC | ✅ 2506.09985 | 预训练 action-free；2-AC 冻结 encoder 训动作条件 predictor + 表征空间 MPC |
| V-JEPA 2.1 | ✅ 2603.14482 | masked 与 visible-context token 都受监督；多层 deep self-supervision；规模扩展 |
| DINO-WM | ✅ 2411.04983 | 在冻结 DINOv2 特征上学动态，用目标特征规划 |
| JEPA 泛化理论 | ✅ 2606.27014 | 条件谱图 / 动作条件共现矩阵低秩分解 → 预训练误差与规划 regret |
| VL-JEPA | ✅ 2512.10942 | 把语言接入同一框架；**论文 2512.10942；仅提方向**，不引用其具体数字 |

**谱系三、具身与评测**

| 工作 | 出处 | 一句话 |
| --- | --- | --- |
| Genie | ✅ 2402.15391 | 11B；视频 tokenizer + 8 码 LAM + 动态模型；MaskGIT 式采样 |
| Genie 3 | ⚠️ 博客 2025-08-05 | 720p / 24 FPS / 数分钟一致性 / 约一分钟视觉记忆；无论文、无公开参数量 |
| Project Genie | ⚠️ 博客 2026-01-29 | 消费者原型；60 秒是产品限制 |
| GameNGen | ✅ 2408.14837 | v1 摘要 ">20 FPS"、v2 正文 20 FPS（单 TPU-v5、4 步 DDIM）；29.4 dB 是**下一帧预测** PSNR（DOOM 模拟数字） |
| UniSim | ✅ 2310.06114 | 统一的动作条件交互式模拟器路线 |
| Navigation World Models | ✅ 2412.03572 | 第一人称视频 + 动作上的 conditional diffusion transformer |
| Cosmos WFM | ✅ 2501.03575 | 自回归与扩散两条路径；Predict / Transfer / Reason 是不同任务 |
| Cosmos-Reason1 | ✅ 2503.15558 | 物理常识与具身推理的 VLM，**不是**动态模型证据 |
| Cosmos-Predict2.5 | ✅ 2511.00062 | 2B / 14B 配置；**不继承**给 Cosmos 3 |
| Cosmos 3 | ✅ 2606.02800 | omnimodal；每层 AR reasoner + diffusion generator 参数组；Edge 4B / Nano 16B / Super 64B |
| GAIA-2 | ✅ 2503.20523 | 驾驶：动作、道路布局、驾驶条件作为条件 |
| GAIA-3 | ⚠️ 仅新闻稿 | 只在发布层面提及 |
| Matrix-Game 3.0 | ✅ 2604.08995 | 5B；异步部署（8 卡 DiT + 1 卡 VAE）最高 40 FPS @ 720p；记忆检索、流式生成 |
| Atlas | ⚠️ World Labs 博客 2026-09-01 | 文本/图像/相机位姿/深度/空间上下文上的 AR diffusion；展示最多一分钟 1440p |
| LAM 瓶颈分析 | ✅ 2506.15691 | 动作变化与外生变化竞争瓶颈容量；线性情形与 PCA 相关 |
| UniPi | ✅ 2302.00111 | 视频当计划，动作靠逆动力学推断 |
| DreamGen | ✅ 2505.12705 | 生成视频 → LAM/IDM 恢复伪动作 → 策略数据 |
| Genie Envisioner | ✅ 2508.05635 | 连接世界模型与策略学习、仿真评估 |
| SIMA 2 | ✅ 2512.04797 | 在 Genie 生成世界里行动的 agent；不等于现实迁移 |
| Physics-IQ | ✅ 2501.09038 | 66 场景 / 396 段视频；五秒续写；IoU 与 MSE 指标 |
| WorldModelBench | ✅ 2502.20694 | 指令遵循 + 物理/常识违规 |
| WorldScore | ✅ 2504.00983 | 可控性 / 质量 / 动态性 |
| WorldArena 2.0 | ✅ 2605.17912 | 视觉触觉、模型内策略优化、真实机器人平台 |
| WorldRoamBench | ✅ 2606.31672 | 交互中的动作跟随、视觉一致、物理合理、记忆稳定 |
| GLP critique | ✅ 2507.05169 | 反对把生成式视频模型直接当通用世界模型 |
| Definition & Roadmap | ✅ 2607.06401 | 定义与路线图提案；其分类是另一套角色划分（Renderer / Simulator / Planner 等），**不是** Predictor / Simulator / Evolver |
| L0–L7 position | ✅ 2606.15032 | 多个交叉轴上的分级提案 |
| Agentic World Modeling | ✅ 2604.22748 | 把世界模型放回 agent 决策回路；**Predictor / Simulator / Evolver 出自这一篇** |

**开放状态**：Cosmos 3 的 Edge、Nano、Super 均已发布（Edge 发布于 2026-07-20，见其 HF 模型卡），许可 OpenMDW-1.1；Genie 3、Project Genie、Atlas、GAIA-3 只有官方博客层面的信息（Cosmos-Predict2.5 **有论文**，arXiv 2511.00062）。其余条目只断言"论文已公开"，代码、权重与数据是否开放**以各自官方页面为准**——不逐条断言，因为这四件事经常不同步。

### A.2　排除清单与理由

| 被排除 | 理由 |
| --- | --- |
| Sora / Sora 2 | 只有高层级公开报告；与"动作后果与决策"这条主线不匹配——在 §5.4 作为对照保留 |
| Oasis | 机制已由 GameNGen 代表；可研究性与篇幅 |
| Hunyuan-GameCraft | 同一技术问题已有代表工作覆盖 |
| Runway GWM / Odyssey | 发布公告层面 |
| HunyuanWorld | **有论文**（HunyuanWorld 1.0，arXiv 2507.21809）；排除是因为其主题偏 3D 场景生成与可漫游世界构建，与"动作后果如何进入决策"这条主线不是同一问题，且篇幅有限 |
| STORM / TransDreamer | 机制与 Dreamer / IRIS 重叠 |
| Meta CWM | 术语碰撞：指代码执行意义上的 "world model" |
| Othello-GPT 一类 LLM 内部世界模型 | 问的是"表征里有没有"，不是"能否预测动作后果" |
| Tesla | 无论文 |

### A.3　Runnable toy

脚本：[`code/world_models_toy.py`](code/world_models_toy.py)，纯 PyTorch，CPU 上几秒跑完。三个实验都有解析答案，可以逐项对照——**结论只覆盖各自的显式设定，不是论文复现**。

**实验 1｜RSSM 线性高斯：posterior collapse 与 balanced KL。** 环境 $x_{t+1}=0.5x_t+a_t+\epsilon$，$\epsilon$ 等概率取 $\pm1$。确定性部分 $h=0.5x_t+a_t$ 可精确算出，残差 $\epsilon=o_{t+1}-h$ 是唯一的新信息。标量参数化 $q(z\mid h,o)=\mathcal N(w\epsilon,1)$、$p(z\mid h)=\mathcal N(v,1)$、$\hat o=h+dz$，于是 $\mathcal L_{rec}=\tfrac12[(1-dw)^2+d^2]$、$\mathbb E D_{KL}=\tfrac12(w^2+v^2)$。对比单一 KL 系数 9 与 balanced $\beta_{dyn}=9,\beta_{rep}=0.1$（**toy 取值，不是论文取值**）：从 $w=d=0.5$、$v=1$ 出发，SGD lr 0.05 跑 1,000 步，单系数版收敛到 $\lvert w\rvert,\lvert d\rvert<10^{-6}$——posterior collapse，重建 MSE ≈ 1，等于完全没用上观测；balanced 版收敛到 $v\to0$、$w^2=1/\sqrt{0.1}-1$、$d=w/(1+w^2)$，MSE $=\sqrt{0.1}$。

**实验 2｜JEPA 的捷径与停梯度。** $x=s$、$y=s+\epsilon$，encoder 是 $E_w(u)=wu$，predictor 取恒等。共享 $w$ 且**不**停梯度时 $\mathcal L=\tfrac12w^2$，lr 0.1 的梯度下降给出 $w_n=0.9^n$：从 $w_0=1$ 跑 100 步，上下文表征方差降到 $0.9^{200}$——塌缩是吸引子。换成停梯度 + EMA 教师 $b\leftarrow0.9b+0.1w$：从 $w=b=1$ 出发梯度为零，$w$ 停在 1，损失 0.5 全部来自不可预测的噪声；但从 $w=b=0$ 出发它同样停住。**这正是"机制不是定理"的可执行版本。**

**实验 3｜网格世界上的坐标 LAM。** 编码器取 $\Delta=s'-s$，4 个码的 VQ，解码器是加性的。farthest-first 初始化后做一轮最近邻分配与质心更新，就能精确恢复四个转移向量：验证集重建误差 0、purity 1、$I(K;A)=2$ bits（在码的置换意义下）。这说明在**这个坐标化、无外生变化**的设定里瓶颈确实抓到了动作——一旦加入与动作无关的高方差变化，§5.1 说的竞争就会开始。
