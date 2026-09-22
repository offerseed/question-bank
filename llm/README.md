# 大语言模型（LLM）题库

> 系统整理 LLM 相关面试题，覆盖预训练、微调、对齐、推理、应用和 Agent。

## 基础题目

| 序号 | 题目 | 难度 | 标签 |
| :---: | :--- | :---: | :--- |
| 01 | [GPT 的预训练任务是什么？](./预训练与微调.md#01-预训练) | ⭐⭐ | 预训练 |
| 02 | [LoRA / QLoRA 微调原理](./预训练与微调.md#02-lora) | ⭐⭐⭐ | 微调 |
| 03 | [RLHF 和 DPO 的区别](./对齐技术.md#03-rlhf-dpo) | ⭐⭐⭐ | 对齐 |
| 04 | [位置编码演进（RoPE/ALiBi）](./架构进阶.md#04-位置编码) | ⭐⭐⭐ | 架构 |
| 05 | [KV Cache 原理和优化](./推理优化.md#05-kv-cache) | ⭐⭐⭐ | 推理 |
| 06 | [PagedAttention / vLLM](./推理优化.md#06-pagedattention) | ⭐⭐⭐ | 推理 |
| 07 | [RAG 系统设计](./应用实践.md#07-rag) | ⭐⭐ | 应用 |
| 08 | [Agent 架构设计](./应用实践.md#08-agent) | ⭐⭐⭐ | 应用 |

## ARIS 精选面试题（每篇 25 题，L1/L2/L3 分层）

> 来源：[ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer)（已授权）

| 主题 | 文件 |
| :--- | :--- |
| 分词器 Tokenization | [aris-tokenization_tutorial.md](./aris-tokenization_tutorial.md) |
| RLHF / DPO / GRPO / PPO | [aris-rlhf_dpo_grpo_ppo_tutorial.md](./aris-rlhf_dpo_grpo_ppo_tutorial.md) |
| RLHF 中的 KL 散度 | [aris-kl_divergence_rlhf_tutorial.md](./aris-kl_divergence_rlhf_tutorial.md) |
| LLM 在线蒸馏 OPD | [aris-llm_opd_tutorial.md](./aris-llm_opd_tutorial.md) |
| 推理模型 Reasoning | [aris-reasoning_models_tutorial.md](./aris-reasoning_models_tutorial.md) |
| KV Cache 与投机解码 | [aris-kv_cache_speculative_decoding_tutorial.md](./aris-kv_cache_speculative_decoding_tutorial.md) |
| 长上下文 RoPE/YaRN/MLA | [aris-long_context_rope_yarn_mla_tutorial.md](./aris-long_context_rope_yarn_mla_tutorial.md) |
| MoE 混合专家 | [aris-moe_tutorial.md](./aris-moe_tutorial.md) |
| 量化 Quantization | [aris-quantization_tutorial.md](./aris-quantization_tutorial.md) |
| LLM 推理服务 | [aris-llm_inference_serving_tutorial.md](./aris-llm_inference_serving_tutorial.md) |
| LLM 评估与基准 | [aris-llm_evaluation_benchmarking_tutorial.md](./aris-llm_evaluation_benchmarking_tutorial.md) |
| LLM 预训练流程 | [aris-llm_pretraining_pipeline_tutorial.md](./aris-llm_pretraining_pipeline_tutorial.md) |
| LoRA 与 PEFT | [aris-lora_peft_tutorial.md](./aris-lora_peft_tutorial.md) |
| RAG 与文本嵌入检索 | [aris-rag_embedding_retrieval_tutorial.md](./aris-rag_embedding_retrieval_tutorial.md) |
| Agent 基础 | [aris-agent_foundations_tutorial.md](./aris-agent_foundations_tutorial.md) |
| Agentic RL | [aris-agentic_rl_tutorial.md](./aris-agentic_rl_tutorial.md) |
| 自进化 Agent | [aris-self_evolving_agents_tutorial.md](./aris-self_evolving_agents_tutorial.md) |
| 多 Agent 长时程 | [aris-multi_agent_long_horizon_tutorial.md](./aris-multi_agent_long_horizon_tutorial.md) |

## 推荐学习路线

1. [预训练与微调](./预训练与微调.md) — 理解模型如何训练
2. [对齐技术](./对齐技术.md) — RLHF/DPO 如何让模型安全有用
3. [架构进阶](./架构进阶.md) — RoPE、GQA、SwiGLU 等改进
4. [推理优化](./推理优化.md) — KV Cache、PagedAttention、量化
5. [应用实践](./应用实践.md) — RAG 和 Agent 系统
6. ARIS 系列：Tokenization → 预训练 → LoRA/PEFT → RLHF/DPO → 推理服务 → Agent
