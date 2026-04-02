# Qwen2.5-1.5B Training Project

> 基于 DeepSpeed ZeRO 分布式训练 + 内存优化方案的大模型预训练/微调实战

[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO%203-orange)](https://github.com/microsoft/DeepSpeed)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-blue)](https://github.com/huggingface/transformers)
[![SwanLab](https://img.shields.io/badge/SwanLab-Tracking-green)](https://swanlab.cn)
[![License](https://img.shields.io/badge/License-Apache--2.0-yellow)](LICENSE)

---

## 📌 项目概述

本项目基于 **Qwen2.5-1.5B**（阿里巴巴通义千问团队开源的 15 亿参数大语言模型），实现了：

- **预训练** — 在中文开源语料 Mobvoi Seq Monkey 上的完整预训练
- **SFT 微调** — 使用 BelleGroup 3.5M 中文指令数据进行有监督微调
- **ZeRO 分布式训练** — DeepSpeed ZeRO-2 / ZeRO-3 显存优化配置
- **CPU Offload** — 突破单卡显存瓶颈的内存优化方案
- **显存估算工具** — 训练前预估 GPU 显存需求

---

## 🗂️ 项目结构

```
Qwen_2.5_1.5B/
├── model/Qwen/Qwen2.5-1.5B/      # Qwen2.5-1.5B 模型文件
│   ├── config.json               # 模型架构配置
│   ├── model.safetensors          # 模型权重 (~3GB)
│   └── tokenizer.json            # Tokenizer
│
├── scripts/
│   ├── pretrain.sh               # 预训练 (ZeRO-2)
│   ├── pretrain_optimized.sh     # 预训练优化版 (ZeRO-3 + CPU Offload)
│   ├── finetune_zero3.sh         # 微调 (ZeRO-3)
│   └── finetune_zero3_offload.sh # 微调 (ZeRO-3 + CPU Offload)
│
├── configs/                          # DeepSpeed 配置文件
│   ├── ds_config_zero2.json          # ZeRO Stage 2
│   ├── ds_config_zero3.json          # ZeRO Stage 3
│   └── ds_config_zero3_offload.json  # ZeRO Stage 3 + CPU Offload
│
├── pretrain.py                    # 预训练主脚本
├── finetune.py                    # 微调主脚本
├── eval.py                        # 模型评估脚本
├── download_model.py              # ModelScope 模型下载
├── alternative_download.py         # 替代下载方案
├── login_hf.py                    # HuggingFace 认证
├── memory_estimation.py           # 显存需求估算工具
│
├── output/                        # 预训练输出
├── output_optimized/              # 优化版预训练输出
├── output_zero3/                  # ZeRO-3 微调输出
├── output_zero3_offload/          # CPU Offload 微调输出
│
├── wandb/                         # SwanLab 实验日志
└── README.md
```

---

## 🧠 模型架构

| 参数 | 值 |
|------|-----|
| **模型规模** | 1.5B 参数 |
| **架构** | Qwen2ForCausalLM |
| **隐藏层维度** | 1536 |
| **层数** | 28 |
| **注意力头数** | 12 |
| **KV 头数** | 2 (GQA) |
| **上下文长度** | 131,072 |
| **词表大小** | 151,936 |
| **训练精度** | BF16 |

---

## 🚀 快速开始

### 环境依赖

```bash
pip install transformers deepspeed datasets swanlab torch
pip install modelscope                      # ModelScope 模型下载
```

### 1. 下载模型

```bash
# 方案一：ModelScope (推荐国内用户)
python download_model.py

# 方案二：HuggingFace
python login_hf.py
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir ./model/Qwen/Qwen2.5-1.5B

# 方案三：git LFS
bash git clone https://huggingface.co/Qwen/Qwen2.5-1.5B ./model/Qwen/Qwen2.5-1.5B
```

### 2. 预训练

```bash
# 标准预训练 (ZeRO-2)
bash scripts/pretrain.sh

# 显存不足时：优化版预训练 (ZeRO-3 + CPU Offload)
bash scripts/pretrain_optimized.sh
```

### 3. 微调

```bash
# ZeRO-3 微调
bash scripts/finetune_zero3.sh

# 显存严重不足时：ZeRO-3 + CPU Offload
bash scripts/finetune_zero3_offload.sh
```

### 4. 评估

```bash
python eval.py
```

---

## 💾 显存优化方案

### 遇到 OOM？用这套方案

原始配置直接训练 1.5B 模型需要约 **21 GiB GPU 显存**，很容易爆显存。以下是渐进式优化方案：

### 三档配置对比

| 配置 | GPU 显存 | 适用场景 |
|------|---------|---------|
| **ZeRO-2** | ~12 GiB | 单卡 16 GiB 以上 |
| **ZeRO-3** | ~6 GiB | 单卡 8 GiB 以上 |
| **ZeRO-3 + CPU Offload** | ~1.6 GiB | 显存严重不足 |

> 优化后显存减少约 **92.5%**，可在 2 张消费级 GPU 上运行 15 亿参数模型。

### 核心优化技术

#### 1. ZeRO Stage 3 — 模型状态分片

将模型参数、梯度和优化器状态分片到多 GPU，每张卡只保存 1/N：

```json
// ds_config_zero3.json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto"
  }
}
```

#### 2. CPU Offload — 卸荷到内存

将优化器状态和参数 offload 到系统内存，彻底释放 GPU 显存：

```json
// ds_config_zero3_offload.json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu" },
    "offload_param": { "device": "cpu" }
  }
}
```

#### 3. 梯度检查点 (Gradient Checkpointing)

用计算换内存 — 反向传播时重新计算激活值，激活内存减少约 90%：

```bash
--gradient_checkpointing
```

#### 4. 批量大小与序列长度调优

| 参数 | 原始 | 优化后 |
|------|-----|--------|
| `per_device_train_batch_size` | 16 | 4 |
| `gradient_accumulation_steps` | 4 | 16 |
| `block_size` (序列长度) | 2048 | 1024 |

### 显存估算

训练前使用脚本预估显存需求：

```bash
python memory_estimation.py
```

输出示例：
```
📊 原始配置 (OOM):
  总GPU内存需求估算: 21.26 GiB ❌

🚀 优化配置 (ZeRO-3 + Offload):
  总GPU内存需求估算: 1.59 GiB ✅

📈 优化效果:
  内存减少: 92.5%
```

---

## 📋 关键训练参数

### 预训练

| 参数 | 值 |
|------|-----|
| 预训练语料 | Mobvoi Seq Monkey General Open Corpus |
| 学习率 | 1e-4 |
| Warmup Steps | 200 |
| Batch Size (per device) | 8 |
| 梯度累积步数 | 16 |
| 序列长度 | 1024 |
| 训练精度 | BF16 |
| 梯度检查点 | ✅ 启用 |
| DeepSpeed | ZeRO-3 + CPU Offload |
| 日志 | SwanLab |

### 微调 (SFT)

| 参数 | 值 |
|------|-----|
| 微调语料 | BelleGroup train_3.5M_CN (中文指令数据) |
| 学习率 | 1e-4 |
| Batch Size (per device) | 4 |
| 梯度累积步数 | 8 / 16 |
| 序列长度 | 512 / 1024 |
| 训练精度 | BF16 |

---

## 📊 训练日志

使用 **SwanLab** 跟踪训练过程：

| Step | Loss | Learning Rate | Grad Norm |
|------|------|--------------|-----------|
| 6500 | 5.8204 | 4.96e-05 | 2.11 |

```bash
# 实时查看训练进度
swanlab watch --project pretrain
```

---

## 🛠️ 工具脚本

| 脚本 | 功能 |
|------|------|
| `download_model.py` | ModelScope 模型下载，支持断点续传 |
| `alternative_download.py` | 多方案下载：git LFS / 手动下载 / ModelScope |
| `login_hf.py` | HuggingFace 认证 |
| `memory_estimation.py` | 显存需求估算，比较不同配置 |
| `eval.py` | 基于 EvalScope 的模型评估，支持 OpenAI API |

---

## 🙏 致谢

- **Qwen Team** — 开源 Qwen2.5-1.5B 模型 [@ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B)
- **HuggingFace** — Transformers 生态 [@GitHub](https://github.com/huggingface/transformers)
- **Microsoft DeepSpeed** — 分布式训练优化 [@GitHub](https://github.com/microsoft/DeepSpeed)
- **SwanLab** — 实验跟踪工具 [@swanlab](https://swanlab.cn)

---

## 📝 License

Apache License 2.0
