#!/bin/bash

# 使用所有可用的GPU
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 调整参数以减少内存使用
# 原始配置：per_device_train_batch_size=16, gradient_accumulation_steps=4, block_size=2048
# 调整为更小的批量大小和序列长度以减少内存

deepspeed finetune.py \
    --model_name_or_path ./output_optimized \
    --train_files ./dataset/sft_data/BelleGroup/train_3.5M_CN.json \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --output_dir ./output_zero3 \
    --eval_strategy no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir ./output_zero3/sft/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 1024 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero3.json \
    --report_to swanlab

# 可选：如果仍然内存不足，可以进一步调整：
# 1. 使用CPU offload版本：--deepspeed ./ds_config_zero3_offload.json
# 2. 进一步减少批量大小：--per_device_train_batch_size 4
# 3. 进一步减少序列长度：--block_size 512
# 4. 增加梯度累积步数：--gradient_accumulation_steps 16