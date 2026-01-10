#!/bin/bash

# 使用DeepSpeed Zero-3 with CPU offload的微调脚本
# 适用于内存严重不足的情况

deepspeed finetune.py \
    --model_name_or_path ./output_optimized \
    --train_files ./dataset/sft_data/BelleGroup/train_3.5M_CN.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --do_train \
    --output_dir ./output_zero3_offload \
    --eval_strategy no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir ./output_zero3_offload/sft/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 512 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero3_offload.json \
    --report_to swanlab

echo "训练完成！输出保存在: ./output_zero3_offload"

# 内存估算：
# 使用Zero-3 with CPU offload可以将大部分参数和优化器状态offload到CPU
# 每个GPU只需要存储当前正在处理的分片参数
# 这可以显著减少GPU内存使用，但可能会稍微降低训练速度（由于CPU-GPU数据传输）