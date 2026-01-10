#!/bin/bash

# 使用优化的DeepSpeed配置和训练参数
# 解决CUDA内存不足问题

CUDA_VISIBLE_DEVICES=2,5

deepspeed pretrain.py \
    --config_name ./model/Qwen/Qwen2.5-1.5B \
    --tokenizer_name ./model/Qwen/Qwen2.5-1.5B \
    --train_files ./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 \
    --per_device_train_batch_size 8 \          # 大幅减少批量大小
    --gradient_accumulation_steps 16 \         # 增加梯度累积步数以保持有效批量大小
    --do_train \
    --output_dir ./output_optimized \
    --eval_strategy no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir ./output_optimized/pretrain/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 500 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 1024 \                        # 减少序列长度
    --bf16 \
    --gradient_checkpointing \                 # 启用梯度检查点
    --deepspeed ./ds_config_zero3_offload.json \  # 使用优化的DeepSpeed配置
    --report_to swanlab \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --group_by_length \
    --length_column_name length