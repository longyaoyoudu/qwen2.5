CUDA_VISIBLE_DEVICES=4,5

deepspeed finetune.py \
    --model_name_or_path ./output_optimized \
    --train_files ./dataset/sft_data/BelleGroup/train_3.5M_CN.json \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir ./output \
    --eval_strategy no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir ./output/sft/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab