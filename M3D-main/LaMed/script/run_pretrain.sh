#!/bin/bash

# Set GPU to the empty Quadro GV100 (ID 0)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Run training
torchrun --nnodes=1 --nproc_per_node=1 LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path ./LaMed/pretrained_model/llama-2-7b-chat \
    --model_type llama2 \
    --vision_tower vit3d \
    --pretrain_vision_model ./LaMed/pretrained_model/pretrained_ViT.bin \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-llama2-7B-pretrain-0000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --eval_accumulation_steps 1 \
    --eval_steps 1 \
    --save_strategy steps \
    --save_steps 65 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8 \
    --report_to tensorboard