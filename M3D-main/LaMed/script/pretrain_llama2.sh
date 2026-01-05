#!/bin/bash

# run "accelerate config" first!

# changed by codex
# makes the script more robust so it doesn’t fail on machines where only one of them exists.
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    echo "[pretrain_llama2.sh] Python interpreter not found." >&2
    exit 1
fi

# Ensure the repo root is on PYTHONPATH so `LaMed` imports resolve no matter the cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# If GPU exists and bf16 supported → BF16_MODE=gpu_bf16 → keeps --bf16 True
# If GPU exists but bf16 NOT supported → BF16_MODE=gpu_only → sets --bf16 False
# If no GPU → BF16_MODE=cpu → sets --bf16 False and forces CPU mode
if BF16_MODE=$("$PYTHON_BIN" <<'PY'
import torch
mode = "cpu"
if torch.cuda.is_available():
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    supported = False
    if callable(checker):
        try:
            supported = checker()
        except Exception:
            supported = False
    mode = "gpu_bf16" if supported else "gpu_only"
print(mode)
PY
); then
    :
else
    BF16_MODE="unknown"
fi

BF16_FLAG="--bf16 True"
CPU_FLAG=""

# When CUDA isn’t detected:
# You pass --use_cpu True
# You export ACCELERATE_USE_CPU=true
# You stop using accelerate launch and run train.py via plain python
case "$BF16_MODE" in
    gpu_only)
        echo "[pretrain_llama2.sh] GPU detected but bfloat16 unsupported; disabling --bf16." >&2
        BF16_FLAG="--bf16 False"
        ;;
    cpu)
        echo "[pretrain_llama2.sh] CUDA GPU not detected; forcing CPU mode and disabling --bf16." >&2
        BF16_FLAG="--bf16 False"
        CPU_FLAG="--use_cpu True"
        export ACCELERATE_USE_CPU=true
        ;;
    unknown)
        echo "[pretrain_llama2.sh] Could not determine bf16 capability; defaulting to --bf16 False." >&2
        BF16_FLAG="--bf16 False"
        ;;
esac

LAUNCH_CMD="accelerate launch"
if [ "$BF16_MODE" = "cpu" ]; then
    LAUNCH_CMD="$PYTHON_BIN"
fi
# changed it to no under folder "M3D-CLIP"
# deleted bf16 True \, added the two lines below

$LAUNCH_CMD LaMed/src/train/train2.py \
    --version v0 \
    --model_name_or_path ./LaMed/pretrained_model/llama-2-7b-chat \
    --model_type llama2 \
    --vision_tower vit3d \
    --pretrain_vision_model ./LaMed/pretrained_model/pretrained_ViT.bin \
    --tune_mm_mlp_adapter True \
    $CPU_FLAG \
    $BF16_FLAG \
    --output_dir ./LaMed/output/LaMed-llama2-7B-pretrain-0000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 1 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8 \
    --report_to tensorboard
