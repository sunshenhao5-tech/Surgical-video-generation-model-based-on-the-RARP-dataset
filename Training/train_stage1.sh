#!/usr/bin/env bash
# Stage-1 training: DualFlowControlNet on RARP multimodal dataset.
# Target: ≤ 2 days on a single GPU.
#
# Runtime estimate:
#   1264 clips, batch=1, grad_accum=4 → 316 optimizer steps/epoch
#   ~8-10 s/step → 316 × 9s ≈ 47 min/epoch → ~60 epochs in 48 h
#   max_train_steps=18000 (≈57 epochs) fits comfortably in 2 days.
#
# Run from project root: bash Training/train_stage1.sh

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

EXP_NAME="train_stage1_rarp"
SVD_PATH="/models/svd/Training/ckpts/stable-video-diffusion-img2vid-xt-1-1"
DATA_ROOT="/opt/data/private/RarpSora/data"

cd "$(dirname "$0")/.." || exit 1

accelerate launch \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    Training/train_stage1.py \
    --pretrained_model_name_or_path="${SVD_PATH}" \
    --data_root="${DATA_ROOT}" \
    --output_dir="logs/${EXP_NAME}/" \
    --width=320 \
    --height=192 \
    --seed=42 \
    --num_frames=21 \
    --sample_stride=1 \
    --max_flow=20.0 \
    --val_split=0.1 \
    --learning_rate=2e-5 \
    --per_gpu_batch_size=1 \
    --num_train_epochs=100 \
    --max_train_steps=18000 \
    --mixed_precision="bf16" \
    --gradient_accumulation_steps=4 \
    --checkpointing_steps=3000 \
    --checkpoints_total_limit=3 \
    --validation_steps=3000 \
    --num_validation_images=1 \
    --num_workers=4 \
    --lambda_fg=2.0 \
    --lambda_flow_tv=0.01 \
    --lambda_depth_tv=0.05 \
    --resume_from_checkpoint="latest" \
    --report_to="tensorboard"
