#!/usr/bin/env bash
# Stage-2 training: fine-tune remaining DualFlowControlNet parameters.
# Requires Stage-1 checkpoint at logs/train_stage1_rarp/checkpoint-<N>/
#
# Run from project root: bash Training/train_stage2.sh

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

EXP_NAME="train_stage2_rarp"
SVD_PATH="/models/svd/Training/ckpts/stable-video-diffusion-img2vid-xt-1-1"
DATA_ROOT="/opt/data/private/RarpSora/data"

# ── Point these to your Stage-1 output ──────────────────────────────────────
# After stage1 finishes, set STAGE1_CKPT to the best checkpoint directory,
# e.g. logs/train_stage1_rarp/checkpoint-18000
STAGE1_CKPT="logs/train_stage1_rarp/checkpoint-18000"
STAGE1_SEG="logs/train_stage1_rarp/checkpoint-18000/seg_encoder.pth"
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.." || exit 1

accelerate launch \
    --main_process_port 29501 \
    --mixed_precision bf16 \
    Training/train_stage2.py \
    --pretrained_model_name_or_path="${SVD_PATH}" \
    --stage1_controlnet_path="${STAGE1_CKPT}/controlnet" \
    --stage1_seg_encoder_path="${STAGE1_SEG}" \
    --data_root="${DATA_ROOT}" \
    --output_dir="logs/${EXP_NAME}/" \
    --width=320 \
    --height=192 \
    --seed=42 \
    --num_frames=21 \
    --sample_stride=1 \
    --max_flow=20.0 \
    --val_split=0.1 \
    --learning_rate=1e-5 \
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
