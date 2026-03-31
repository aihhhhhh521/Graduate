#!/usr/bin/env bash
set -euo pipefail

# 用法示例:
#   bash scripts/train_gater.sh
#   DATA_PATH=... IMAGE_FOLDER=... VIDEO_FOLDER=... MODEL_PATH=... OUT_DIR=... bash scripts/train_gater.sh

# ===== 基础路径配置 =====
DATA_PATH="${DATA_PATH:-./data/data/evt_track_videos/stt_expert_train/uninavid_track_dataset/track_dataset.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-./data/Nav-Finetune}"
VIDEO_FOLDER="${VIDEO_FOLDER:-./data/Nav-Finetune}"
MODEL_PATH="${MODEL_PATH:-./model_zoo/uninavid-7b-full-224-video-fps-1-grid-2}"
VISION_TOWER="${VISION_TOWER:-./model_zoo/eva_vit_g.pth}"
IMAGE_PROCESSOR="${IMAGE_PROCESSOR:-./uninavid/processor/clip-patch14-224}"
OUT_DIR="${OUT_DIR:-./exp_results/gater_only}"

# ===== 训练资源配置 =====
NUM_GPUS="${NUM_GPUS:-1}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-./scripts/zero2.json}"

# ===== Gater(MLP) 专项训练超参 =====
GATER_K="${GATER_K:-16}"
GATER_HIDDEN_DIM="${GATER_HIDDEN_DIM:-256}"
GATER_TEMPERATURE="${GATER_TEMPERATURE:-1.0}"
GATER_LOSS_WEIGHT="${GATER_LOSS_WEIGHT:-0.1}"
GATER_SPARSITY_WEIGHT="${GATER_SPARSITY_WEIGHT:-1.0}"
GATER_BINARY_WEIGHT="${GATER_BINARY_WEIGHT:-0.01}"

# ===== 通用训练超参 =====
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-2}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOG_STEPS="${LOG_STEPS:-10}"


echo "[Gater-Only Training]"
echo "  DATA_PATH=${DATA_PATH}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  NUM_GPUS=${NUM_GPUS}"


deepspeed --num_gpus="${NUM_GPUS}" uninavid/train/train_mem.py \
  --deepspeed "${DEEPSPEED_CFG}" \
  --model_name_or_path "${MODEL_PATH}" \
  --version imgsp_v1 \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --video_folder "${VIDEO_FOLDER}" \
  --vision_tower "${VISION_TOWER}" \
  --image_processor "${IMAGE_PROCESSOR}" \
  --compress_type "grid:2" \
  --run_type train \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_vision_select_feature patch \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --group_by_modality_length True \
  --video_fps 1 \
  --image_aspect_ratio pad \
  --lazy_preprocess True \
  \
  --mm_use_gater True \
  --gater_mode soft \
  --gater_k "${GATER_K}" \
  --gater_hidden_dim "${GATER_HIDDEN_DIM}" \
  --gater_temperature "${GATER_TEMPERATURE}" \
  --gater_loss_weight "${GATER_LOSS_WEIGHT}" \
  --gater_ratio_weight "${GATER_SPARSITY_WEIGHT}" \
  --gater_entropy_weight "${GATER_BINARY_WEIGHT}" \
  --tune_mm_gater True \
  \
  --freeze_backbone True \
  --tune_mm_mlp_adapter False \
  --tune_vision_encoder False \
  \
  --output_dir "${OUT_DIR}" \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --learning_rate "${LR}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOG_STEPS}" \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --bf16 True \
  --tf32 True \
  --report_to none


echo "Done. Gater adapter checkpoint is expected at: ${OUT_DIR}/token_gater.bin"