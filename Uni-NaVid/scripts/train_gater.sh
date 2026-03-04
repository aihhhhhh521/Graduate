#!/usr/bin/env bash
set -e

DATA_PATH="${DATA_PATH:-./data/nav_finetune}"
MODEL_PATH="${MODEL_PATH:-./model_zoo/vicuna-7b-v1.5}"
OUT_DIR="${OUT_DIR:-./exp_results/gater_only}"

deepspeed --num_gpus=1 \
  uninavid/train/train_mem.py \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUT_DIR}" \
  --compress_type "grid:2" \
  --run_type "train" \
  --version "imgsp_v1" \
  --vision_tower "./uninavid/processor/clip-patch14-224" \
  --mm_projector_type "mlp2x_gelu" \
  --mm_vision_select_feature "patch" \
  --mm_use_gater True \
  --gater_k 16 \
  --gater_hidden_dim 256 \
  --gater_temperature 1.0 \
  --gater_mode "soft" \
  --gater_loss_weight 0.1 \
  --gater_ratio_weight 1.0 \
  --gater_entropy_weight 0.01 \
  --tune_mm_gater True \
  --freeze_backbone True \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True
