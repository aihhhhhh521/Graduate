#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
# bash eval_fastv_sparsevlm_stepstats.sh stt 30 8
# bash eval_fastv_sparsevlm_stepstats.sh at 30 8

TASK_TYPE="${1:-stt}"          # stt | at
CHUNKS="${2:-30}"
NUM_PARALLEL="${3:-8}"
SEED="${SEED:-100}"

MODEL_PATH="${MODEL_PATH:-../Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2}"
OUT_ROOT="${OUT_ROOT:-exp_results/ablation}"

if [[ "$TASK_TYPE" == "stt" ]]; then
  EXP_CONFIG="habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml"
elif [[ "$TASK_TYPE" == "at" ]]; then
  EXP_CONFIG="habitat-lab/habitat/config/benchmark/nav/track/track_infer_at.yaml"
else
  echo "[ERROR] TASK_TYPE must be stt or at, got: $TASK_TYPE"
  exit 1
fi

run_method () {
  local method_name="$1"
  local ablation_mode="$2"
  local save_path="${OUT_ROOT}/${method_name}/${TASK_TYPE}"

  local idx=0
  while [[ $idx -lt $CHUNKS ]]; do
    for ((i=0; i<NUM_PARALLEL && idx<CHUNKS; i++)); do
      echo "[$method_name] Launch split ${idx} on GPU ${i}"
      CUDA_VISIBLE_DEVICES="$i" PYTHONPATH="habitat-lab" python run_patched_stepstats.py \
        --run-type eval \
        --split-num "$CHUNKS" \
        --split-id "$idx" \
        --exp-config "$EXP_CONFIG" \
        --save-path "${save_path}/split${idx}" \
        --model-path "$MODEL_PATH" \
        --model-name uni-navid \
        --enable-step-stats \
        --log-every-n-steps 1 \
        --seed "$SEED" \
        --token-ablation-mode "$ablation_mode" \
        --online-cache-prune-mode step_window &
      idx=$((idx+1))
    done
    wait
  done

  echo "[$method_name] done. Results: ${save_path}"
}

run_method "fastv_like" "pool_all_2x2_to_1x1"
run_method "sparsevlm_like" "drop_history_keep_latest_nav64"

echo "All done. You can aggregate task metrics with:"
echo "python analyze_results.py --path ${OUT_ROOT}/fastv_like/${TASK_TYPE}"
echo "python analyze_results.py --path ${OUT_ROOT}/sparsevlm_like/${TASK_TYPE}"
