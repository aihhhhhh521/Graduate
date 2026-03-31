#!/usr/bin/env bash
set -euo pipefail

EXP_CFG="${EXP_CFG:-habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml}"
MODEL_PATH="${MODEL_PATH:-model_zoo/uninavid-7b-full-224-video-fps-1-grid-2}"
SAVE_ROOT="${SAVE_ROOT:-data/evt_track_videos/stt-expert-train}"
SCENES_DIR="${SCENES_DIR:-data/scene_datasets}"
SPLITS="${SPLITS:-8}"
START_SPLIT="${START_SPLIT:-0}"
GPU_LIST="${GPU_LIST:-}"

# UniNaVid runtime options
PRUNE_MODE="${PRUNE_MODE:-step_window}"   # step_window | episode_end | off
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DO_SAMPLE="${DO_SAMPLE:-0}"               # 1 -> --do-sample
TEMPERATURE="${TEMPERATURE:-0.2}"

# job control
DRY_RUN="${DRY_RUN:-0}"
SYNC_RUN="${SYNC_RUN:-0}"                 # 1: serial run, 0: background all

declare -A GPU_ACTIVE_PID=()
declare -A GPU_ACTIVE_SPLIT=()

wait_gpu_slot_if_busy() {
  local gpu="$1"
  local pid="${GPU_ACTIVE_PID[$gpu]:-}"
  local running_split="${GPU_ACTIVE_SPLIT[$gpu]:-}"
  if [[ -n "${pid}" ]]; then
    echo "[render_bash] GPU ${gpu} is busy (split_${running_split}, pid=${pid}), waiting..."
    wait "${pid}"
    echo "[render_bash] GPU ${gpu} released."
    unset GPU_ACTIVE_PID["$gpu"]
    unset GPU_ACTIVE_SPLIT["$gpu"]
  fi
}


usage() {
  cat <<USAGE
Usage: bash render_bash.sh [options]
  --exp-cfg PATH
  --model-path PATH
  --save-root PATH
  --scenes-dir PATH
  --splits N
  --start-split N
  --gpu-list "0,1,2,3"     (default: split_id itself)
  --prune-mode MODE         (step_window|episode_end|off)
  --max-new-tokens N
  --do-sample               (default off)
  --temperature FLOAT
  --sync-run                (run splits serially)
  --dry-run                 (print commands only)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-cfg) EXP_CFG="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --scenes-dir) SCENES_DIR="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --start-split) START_SPLIT="$2"; shift 2 ;;
    --gpu-list) GPU_LIST="$2"; shift 2 ;;
    --prune-mode) PRUNE_MODE="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --do-sample) DO_SAMPLE=1; shift ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --sync-run) SYNC_RUN=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "${SAVE_ROOT}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

echo "[render_bash] EXP_CFG=${EXP_CFG}"
echo "[render_bash] MODEL_PATH=${MODEL_PATH}"
echo "[render_bash] SAVE_ROOT=${SAVE_ROOT}"
echo "[render_bash] SCENES_DIR=${SCENES_DIR}"
echo "[render_bash] SPLITS=${SPLITS} START_SPLIT=${START_SPLIT}"
echo "[render_bash] PRUNE_MODE=${PRUNE_MODE} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} DO_SAMPLE=${DO_SAMPLE} TEMPERATURE=${TEMPERATURE}"

for ((sid=START_SPLIT; sid<SPLITS; sid++)); do
  gpu_id="${sid}"
  if [[ -n "${GPU_LIST}" ]]; then
    idx=$(( (sid - START_SPLIT) % ${#GPUS[@]} ))
    gpu_id="${GPUS[$idx]}"
  fi

  cmd=(
    python render_train_split.py
      --exp-config "${EXP_CFG}"
      --model-path "${MODEL_PATH}"
      --save-path "${SAVE_ROOT}/split_${sid}"
      --split-num "${SPLITS}"
      --split-id "${sid}"
      --scenes-dir "${SCENES_DIR}"
      --online-cache-prune-mode "${PRUNE_MODE}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --temperature "${TEMPERATURE}"
      --output-config-dirname "configs"
      --output-video-dirname "videos"
      --output-json-name "track_train.json"
  )
  if [[ "${DO_SAMPLE}" == "1" ]]; then
    cmd+=(--do-sample)
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
    continue
  fi

  if [[ "${SYNC_RUN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" 2>&1 | tee "${SAVE_ROOT}/split_${sid}.log"
  else
    wait_gpu_slot_if_busy "${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" > "${SAVE_ROOT}/split_${sid}.log" 2>&1 &
    pid=$!
    GPU_ACTIVE_PID["$gpu_id"]="${pid}"
    GPU_ACTIVE_SPLIT["$gpu_id"]="${sid}"
    echo "[render_bash] started split_${sid} on GPU ${gpu_id} (pid=${pid})"
  fi
done

if [[ "${DRY_RUN}" != "1" && "${SYNC_RUN}" != "1" ]]; then
  for gpu in "${!GPU_ACTIVE_PID[@]}"; do
    pid="${GPU_ACTIVE_PID[$gpu]}"
    sid="${GPU_ACTIVE_SPLIT[$gpu]}"
    echo "[render_bash] waiting split_${sid} on GPU ${gpu} (pid=${pid})"
    wait "${pid}"
  done
fi

echo "[render_bash] done."