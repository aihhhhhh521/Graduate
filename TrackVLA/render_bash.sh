EXP_CFG="habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml"
MODEL_PATH="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
SAVE_ROOT="data/evt_track_videos/stt-expert-train"
SCENES_DIR="data/scene_datasets"
SPLITS=8

# UniNaVid runtime options
PRUNE_MODE="step_window"            # avoid online-cache OOM
MAX_NEW_TOKENS=1024
# if you need stochastic decoding, set DO_SAMPLE_FLAG="--do-sample"
DO_SAMPLE_FLAG=""

# Inference strategy:
# default is per-step inference (normal UniNaVid behavior).
# set REUSE_HORIZON_FLAG="--reuse-action-horizon" to infer every 4 steps and reuse horizon actions.
REUSE_HORIZON_FLAG=""

for i in $(seq 0 $((SPLITS - 1))); do
  CUDA_VISIBLE_DEVICES=${i} \
  python render_train_split.py \
    --exp-config "${EXP_CFG}" \
    --model-path "${MODEL_PATH}" \
    --save-path "${SAVE_ROOT}/split_${i}" \
    --split-num ${SPLITS} \
    --split-id ${i} \
    --scenes-dir "${SCENES_DIR}" \
    --habitat-gpu-id 0 \
    --model-gpu-id 0 \
    --online-cache-prune-mode "${PRUNE_MODE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --output-config-dirname "configs" \
    --output-video-dirname "videos" \
    --output-json-name "track_train.json" \
    ${DO_SAMPLE_FLAG} \
    ${REUSE_HORIZON_FLAG} \
    > "${SAVE_ROOT}/split_${i}.log" 2>&1 &
done

wait
echo "All 8 splits finished."