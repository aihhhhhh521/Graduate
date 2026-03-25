EXP_CFG="habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml"
MODEL_PATH="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
SAVE_ROOT="data/evt_track_videos/stt-expert-train"
SCENES_DIR="data/scene_datasets"

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=${i} \
  python render_train_split.py \
    --exp-config "${EXP_CFG}" \
    --model-path "${MODEL_PATH}" \
    --save-path "${SAVE_ROOT}/split_${i}" \
    --split-num 8 \
    --split-id ${i} \
    --scenes-dir "${SCENES_DIR}" \
    --habitat-gpu-id 0 \
    --model-gpu-id 0 \
    > "${SAVE_ROOT}/split_${i}.log" 2>&1 &
done

wait
echo "All 8 splits finished."