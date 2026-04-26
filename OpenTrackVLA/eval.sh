CHUNKS=30
NUM_PARALLEL=1
SAVE_PATH="sim_data/eval/stt"
DINO_MODEL=facebook/dinov3-vits16-pretrain-lvd1689m

if [ -n "${HF_MODEL_DIR:-}" ]; then
    export HF_MODEL_DIR
    echo "[eval] Using HuggingFace planner weights from ${HF_MODEL_DIR}"
fi

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        #CUDA_VISIBLE_DEVICES=$((i)) SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run_eval.py \
        CUDA_VISIBLE_DEVICES=1 SAVE_VIDEO=1 PYTHONPATH="habitat-lab" python run_eval.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH &
        ((IDX++))
    done
    wait
done
