CHUNKS=30
NUM_PARALLEL=8
SAVE_PATH="exp_results/uninavid_all/stt1"
MODEL_PATH="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        CUDA_VISIBLE_DEVICES=$((i)) PYTHONPATH="habitat-lab" python run_patched_stepstats.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH \
            --model-path $MODEL_PATH \
            --model-name 'uni-navid' &
        ((IDX++))
    done
    wait
done