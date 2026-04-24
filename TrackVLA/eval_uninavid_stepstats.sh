CHUNKS=30
# GPUs to use for this run. Edit this list to restrict to a subset of cards
# (e.g. GPUS=(0 2 5) for a 3-GPU smoke test). NUM_PARALLEL is derived from it.
GPUS=(0 1)
NUM_PARALLEL=${#GPUS[@]}
SAVE_PATH="exp_results/uninavid_ttt/stt3"
MODEL_PATH="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"

# Notes on flag semantics (matches TrackVLA_origin/eval_uninavid.sh inference):
#   * FastV is intentionally NOT passed (fastv_k defaults to None -> FastV disabled).
#   * --online-cache-prune-mode off: keep per-episode feat_cache accumulation like
#     Uni-NaVid_origin; step_window would trim the cache every step.
#   * --token-ablation-mode is NOT passed (defaults to None -> no ablation).
#   * --ttt-* enables In-Place TTT on the LLM backbone (LLaMA-2-7B, 32 layers).

IDX=0
while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        GPU_ID=${GPUS[$i]}
        echo "Launching job IDX=$IDX on GPU=$GPU_ID"
        CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="habitat-lab" python run_patched_stepstats.py \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH \
            --model-path $MODEL_PATH \
            --model-name 'uni-navid' \
            --enable-step-stats \
            --log-every-n-steps 1 \
            --ttt-mode \
            --ttt-layers 0,6,12,18,24,30 \
            --ttt-lr 0.1 \
            --ttt-no-proj \
            --ttt-chunk 512 \
            --ttt-target hidden_states&
        ((IDX++))
    done
    wait
done