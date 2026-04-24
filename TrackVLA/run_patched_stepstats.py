import argparse
import numpy as np
import random
import os
import json
import subprocess


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def write_run_meta(
    save_path: str,
    model_path: str,
    split_id: int,
    split_num: int,
    ablation_config: dict,
    seed: int,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    meta = {
        "git_commit": get_git_commit_hash(),
        "model_path": model_path,
        "token_ablation": ablation_config,
        "seed": seed,
        "split_config": {"split_id": split_id, "split_num": split_num},
    }
    with open(os.path.join(save_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["eval", "train"],
        required=True,
        help="run type",
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--split-num",
        type=int,
        default=7,
        required=False,
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--enable-step-stats",
        action="store_true",
        help="Write per-step latency/FPS JSONL + per-split SR/TR/CR summary JSONs.",
    )

    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1,
        help="Log step stats every N env steps (default: 1 = every step).",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Unified random seed for Python/NumPy/Torch/CUDA.",
    )

    parser.add_argument(
        "--token-ablation-mode",
        type=str,
        default=None,
        choices=["pool_all_2x2_to_1x1", "drop_history_keep_latest_nav64"],
        help=(
            "Token-ablation experiment mode. "
            "pool_all_2x2_to_1x1: pool every history 2x2 block to 1x1; "
            "drop_history_keep_latest_nav64: drop all history tokens and keep latest 8x8 nav tokens only."
        ),
    )
    
    parser.add_argument(
        "--online-cache-prune-mode",
        type=str,
        default="step_window",
        choices=["step_window", "episode_end", "off"],
        help=(
            "Online visual cache pruning strategy. "
            "step_window: trim short-term cache every step (recommended for stable VRAM); "
            "episode_end: keep full episode cache and clear at episode reset; "
            "off: disable trimming (debug only)."
        ),
    )

    parser.add_argument(
        "--fastv-k",
        type=int,
        default=None,
        help=(
            "FastV pruning layer index K. "
            "Attention is captured at layer K-1 and visual tokens are pruned at layer K. "
            "Typical value: 3. Set to None to disable FastV."
        ),
    )

    parser.add_argument(
        "--fastv-r",
        type=float,
        default=0.5,
        help=(
            "FastV pruning ratio R (fraction of visual tokens to DROP). "
            "0.5 means keep the top-50%% most-attended visual tokens. "
            "Range: (0, 1). Default: 0.5."
        ),
    )

    # ------------------------------------------------------------------
    # In-Place TTT (ported from ByteDance-Seed/In-Place-TTT).
    # All defaults keep TTT disabled so runs without --ttt-mode behave like
    # the original Uni-NaVid pipeline.
    # ------------------------------------------------------------------
    parser.add_argument(
        "--ttt-mode",
        action="store_true",
        help="Enable In-Place TTT (fast-weight update of MLP.down_proj at "
             "inference). When absent, TTT is fully bypassed.",
    )
    parser.add_argument(
        "--ttt-layers",
        type=str,
        default="0,6,12,18,24,30",
        help="Comma-separated decoder-layer indices to insert TTT at. "
             "Official LLaMA-3.1-8B recommendation: 0,6,12,18,24,30 (stride 6).",
    )
    parser.add_argument(
        "--ttt-lr",
        type=float,
        default=0.3,
        help="Fast-weight update step size for TTT (official inference default 0.3).",
    )
    parser.add_argument(
        "--ttt-chunk",
        type=int,
        default=512,
        help="Chunk size for chunk-wise TTT updates. When prefill length < chunk, "
             "the MLP bypasses the update and only reads the carried weight.",
    )
    parser.add_argument(
        "--ttt-no-proj",
        action="store_true",
        help="Disable the TTT projection A_omega (ttt_proj). Default: enabled "
             "(matches official config ttt_proj=true).",
    )
    parser.add_argument(
        "--ttt-target",
        type=str,
        default="hidden_states",
        choices=["hidden_states", "input_embed"],
        help="Target states for the TTT objective. Use 'hidden_states' for "
             "continual-trained checkpoints; 'input_embed' is for from-scratch.",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    run_type: str,
    exp_config: str,
    split_id: int,
    split_num: int,
    save_path: str,
    model_path: str,
    model_name: str,
    enable_step_stats: bool = False,
    log_every_n_steps: int = 1,
    seed: int = None,
    token_ablation_mode: str = None,
    online_cache_prune_mode: str = "step_window",
    fastv_k: int = None,
    fastv_r: float = 0.5,
    ttt_mode: bool = False,
    ttt_layers: str = "0,6,12,18,24,30",
    ttt_lr: float = 0.3,
    ttt_chunk: int = 1024,
    ttt_no_proj: bool = False,
    ttt_target: str = "hidden_states",
    opts=None,
) -> None:
    if run_type == "eval":
        if model_name == "uni-navid":
            from evt_bench.default import get_config
            from habitat.datasets import make_dataset
            from agent_uninavid import evaluate_agent

            config = get_config(exp_config)
            effective_seed = int(seed if seed is not None else config.habitat.simulator.seed)
            set_global_seed(effective_seed)

            dataset = make_dataset(
                id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
            )
            dataset_split = dataset.get_splits(split_num)[split_id]

            # Parse ttt_layers comma string to list[int] once here.
            try:
                ttt_layers_list = [int(s) for s in str(ttt_layers).split(",") if s.strip() != ""]
            except Exception:
                raise ValueError(f"--ttt-layers must be a comma list of ints, got: {ttt_layers!r}")

            write_run_meta(
                save_path=save_path,
                model_path=model_path,
                split_id=split_id,
                split_num=split_num,
                ablation_config={
                    "token_ablation_mode": token_ablation_mode,
                    "online_cache_prune_mode": online_cache_prune_mode,
                    "fastv_k": fastv_k,
                    "fastv_r": fastv_r,
                    "ttt_mode": ttt_mode,
                    "ttt_layers": ttt_layers_list,
                    "ttt_lr": ttt_lr,
                    "ttt_chunk": ttt_chunk,
                    "ttt_proj": (not ttt_no_proj),
                    "ttt_target": ttt_target,
                },
                seed=effective_seed,
            )

            evaluate_agent(
                config,
                model_path,
                dataset_split,
                save_path,
                split_id=split_id,
                enable_step_stats=enable_step_stats,
                log_every_n_steps=log_every_n_steps,
                seed=effective_seed,
                token_ablation_mode=token_ablation_mode,
                online_cache_prune_mode=online_cache_prune_mode,
                fastv_k=fastv_k,
                fastv_r=fastv_r,
                ttt_mode=ttt_mode,
                ttt_layers=ttt_layers_list,
                ttt_lr=ttt_lr,
                ttt_chunk=ttt_chunk,
                ttt_proj=(not ttt_no_proj),
                ttt_target=ttt_target,
            )
        elif model_name == "baseline":
            from evt_bench.default import get_config
            from habitat.datasets import make_dataset
            from baseline_agent import evaluate_agent

            config = get_config(exp_config)
            effective_seed = int(seed if seed is not None else config.habitat.simulator.seed)
            set_global_seed(effective_seed)
            dataset = make_dataset(
                id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
            )
            dataset_split = dataset.get_splits(split_num)[split_id]
            evaluate_agent(config, dataset_split, save_path)
        else:
            raise ValueError(f"The model name: {model_name} is not supported")
    else:
        raise ValueError("Not supported now")

    return


if __name__ == "__main__":
    main()
