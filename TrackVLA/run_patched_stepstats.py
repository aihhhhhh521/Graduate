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
        default=None,
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
    opts=None,
) -> None:
    if run_type == "eval":
        if model_name == "uni-navid":
            from evt_bench.default import get_config
            from habitat.datasets import make_dataset
            from agent_uninavid_patched_stepstats import evaluate_agent

            config = get_config(exp_config)
            effective_seed = int(seed if seed is not None else config.habitat.simulator.seed)
            set_global_seed(effective_seed)

            dataset = make_dataset(
                id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
            )
            dataset_split = dataset.get_splits(split_num)[split_id]

            write_run_meta(
                save_path=save_path,
                model_path=model_path,
                split_id=split_id,
                split_num=split_num,
                ablation_config={
                    "token_ablation_mode": token_ablation_mode,
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
