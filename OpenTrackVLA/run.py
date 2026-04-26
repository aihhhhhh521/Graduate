from baseline_agent import evaluate_agent
import argparse
import habitat
from habitat.datasets import make_dataset
import evt_bench
import numpy as np
import random
import os
import json

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
        "--target-id",
        type=int,
        required=False,
        help="Semantic id of the specific person to follow (overrides name)",
    )

    parser.add_argument(
        "--target-name",
        type=str,
        required=False,
        help="Name of the person to follow, resolved via humanoid_infos.json",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(run_type: str, exp_config: str, split_id: int, split_num: int, save_path: str, opts: None, target_id: int = None, target_name: str = None) -> None:
    config = habitat.get_config(exp_config, opts)
    random.seed(config.habitat.simulator.seed)
    np.random.seed(config.habitat.simulator.seed)

    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    dataset_split = dataset.get_splits(split_num)[split_id]

    # Resolve target human semantic id
    resolved_target_id = target_id
    if resolved_target_id is None and target_name:
        try:
            infos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "humanoid_infos.json"))
            with open(infos_path, "r") as f:
                infos = json.load(f)
            for entry in infos:
                if entry.get("name") == target_name:
                    resolved_target_id = int(entry.get("semantic_id"))
                    break
        except Exception:
            resolved_target_id = None

    if run_type == "eval":
        evaluate_agent(config, dataset_split, save_path, resolved_target_id)
    else:
        raise ValueError("Not supported now")
    
    return
 

if __name__ == "__main__":
    main()
