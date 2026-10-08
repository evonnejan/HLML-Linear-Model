"""Dense multi-seed initial-learning-rate comparison for the 13-feature model.

This deliberately fixes the learning-rate schedule to exp(gamma=0.85), uses
the current historical dataset/split, and never constructs a test loader.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_v13_feature_selection import (
    base_config,
    hashes,
    rank,
    read_results,
    run_one,
    save_json,
)

# Six new values make the transition between the earlier 0.0005 and 0.001
# controls dense, while 0.0004 and 0.0015 test the adjacent shoulders.
LRS = (0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010, 0.0012, 0.0015)
SEEDS = (42, 43, 44, 45, 46)
SOURCE_PATHS = [
    "run.py", "models/DLinearMix2.py", "data_provider/Data_Loader.py",
    "data_provider/Data_Factory.py", "exp/exp_Main2.py", "utils/tools.py",
    "dataset/train_old.csv", "dataset/splits_train_old.csv", "dataset/splits_train_old.json",
    "scripts/run_v13_feature_selection.py", "scripts/run_v13_exp_lr_multiseed.py",
]


def config_id(lr: float) -> str:
    return f"lr{lr:g}_exp"


def items() -> list[dict]:
    return [
        {
            "key": f"{config_id(lr)}_s{seed}", "config_id": config_id(lr),
            "learning_rate": lr, "seed": seed, "lradj": "exp", "exp_decay_gamma": 0.85,
        }
        for lr in LRS for seed in SEEDS
    ]


def verify_frozen(plan: dict) -> None:
    current = hashes(SOURCE_PATHS)
    if current != plan["source_hashes"]:
        changed = [name for name, digest in current.items() if plan["source_hashes"].get(name) != digest]
        raise RuntimeError(f"Frozen source/data changed; refuse to mix runs: {changed}")


def initialise(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing study: {root}")
    config = base_config()
    if not config["validation_only"] or config["lradj"] != "exp" or config["exp_decay_gamma"] != 0.85:
        raise AssertionError("Study setup must be validation-only exp(gamma=0.85)")
    root.mkdir(parents=True)
    plan = {
        "study": "13-feature dense initial-LR comparison with fixed exp schedule",
        "created_at_unix": time.time(),
        "base_config": config,
        "schedule": {"name": "exp", "gamma": 0.85},
        "learning_rates": list(LRS), "seeds": list(SEEDS), "runs": items(),
        "source_hashes": hashes(SOURCE_PATHS),
        "git_status_before": subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.splitlines(),
        "checkpoint_rule": "Within each run, retain the epoch with highest raw validation correlation.",
        "comparison_rule": "Rank learning rates by mean anchored validation correlation across five seeds; break ties by mean anchored MAE, then RMSE, then learning rate.",
        "scope": "13 features; current train_old.csv and fixed split; train/validation only; no test loader, test metric or test checkpoint selection.",
    }
    save_json(root / "search_plan.json", plan)
    (root / "search_plan.md").write_text("\n".join([
        "# Dense 13-feature initial-LR comparison", "",
        "- Fixed schedule: `exp`, gamma = 0.85.",
        "- Initial LRs: " + ", ".join(f"{lr:g}" for lr in LRS) + ".",
        "- Seeds: " + ", ".join(map(str, SEEDS)) + ".",
        "- 45 independent train/validation runs; no test loader or test metrics.",
        "- Per-run checkpoint: greatest raw validation correlation.",
        "- Select LR: greatest mean anchored validation correlation; ties use anchored MAE, RMSE, then LR.",
        "- Save exact epoch-8 and epoch-10 checkpoints for every run.",
        "",
    ]), encoding="utf-8")


def smoke(root: Path, plan: dict) -> None:
    verify_frozen(plan)
    smoke_root = root / "smoke"
    first = run_one(smoke_root, {
        "key": "repro_a", "config_id": "smoke", "learning_rate": 0.0008,
        "seed": 42, "lradj": "exp", "exp_decay_gamma": 0.85,
    }, epochs=1, fixed_checkpoints=False)
    second = run_one(smoke_root, {
        "key": "repro_b", "config_id": "smoke", "learning_rate": 0.0008,
        "seed": 42, "lradj": "exp", "exp_decay_gamma": 0.85,
    }, epochs=1, fixed_checkpoints=False)
    checks = {
        "same_checkpoint_hash": first["checkpoint_sha256"] == second["checkpoint_sha256"],
        "same_raw_metrics": first["raw"] == second["raw"],
        "same_anchored_metrics": first["anchored"] == second["anchored"],
        "same_best_epoch": first["best_epoch"] == second["best_epoch"],
    }
    if not all(checks.values()):
        raise AssertionError(f"Same-seed reproducibility failed: {checks}")
    save_json(root / "smoke_report.json", {"checks": checks, "runs": [first, second]})


def write_progress(path: Path, results: list[dict]) -> None:
    rows = []
    for result in rank(results):
        rows.append({
            "key": result["key"], "config_id": result["config_id"], "learning_rate": result["learning_rate"],
            "seed": result["seed"], "best_epoch": result["best_epoch"], "epochs": result["epochs"],
            "anchored_corr": result["anchored"]["corr"], "anchored_mae_mm": result["anchored"]["mae"],
            "anchored_rmse_mm": result["anchored"]["rmse"], "raw_corr": result["raw"]["corr"],
            "seconds": result["seconds"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def run(root: Path, plan: dict) -> None:
    verify_frozen(plan)
    if not (root / "smoke_report.json").is_file():
        raise RuntimeError("Run the reproducibility smoke test before full training")
    folder = root / "runs"
    folder.mkdir(exist_ok=True)
    for item in plan["runs"]:
        if (folder / item["key"] / "result.json").is_file():
            continue
        run_one(folder, item)
        write_progress(folder / "progress.csv", read_results(folder))
    results = read_results(folder)
    if len(results) != len(plan["runs"]):
        raise AssertionError(f"Study incomplete: expected {len(plan['runs'])}, got {len(results)}")
    write_progress(folder / "ranking.csv", results)


def finalize(root: Path, plan: dict) -> None:
    verify_frozen(plan)
    results = read_results(root / "runs")
    if len(results) != len(plan["runs"]):
        raise RuntimeError("Complete every planned run before finalizing")
    summaries = []
    for lr in LRS:
        members = [result for result in results if result["learning_rate"] == lr]
        if {result["seed"] for result in members} != set(SEEDS):
            raise AssertionError(f"Missing seed(s) for LR {lr:g}")
        summaries.append({
            "learning_rate": lr, "config_id": config_id(lr), "n_seeds": len(members),
            "mean_anchored_corr": float(np.mean([r["anchored"]["corr"] for r in members])),
            "std_anchored_corr": float(np.std([r["anchored"]["corr"] for r in members], ddof=1)),
            "mean_anchored_mae_mm": float(np.mean([r["anchored"]["mae"] for r in members])),
            "std_anchored_mae_mm": float(np.std([r["anchored"]["mae"] for r in members], ddof=1)),
            "mean_anchored_rmse_mm": float(np.mean([r["anchored"]["rmse"] for r in members])),
            "std_anchored_rmse_mm": float(np.std([r["anchored"]["rmse"] for r in members], ddof=1)),
            "seeds": members,
        })
    summaries.sort(key=lambda group: (-group["mean_anchored_corr"], group["mean_anchored_mae_mm"],
                                      group["mean_anchored_rmse_mm"], group["learning_rate"]))
    winner = summaries[0]
    best_seed = rank(winner["seeds"])[0]
    total_seconds = sum(result["seconds"] for result in results)
    report = {
        "final_configuration": {key: value for key, value in winner.items() if key != "seeds"},
        "best_seed_checkpoint": {"key": best_seed["key"], "seed": best_seed["seed"],
                                  "path": best_seed["source"], "best_epoch": best_seed["best_epoch"]},
        "ranking": summaries,
        "run_count": len(results), "total_training_seconds": total_seconds,
        "scope": plan["scope"], "source_hashes_unchanged": hashes(SOURCE_PATHS) == plan["source_hashes"],
    }
    save_json(root / "final_report.json", report)
    pd.DataFrame([{key: value for key, value in row.items() if key != "seeds"} for row in summaries]).to_csv(
        root / "final_ranking.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "smoke", "run", "finalize", "status"))
    parser.add_argument("--output", default="runs/v13feat_exp_lr_multiseed_20261005")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = (ROOT / args.output).resolve()
    if args.mode == "init":
        initialise(root)
        print(root)
        return
    plan = json.loads((root / "search_plan.json").read_text(encoding="utf-8"))
    if args.mode == "smoke":
        smoke(root, plan)
    elif args.mode == "run":
        run(root, plan)
    elif args.mode == "finalize":
        finalize(root, plan)
    else:
        completed = len(read_results(root / "runs")) if (root / "runs").is_dir() else 0
        print(f"runs: {completed}/{len(plan['runs'])}")
        for name in ("smoke_report.json", "final_report.json"):
            print(f"{name}: {'present' if (root / name).is_file() else 'missing'}")


if __name__ == "__main__":
    main()
