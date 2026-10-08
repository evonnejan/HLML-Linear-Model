"""Frozen, validation-only hyperparameter selection for the 13-feature model.

This study retains the current train_old.csv and split assignments, removes
Now only from exog_col, and never constructs a test loader.  It deliberately
uses a new output directory and leaves all legacy artifacts untouched.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from exp.exp_Main2 import Exp_Main
from run import _configure_mix_model_args, _set_seed
from utils.tools import adjust_learning_rate


WATER = ["HL02", "HL03", "HL04", "HL05", "HL06"]
EXOG = [
    "Past10Min", "Past1Hr",
    "north_gate_opening_1", "north_gate_opening_2", "north_gate_opening_3",
    "south_gate_opening_1", "south_gate_opening_2", "south_gate_opening_3",
]
LRS = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.004]
SCHEDULES = ["type1", "exp", "warmup_exp"]
BASELINE = (0.001, "exp")

SOURCE_PATHS = [
    "run.py", "models/DLinearMix2.py", "data_provider/Data_Loader.py",
    "data_provider/Data_Factory.py", "exp/exp_Main2.py", "utils/tools.py",
    "dataset/train_old.csv", "dataset/splits_train_old.csv", "dataset/splits_train_old.json",
    "scripts/run_v13_feature_selection.py",
]
PROTECTED_PATHS = [
    "dataset/train_old.csv", "dataset/splits_train_old.csv", "dataset/splits_train_old.json",
    "project_stage1/inference/input_schema.json", "project_stage1/inference/validation.py",
    "project_stage1/INPUT_MANUAL.md",
    "runs/delivery_lr_20260926/lr0.001_exp/run_args.json",
    "runs/delivery_lr_20260926/lr0.001_exp/checkpoints/checkpoint.pth",
    "runs/delivery_lr_20260926/lr0.001_exp/scaler.npz",
    "runs/delivery_lr_20260926/lr0.001_exp/result.json",
    "runs/delivery_lr_20260926/lr0.001_exp/validation_predictions.npz",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes(paths: list[str]) -> dict[str, str]:
    result = {}
    for text in paths:
        path = ROOT / text
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact is missing: {path}")
        result[text] = sha256(path)
    return result


def save_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def base_config() -> dict:
    return {
        "model": "DLinearMix2", "data": "custom", "root_path": str(ROOT / "dataset"),
        "data_path": "train_old.csv", "features": "S", "target": "HL01", "freq": "min",
        "embed": "timeF", "input_col": ",".join(WATER), "exog_col": ",".join(EXOG),
        "segment_col": "segment_id", "split_mode": "file",
        "split_file": str(ROOT / "dataset/splits_train_old.csv"), "fold": None,
        "seq_len": 60, "label_len": 30, "pred_len": 15, "stride_train": 1,
        "stride_eval": 1, "individual": True, "dlinear_kernel_size": 25,
        "mix_in": None, "branch_in": None, "exog_in": None, "flatten_fusion": True,
        "fusion_hidden_dim": 32, "exog_emb_dim": 16, "dropout": 0.1,
        "train_epochs": 80, "batch_size": 64, "patience": 15, "num_workers": 0,
        "train_only": False, "validation_only": True, "learning_rate": 0.001,
        "lradj": "exp", "criterion": "mse", "huber_beta": 1.0,
        "warmup_epochs": 8, "exp_decay_gamma": 0.85, "lr_decay_gamma": 0.85,
        "lr_floor_ratio": 0.01, "early_stop_metric": "corr", "seed": 42,
        "use_amp": False, "use_gpu": False, "use_multi_gpu": False, "gpu": 0,
    }


def schedule_fields(schedule: str) -> dict:
    if schedule == "type1":
        return {"lradj": schedule}
    if schedule == "exp":
        return {"lradj": schedule, "exp_decay_gamma": 0.85}
    if schedule == "warmup_exp":
        return {"lradj": schedule, "warmup_epochs": 8, "lr_decay_gamma": 0.85}
    raise ValueError(f"Unsupported schedule: {schedule}")


def config_id(lr: float, schedule: str) -> str:
    return f"lr{lr:g}_{schedule}"


def stage1_items() -> list[dict]:
    pairs = [BASELINE] + [
        (lr, schedule) for schedule in SCHEDULES for lr in LRS
        if (lr, schedule) != BASELINE
    ]
    return [
        {
            "key": f"{config_id(lr, schedule)}_s42", "config_id": config_id(lr, schedule),
            "learning_rate": lr, "seed": 42, **schedule_fields(schedule),
        }
        for lr, schedule in pairs
    ]


def argument_object(config: dict, item: dict, folder: Path, *, verify_13: bool = True) -> SimpleNamespace:
    values = config | {k: v for k, v in item.items() if k not in {"key", "config_id"}}
    args = SimpleNamespace(**values)
    args.run_dir = str(folder)
    args.setting = item["key"]
    _configure_mix_model_args(args)
    if verify_13:
        if args.input_cols != WATER or args.exog_cols != EXOG:
            raise AssertionError(f"Feature order mismatch: {args.input_cols=} {args.exog_cols=}")
        if (args.branch_in, args.exog_in, args.mix_in) != (5, 8, 13):
            raise AssertionError(f"Expected branch/exog/total dimensions 5/8/13, got "
                                 f"{args.branch_in}/{args.exog_in}/{args.mix_in}")
    if not args.validation_only:
        raise AssertionError("Search must set validation_only=True")
    return args


def scorer(pred: np.ndarray, true: np.ndarray) -> dict:
    if pred.shape != true.shape or pred.ndim != 3 or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Predictions/targets must be finite matching [N, H, 1] arrays")
    corrs = []
    for horizon in range(pred.shape[1]):
        x, y = pred[:, horizon, 0], true[:, horizon, 0]
        if x.std() == 0 or y.std() == 0:
            corrs.append(float("nan"))
        else:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
    if not all(np.isfinite(value) for value in corrs):
        raise ValueError(f"Undefined validation correlation by horizon: {corrs}")
    error = pred.astype(np.float64) - true.astype(np.float64)
    return {
        "corr": float(np.mean(corrs)), "horizon_corr": corrs,
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
        "valid_corr_horizons": len(corrs),
    }


def inverse_validate(exp: Exp_Main, folder: Path) -> dict:
    dataset, loader = exp._get_data("val")
    raw, true, anchor, _ = exp._run_inference(dataset, loader)
    shape = raw.shape
    inverse = lambda array: dataset.inverse_transform(array.reshape(-1, shape[-1])).reshape(array.shape)
    raw, true = inverse(raw), inverse(true)
    anchor = inverse(anchor[:, :1, :])
    anchored = anchor + raw - raw[:, :1, :]
    first_step_error = float(np.abs(anchored[:, :1, :] - anchor).max())
    # raw/anchor are float32 after inverse scaling.  The subtract-and-add anchor
    # reconstruction can accumulate several ULPs at physical water-level scale.
    max_physical_value = max(1.0, float(np.abs(np.concatenate([raw.ravel(), anchor.ravel()])).max()))
    first_step_tolerance = float(3 * np.spacing(np.float32(max_physical_value)))
    if first_step_error > first_step_tolerance:
        raise AssertionError(
            f"Anchored first-horizon invariant failed: {first_step_error} > {first_step_tolerance}"
        )
    origins = dataset.dates[dataset.valid_starts + dataset.seq_len - 1].astype(str)
    np.savez_compressed(folder / "validation_predictions.npz", raw=raw, true=true, anchor=anchor,
                        anchored=anchored, origins=origins)
    np.savez(folder / "scaler.npz", x_mean=dataset.scaler_x.mean_, x_scale=dataset.scaler_x.scale_,
             y_mean=dataset.scaler_y.mean_, y_scale=dataset.scaler_y.scale_,
             features=np.asarray(dataset.x_cols))
    return {
        "raw": scorer(raw, true), "anchored": scorer(anchored, true), "n_val": len(dataset),
        "anchored_first_step": {"max_error": first_step_error, "tolerance": first_step_tolerance},
    }


def train_validation_only(exp: Exp_Main, *, fixed_checkpoints: bool) -> dict:
    """Exp_Main.train equivalent with explicit epoch-8/10 snapshots and no test request."""
    train_data, train_loader = exp._get_data("train")
    val_data, val_loader = exp._get_data("val")
    checkpoint_dir = Path(exp._checkpoint_dir())
    optimizer = exp._select_optimizer()
    if str(exp.args.lradj).startswith("warmup"):
        for group in optimizer.param_groups:
            group["lr"] = exp.args.learning_rate * 0.1
    criterion = exp._select_criterion()
    best_corr, best_epoch, stale = -float("inf"), None, 0
    history = []
    started = time.monotonic()
    for epoch in range(1, int(exp.args.train_epochs) + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = exp._run_train_epoch(train_loader, optimizer, criterion, None, epoch - 1, exp.args.train_epochs)
        metrics = exp.vali(val_loader)
        if not all(np.isfinite(value) for value in (train_loss, metrics["mse"], metrics["mae"], metrics["corr"])):
            raise ValueError(f"Non-finite train/validation metrics at epoch {epoch}: {metrics}")
        history.append({"epoch": epoch, "learning_rate": learning_rate, "train_loss": train_loss,
                        **{f"val_{name}": value for name, value in metrics.items()}})
        if fixed_checkpoints and epoch in {8, 10}:
            torch.save(exp.model.state_dict(), checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")
        if metrics["corr"] > best_corr:
            best_corr, best_epoch, stale = metrics["corr"], epoch, 0
            torch.save(exp.model.state_dict(), checkpoint_dir / "checkpoint.pth")
        else:
            stale += 1
        if stale >= int(exp.args.patience):
            break
        adjust_learning_rate(optimizer, epoch, exp.args)
    if best_epoch is None:
        raise AssertionError("No checkpoint was selected")
    if fixed_checkpoints:
        for epoch in (8, 10):
            if not (checkpoint_dir / f"checkpoint_epoch_{epoch}.pth").is_file():
                raise AssertionError(f"Required epoch-{epoch} checkpoint was not saved")
    exp.model.load_state_dict(torch.load(checkpoint_dir / "checkpoint.pth", weights_only=True))
    exp.training_history = history
    exp.best_epoch = best_epoch
    exp.epochs_trained = len(history)
    pd.DataFrame(history).to_csv(checkpoint_dir / "training_history.csv", index=False)
    return {"counts": {"train": len(train_data), "val": len(val_data)}, "seconds": time.monotonic() - started}


def run_one(root: Path, item: dict, *, epochs: int = 80, fixed_checkpoints: bool = True) -> dict:
    folder = root / item["key"]
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {folder}")
    folder.mkdir(parents=True)
    args = argument_object(base_config() | {"train_epochs": epochs}, item, folder)
    save_json(folder / "run_args.json", vars(args))
    with (folder / "train.log").open("w", buffering=1, encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        _set_seed(args.seed)
        exp = Exp_Main(args)
        training = train_validation_only(exp, fixed_checkpoints=fixed_checkpoints)
        metrics = inverse_validate(exp, folder)
    result = item | metrics | training | {
        "best_epoch": exp.best_epoch, "epochs": exp.epochs_trained,
        "parameter_count": int(sum(parameter.numel() for parameter in exp.model.parameters())),
        "checkpoint_sha256": sha256(folder / "checkpoints/checkpoint.pth"),
    }
    save_json(folder / "result.json", result)
    return result


def window_keys(dataset) -> set[tuple[str, str]]:
    starts = dataset.valid_starts
    origins = dataset.dates[starts + dataset.seq_len - 1].astype(str)
    return set(zip(dataset.window_segment_ids.astype(str), origins, strict=True))


def load_dataset_counts(config: dict) -> tuple[dict, dict, dict]:
    """Return 13-feature counts, 14-feature counts, and detail for the Now impact."""
    detail, counts13, counts14 = {}, {}, {}
    legacy = config | {"exog_col": "Past10Min,Past1Hr,Now," + ",".join(EXOG[2:])}
    for flag in ("train", "val"):
        new_args = argument_object(config, {"key": f"preflight_{flag}", "config_id": "preflight", "seed": 42}, ROOT / ".preflight")
        old_args = argument_object(legacy, {"key": f"preflight_legacy_{flag}", "config_id": "preflight_legacy", "seed": 42}, ROOT / ".preflight", verify_13=False)
        new_ds, _ = Exp_Main(new_args)._get_data(flag)
        old_ds, _ = Exp_Main(old_args)._get_data(flag)
        new_keys, old_keys = window_keys(new_ds), window_keys(old_ds)
        counts13[flag], counts14[flag] = len(new_ds), len(old_ds)
        detail[flag] = {
            "13_feature_windows": len(new_ds), "14_feature_windows": len(old_ds),
            "added_after_removing_Now": len(new_keys - old_keys),
            "unexpectedly_removed_after_removing_Now": len(old_keys - new_keys),
        }
    return counts13, counts14, detail


def preflight(root: Path, frozen: dict) -> None:
    verify_frozen(frozen)
    args = argument_object(base_config(), {"key": "preflight", "config_id": "preflight", "seed": 42}, ROOT / ".preflight")
    exp = Exp_Main(args)
    train_ds, train_loader = exp._get_data("train")
    batch_x, batch_y, *_ = next(iter(train_loader))
    if tuple(batch_x.shape[1:]) != (60, 13):
        raise AssertionError(f"Expected [batch, 60, 13] input, got {tuple(batch_x.shape)}")
    with torch.no_grad():
        output = exp.model(batch_x.float())
    if tuple(output.shape[1:]) != (15, 1):
        raise AssertionError(f"Expected [batch, 15, 1] output, got {tuple(output.shape)}")
    starts = train_ds.valid_starts
    origin = pd.Timestamp(train_ds.dates[int(starts[0]) + 59])
    labels = pd.DatetimeIndex(train_ds.dates[int(starts[0]) + 60:int(starts[0]) + 75])
    if len(labels) != 15 or labels[0] != origin + pd.Timedelta(minutes=1) or labels[-1] != origin + pd.Timedelta(minutes=15):
        raise AssertionError("Target timestamps are not t+1 through t+15")
    counts13, counts14, now_effect = load_dataset_counts(base_config())
    with np.load(ROOT / "runs/delivery_lr_20260926/lr0.001_exp/scaler.npz", allow_pickle=False) as legacy_scaler:
        old_features = list(legacy_scaler["features"].astype(str))
        comparisons = {}
        for index, name in enumerate(train_ds.x_cols):
            old_index = old_features.index(name)
            comparisons[name] = {
                "mean_delta": float(train_ds.scaler_x.mean_[index] - legacy_scaler["x_mean"][old_index]),
                "scale_delta": float(train_ds.scaler_x.scale_[index] - legacy_scaler["x_scale"][old_index]),
            }
    hard_coded = {}
    for relative in ("models/DLinearMix2.py", "data_provider/Data_Loader.py", "exp/exp_Main2.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        hard_coded[relative] = re.findall(r"(?<!\d)5\s*:\s*14(?!\d)", text)
    report = {
        "feature_order": train_ds.x_cols, "branch_in": args.branch_in, "exog_in": args.exog_in,
        "parameter_count": int(sum(parameter.numel() for parameter in exp.model.parameters())),
        "batch_x_shape": list(batch_x.shape), "batch_y_shape": list(batch_y.shape),
        "output_shape": list(output.shape), "target_alignment": {
            "origin": origin.isoformat(), "first_label": labels[0].isoformat(), "last_label": labels[-1].isoformat(),
        },
        "window_counts": {"13_feature": counts13, "14_feature": counts14, "Now_indirect_effect": now_effect},
        "scaler_comparison_to_legacy": comparisons,
        "hard_coded_5_to_14_matches": hard_coded,
        "future_exog_filter_note": "Current loader checks input/exog/target over all 75 minutes; this is recorded as a deferred fix, not changed by this study.",
        "test_loader_policy": "validation_only=True; this script requests only train and val datasets.",
    }
    if any(hard_coded.values()):
        raise AssertionError(f"Hard-coded 5:14 slice found: {hard_coded}")
    save_json(root / "preflight_report.json", report)


def smoke(root: Path, frozen: dict) -> None:
    verify_frozen(frozen)
    smoke_root = root / "smoke"
    item = {"key": "repro_a", "config_id": "smoke", "learning_rate": 0.001, "seed": 42, **schedule_fields("exp")}
    first = run_one(smoke_root, item, epochs=1, fixed_checkpoints=False)
    second = run_one(smoke_root, item | {"key": "repro_b"}, epochs=1, fixed_checkpoints=False)
    checks = {
        "same_checkpoint_hash": first["checkpoint_sha256"] == second["checkpoint_sha256"],
        "same_raw_metrics": first["raw"] == second["raw"],
        "same_anchored_metrics": first["anchored"] == second["anchored"],
        "same_best_epoch": first["best_epoch"] == second["best_epoch"],
    }
    if not all(checks.values()):
        raise AssertionError(f"Same-seed reproducibility failed: {checks}")
    per_epoch_seconds = (first["seconds"] + second["seconds"]) / 2
    save_json(root / "smoke_report.json", {
        "checks": checks, "runs": [first, second],
        "estimated_upper_bound_seconds": per_epoch_seconds * 80 * 29,
        "estimate_basis": "mean CPU time for two one-epoch baseline runs × 80 epochs × maximum 29 search runs; early stopping lowers actual time.",
    })


def rank(results: list[dict]) -> list[dict]:
    return sorted(results, key=lambda r: (-r["anchored"]["corr"], r["anchored"]["mae"],
                                          r["anchored"]["rmse"], r["seconds"], r["config_id"]))


def read_results(folder: Path) -> list[dict]:
    results = []
    for result_path in sorted(folder.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if sha256(result_path.parent / "checkpoints/checkpoint.pth") != result["checkpoint_sha256"]:
            raise AssertionError(f"Checkpoint changed after result creation: {result_path.parent}")
        result["source"] = str(result_path.parent.relative_to(ROOT))
        results.append(result)
    return results


def write_summary(path: Path, results: list[dict]) -> None:
    rows = []
    for result in rank(results):
        rows.append({
            "key": result["key"], "config_id": result["config_id"], "seed": result["seed"],
            "learning_rate": result["learning_rate"], "schedule": result["lradj"],
            "best_epoch": result["best_epoch"], "epochs": result["epochs"], "seconds": result["seconds"],
            "anchored_corr": result["anchored"]["corr"], "anchored_mae_mm": result["anchored"]["mae"],
            "anchored_rmse_mm": result["anchored"]["rmse"], "anchored_h2_15_corr": float(np.mean(result["anchored"]["horizon_corr"][1:])),
            "raw_corr": result["raw"]["corr"],
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def stage1(root: Path, frozen: dict) -> None:
    verify_frozen(frozen)
    if not (root / "preflight_report.json").is_file() or not (root / "smoke_report.json").is_file():
        raise RuntimeError("Run preflight and smoke successfully before stage1")
    folder = root / "stage1"
    folder.mkdir(exist_ok=True)
    for item in frozen["stage1"]:
        if (folder / item["key"] / "result.json").is_file():
            continue
        run_one(folder, item)
        write_summary(folder / "ranking.csv", read_results(folder))
    results = read_results(folder)
    if len(results) != 21:
        raise AssertionError(f"Stage 1 incomplete: expected 21 results, got {len(results)}")
    write_summary(folder / "ranking.csv", results)


def stage2_plan(root: Path, frozen: dict) -> dict:
    path = root / "stage2_plan.json"
    stage1_results = read_results(root / "stage1")
    if len(stage1_results) != 21:
        raise RuntimeError("Stage 1 must complete before stage 2 planning")
    selected = rank(stage1_results)[:3]
    baseline_id = config_id(*BASELINE)
    if baseline_id not in {result["config_id"] for result in selected}:
        selected.append(next(result for result in stage1_results if result["config_id"] == baseline_id))
    chosen = [{key: result[key] for key in ("config_id", "learning_rate", "lradj", "exp_decay_gamma", "lr_decay_gamma", "warmup_epochs") if key in result}
              for result in selected]
    runs = []
    for config in chosen:
        for seed in (43, 44):
            runs.append({"key": f"{config['config_id']}_s{seed}", "seed": seed, **config,
                         **schedule_fields(config["lradj"])})
    plan = {
        "selection_rule": "Top 3 stage-1 configurations by anchored corr, then MAE, RMSE, runtime, config ID; add baseline if absent.",
        "configs": chosen, "runs": runs, "new_runs": len(runs), "frozen_source_hashes": frozen["source_hashes"],
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise AssertionError("Existing stage-2 plan differs from the frozen selection")
    else:
        save_json(path, plan)
    return plan


def stage2(root: Path, frozen: dict) -> None:
    verify_frozen(frozen)
    plan = stage2_plan(root, frozen)
    if plan["new_runs"] > 8:
        raise AssertionError(f"Stage 2 run cap exceeded: {plan['new_runs']}")
    folder = root / "stage2"
    folder.mkdir(exist_ok=True)
    for item in plan["runs"]:
        if (folder / item["key"] / "result.json").is_file():
            continue
        run_one(folder, item)
        write_summary(folder / "progress.csv", read_results(folder))
    results = read_results(folder)
    if len(results) != plan["new_runs"]:
        raise AssertionError("Stage 2 incomplete")
    write_summary(folder / "ranking.csv", results)


def finalize(root: Path, frozen: dict) -> None:
    verify_frozen(frozen)
    stage1_results, stage2_results = read_results(root / "stage1"), read_results(root / "stage2")
    plan = json.loads((root / "stage2_plan.json").read_text(encoding="utf-8"))
    groups = []
    for config in plan["configs"]:
        cid = config["config_id"]
        members = [result for result in stage1_results + stage2_results if result["config_id"] == cid]
        if {result["seed"] for result in members} != {42, 43, 44}:
            raise AssertionError(f"Missing seed for {cid}")
        groups.append({
            "config_id": cid,
            "mean_anchored_corr": float(np.mean([r["anchored"]["corr"] for r in members])),
            "std_anchored_corr": float(np.std([r["anchored"]["corr"] for r in members], ddof=1)),
            "mean_mae_mm": float(np.mean([r["anchored"]["mae"] for r in members])),
            "mean_rmse_mm": float(np.mean([r["anchored"]["rmse"] for r in members])),
            "seeds": members,
        })
    final = sorted(groups, key=lambda r: (-r["mean_anchored_corr"], r["mean_mae_mm"], r["mean_rmse_mm"], r["config_id"]))[0]
    winning_seed = rank(final["seeds"])[0]
    total_seconds = sum(result["seconds"] for result in stage1_results + stage2_results)
    protected_now = hashes(PROTECTED_PATHS)
    report = {
        "final_configuration": {key: value for key, value in final.items() if key != "seeds"},
        "best_seed_checkpoint": {"key": winning_seed["key"], "seed": winning_seed["seed"],
                                  "path": winning_seed["source"], "best_epoch": winning_seed["best_epoch"]},
        "stage1_ranking": [result["key"] for result in rank(stage1_results)],
        "stage2_results": groups, "best_epoch_distribution": [result["best_epoch"] for result in stage1_results + stage2_results],
        "run_count": {"stage1": len(stage1_results), "stage2_new": len(stage2_results), "total_search": len(stage1_results) + len(stage2_results)},
        "total_training_seconds": total_seconds,
        "protected_hashes_unchanged": protected_now == frozen["protected_hashes"],
        "protected_hashes_after": protected_now,
        "scope": "13 features; current training data and fixed split; train/validation only; no test loader/evaluation.",
    }
    save_json(root / "final_report.json", report)
    pd.DataFrame([{key: value for key, value in group.items() if key != "seeds"} for group in groups]).sort_values(
        "mean_anchored_corr", ascending=False).to_csv(root / "final_ranking.csv", index=False)


def verify_frozen(frozen: dict) -> None:
    current = hashes(SOURCE_PATHS)
    if current != frozen["source_hashes"]:
        changed = [path for path, digest in current.items() if frozen["source_hashes"].get(path) != digest]
        raise RuntimeError(f"Frozen source/data changed; stop instead of mixing results: {changed}")


def initialise(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"Study path already exists: {root}")
    root.mkdir(parents=True)
    frozen = {
        "study": "13-feature DLinearMix2 selection on current historical data",
        "created_at_unix": time.time(), "base_config": base_config(), "stage1": stage1_items(),
        "source_hashes": hashes(SOURCE_PATHS), "protected_hashes": hashes(PROTECTED_PATHS),
        "git_status_before": subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True,
                                               check=True, capture_output=True).stdout.splitlines(),
        "selection": "Best raw-validation-correlation checkpoint per run; rank configurations by anchored validation correlation, then MAE, RMSE, runtime, config ID.",
    }
    save_json(root / "search_plan.json", frozen)
    text = "\n".join([
        "# 13-feature hyperparameter selection plan", "",
        "This plan is frozen before search training. It uses the current `train_old.csv` and existing split assignments; only `Now` is removed from the 14-feature model input.", "",
        "## Fixed setup", "",
        "- Features: HL02–HL06; Past10Min, Past1Hr, and six gate fields (5 + 8).",
        "- DLinearMix2, seq_len 60, pred_len 15, MSE, batch 64, max 80 epochs, patience 15.",
        "- Train/validation only: `validation_only=True`; no test loader or test metric.",
        "- Per-run checkpoint: highest raw validation correlation. Comparison: anchored validation correlation, then MAE, RMSE, runtime, configuration ID.",
        "- Stage 1: seven LRs (0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.004) × type1 / exp(0.85) / warmup_exp(warmup 8, gamma 0.85), seed 42; 21 runs.",
        "- Stage 2: top three, plus the baseline if absent; add seeds 43 and 44, at most eight new runs. Retain all seeds.",
        "- Save exact epoch-8 and epoch-10 snapshots for every search run; do not use them to select in this study.",
        "",
        "## Checks before Stage 1", "",
        "Feature order and 5/8 dimensions; parameter count; scaler comparison; Now-related window-count change; target time alignment; one-epoch smoke and same-seed replay; schedule and legacy-hash verification.",
        "",
        "## Runtime estimate", "",
        "A two-run one-epoch CPU calibration is saved in `smoke_report.json` before Stage 1. Its conservative 80-epoch × 29-run upper bound is recorded there; early stopping reduces actual runtime.",
    ]) + "\n"
    (root / "search_plan.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["init", "preflight", "smoke", "stage1", "stage2", "finalize", "status"])
    parser.add_argument("--output", default="runs/v13feat_currentdata_selection_20261004")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = (ROOT / args.output).resolve()
    if args.mode == "init":
        initialise(root)
        print(root)
        return
    frozen = json.loads((root / "search_plan.json").read_text(encoding="utf-8"))
    if args.mode == "preflight":
        preflight(root, frozen)
    elif args.mode == "smoke":
        smoke(root, frozen)
    elif args.mode == "stage1":
        stage1(root, frozen)
    elif args.mode == "stage2":
        stage2(root, frozen)
    elif args.mode == "finalize":
        finalize(root, frozen)
    else:
        for name in ("preflight_report.json", "smoke_report.json", "stage2_plan.json", "final_report.json"):
            path = root / name
            if path.exists():
                print(f"{name}: present")
        for name in ("stage1", "stage2"):
            folder = root / name
            print(f"{name}: {len(read_results(folder)) if folder.exists() else 0} completed")


if __name__ == "__main__":
    main()
