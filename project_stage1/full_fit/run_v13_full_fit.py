"""Run the approved 13-feature full-data fit and export its predictor bundle.

The configuration in this file is intentionally fixed: DLinearMix2, the
approved 13 ordered features, seed 44, LR 0.0012, exp(gamma=0.85), and exactly
11 epochs.  It is not a hyperparameter-search entry point.

Typical sequence (each command refuses to overwrite completed artifacts):

    python -m project_stage1.full_fit.run_v13_full_fit init
    python -m project_stage1.full_fit.run_v13_full_fit preflight
    python -m project_stage1.full_fit.run_v13_full_fit fit
    python -m project_stage1.full_fit.run_v13_full_fit export
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from exp.exp_Main2 import Exp_Main
from project_stage1.full_fit.export_model_bundle import export_bundle
from run import _configure_mix_model_args, _set_seed


RUN_ID = "v13_fullfit_lr0.0012_exp_s44_e11"
DEFAULT_OUTPUT = Path("runs") / RUN_ID
DEFAULT_BUNDLE = Path("project_stage1/inference/model_bundle")
WATER = ["HL02", "HL03", "HL04", "HL05", "HL06"]
EXOG = [
    "Past10Min", "Past1Hr",
    "north_gate_opening_1", "north_gate_opening_2", "north_gate_opening_3",
    "south_gate_opening_1", "south_gate_opening_2", "south_gate_opening_3",
]
SOURCE_PATHS = [
    "run.py",
    "models/DLinearMix2.py",
    "data_provider/Data_Loader.py",
    "data_provider/Data_Factory.py",
    "exp/exp_Main2.py",
    "utils/tools.py",
    "dataset/train_old.csv",
    "dataset/splits_train_old.csv",
    "project_stage1/inference/input_schema.json",
    "project_stage1/inference/model.py",
    "project_stage1/full_fit/export_model_bundle.py",
    "project_stage1/full_fit/run_v13_full_fit.py",
]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashes(paths: list[str]) -> dict[str, str]:
    result = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source/data artifact is missing: {path}")
        result[relative] = _sha256(path)
    return result


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def base_config() -> dict:
    """The reviewed deployment configuration; do not expose tuning knobs here."""
    return {
        "model": "DLinearMix2",
        "data": "custom",
        "root_path": str(ROOT / "dataset"),
        "data_path": "train_old.csv",
        "features": "S",
        "target": "HL01",
        "freq": "min",
        "embed": "timeF",
        "input_col": ",".join(WATER),
        "exog_col": ",".join(EXOG),
        "segment_col": "segment_id",
        # The file is retained as provenance.  With train_only=True the loader
        # deliberately uses every segment instead of its train/val/test labels.
        "split_mode": "file",
        "split_file": str(ROOT / "dataset/splits_train_old.csv"),
        "fold": None,
        "seq_len": 60,
        "label_len": 30,
        "pred_len": 15,
        "stride_train": 1,
        "stride_eval": 1,
        "individual": True,
        "dlinear_kernel_size": 25,
        "mix_in": None,
        "branch_in": None,
        "exog_in": None,
        "flatten_fusion": True,
        "fusion_hidden_dim": 32,
        "exog_emb_dim": 16,
        "dropout": 0.1,
        "train_epochs": 11,
        "batch_size": 64,
        "patience": 0,
        "num_workers": 0,
        "train_only": True,
        "validation_only": False,
        "learning_rate": 0.0012,
        "lradj": "exp",
        "criterion": "mse",
        "huber_beta": 1.0,
        "warmup_epochs": 0,
        "exp_decay_gamma": 0.85,
        "lr_decay_gamma": 0.85,
        "lr_floor_ratio": 0.01,
        "early_stop_metric": "not_used_fixed_full_fit",
        "seed": 44,
        "use_amp": False,
        "use_gpu": False,
        "use_multi_gpu": False,
        "gpu": 0,
    }


def _args_for(run_dir: Path) -> SimpleNamespace:
    args = SimpleNamespace(**base_config())
    args.run_dir = str(run_dir.resolve())
    args.setting = RUN_ID
    _configure_mix_model_args(args)
    if args.input_cols != WATER or args.exog_cols != EXOG:
        raise AssertionError(f"Unexpected feature order: {args.input_cols=} {args.exog_cols=}")
    if (args.branch_in, args.exog_in, args.mix_in) != (5, 8, 13):
        raise AssertionError(
            "Expected 5 branch + 8 exogenous = 13 features, got "
            f"{args.branch_in} + {args.exog_in} = {args.mix_in}"
        )
    if not args.train_only or args.validation_only:
        raise AssertionError("Full fit must use train_only=True and validation_only=False")
    return args


def _load_plan(root: Path) -> dict:
    path = root / "full_fit_plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run init before this step: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    current = _hashes(SOURCE_PATHS)
    if plan["source_hashes"] != current:
        changed = [name for name, digest in current.items() if plan["source_hashes"].get(name) != digest]
        raise RuntimeError(f"Frozen source/data changed; refuse to mix artifacts: {changed}")
    return plan


def initialise(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite full-fit run directory: {root}")
    root.mkdir(parents=True)
    config = base_config()
    plan = {
        "run_id": RUN_ID,
        "created_at": _now(),
        "configuration": config,
        "feature_order": WATER + EXOG,
        "fit_contract": {
            "data_scope": "All valid windows from all segments in dataset/train_old.csv.",
            "scaler_scope": "Refit X and Y scalers on clean rows from that same full-data scope.",
            "loader_policy": "Only the train loader is constructed, with train_only=True.",
            "model_selection": "None: run exactly 11 epochs and save the final epoch weights.",
            "validation_or_test_metrics": "Not constructed or reported by this full-fit run.",
        },
        "source_hashes": _hashes(SOURCE_PATHS),
        "git_status_before": subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines(),
    }
    _write_json(root / "full_fit_plan.json", plan)


def preflight(root: Path, plan: dict) -> None:
    """Validate the full-data loader and model once without creating val/test loaders."""
    args = _args_for(root)
    _set_seed(args.seed)
    experiment = Exp_Main(args)
    dataset, loader = experiment._get_data("train")
    if len(dataset) == 0 or len(loader) == 0:
        raise ValueError("Full-data loader has no valid windows/batches")
    batch_x, batch_y, *_ = next(iter(loader))
    with torch.no_grad():
        output = experiment._forward(batch_x.float().to(experiment.device))
    if tuple(batch_x.shape[1:]) != (60, 13):
        raise AssertionError(f"Expected X shape [batch, 60, 13], got {tuple(batch_x.shape)}")
    if tuple(output.shape[1:]) != (15, 1):
        raise AssertionError(f"Expected model output [batch, 15, 1], got {tuple(output.shape)}")

    starts = dataset.valid_starts
    origin = pd.Timestamp(dataset.dates[int(starts[0]) + args.seq_len - 1])
    labels = pd.DatetimeIndex(
        dataset.dates[int(starts[0]) + args.seq_len:int(starts[0]) + args.seq_len + args.pred_len]
    )
    if len(labels) != 15 or labels[0] != origin + pd.Timedelta(minutes=1) or labels[-1] != origin + pd.Timedelta(minutes=15):
        raise AssertionError("Labels are not aligned to t+1 through t+15")

    raw = pd.read_csv(ROOT / "dataset/train_old.csv", usecols=["segment_id"])
    all_segments = {str(value) for value in raw["segment_id"].unique()}
    window_segments = {str(value) for value in dataset.window_segment_ids}
    report = {
        "checked_at": _now(),
        "run_id": plan["run_id"],
        "feature_order": dataset.x_cols,
        "dimensions": {"branch_in": args.branch_in, "exog_in": args.exog_in, "total_features": args.mix_in},
        "train_windows": len(dataset),
        "train_rows": len(dataset.data_x),
        "segments": {
            "all_in_source": len(all_segments),
            "with_at_least_one_valid_window": len(window_segments),
            "without_valid_window": sorted(all_segments - window_segments),
        },
        "batch_x_shape": list(batch_x.shape),
        "batch_y_shape": list(batch_y.shape),
        "output_shape": list(output.shape),
        "target_alignment": {
            "origin": origin.isoformat(),
            "first_label": labels[0].isoformat(),
            "last_label": labels[-1].isoformat(),
        },
        "loader_requests": ["train"],
        "validation_loader_constructed": False,
        "test_loader_constructed": False,
        "parameter_count": int(sum(parameter.numel() for parameter in experiment.model.parameters())),
    }
    _write_json(root / "preflight_report.json", report)


def fit(root: Path, plan: dict) -> None:
    if not (root / "preflight_report.json").is_file():
        raise RuntimeError("Run preflight successfully before fit")
    if (root / "full_fit_report.json").exists():
        raise FileExistsError("This run directory already contains a completed full fit")

    args = _args_for(root)
    _write_json(root / "run_args.json", vars(args))
    started = time.monotonic()
    with (root / "train.log").open("w", buffering=1, encoding="utf-8") as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        _set_seed(args.seed)
        experiment = Exp_Main(args)
        experiment.train_fixed_epochs()
    elapsed_seconds = time.monotonic() - started

    dataset = experiment.fixed_fit_train_data
    if dataset.scaler_x is None or dataset.scaler_y is None:
        raise AssertionError("Full-data fit did not produce both X and Y scalers")
    if experiment.epochs_trained != args.train_epochs or len(experiment.training_history) != args.train_epochs:
        raise AssertionError("Fixed fit did not complete exactly the configured number of epochs")
    history_path = root / "checkpoints/training_history.csv"
    checkpoint_path = root / "checkpoints/checkpoint.pth"
    if not history_path.is_file() or not checkpoint_path.is_file():
        raise AssertionError("Fixed fit did not write its required history/checkpoint artifacts")
    np.savez(
        root / "scaler.npz",
        x_mean=dataset.scaler_x.mean_,
        x_scale=dataset.scaler_x.scale_,
        y_mean=dataset.scaler_y.mean_,
        y_scale=dataset.scaler_y.scale_,
        features=np.asarray(dataset.x_cols),
    )
    report = {
        "completed_at": _now(),
        "run_id": plan["run_id"],
        "configuration": {
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "schedule": args.lradj,
            "exp_decay_gamma": args.exp_decay_gamma,
            "epochs": args.train_epochs,
        },
        "feature_order": dataset.x_cols,
        "train_windows": len(dataset),
        "train_rows": len(dataset.data_x),
        "parameter_count": int(sum(parameter.numel() for parameter in experiment.model.parameters())),
        "loader_requests": ["train"],
        "validation_loader_constructed": False,
        "test_loader_constructed": False,
        "checkpoint_policy": "checkpoint.pth contains the final epoch (epoch 11), not a selected best checkpoint.",
        "epochs_completed": experiment.epochs_trained,
        "final_train_loss": experiment.training_history[-1]["train_loss"],
        "elapsed_seconds": elapsed_seconds,
        "artifacts": {
            "checkpoint": "checkpoints/checkpoint.pth",
            "checkpoint_sha256": _sha256(checkpoint_path),
            "training_history": "checkpoints/training_history.csv",
            "scaler": "scaler.npz",
            "scaler_sha256": _sha256(root / "scaler.npz"),
            "log": "train.log",
        },
    }
    _write_json(root / "full_fit_report.json", report)


def export(root: Path, plan: dict, bundle_dir: Path) -> None:
    if not (root / "full_fit_report.json").is_file():
        raise RuntimeError("Run fit successfully before export")
    report_path = root / "export_report.json"
    if report_path.exists():
        raise FileExistsError(f"This run has already been exported: {report_path}")
    bundle = export_bundle(root, bundle_dir)
    report = {
        "exported_at": _now(),
        "run_id": plan["run_id"],
        "bundle_dir": _relative(bundle),
        "files": {
            name: _sha256(bundle / name)
            for name in ("config.json", "scaler.npz", "checkpoint.pth")
        },
    }
    _write_json(report_path, report)


def status(root: Path) -> None:
    print(f"run directory: {_relative(root)}")
    for name in ("full_fit_plan.json", "preflight_report.json", "run_args.json", "full_fit_report.json", "export_report.json"):
        print(f"{name}: {'present' if (root / name).is_file() else 'missing'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["init", "preflight", "fit", "export", "status"])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="run directory, relative to repository root")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE), help="new inference bundle directory")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = _resolve(args.output)
    if args.mode == "init":
        initialise(root)
        print(root)
        return
    plan = _load_plan(root)
    if args.mode == "preflight":
        preflight(root, plan)
    elif args.mode == "fit":
        fit(root, plan)
    elif args.mode == "export":
        export(root, plan, _resolve(args.bundle))
    else:
        status(root)


if __name__ == "__main__":
    main()
