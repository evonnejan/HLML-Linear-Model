"""Export one completed fixed-fit run as the inference model bundle.

The bundle holds exactly what ``project_stage1.inference.predictor.Predictor``
loads: ``config.json`` (model settings and ordered features), ``scaler.npz``
and ``checkpoint.pth``.

Example:
    python -m project_stage1.full_fit.export_model_bundle \
        runs/<run_dir> project_stage1/inference/model_bundle
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

MODEL_KEYS = [
    "seq_len", "pred_len", "dlinear_kernel_size", "flatten_fusion",
    "fusion_hidden_dim", "exog_emb_dim", "dropout",
]
SCALER_KEYS = ["x_mean", "x_scale", "y_mean", "y_scale", "features"]


def _columns(args: dict, list_key: str, text_key: str) -> list[str]:
    if args.get(list_key):
        return [str(name) for name in args[list_key]]
    return [name.strip() for name in str(args[text_key]).split(",") if name.strip()]


def export_bundle(
    run_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str = "checkpoints/checkpoint.pth",
) -> Path:
    """Copy one verified run into a new, non-populated predictor bundle.

    Refusing a populated output directory prevents a checkpoint from one run
    being silently mixed with scalers or configuration from another run.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    args_path = run_dir / "run_args.json"
    scaler_path = run_dir / "scaler.npz"
    checkpoint_path = run_dir / checkpoint
    for path in (args_path, scaler_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required run artifact is missing: {path}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated model bundle: {out_dir}")

    args = json.loads(args_path.read_text(encoding="utf-8"))
    if args.get("model") != "DLinearMix2":
        raise ValueError(f"Expected a DLinearMix2 run, got {args.get('model')!r}")
    try:
        config = {key: args[key] for key in MODEL_KEYS}
    except KeyError as error:
        raise ValueError(f"run_args.json is missing required model setting: {error.args[0]}") from error
    config["branch_features"] = _columns(args, "input_cols", "input_col")
    config["exog_features"] = _columns(args, "exog_cols", "exog_col")

    with np.load(scaler_path, allow_pickle=False) as scaler:
        missing = [key for key in SCALER_KEYS if key not in scaler.files]
        if missing:
            raise ValueError(f"scaler.npz is missing required arrays: {missing}")
        arrays = {key: scaler[key] for key in SCALER_KEYS}
    expected_features = config["branch_features"] + config["exog_features"]
    if [str(name) for name in arrays["features"]] != expected_features:
        raise ValueError("scaler.npz feature order differs from run_args.json input_col + exog_col")
    if arrays["x_mean"].shape != arrays["x_scale"].shape or arrays["x_mean"].size != len(expected_features):
        raise ValueError("scaler.npz X-scaler shape does not match the configured feature order")
    if arrays["y_mean"].size != 1 or arrays["y_scale"].size != 1:
        raise ValueError("scaler.npz Y-scaler must contain exactly one target scale")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez(out_dir / "scaler.npz", **arrays)
    shutil.copyfile(checkpoint_path, out_dir / "checkpoint.pth")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--checkpoint", default="checkpoints/checkpoint.pth",
        help="checkpoint path relative to run_dir",
    )
    args = parser.parse_args()
    print(export_bundle(args.run_dir, args.out_dir, args.checkpoint))


if __name__ == "__main__":
    main()
