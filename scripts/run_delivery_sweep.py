"""Old/base-split LR study; validation only, baseline must pass before sweep.

Run from repo root with the training environment. Artifacts live in runs/.
No test loader, test predictions or persistence comparison are constructed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
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

FEATURES = ["HL02", "HL03", "HL04", "HL05", "HL06"]
EXOG = ["Past10Min", "Past1Hr", "Now", "north_gate_opening_1",
        "north_gate_opening_2", "north_gate_opening_3", "south_gate_opening_1",
        "south_gate_opening_2", "south_gate_opening_3"]
RATES = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.003]
SCHEDULES = ["type1", "exp", "warmup_exp"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def configuration():
    return dict(model="DLinearMix2", data="custom", root_path=str(ROOT / "dataset"),
        data_path="train_old.csv", features="S", target="HL01", freq="min", embed="timeF",
        input_col=",".join(FEATURES), exog_col=",".join(EXOG), segment_col="segment_id",
        split_mode="file", split_file=str(ROOT / "dataset/splits_train_old.csv"), fold=None,
        seq_len=60, label_len=30, pred_len=15, stride_train=1, stride_eval=1,
        individual=True, dlinear_kernel_size=25, mix_in=None, branch_in=None, exog_in=None,
        flatten_fusion=True, fusion_hidden_dim=32, exog_emb_dim=16, dropout=0.1,
        train_epochs=80, batch_size=64, patience=15, num_workers=0, train_only=False,
        learning_rate=0.001, lradj="type1", criterion="mse", huber_beta=1.0,
        warmup_epochs=8, lr_decay_gamma=0.92, lr_floor_ratio=0.01,
        early_stop_metric="corr", seed=42, use_amp=False, use_gpu=False,
        use_multi_gpu=False, gpu=0, validation_only=True)


def plan():
    files = ["dataset/train_old.csv", "dataset/splits_train_old.csv", "dataset/splits_train_old.json",
             "models/DLinearMix2.py", "data_provider/Data_Loader.py", "data_provider/Data_Factory.py",
             "exp/exp_Main2.py", "exp/exp_Basic.py", "utils/tools.py", "run.py",
             "scripts/run_delivery_sweep.py"]
    cfg = configuration()
    side = json.loads((ROOT / "dataset/splits_train_old.json").read_text())
    if side["data_sha256"] != sha(ROOT / "dataset/train_old.csv") or side["split_sha256"] != sha(cfg["split_file"]):
        raise ValueError("Split sidecar does not match current data/split")
    if (side["seq_len"], side["pred_len"], side["stride"]) != (60, 15, 1):
        raise ValueError("Split window configuration mismatch")
    combos = [(0.001, "type1")] + [(lr, sc) for sc in SCHEDULES for lr in RATES if (lr, sc) != (0.001, "type1")]
    return {"config": cfg, "runs": [{"key": f"lr{lr:g}_{sc}", "learning_rate": lr, "lradj": sc} for lr, sc in combos],
            "hashes": {p: sha(ROOT / p) for p in files},
            "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "device": "cpu", "torch_threads": 1},
            "selection": "best raw val corr checkpoint; rank anchored val corr, report physical MAE/RMSE",
            "note": "Explicit 14 features; omits constant north_gate_opening_4. Split counts are audited against loader before training."}


def score(pred, true):
    if pred.shape != true.shape or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("Invalid prediction/target arrays")
    error = pred.astype(np.float64) - true
    corrs = []
    for h in range(pred.shape[1]):
        x, y = pred[:, h, 0], true[:, h, 0]
        if x.std() > 0 and y.std() > 0:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
    if not corrs or not np.isfinite(corrs).all():
        raise ValueError("No valid correlation to rank")
    return dict(corr=float(np.mean(corrs)), valid_corr_horizons=len(corrs),
                mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error ** 2).mean())))


def validate_checkpoint(exp, folder):
    ds, loader = exp._get_data("val")
    raw, true, anchor, _ = exp._run_inference(ds, loader)
    for arr in (raw, true, anchor):
        if not np.isfinite(arr).all():
            raise ValueError("Non-finite validation output")
    shape = raw.shape
    inverse = lambda a: ds.inverse_transform(a.reshape(-1, shape[-1])).reshape(a.shape)
    raw, true, anchor = inverse(raw), inverse(true), inverse(anchor[:, :1, :])
    anchored = raw - (raw[:, :1, :] - anchor)
    if not np.allclose(anchored[:, :1], anchor, atol=1e-5):
        raise ValueError("Anchored first-step invariant failed")
    np.savez_compressed(folder / "validation_predictions.npz", raw=raw, true=true, anchor=anchor,
                        anchored=anchored, origins=ds.dates[ds.valid_starts + ds.seq_len - 1].astype(str))
    # Persist exact fitted preprocessing, not a refit on customer input.
    np.savez(folder / "scaler.npz", x_mean=ds.scaler_x.mean_, x_scale=ds.scaler_x.scale_,
             y_mean=ds.scaler_y.mean_, y_scale=ds.scaler_y.scale_, features=np.array(ds.x_cols))
    raw_scores, adj_scores = score(raw, true), score(anchored, true)
    return {"n_val": len(ds), "raw": raw_scores, "anchored": adj_scores}


def verify_windows(exp):
    split = pd.read_csv(exp.args.split_file)
    counts = {}
    for flag in ("train", "val"):
        ds, _ = exp._get_data(flag)
        wanted = split[split["split"] == flag]
        if set(ds.window_segment_ids) - set(wanted.segment_id):
            raise ValueError("Unexpected segment in dataset")
        actual = pd.Series(ds.window_segment_ids).value_counts().to_dict()
        if any(actual.get(r.segment_id, 0) != r.n_windows for r in wanted.itertuples()):
            raise ValueError("Loader and split per-segment window counts differ")
        counts[flag] = len(ds)
    return counts


def run_one(base, item, root):
    folder = root / item["key"]
    if folder.exists():
        raise ValueError(f"Incomplete/existing run: {folder}; do not overwrite")
    folder.mkdir()
    args = SimpleNamespace(**(base | {k: v for k, v in item.items() if k != "key"}))
    args.run_dir, args.setting = str(folder), item["key"]
    _configure_mix_model_args(args)
    save_json(folder / "run_args.json", vars(args))
    started = time.monotonic()
    with (folder / "train.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        _set_seed(args.seed)
        exp = Exp_Main(args)
        counts = verify_windows(exp)
        exp.train(args.setting)
        metrics = validate_checkpoint(exp, folder)
        trace = exp.training_history
        expected_best = max(trace, key=lambda r: (r["val_corr"], r["epoch"]))["epoch"]
        if exp.best_epoch != expected_best:
            raise ValueError("Best checkpoint epoch inconsistent with trace")
        if abs(metrics["raw"]["corr"] - trace[exp.best_epoch - 1]["val_corr"]) > 1e-5:
            raise ValueError("Reloaded checkpoint validation correlation mismatch")
    result = item | metrics | {"counts": counts, "best_epoch": exp.best_epoch,
        "epochs": exp.epochs_trained, "seconds": time.monotonic() - started,
        "checkpoint_sha256": sha(folder / "checkpoints/checkpoint.pth")}
    save_json(folder / "result.json", result)
    return result


def collect(root, frozen):
    results = []
    for item in frozen["runs"]:
        p = root / item["key"] / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r["checkpoint_sha256"] != sha(p.parent / "checkpoints/checkpoint.pth"):
            raise ValueError("Completed checkpoint changed")
        results.append({k: r[k] for k in ("key", "learning_rate", "lradj", "best_epoch", "epochs", "seconds")} |
                       {f"anchored_val_{k}": v for k, v in r["anchored"].items()} |
                       {f"raw_val_{k}": v for k, v in r["raw"].items()})
    if results:
        pd.DataFrame(results).sort_values("anchored_val_corr", ascending=False).to_csv(root / "summary.csv", index=False)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "baseline", "remaining", "status"])
    ap.add_argument("--output", default="runs/delivery_lr_20260926")
    a = ap.parse_args()
    torch.set_num_threads(1)
    root = Path(a.output).resolve()
    if a.mode == "plan":
        root.mkdir(parents=True, exist_ok=True)
        if (root / "plan.json").exists():
            raise ValueError("Plan already exists; use another output directory")
        save_json(root / "plan.json", plan())
        print(f"Saved 21-run plan to {root}", flush=True)
        return
    frozen = json.loads((root / "plan.json").read_text())
    if a.mode == "status":
        print(json.dumps(collect(root, frozen), indent=2))
        return
    if frozen != plan():
        raise ValueError("Code/data/environment/config changed since plan; start a new study")
    if a.mode == "remaining" and not (root / frozen["runs"][0]["key"] / "result.json").exists():
        raise ValueError("Baseline must finish and pass first")
    items = frozen["runs"][:1] if a.mode == "baseline" else frozen["runs"][1:]
    collect(root, frozen)
    for item in items:
        if (root / item["key"] / "result.json").exists():
            continue
        print(f"START {item['key']}", flush=True)
        try:
            result = run_one(frozen["config"], item, root)
        except Exception as exc:
            save_json(root / "failure.json", {"key": item["key"], "error": repr(exc)})
            raise
        collect(root, frozen)
        print(f"DONE {item['key']} best_ep={result['best_epoch']} anchored_val={result['anchored']}", flush=True)


if __name__ == "__main__":
    main()
