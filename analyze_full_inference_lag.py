import argparse
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_provider.Data_Loader import Dataset_Custom
from exp.exp_Main import Exp_Main


@dataclass
class RunInfo:
    run_dir: str
    setting: str
    model: str
    input_col: str
    target: str
    seq_len: int
    pred_len: int
    kernel_size: Optional[int]
    run_args: Dict


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _parse_csv_list(text: Optional[str], cast=str):
    if text is None or str(text).strip() == "":
        return None
    items = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            items.append(cast(part))
    return items if items else None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def discover_runs(runs_root: str) -> List[str]:
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"runs_root not found: {runs_root}")
    run_dirs = [
        os.path.join(runs_root, name)
        for name in os.listdir(runs_root)
        if os.path.isdir(os.path.join(runs_root, name))
    ]
    run_dirs.sort()
    return run_dirs


def load_run_info(run_dir: str) -> Optional[RunInfo]:
    args_path = os.path.join(run_dir, "run_args.json")
    if not os.path.exists(args_path):
        return None

    with open(args_path, "r", encoding="utf-8") as file:
        run_args = json.load(file)

    setting = str(run_args.get("setting", os.path.basename(run_dir)))
    model = str(run_args.get("model", ""))
    input_col = str(run_args.get("input_col") or run_args.get("target") or "")
    target = str(run_args.get("target", ""))
    seq_len = _safe_int(run_args.get("seq_len"), -1)
    pred_len = _safe_int(run_args.get("pred_len"), -1)
    kernel_size = _safe_int(run_args.get("dlinear_kernel_size"), None)

    return RunInfo(
        run_dir=run_dir,
        setting=setting,
        model=model,
        input_col=input_col,
        target=target,
        seq_len=seq_len,
        pred_len=pred_len,
        kernel_size=kernel_size,
        run_args=run_args,
    )


def pass_filters(
    info: RunInfo,
    models: Optional[List[str]],
    input_cols: Optional[List[str]],
    targets: Optional[List[str]],
    seq_lens: Optional[List[int]],
    kernels: Optional[List[int]],
):
    if models is not None and info.model not in models:
        return False
    if input_cols is not None and info.input_col not in input_cols:
        return False
    if targets is not None and info.target not in targets:
        return False
    if seq_lens is not None and info.seq_len not in seq_lens:
        return False
    if kernels is not None and info.kernel_size not in kernels:
        return False
    return True


def select_best_runs(rows_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for pair, group in rows_df.groupby("pair"):
        best_rows.append(group.sort_values("mse_test", ascending=True).iloc[0])
    return pd.DataFrame(best_rows).reset_index(drop=True)


def topk_per_pair(df: pd.DataFrame, k: int, sort_col: str, ascending: bool, dropna_col: Optional[str] = None) -> pd.DataFrame:
    work = df.copy()
    if dropna_col is not None:
        work = work.dropna(subset=[dropna_col]).copy()

    out_rows = []
    for pair, group in work.groupby("pair"):
        topk = group.sort_values(sort_col, ascending=ascending).head(k).copy()
        topk.insert(0, "rank_within_pair", np.arange(1, len(topk) + 1))
        out_rows.append(topk)

    if len(out_rows) == 0:
        return pd.DataFrame(columns=["rank_within_pair"] + list(df.columns))
    return pd.concat(out_rows, axis=0, ignore_index=True)


def pair_name(input_col: str, target: str) -> str:
    return f"{input_col}->{target}"


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or y.size < 2:
        return np.nan
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def run_full_inference(info: RunInfo) -> pd.DataFrame:
    args = dict(info.run_args)
    args["setting"] = info.setting
    args["run_dir"] = info.run_dir
    args["checkpoints"] = os.path.join(info.run_dir, "checkpoints")

    exp_args = SimpleNamespace(**args)
    exp = Exp_Main(exp_args)

    ckpt_path = os.path.join(info.run_dir, "checkpoints", "checkpoint.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))

    timeenc = 0 if getattr(exp_args, "embed", "timeF") != "timeF" else 1

    dataset = Dataset_Custom(
        root_path=exp_args.root_path,
        data_path=exp_args.data_path,
        flag="train",
        size=[exp_args.seq_len, exp_args.label_len, exp_args.pred_len],
        features=exp_args.features,
        input_col=getattr(exp_args, "input_col", None),
        segment_col=getattr(exp_args, "segment_col", None),
        target=exp_args.target,
        stride=getattr(exp_args, "stride_eval", 1),
        scale=True,
        timeenc=timeenc,
        freq=exp_args.freq,
        train_only=True,
    )

    if dataset.window_segment_ids is None or len(dataset.window_segment_ids) == 0:
        raise RuntimeError(f"No segment windows found in full inference for setting={info.setting}")

    loader = DataLoader(
        dataset,
        batch_size=int(getattr(exp_args, "batch_size", 32)),
        shuffle=False,
        num_workers=int(getattr(exp_args, "num_workers", 0)),
        drop_last=False,
    )

    preds = []
    trues = []

    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -exp_args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :exp_args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            if "Linear" in exp_args.model:
                outputs = exp.model(batch_x)
            else:
                if getattr(exp_args, "output_attention", False):
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if exp_args.features == "MS" else 0
            outputs = outputs[:, -exp_args.pred_len:, f_dim:]
            batch_y = batch_y[:, -exp_args.pred_len:, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    if getattr(dataset, "scale", False):
        shape_preds = preds.shape
        preds = dataset.inverse_transform(preds.reshape(-1, shape_preds[-1])).reshape(shape_preds)
        trues = dataset.inverse_transform(trues.reshape(-1, shape_preds[-1])).reshape(shape_preds)

    sq_err = np.square(preds - trues)
    segment_ids = dataset.window_segment_ids

    rows = []
    for seg_id in pd.unique(segment_ids):
        mask = segment_ids == seg_id
        seg_pred = preds[mask]
        seg_true = trues[mask]
        seg_sq = sq_err[mask]
        if seg_sq.ndim == 3:
            seg_h_mse = seg_sq.mean(axis=(0, 2))
        else:
            seg_h_mse = seg_sq.mean(axis=0)

        pred_len = int(seg_h_mse.shape[0])
        for horizon in range(1, pred_len + 1):
            h_idx = horizon - 1
            h_pred = seg_pred[:, h_idx, :].reshape(-1)
            h_true = seg_true[:, h_idx, :].reshape(-1)

            rows.append(
                {
                    "pair": pair_name(info.input_col, info.target),
                    "setting": info.setting,
                    "input_col": info.input_col,
                    "target": info.target,
                    "segment": seg_id,
                    "horizon": horizon,
                    "num_points": int(h_true.size),
                    "mse": float(seg_h_mse[h_idx]),
                    "corr": _safe_corr(h_pred, h_true),
                }
            )

    return pd.DataFrame(rows)


def plot_best_horizon_mse_boxplot(best_seg_df: pd.DataFrame, out_path: str):
    if len(best_seg_df) == 0:
        return

    pairs = sorted(best_seg_df["pair"].unique().tolist())
    n = len(pairs)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, max(3.5, 3.2 * n)), squeeze=False)

    for idx, pair in enumerate(pairs):
        ax = axes[idx, 0]
        sub = best_seg_df[best_seg_df["pair"] == pair].copy()
        sub["best_horizon"] = sub["best_horizon"].astype(int)

        horizons = sorted(sub["best_horizon"].unique().tolist())
        box_data = [sub[sub["best_horizon"] == h]["mse_at_best_h"].astype(float).values for h in horizons]

        ax.boxplot(box_data, positions=horizons, widths=0.6, showfliers=False)
        ax.set_xticks(horizons)
        ax.set_title(f"{pair} | Best Horizon MSE Boxplot (Full Inference)")
        ax.set_xlabel("Best Horizon (argmin segment-horizon MSE)")
        ax.set_ylabel("MSE at Best Horizon")
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Full-data inference lag distribution using best checkpoints")
    parser.add_argument("--runs_root", type=str, default="./runs")
    parser.add_argument("--out_dir", type=str, default="./analysis")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--input_cols", type=str, default=None)
    parser.add_argument("--targets", type=str, default=None)
    parser.add_argument("--seq_lens", type=str, default=None)
    parser.add_argument("--kernels", type=str, default=None)
    parser.add_argument("--max_runs", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    models = _parse_csv_list(args.models, str)
    input_cols = _parse_csv_list(args.input_cols, str)
    targets = _parse_csv_list(args.targets, str)
    seq_lens = _parse_csv_list(args.seq_lens, int)
    kernels = _parse_csv_list(args.kernels, int)

    ensure_dir(args.out_dir)

    run_dirs = discover_runs(args.runs_root)

    rows = []
    info_by_setting: Dict[str, RunInfo] = {}
    used = 0
    skipped = 0

    for run_dir in run_dirs:
        info = load_run_info(run_dir)
        if info is None:
            skipped += 1
            continue

        if not pass_filters(info, models, input_cols, targets, seq_lens, kernels):
            continue

        metrics_path = os.path.join(run_dir, "outputs", "metrics.npy")
        if not os.path.exists(metrics_path):
            skipped += 1
            continue

        arr = np.load(metrics_path)
        mse = float(arr[1]) if len(arr) > 1 else np.nan
        rows.append(
            {
                "pair": pair_name(info.input_col, info.target),
                "setting": info.setting,
                "mse_test": mse,
            }
        )
        info_by_setting[info.setting] = info

        used += 1
        if args.max_runs > 0 and used >= args.max_runs:
            break

    if len(rows) == 0:
        raise RuntimeError("No valid runs found for lag analysis.")

    all_df = pd.DataFrame(rows)
    best_df = select_best_runs(all_df)

    lag_rows_all = []
    for item in best_df.itertuples(index=False):
        setting = item.setting
        info = info_by_setting[setting]
        print(f"[full-infer] {info.input_col}->{info.target} | setting={setting}")
        lag_rows_all.append(run_full_inference(info))

    lag_metrics_df = pd.concat(lag_rows_all, axis=0, ignore_index=True)

    metrics_csv = os.path.join(args.out_dir, "full_segment_horizon_metrics.csv")
    lag_metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")

    best_seg_df = (
        lag_metrics_df.sort_values(["pair", "setting", "segment", "mse"], ascending=[True, True, True, True])
        .groupby(["pair", "input_col", "target", "setting", "segment"], as_index=False)
        .first()
        .rename(columns={"horizon": "best_horizon", "mse": "mse_at_best_h", "corr": "corr_at_best_h"})
    )

    top30_mse_df = topk_per_pair(lag_metrics_df, k=30, sort_col="mse", ascending=True)
    top30_corr_df = topk_per_pair(lag_metrics_df, k=30, sort_col="corr", ascending=False, dropna_col="corr")

    mse_hits = (
        top30_mse_df.groupby(["pair", "input_col", "target", "setting", "segment"], as_index=False)
        .agg(
            mse_hit_count=("horizon", "count"),
            mse_best_rank=("rank_within_pair", "min"),
            mse_best_value=("mse", "min"),
        )
    )

    corr_hits = (
        top30_corr_df.groupby(["pair", "input_col", "target", "setting", "segment"], as_index=False)
        .agg(
            corr_hit_count=("horizon", "count"),
            corr_best_rank=("rank_within_pair", "min"),
            corr_best_value=("corr", "max"),
        )
    )

    overlap_segments_df = (
        mse_hits.merge(
            corr_hits,
            on=["pair", "input_col", "target", "setting", "segment"],
            how="inner",
        )
        .sort_values(["pair", "mse_best_rank", "corr_best_rank", "segment"])
        .reset_index(drop=True)
    )

    top30_mse_csv = os.path.join(args.out_dir, "top30_mse_full_inference.csv")
    top30_corr_csv = os.path.join(args.out_dir, "top30_corr_full_inference.csv")
    top30_overlap_csv = os.path.join(args.out_dir, "top30_overlap_segments_full_inference.csv")
    top30_mse_df.to_csv(top30_mse_csv, index=False, encoding="utf-8-sig")
    top30_corr_df.to_csv(top30_corr_csv, index=False, encoding="utf-8-sig")
    overlap_segments_df.to_csv(top30_overlap_csv, index=False, encoding="utf-8-sig")

    dist_df = (
        best_seg_df.groupby(["pair", "input_col", "target", "setting", "best_horizon"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    dist_df["total_segments"] = dist_df.groupby("pair")["count"].transform("sum")
    dist_df["ratio"] = dist_df["count"] / dist_df["total_segments"]
    dist_df = dist_df.sort_values(["pair", "best_horizon"]).reset_index(drop=True)

    dist_csv = os.path.join(args.out_dir, "lag_horizon_distribution.csv")
    dist_png = os.path.join(args.out_dir, "lag_horizon_distribution_boxplot.png")
    best_seg_csv = os.path.join(args.out_dir, "best_horizon_segment_metrics.csv")

    dist_df.to_csv(dist_csv, index=False, encoding="utf-8-sig")
    best_seg_df.to_csv(best_seg_csv, index=False, encoding="utf-8-sig")
    plot_best_horizon_mse_boxplot(best_seg_df, dist_png)

    print(f"Done. scanned_runs={used}, skipped_runs={skipped}")
    print(f"Full metrics CSV: {os.path.abspath(metrics_csv)}")
    print(f"Best segment CSV: {os.path.abspath(best_seg_csv)}")
    print(f"Output CSV: {os.path.abspath(dist_csv)}")
    print(f"Boxplot PNG: {os.path.abspath(dist_png)}")
    print(f"Top30 MSE CSV: {os.path.abspath(top30_mse_csv)}")
    print(f"Top30 Corr CSV: {os.path.abspath(top30_corr_csv)}")
    print(f"Top30 overlap CSV: {os.path.abspath(top30_overlap_csv)}")


if __name__ == "__main__":
    main()
