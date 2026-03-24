import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        args = json.load(file)

    setting = str(args.get("setting", os.path.basename(run_dir)))
    model = str(args.get("model", ""))
    input_col = str(args.get("input_col") or args.get("target") or "")
    target = str(args.get("target", ""))
    seq_len = _safe_int(args.get("seq_len"), -1)
    pred_len = _safe_int(args.get("pred_len"), -1)
    kernel_size = _safe_int(args.get("dlinear_kernel_size"), None)

    return RunInfo(
        run_dir=run_dir,
        setting=setting,
        model=model,
        input_col=input_col,
        target=target,
        seq_len=seq_len,
        pred_len=pred_len,
        kernel_size=kernel_size,
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


def pair_name(input_col: str, target: str) -> str:
    return f"{input_col}->{target}"


def plot_horizon_overlay(df_h: pd.DataFrame, out_path: str):
    if len(df_h) == 0:
        return

    plt.figure(figsize=(9.5, 5.2))
    for input_col, group in df_h.groupby("input_col"):
        group = group.sort_values("horizon")
        plt.plot(group["horizon"], group["mse"], marker="o", linewidth=2, label=str(input_col))

    plt.xlabel("Horizon (t+k)")
    plt.ylabel("MSE")
    plt.title("Horizon-wise MSE Overlay (Best Model Per Input Point)")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Input", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_horizon_relative_table(df_h: pd.DataFrame) -> pd.DataFrame:
    if len(df_h) == 0:
        return df_h.copy()

    rows = []
    for input_col, group in df_h.groupby("input_col"):
        group = group.sort_values("horizon").copy()
        base_row = group[group["horizon"].astype(int) == 1]
        if len(base_row) == 0:
            base_mse = float(group.iloc[0]["mse"])
        else:
            base_mse = float(base_row.iloc[0]["mse"])

        safe_base = base_mse if abs(base_mse) > 1e-12 else 1e-12
        group["mse_ratio_to_h1"] = group["mse"].astype(float) / safe_base
        group["mse_pct_change_to_h1"] = (group["mse_ratio_to_h1"] - 1.0) * 100.0
        rows.append(group)

    return pd.concat(rows, axis=0, ignore_index=True)


def plot_horizon_overlay_relative(df_h_rel: pd.DataFrame, out_path: str):
    if len(df_h_rel) == 0:
        return

    plt.figure(figsize=(9.5, 5.2))
    for input_col, group in df_h_rel.groupby("input_col"):
        group = group.sort_values("horizon")
        plt.plot(group["horizon"], group["mse_ratio_to_h1"], marker="o", linewidth=2, label=str(input_col))

    plt.xlabel("Horizon (t+k)")
    plt.ylabel("Relative MSE (MSE / MSE@t+1)")
    plt.title("Horizon-wise Relative MSE Overlay (Best Model Per Input Point)")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Input", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_segment_fan(points_df: pd.DataFrame, segment_value, title: str, out_path: str):
    part = points_df[points_df["segment"].astype(str) == str(segment_value)].copy()
    if len(part) == 0:
        return

    part["target_time"] = pd.to_datetime(part["target_time"])

    true_df = (
        part.groupby("target_time", as_index=False)["true"]
        .mean()
        .sort_values("target_time")
    )

    pred_df = (
        part.groupby(["target_time", "horizon"], as_index=False)["pred"]
        .mean()
        .sort_values(["horizon", "target_time"])
    )

    show_horizons = [1, 5, 10, 15]
    pred_df = pred_df[pred_df["horizon"].astype(int).isin(show_horizons)].copy()

    plt.figure(figsize=(12, 4.8))
    plt.plot(true_df["target_time"], true_df["true"], color="black", linewidth=2.2, label="GT")

    horizons = sorted(pred_df["horizon"].astype(int).unique().tolist())
    cmap = plt.get_cmap("viridis", max(len(horizons), 2))
    for idx, horizon in enumerate(horizons):
        sub = pred_df[pred_df["horizon"].astype(int) == int(horizon)]
        label = f"t+{horizon}"
        plt.plot(
            sub["target_time"],
            sub["pred"],
            color=cmap(idx),
            linewidth=1.5,
            alpha=0.85,
            label=label,
        )

    plt.title(f"{title} | shown: t+1, t+5, t+10, t+15")
    plt.xlabel("Target Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Overview analysis using best model per point")
    parser.add_argument("--runs_root", type=str, default="./runs")
    parser.add_argument("--out_dir", type=str, default="./analysis")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--input_cols", type=str, default=None)
    parser.add_argument("--targets", type=str, default=None)
    parser.add_argument("--seq_lens", type=str, default=None)
    parser.add_argument("--kernels", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3, help="top/bottom K segments for fan plots")
    parser.add_argument("--max_runs", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    models = _parse_csv_list(args.models, str)
    input_cols = _parse_csv_list(args.input_cols, str)
    targets = _parse_csv_list(args.targets, str)
    seq_lens = _parse_csv_list(args.seq_lens, int)
    kernels = _parse_csv_list(args.kernels, int)

    ensure_dir(args.out_dir)
    fanplot_dir = os.path.join(args.out_dir, "segment_fanplots")
    ensure_dir(fanplot_dir)

    run_dirs = discover_runs(args.runs_root)

    rows = []
    horizon_tables: List[pd.DataFrame] = []
    segment_tables: Dict[str, pd.DataFrame] = {}
    points_tables: Dict[str, pd.DataFrame] = {}

    used = 0
    skipped = 0

    for run_dir in run_dirs:
        info = load_run_info(run_dir)
        if info is None:
            skipped += 1
            continue

        if not pass_filters(info, models, input_cols, targets, seq_lens, kernels):
            continue

        outputs_dir = os.path.join(run_dir, "outputs")
        metrics_path = os.path.join(outputs_dir, "metrics.npy")
        horizon_path = os.path.join(outputs_dir, "mse_horizon.csv")
        segment_path = os.path.join(outputs_dir, "mse_segment_combined.csv")
        points_path = os.path.join(outputs_dir, "segment_horizon_points.csv.gz")

        if not (os.path.exists(metrics_path) and os.path.exists(horizon_path) and os.path.exists(segment_path) and os.path.exists(points_path)):
            skipped += 1
            continue

        metrics_arr = np.load(metrics_path)
        mae = float(metrics_arr[0]) if len(metrics_arr) > 0 else np.nan
        mse = float(metrics_arr[1]) if len(metrics_arr) > 1 else np.nan

        pair = pair_name(info.input_col, info.target)
        rows.append(
            {
                "pair": pair,
                "setting": info.setting,
                "model": info.model,
                "seq_len": info.seq_len,
                "kernel_size": info.kernel_size,
                "mse_test": mse,
                "mae_test": mae,
                "run_dir": info.run_dir,
                "input_col": info.input_col,
                "target": info.target,
            }
        )

        df_h = pd.read_csv(horizon_path)
        df_h["pair"] = pair
        df_h["setting"] = info.setting
        df_h["input_col"] = info.input_col
        df_h["target"] = info.target
        horizon_tables.append(df_h)

        df_s = pd.read_csv(segment_path)
        segment_tables[info.setting] = df_s
        points_tables[info.setting] = pd.read_csv(points_path, compression="gzip")

        used += 1
        if args.max_runs > 0 and used >= args.max_runs:
            break

    if len(rows) == 0:
        raise RuntimeError("No valid runs found for analysis.")

    all_df = pd.DataFrame(rows)

    best_rows = []
    for pair, group in all_df.groupby("pair"):
        best_rows.append(group.sort_values("mse_test", ascending=True).iloc[0])
    best_df = pd.DataFrame(best_rows).reset_index(drop=True)

    overview_df = best_df[["pair", "setting", "model", "seq_len", "kernel_size", "mse_test", "mae_test"]].copy()
    overview_df.to_csv(os.path.join(args.out_dir, "run_overview.csv"), index=False, encoding="utf-8-sig")

    best_settings = set(best_df["setting"].tolist())
    horizon_concat = pd.concat(horizon_tables, axis=0, ignore_index=True)
    horizon_best = horizon_concat[horizon_concat["setting"].isin(best_settings)].copy()
    horizon_best = horizon_best[["pair", "input_col", "target", "setting", "horizon", "mse"]].sort_values(["input_col", "horizon"])

    horizon_best.to_csv(os.path.join(args.out_dir, "horizon_mse_overlay.csv"), index=False, encoding="utf-8-sig")
    plot_horizon_overlay(horizon_best, os.path.join(args.out_dir, "horizon_mse_overlay.png"))

    horizon_rel = build_horizon_relative_table(horizon_best)
    horizon_rel.to_csv(os.path.join(args.out_dir, "horizon_mse_overlay_relative.csv"), index=False, encoding="utf-8-sig")
    plot_horizon_overlay_relative(horizon_rel, os.path.join(args.out_dir, "horizon_mse_overlay_relative.png"))

    fanplot_records = []

    for row in best_df.itertuples(index=False):
        setting = row.setting
        pair = row.pair
        safe_pair = str(pair).replace("->", "_to_")

        df_seg = segment_tables.get(setting)
        points_df = points_tables.get(setting)
        if df_seg is None or points_df is None or len(df_seg) == 0 or len(points_df) == 0:
            continue

        seg_overall = df_seg[df_seg["horizon"].astype(str) == "all"].copy()
        if len(seg_overall) == 0:
            continue

        topk = seg_overall.sort_values("mse", ascending=True).head(args.topk)
        bottomk = seg_overall.sort_values("mse", ascending=False).head(args.topk)

        for rank_idx, item in enumerate(topk.itertuples(index=False), start=1):
            seg = item.segment
            mse_val = float(item.mse)
            out_name = f"{safe_pair}_best_{rank_idx:02d}_seg-{seg}.png"
            title = f"{pair} | BEST #{rank_idx} segment={seg} overall_mse={mse_val:.4f}"
            plot_segment_fan(points_df, seg, title, os.path.join(fanplot_dir, out_name))
            fanplot_records.append(
                {
                    "pair": pair,
                    "setting": setting,
                    "segment": seg,
                    "category": "best",
                    "rank": rank_idx,
                    "overall_mse": mse_val,
                    "plot_file": out_name,
                }
            )

        for rank_idx, item in enumerate(bottomk.itertuples(index=False), start=1):
            seg = item.segment
            mse_val = float(item.mse)
            out_name = f"{safe_pair}_worst_{rank_idx:02d}_seg-{seg}.png"
            title = f"{pair} | WORST #{rank_idx} segment={seg} overall_mse={mse_val:.4f}"
            plot_segment_fan(points_df, seg, title, os.path.join(fanplot_dir, out_name))
            fanplot_records.append(
                {
                    "pair": pair,
                    "setting": setting,
                    "segment": seg,
                    "category": "worst",
                    "rank": rank_idx,
                    "overall_mse": mse_val,
                    "plot_file": out_name,
                }
            )

    if len(fanplot_records) > 0:
        fanplot_records_df = pd.DataFrame(fanplot_records)
        fanplot_records_df.to_csv(
            os.path.join(args.out_dir, "segment_fanplots", "segment_fanplot_draw_records.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        fanplot_counts = (
            fanplot_records_df.assign(
                best_count=lambda d: (d["category"] == "best").astype(int),
                worst_count=lambda d: (d["category"] == "worst").astype(int),
            )
            .groupby(["segment"], as_index=False)[["best_count", "worst_count"]]
            .sum()
        )
        fanplot_counts["total_count"] = fanplot_counts["best_count"] + fanplot_counts["worst_count"]
        fanplot_counts = fanplot_counts.sort_values(["total_count", "segment"], ascending=[False, True])
        fanplot_counts.to_csv(
            os.path.join(args.out_dir, "segment_fanplots", "segment_fanplot_counts.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print(f"Done. analyzed_runs={used}, skipped_runs={skipped}")
    print(f"Output dir: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
