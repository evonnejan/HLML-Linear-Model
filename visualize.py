import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_interactive_backend():
    backend = str(plt.get_backend()).lower()
    if "agg" not in backend:
        return

    for candidate in ["MacOSX", "TkAgg", "QtAgg"]:
        try:
            plt.switch_backend(candidate)
            return
        except Exception:
            continue


def _is_run_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "run_args.json"))


def _iter_run_dirs(output_root: str):
    """Yield every directory under output_root that looks like a run dir.

    Supports both legacy layout (`runs/<setting>/`) and new layout that adds a
    model-name subfolder (`runs/<MODEL>/<setting>/`).
    """
    if not os.path.isdir(output_root):
        return
    for name in os.listdir(output_root):
        first = os.path.join(output_root, name)
        if not os.path.isdir(first):
            continue
        if _is_run_dir(first):
            yield first
            continue
        for sub in os.listdir(first):
            second = os.path.join(first, sub)
            if _is_run_dir(second):
                yield second


def get_latest_run(output_root: str) -> str:
    if not os.path.isdir(output_root):
        raise FileNotFoundError(f"output_root not found: {output_root}")
    dirs = list(_iter_run_dirs(output_root))
    if not dirs:
        raise FileNotFoundError(f"No run folders under: {output_root}")
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs[0]


def describe_run(run_dir: str) -> str:
    args_path = os.path.join(run_dir, "run_args.json")
    if not os.path.exists(args_path):
        return f"(no run_args.json) {run_dir}"
    try:
        import json
        with open(args_path, "r", encoding="utf-8") as f:
            args = json.load(f)
    except Exception:
        return run_dir
    model = args.get("model", "?")
    input_col = args.get("input_col") or args.get("target") or "?"
    target = args.get("target", "?")
    seq_len = args.get("seq_len", "?")
    pred_len = args.get("pred_len", "?")
    return f"model={model} | input={input_col} -> target={target} | seq_len={seq_len} pred_len={pred_len}"


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def read_outputs(run_dir: str):
    outputs_dir = os.path.join(run_dir, "outputs")
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"outputs folder not found: {outputs_dir}")

    result = {
        "outputs_dir": outputs_dir,
        # Newer runs write metrics_*.csv; older ones write mse_horizon.csv /
        # mse_segment_combined.csv. Resolve whichever exists.
        "metrics_horizon": _first_existing(
            os.path.join(outputs_dir, "metrics_horizon.csv"),
            os.path.join(outputs_dir, "mse_horizon.csv"),
        ),
        "metrics_segment": _first_existing(
            os.path.join(outputs_dir, "metrics_segment.csv"),
            os.path.join(outputs_dir, "mse_segment_combined.csv"),
        ),
        "points": os.path.join(outputs_dir, "segment_horizon_points.csv.gz"),
        "points_full": os.path.join(outputs_dir, "segment_horizon_points_full.csv.gz"),
        "rank": os.path.join(outputs_dir, "segment_horizon_rank.csv"),
        "meeting": os.path.join(outputs_dir, "meeting.csv"),
    }
    return result


def _points_has_segment_horizon(points_csv: str, segment, horizon: int) -> bool:
    if not os.path.exists(points_csv):
        return False
    try:
        df = pd.read_csv(points_csv, compression="gzip")
    except Exception:
        return False
    sub = df[(df["segment"].astype(str) == str(segment)) & (df["horizon"].astype(int) == int(horizon))]
    return len(sub) > 0


def resolve_points_file(run_dir: str, files: dict, segment, horizon: int, points_source: str = "full") -> str:
    test_points = files["points"]
    full_points = files["points_full"]

    candidates = []
    source = str(points_source).lower()
    if source == "full":
        candidates = [full_points]
    elif source == "test":
        candidates = [test_points]
    else:
        candidates = [full_points, test_points]

    for candidate in candidates:
        if _points_has_segment_horizon(candidate, segment, horizon):
            return candidate

    raise ValueError(
        f"No data for segment={segment}, horizon={horizon} in selected points source ({points_source}). "
        f"Please generate full points first via analyze_full_inference.py"
    )


def plot_horizon_mse(csv_path: str, out_path: str = None):
    df = pd.read_csv(csv_path)
    if out_path is None:
        _ensure_interactive_backend()
    plt.figure(figsize=(8, 4))
    plt.plot(df["horizon"], df["MSE"], marker="o")
    plt.xlabel("Horizon (t+k)")
    plt.ylabel("MSE")
    plt.title("Horizon-wise MSE")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
    else:
        plt.show(block=True)
    plt.close()


def _parse_horizons(horizon_spec):
    """Accept '1', '15', '1,15', or int → return list[int]."""
    if isinstance(horizon_spec, int):
        return [horizon_spec]
    if isinstance(horizon_spec, str):
        return [int(x.strip()) for x in horizon_spec.split(",") if x.strip()]
    return [int(x) for x in horizon_spec]


def _read_segment_bounds(data_path: str, segment):
    """Lookup SegmentStart / SegmentEnd from the raw dataset CSV (if available)."""
    if not data_path or not os.path.exists(data_path):
        return None, None
    try:
        ds = pd.read_csv(data_path, usecols=["segment_id", "SegmentStart", "SegmentEnd"])
    except Exception:
        return None, None
    rows = ds[ds["segment_id"].astype(str) == str(segment)]
    if len(rows) == 0:
        return None, None
    return pd.to_datetime(rows["SegmentStart"].iloc[0]), pd.to_datetime(rows["SegmentEnd"].iloc[0])


def plot_segment_curve(points_csv: str, segment, horizon, out_path: str = None,
                       data_path: str = None):
    """Plot ground truth + per-horizon predictions for one segment.

    `horizon` accepts a single int or a comma-separated string of horizons
    (e.g. "1,15"). When multiple horizons are given, ground truth is plotted
    once and each horizon gets its own prediction line.

    If `data_path` is provided (and the raw dataset has SegmentStart /
    SegmentEnd columns), vertical dashed lines mark the rain on/off boundaries.
    """
    horizons = _parse_horizons(horizon)
    df = pd.read_csv(points_csv, compression="gzip")
    df_seg = df[df["segment"].astype(str) == str(segment)].copy()
    if df_seg.empty:
        raise ValueError(f"No data for segment={segment}")
    df_seg["target_time"] = pd.to_datetime(df_seg["target_time"])

    # Sanity check that all requested horizons exist for this segment.
    available_h = set(df_seg["horizon"].astype(int).unique())
    missing = [h for h in horizons if h not in available_h]
    if missing:
        raise ValueError(f"No data for segment={segment}, horizon(s)={missing}")

    if out_path is None:
        _ensure_interactive_backend()

    fig, ax = plt.subplots(figsize=(14, 5))

    # Ground truth: combine across requested horizons (true is the same per target_time;
    # different horizons cover slightly different target_time ranges, so union gives the
    # widest possible truth coverage).
    truth_df = (df_seg[df_seg["horizon"].astype(int).isin(horizons)]
                [["target_time", "true"]]
                .drop_duplicates(subset=["target_time"])
                .sort_values("target_time"))
    ax.plot(truth_df["target_time"], truth_df["true"],
            label="GroundTruth", linewidth=2.0, color="black")

    # Predictions, one line per horizon.
    palette = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple"]
    title_lines = [f"segment={segment}  |  horizons={horizons}"]
    for i, h in enumerate(horizons):
        sub = df_seg[df_seg["horizon"].astype(int) == h].sort_values("target_time")
        mse = float(np.mean((sub["pred"].to_numpy() - sub["true"].to_numpy()) ** 2))
        if len(sub) >= 2:
            corr = float(np.corrcoef(sub["pred"].to_numpy(), sub["true"].to_numpy())[0, 1])
        else:
            corr = float("nan")
        ax.plot(sub["target_time"], sub["pred"],
                label=f"Pred h={h}  (mse={mse:.1f}, corr={corr:.3f})",
                linewidth=1.5, color=palette[i % len(palette)], alpha=0.85)

    # Vertical lines: SegmentStart (rain on) / SegmentEnd (rain off).
    seg_start, seg_end = _read_segment_bounds(data_path, segment)
    if seg_start is not None:
        ax.axvline(seg_start, color="green", linestyle="--", linewidth=1.2,
                   alpha=0.7, label="SegmentStart (rain on)")
    if seg_end is not None:
        ax.axvline(seg_end, color="red", linestyle="--", linewidth=1.2,
                   alpha=0.7, label="SegmentEnd (rain off)")

    ax.set_title(" | ".join(title_lines))
    ax.set_xlabel("Target Time")
    ax.set_ylabel("HL01 (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
    else:
        plt.show(block=True)
    plt.close(fig)


def print_topk(rank_csv: str, k: int = 10):
    df = pd.read_csv(rank_csv)
    print("\n=== Top by lowest MSE ===")
    print(df.sort_values("mse", ascending=True).head(k).to_string(index=False))

    if "corr" in df.columns:
        df_corr = df.dropna(subset=["corr"]).sort_values("corr", ascending=False)
        print("\n=== Top by highest CORR ===")
        print(df_corr.head(k).to_string(index=False))


def print_meeting(meeting_csv: str):
    df = pd.read_csv(meeting_csv)
    for category in ["best_mse", "worst_mse", "best_corr", "worst_corr"]:
        sub = df[df["category"] == category]
        if len(sub) == 0:
            continue
        print(f"\n=== {category} ===")
        print(sub.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Visualization utility for runs/<setting>/outputs")
    parser.add_argument("--output_root", type=str, default="./runs")
    parser.add_argument("--run_dir", type=str, default=None, help="Specific run folder; default uses latest under output_root")
    parser.add_argument("--mode", type=str, default="topk", choices=["topk", "meeting", "horizon", "segment"])
    parser.add_argument("--segment", type=str, default=None)
    parser.add_argument("--horizon", type=str, default=None,
                        help='Horizon(s) to plot for --mode segment. Single int (e.g. "15") or comma-separated list (e.g. "1,15").')
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--points_source", type=str, default="full", choices=["full", "test", "auto"])
    parser.add_argument("--data_path", type=str, default="dataset/water_level_rain_gate_all.csv",
                        help="Raw dataset CSV with SegmentStart/SegmentEnd columns; used to draw rain-on/off vertical lines.")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else get_latest_run(args.output_root)
    files = read_outputs(run_dir)

    print(f"Using run_dir: {run_dir}")
    print(f"  -> {describe_run(run_dir)}")

    if args.mode == "topk":
        if not os.path.exists(files["rank"]):
            raise FileNotFoundError(f"rank file not found: {files['rank']}")
        print_topk(files["rank"], k=args.k)
        return

    if args.mode == "meeting":
        if not os.path.exists(files["meeting"]):
            raise FileNotFoundError(f"meeting file not found: {files['meeting']}")
        print_meeting(files["meeting"])
        return

    if args.mode == "horizon":
        if not os.path.exists(files["metrics_horizon"]):
            raise FileNotFoundError(f"horizon csv not found: {files['metrics_horizon']}")
        plot_horizon_mse(files["metrics_horizon"], out_path=args.save)
        return

    if args.mode == "segment":
        if args.segment is None or args.horizon is None:
            raise ValueError("--mode segment requires --segment and --horizon")
        horizons = _parse_horizons(args.horizon)
        # resolve_points_file checks existence per (segment, horizon); pick first horizon
        # as the probe — points file contains ALL horizons together.
        points_file = resolve_points_file(
            run_dir,
            files,
            segment=args.segment,
            horizon=horizons[0],
            points_source=args.points_source,
        )
        print(f"Using points source: {points_file}")
        plot_segment_curve(points_file, segment=args.segment, horizon=args.horizon,
                           out_path=args.save, data_path=args.data_path)
        return


if __name__ == "__main__":
    main()
