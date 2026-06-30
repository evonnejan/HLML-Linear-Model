"""Evaluate a trained model on dry segments (zero rainfall).

Pipeline (matches design spec docs/superpowers/specs/2026-06-01-dry-segment-eval-design.md):

1. Load trained model (run_args.json + checkpoints/checkpoint.pth from --run_dir)
2. Re-fit scalers from the training CSV (via Dataset_Custom, flag='train')
3. Read all_minute_wide.csv + rain_segments_meta.csv
4. Label each minute: in_rain_window, minutes_since_last_rain
5. Find continuous in_rain_window=False blocks >= --min_dry_minutes; assign dry_segment_id
6. Per-(dry_segment_id) ffill on gate cols (mirrors merge_gate_data.py:83-86 for rain segments)
7. Set isRain=0 (dry segments are outside any rain window by definition)
8. Slice sliding windows (stride=1); drop windows with NaN in input/exog/target
9. Group windows by `min_since_rain_at_anchor`: post_rain (0-180), mid_dry (180-360), pure_dry (>360 or NaN)
10. Batched forward → inverse scale → MSE/MAE/RMSE/Corr per (group, horizon, dry_segment)
11. Persistence baseline (y_pred[k] = input[-1]) for comparison
12. Visualisations: fanplot, per-horizon bar, per-segment box, time series overlay
13. Output to <output_dir>/

Outputs:
  <output_dir>/
    eval_dry.log               — full run log (this script's stdout)
    dry_metrics.csv            — one row per group with overall metrics
    per_horizon.csv            — per (group, horizon) metrics
    per_segment.csv            — per (group, dry_segment_id) metrics
    predictions.npz            — y_true, y_pred, meta arrays for further analysis
    figures/
      fanplot_<group>.png
      per_horizon_mse.png
      per_segment_mse_box.png
      time_series_<group>.png
      hist_min_since_rain.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Reuse project modules (script must be run from repo root)
from data_provider.Data_Loader import Dataset_Custom
from models import DLinear, DLinearMix, DLinearMix2, Linear, NLinear

_MODEL_DICT = {
    "DLinear": DLinear,
    "DLinearMix": DLinearMix,
    "NLinear": NLinear,
    "Linear": Linear,
    "DLinearMix2": DLinearMix2,
}

GATE_COLS = [
    "north_gate_opening_1",
    "north_gate_opening_2",
    "north_gate_opening_3",
    "north_gate_opening_4",
    "south_gate_opening_1",
    "south_gate_opening_2",
    "south_gate_opening_3",
]

GROUP_BOUNDS = {
    "post_rain": (0, 180),    # 0–3hr after last SegmentEnd: drainage dynamics regime
    "pure_dry": (180, float("inf")),  # >3hr (or NaN, set elsewhere): baseline / false-alarm regime
}


# --------------------------------------------------------------------------- #
#  CLI + logging                                                              #
# --------------------------------------------------------------------------- #

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str, required=True,
                   help="Path to a single run directory (contains run_args.json + checkpoints/checkpoint.pth)")
    p.add_argument("--all_minute_csv", type=str, default="dataset/all_minute_wide.csv")
    p.add_argument("--segments_meta_csv", type=str, default="dataset/rain_segments_meta.csv")
    p.add_argument("--min_dry_minutes", type=int, default=180,
                   help="Minimum length (minutes) for a candidate dry segment")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: <run_dir>/eval_dry)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default=None,
                   help="Override device (default: auto = mps/cuda/cpu)")
    p.add_argument("--max_fanplot_samples", type=int, default=6,
                   help="Number of sample windows to draw per group in the fanplot")
    return p.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("eval_dry")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def pick_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
#  Model + scaler loading                                                     #
# --------------------------------------------------------------------------- #

def load_run_args(run_dir: Path) -> SimpleNamespace:
    args_path = run_dir / "run_args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"run_args.json not found: {args_path}")
    with open(args_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SimpleNamespace(**data)


def load_model(args: SimpleNamespace, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    if args.model not in _MODEL_DICT:
        raise ValueError(f"Unknown model '{args.model}'. Supported: {list(_MODEL_DICT.keys())}")
    model = _MODEL_DICT[args.model].Model(args).float()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def fit_scalers_from_train(args: SimpleNamespace, log: logging.Logger):
    """Re-fit scaler_x / scaler_y from the original training CSV.

    Uses Dataset_Custom (flag='train') to guarantee identical preprocessing as
    training. Returns x_cols (input + exog order), bool_x_cols, scaler_x,
    scaler_y (target).
    """
    log.info(f"Fitting scalers from training CSV: {args.root_path}{args.data_path}")
    ds = Dataset_Custom(
        root_path=args.root_path,
        flag="train",
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        input_col=args.input_col,
        exog_col=getattr(args, "exog_col", None),
        segment_col=getattr(args, "segment_col", None),
        target=args.target,
        stride=getattr(args, "stride_train", 1),
        scale=True,
        timeenc=0,
        freq=getattr(args, "freq", "min"),
        train_only=False,
        model_name=args.model,
    )
    x_cols = list(ds.x_cols)
    bool_x_cols = list(ds.bool_x_cols)
    log.info(f"  x_cols ({len(x_cols)}): {x_cols}")
    log.info(f"  bool_x_cols ({len(bool_x_cols)}): {bool_x_cols}")
    log.info(f"  target: {args.target}")
    return ds.scaler_x, ds.scaler_y, x_cols, bool_x_cols


# --------------------------------------------------------------------------- #
#  Phase 2: rain labelling + dry block detection + gate ffill                 #
# --------------------------------------------------------------------------- #

def label_rain_windows(all_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """Add columns:
      - in_rain_window (bool): minute falls within ANY [WinStart, WinEnd]
      - minutes_since_last_rain (float): minutes since the most recent SegmentEnd
                                          (NaN if no prior rain at that minute)
    """
    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    segs = segments_df.copy()
    for c in ("SegmentStart", "SegmentEnd", "WinStart", "WinEnd"):
        segs[c] = pd.to_datetime(segs[c])
    segs = segs.sort_values("WinStart").reset_index(drop=True)

    # ---- in_rain_window via interval search using sorted WinStart / WinEnd ----
    win_starts = segs["WinStart"].to_numpy(dtype="datetime64[ns]")
    win_ends = segs["WinEnd"].to_numpy(dtype="datetime64[ns]")
    dates = df["date"].to_numpy(dtype="datetime64[ns]")

    # For each date, find rightmost WinStart <= date, then check if date <= corresponding WinEnd
    idx = np.searchsorted(win_starts, dates, side="right") - 1
    in_rain = np.zeros(len(dates), dtype=bool)
    valid = idx >= 0
    in_rain[valid] = dates[valid] <= win_ends[idx[valid]]
    df["in_rain_window"] = in_rain

    # ---- minutes_since_last_rain via sorted SegmentEnd ----
    seg_ends = np.sort(segs["SegmentEnd"].to_numpy(dtype="datetime64[ns]"))
    idx2 = np.searchsorted(seg_ends, dates, side="right") - 1
    mins = np.full(len(dates), np.nan, dtype=np.float64)
    valid2 = idx2 >= 0
    deltas_ns = (dates[valid2] - seg_ends[idx2[valid2]]).astype("timedelta64[s]").astype(np.int64)
    mins[valid2] = deltas_ns / 60.0
    df["minutes_since_last_rain"] = mins

    return df


def find_dry_blocks(df: pd.DataFrame, min_dry_minutes: int) -> pd.DataFrame:
    """Assign dry_segment_id to continuous in_rain_window=False blocks of length
    >= min_dry_minutes. Rows not in such a block get dry_segment_id = -1.
    """
    out = df.copy()
    not_rain = (~out["in_rain_window"]).to_numpy()
    if not not_rain.any():
        out["dry_segment_id"] = -1
        return out

    # Boundary indices of False->True or True->False transitions
    diffs = np.diff(not_rain.astype(np.int8), prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]  # exclusive

    seg_id = np.full(len(out), -1, dtype=np.int64)
    next_id = 0
    for s, e in zip(starts, ends):
        if (e - s) >= min_dry_minutes:
            seg_id[s:e] = next_id
            next_id += 1
    out["dry_segment_id"] = seg_id
    return out


def per_dry_segment_ffill(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """ffill `cols` within each dry_segment_id group (no cross-segment fill).

    Mirrors merge_gate_data.py:83-86 (training-time per-rain-segment ffill).
    Rows with dry_segment_id == -1 are left untouched.
    """
    out = df.copy()
    mask = out["dry_segment_id"] >= 0
    if not mask.any():
        return out
    out.loc[mask, cols] = (
        out.loc[mask].groupby("dry_segment_id", sort=False)[cols].transform(lambda c: c.ffill())
    )
    return out


# --------------------------------------------------------------------------- #
#  Phase 2.5: build sliding windows + scaling                                 #
# --------------------------------------------------------------------------- #

def build_windows(
    df: pd.DataFrame,
    args: SimpleNamespace,
    x_cols: list[str],
    bool_x_cols: list[str],
    scaler_x,
    scaler_y,
    log: logging.Logger,
) -> dict:
    """Slice valid sliding windows from dry blocks.

    Returns dict with:
      - x_scaled: (N, seq_len, len(x_cols)) float32
      - y_true:   (N, pred_len, 1) float32  (original scale)
      - persist:  (N, pred_len, 1) float32  (HL01 at last input minute, broadcast)
      - meta:     DataFrame with one row per window (anchor_time, dry_segment_id,
                                                      min_since_rain_at_anchor, group)

    Windows that contain any NaN in (x_cols + target) are dropped (matches
    training Data_Loader._drop_nan_windows behaviour).
    """
    seq_len = int(args.seq_len)
    pred_len = int(args.pred_len)
    target = args.target

    # Filter to dry rows only (dry_segment_id >= 0)
    dry = df.loc[df["dry_segment_id"] >= 0].copy().reset_index(drop=True)
    if dry.empty:
        raise ValueError("No dry segments found above min_dry_minutes threshold.")

    # Ensure isRain column exists (model trained with this exog; dry => always 0)
    if "isRain" in x_cols and "isRain" not in dry.columns:
        dry["isRain"] = False

    # NaN check: any NaN in x_cols or target invalidates that minute
    nan_check_cols = list(x_cols)
    if target not in nan_check_cols:
        nan_check_cols.append(target)
    bad_rows = dry[nan_check_cols].isna().any(axis=1).to_numpy()

    # Build x array (continuous scaled, bool passthrough as int8)
    cont_x_cols = [c for c in x_cols if c not in bool_x_cols]
    cont_idx = [x_cols.index(c) for c in cont_x_cols]

    x_df = dry[x_cols].copy()
    for c in bool_x_cols:
        x_df[c] = x_df[c].astype(np.int8)
    x_df = x_df.fillna(0)

    x_all = x_df.values.astype(np.float32, copy=True)
    if scaler_x is not None and len(cont_x_cols) > 0:
        x_all[:, cont_idx] = scaler_x.transform(x_df[cont_x_cols].values)

    y_all = dry[[target]].fillna(0).values.astype(np.float32, copy=True)
    seg_ids = dry["dry_segment_id"].to_numpy()
    anchor_min_since_rain = dry["minutes_since_last_rain"].to_numpy()
    dates = pd.to_datetime(dry["date"]).to_numpy()

    need = seq_len + pred_len
    n_total = len(dry)

    # For each dry_segment_id (continuous block), enumerate window starts
    valid_starts = []
    for seg in np.unique(seg_ids):
        block_idx = np.where(seg_ids == seg)[0]
        s0, s1 = block_idx[0], block_idx[-1] + 1
        L = s1 - s0
        if L < need:
            continue
        for s in range(s0, s0 + L - need + 1):
            valid_starts.append(s)
    valid_starts = np.asarray(valid_starts, dtype=np.int64)

    # Drop windows that contain a NaN row anywhere in [s, s+need)
    if bad_rows.any() and len(valid_starts) > 0:
        csum = np.concatenate([[0], np.cumsum(bad_rows.astype(np.int32))])
        window_bad = (csum[need:] - csum[:-need]) > 0
        keep = ~window_bad[valid_starts]
        n_before = len(valid_starts)
        valid_starts = valid_starts[keep]
        log.info(f"  Dropped {n_before - len(valid_starts)} / {n_before} windows due to NaN in {nan_check_cols}")
    else:
        log.info(f"  No windows dropped (bad_rows={bad_rows.sum()}/{len(bad_rows)})")

    if len(valid_starts) == 0:
        raise ValueError("All dry windows were dropped due to NaN; check gate ffill coverage.")

    # Build batched arrays
    x_scaled = np.zeros((len(valid_starts), seq_len, len(x_cols)), dtype=np.float32)
    y_true = np.zeros((len(valid_starts), pred_len, 1), dtype=np.float32)
    persist = np.zeros((len(valid_starts), pred_len, 1), dtype=np.float32)

    anchors = np.zeros(len(valid_starts), dtype="datetime64[ns]")
    anchor_segs = np.zeros(len(valid_starts), dtype=np.int64)
    anchor_msr = np.zeros(len(valid_starts), dtype=np.float64)

    for i, s in enumerate(valid_starts):
        s_end = s + seq_len
        r_end = s_end + pred_len
        x_scaled[i] = x_all[s:s_end]
        y_true[i, :, 0] = y_all[s_end:r_end, 0]
        persist[i, :, 0] = y_all[s_end - 1, 0]  # last input HL01, broadcast
        anchors[i] = dates[s_end]
        anchor_segs[i] = seg_ids[s_end]
        anchor_msr[i] = anchor_min_since_rain[s_end]

    # Group label per window
    groups = np.full(len(valid_starts), "pure_dry", dtype=object)
    msr = anchor_msr.copy()
    msr_nan = np.isnan(msr)
    for name, (lo, hi) in GROUP_BOUNDS.items():
        m = (~msr_nan) & (msr > lo) & (msr <= hi if np.isfinite(hi) else np.full_like(msr, True, dtype=bool))
        groups[m] = name
    # NaN min_since_rain (no prior rain ever) -> pure_dry
    groups[msr_nan] = "pure_dry"

    meta = pd.DataFrame({
        "anchor_time": anchors,
        "dry_segment_id": anchor_segs,
        "min_since_rain_at_anchor": anchor_msr,
        "group": groups,
    })

    log.info(f"  Built {len(valid_starts):,} windows.  group counts:")
    for g, n in meta["group"].value_counts().to_dict().items():
        log.info(f"    {g}: {n:,}")

    return {"x_scaled": x_scaled, "y_true": y_true, "persist": persist, "meta": meta}


# --------------------------------------------------------------------------- #
#  Phase 3: inference                                                          #
# --------------------------------------------------------------------------- #

def run_inference(
    model: torch.nn.Module,
    x_scaled: np.ndarray,
    batch_size: int,
    device: torch.device,
    log: logging.Logger,
) -> np.ndarray:
    """Forward x_scaled in batches; returns y_pred_scaled with shape (N, pred_len, 1)."""
    log.info(f"Running inference: {x_scaled.shape[0]:,} windows, batch_size={batch_size}")
    ds = TensorDataset(torch.from_numpy(x_scaled))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.float().to(device)
            yb = model(xb)  # [B, pred_len, 1] for DLinear-style
            outs.append(yb.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------- #
#  Phase 4: metrics + visualisation                                            #
# --------------------------------------------------------------------------- #

def _per_horizon_corr(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-horizon Pearson corr. p, t shape (N, H)."""
    H = p.shape[1]
    out = np.full(H, np.nan, dtype=float)
    for h in range(H):
        ph, th = p[:, h], t[:, h]
        if ph.std() > 0 and th.std() > 0:
            out[h] = float(np.corrcoef(ph, th)[0, 1])
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, persist: np.ndarray, meta: pd.DataFrame):
    """Compute overall, per-horizon, per-segment metrics. Returns dict of DataFrames."""
    pred_len = y_true.shape[1]
    overall_rows = []
    horizon_rows = []
    segment_rows = []

    groups = list(meta["group"].unique()) + ["all"]
    for g in groups:
        if g == "all":
            mask = np.ones(len(meta), dtype=bool)
        else:
            mask = (meta["group"] == g).to_numpy()
        if not mask.any():
            continue

        yt = y_true[mask, :, 0]
        yp = y_pred[mask, :, 0]
        yps = persist[mask, :, 0]
        diff = yp - yt
        diff_p = yps - yt

        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        rmse = float(np.sqrt(mse))
        corr_h = _per_horizon_corr(yp, yt)
        corr_mean = float(np.nanmean(corr_h)) if np.isfinite(corr_h).any() else float("nan")
        p_mse = float((diff_p ** 2).mean())
        p_mae = float(np.abs(diff_p).mean())
        improve_mse_pct = (p_mse - mse) / p_mse * 100.0 if p_mse > 0 else float("nan")
        improve_mae_pct = (p_mae - mae) / p_mae * 100.0 if p_mae > 0 else float("nan")

        overall_rows.append({
            "group": g,
            "n_windows": int(mask.sum()),
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Corr_mean": corr_mean,
            "persist_MSE": p_mse,
            "persist_MAE": p_mae,
            "improve_MSE_pct": improve_mse_pct,
            "improve_MAE_pct": improve_mae_pct,
        })

        # per-horizon
        for h in range(pred_len):
            h_mse = float((diff[:, h] ** 2).mean())
            h_mae = float(np.abs(diff[:, h]).mean())
            h_p_mse = float((diff_p[:, h] ** 2).mean())
            horizon_rows.append({
                "group": g,
                "horizon": h + 1,
                "MSE": h_mse,
                "RMSE": float(np.sqrt(h_mse)),
                "MAE": h_mae,
                "Corr": corr_h[h],
                "persist_MSE": h_p_mse,
                "improve_MSE_pct": (h_p_mse - h_mse) / h_p_mse * 100.0 if h_p_mse > 0 else float("nan"),
            })

        # per-segment (only for non-all groups)
        if g != "all":
            sub_meta = meta.loc[mask]
            mask_positions = np.where(mask)[0]
            for seg_id, sub in sub_meta.groupby("dry_segment_id"):
                # Map absolute meta indices in `sub` to row positions inside the masked yt/yp arrays
                local = np.searchsorted(mask_positions, sub.index.to_numpy())
                yt_s = yt[local]
                yp_s = yp[local]
                d = yp_s - yt_s
                s_mse = float((d ** 2).mean())
                s_mae = float(np.abs(d).mean())
                s_corr = _per_horizon_corr(yp_s, yt_s)
                s_corr_mean = float(np.nanmean(s_corr)) if np.isfinite(s_corr).any() else float("nan")
                segment_rows.append({
                    "group": g,
                    "dry_segment_id": int(seg_id),
                    "n_windows": int(len(sub)),
                    "MSE": s_mse,
                    "MAE": s_mae,
                    "RMSE": float(np.sqrt(s_mse)),
                    "Corr_mean": s_corr_mean,
                })

    return {
        "overall": pd.DataFrame(overall_rows),
        "per_horizon": pd.DataFrame(horizon_rows),
        "per_segment": pd.DataFrame(segment_rows),
    }


def make_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    metrics: dict,
    fig_dir: Path,
    args_cli,
    log: logging.Logger,
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred_len = y_true.shape[1]
    horizons = np.arange(1, pred_len + 1)

    # ---- 1. Per-horizon MSE bar chart (3 groups) ----
    plt.figure(figsize=(9, 4))
    width = 0.25
    group_list = [g for g in GROUP_BOUNDS.keys() if g in metrics["per_horizon"]["group"].unique()]
    for i, g in enumerate(group_list):
        sub = metrics["per_horizon"][metrics["per_horizon"]["group"] == g].sort_values("horizon")
        plt.bar(horizons + (i - 1) * width, sub["MSE"].values, width=width, label=g)
    plt.xlabel("Horizon (t+k min)")
    plt.ylabel("MSE")
    plt.title("Per-horizon MSE by group")
    plt.legend()
    plt.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "per_horizon_mse.png", dpi=180)
    plt.close()
    log.info(f"  saved {fig_dir / 'per_horizon_mse.png'}")

    # ---- 2. Per-segment MSE box plot (one box per group) ----
    plt.figure(figsize=(7, 4))
    data = []
    labels = []
    for g in group_list:
        sub = metrics["per_segment"][metrics["per_segment"]["group"] == g]
        if len(sub) > 0:
            data.append(sub["MSE"].values)
            labels.append(f"{g}\n(n={len(sub)} segs)")
    if data:
        plt.boxplot(data, labels=labels, showfliers=True)
        plt.ylabel("Per-segment MSE")
        plt.title("Per-segment MSE distribution")
        plt.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        plt.savefig(fig_dir / "per_segment_mse_box.png", dpi=180)
        plt.close()
        log.info(f"  saved {fig_dir / 'per_segment_mse_box.png'}")

    # ---- 3. Histogram of min_since_rain_at_anchor ----
    plt.figure(figsize=(8, 4))
    msr = meta["min_since_rain_at_anchor"].to_numpy()
    msr_finite = msr[np.isfinite(msr)]
    plt.hist(msr_finite, bins=60)
    plt.axvline(180, color="red", linestyle="--", label="post_rain | pure_dry")
    plt.xlabel("Minutes since last SegmentEnd (anchor time)")
    plt.ylabel("# windows")
    plt.title(f"Distribution of min_since_rain  ({(~np.isfinite(msr)).sum()} NaN = no prior rain)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "hist_min_since_rain.png", dpi=180)
    plt.close()
    log.info(f"  saved {fig_dir / 'hist_min_since_rain.png'}")

    # ---- 4. Fanplot: random sample windows per group ----
    rng = np.random.default_rng(0)
    for g in group_list:
        mask = (meta["group"] == g).to_numpy()
        if not mask.any():
            continue
        n_samples = min(args_cli.max_fanplot_samples, mask.sum())
        chosen = rng.choice(np.where(mask)[0], size=n_samples, replace=False)
        cols = 2
        rows = (n_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(11, 2.5 * rows), squeeze=False)
        for i, idx in enumerate(chosen):
            ax = axes[i // cols][i % cols]
            t = np.arange(1, pred_len + 1)
            ax.plot(t, y_true[idx, :, 0], "k-", marker="o", label="y_true", markersize=3)
            ax.plot(t, y_pred[idx, :, 0], "C0-", marker="x", label="y_pred", markersize=3)
            ax.set_title(f"anchor={pd.Timestamp(meta['anchor_time'].iloc[idx]).strftime('%Y-%m-%d %H:%M')}  "
                         f"seg={meta['dry_segment_id'].iloc[idx]}  "
                         f"min_since_rain={meta['min_since_rain_at_anchor'].iloc[idx]:.0f}",
                         fontsize=8)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(fontsize=8)
        for j in range(len(chosen), rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle(f"Sample window fanplot — group={g}", y=1.0)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fanplot_{g}.png", dpi=180)
        plt.close(fig)
        log.info(f"  saved {fig_dir / f'fanplot_{g}.png'}")

    # ---- 5. Time series overlay: longest segment per group ----
    for g in group_list:
        mask = (meta["group"] == g).to_numpy()
        if not mask.any():
            continue
        seg_counts = meta.loc[mask].groupby("dry_segment_id").size().sort_values(ascending=False)
        if len(seg_counts) == 0:
            continue
        best_seg = seg_counts.index[0]
        seg_mask = mask & (meta["dry_segment_id"].to_numpy() == best_seg)
        if seg_mask.sum() < 2:
            continue
        anchors = pd.to_datetime(meta.loc[seg_mask, "anchor_time"].to_numpy())
        # Plot t+1 horizon true vs pred over time as a representative curve
        plt.figure(figsize=(12, 4))
        plt.plot(anchors, y_true[seg_mask, 0, 0], "k-", label="y_true (t+1)", alpha=0.8)
        plt.plot(anchors, y_pred[seg_mask, 0, 0], "C0-", label="y_pred (t+1)", alpha=0.8)
        plt.xlabel("Anchor time")
        plt.ylabel("HL01")
        plt.title(f"Time-series overlay — group={g}, dry_segment_id={best_seg}  (longest segment, n={seg_mask.sum()})")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir / f"time_series_{g}.png", dpi=180)
        plt.close()
        log.info(f"  saved {fig_dir / f'time_series_{g}.png'}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def _evaluate_one_checkpoint(
    label: str,
    ckpt_path: Path,
    out_dir: Path,
    args: SimpleNamespace,
    device: torch.device,
    bundle: dict,
    scaler_y,
    cli,
    log: logging.Logger,
):
    """Run inference + metrics + figures for a single checkpoint into out_dir.

    Inputs (bundle) are reused across checkpoints — only the model and its
    predictions change.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 72)
    log.info(f"  Evaluating checkpoint: {label}  ({ckpt_path.name})")
    log.info(f"  Output dir: {out_dir}")
    log.info("=" * 72)

    model = load_model(args, ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {n_params:,}")

    y_pred_scaled = run_inference(model, bundle["x_scaled"], cli.batch_size, device, log)

    if scaler_y is not None:
        shape = y_pred_scaled.shape
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(shape)
    else:
        y_pred = y_pred_scaled

    log.info("Computing metrics ...")
    metrics = compute_metrics(bundle["y_true"], y_pred, bundle["persist"], bundle["meta"])

    log.info(f"Overall metrics ({label}):")
    for _, r in metrics["overall"].iterrows():
        log.info(f"  {r['group']:<10} n={int(r['n_windows']):>7,}  "
                 f"MSE={r['MSE']:.4f}  RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  "
                 f"Corr={r['Corr_mean']:+.4f}  improve_MSE={r['improve_MSE_pct']:+.2f}%")

    metrics["overall"].to_csv(out_dir / "dry_metrics.csv", index=False, encoding="utf-8-sig")
    metrics["per_horizon"].to_csv(out_dir / "per_horizon.csv", index=False, encoding="utf-8-sig")
    metrics["per_segment"].to_csv(out_dir / "per_segment.csv", index=False, encoding="utf-8-sig")
    meta = bundle["meta"]
    np.savez(
        out_dir / "predictions.npz",
        y_true=bundle["y_true"],
        y_pred=y_pred,
        persist=bundle["persist"],
        anchor_time=meta["anchor_time"].to_numpy().astype("datetime64[ns]"),
        dry_segment_id=meta["dry_segment_id"].to_numpy(),
        min_since_rain_at_anchor=meta["min_since_rain_at_anchor"].to_numpy(),
        group=meta["group"].to_numpy(),
    )
    log.info(f"Saved dry_metrics.csv / per_horizon.csv / per_segment.csv / predictions.npz")

    log.info("Drawing figures ...")
    make_figures(bundle["y_true"], y_pred, meta, metrics, out_dir / "figures", cli, log)


def main():
    cli = parse_cli()
    run_dir = Path(cli.run_dir).resolve()
    base_out = Path(cli.output_dir).resolve() if cli.output_dir else (run_dir / "eval_dry")
    base_out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(base_out / "eval_dry.log")

    log.info("=" * 72)
    log.info("  eval_dry.py — dry-segment inference evaluation")
    log.info("=" * 72)

    # ---- Load args + detect available checkpoints ----
    args = load_run_args(run_dir)
    primary_metric = str(getattr(args, "early_stop_metric", "mse")).lower()
    alt_metric = "corr" if primary_metric in ("mse", "mae") else "mse"

    ckpt_dir = run_dir / "checkpoints"
    primary_ckpt = ckpt_dir / "checkpoint.pth"
    alt_ckpt = ckpt_dir / "checkpoint_alt.pth"

    if not primary_ckpt.exists():
        raise FileNotFoundError(f"checkpoint.pth not found: {primary_ckpt}")

    # Always write into subdirs (best_<metric>/) for consistency, mirroring exp_Main2.test()
    runs_to_eval: list[tuple[str, Path, Path]] = [
        (f"best_{primary_metric}", primary_ckpt, base_out / f"best_{primary_metric}"),
    ]
    if alt_ckpt.exists():
        runs_to_eval.append(
            (f"best_{alt_metric}", alt_ckpt, base_out / f"best_{alt_metric}")
        )

    log.info(f"Model:    {args.model}")
    log.info(f"Setting:  {getattr(args, 'setting', '(unknown)')}")
    log.info(f"Run dir:  {run_dir}")
    log.info(f"Primary metric (controls early stopping during training): {primary_metric}")
    log.info(f"Detected checkpoints to evaluate:")
    for label, ckpt_path, out_dir in runs_to_eval:
        log.info(f"  - {label:<14}  →  {out_dir}   (from {ckpt_path.name})")

    device = pick_device(cli.device)
    log.info(f"Device:   {device}")

    # ---- Fit scalers from training CSV (shared) ----
    scaler_x, scaler_y, x_cols, bool_x_cols = fit_scalers_from_train(args, log)

    # ---- Phase 2: load + label + ffill + slice (shared) ----
    log.info("Loading dry evaluation data ...")
    all_df = pd.read_csv(cli.all_minute_csv, parse_dates=["date"])
    segs_df = pd.read_csv(cli.segments_meta_csv,
                          parse_dates=["SegmentStart", "SegmentEnd", "WinStart", "WinEnd"])
    log.info(f"  all_minute_wide: {len(all_df):,} rows, {all_df['date'].min()} ~ {all_df['date'].max()}")
    log.info(f"  rain_segments_meta: {len(segs_df):,} segments")

    log.info("Labelling rain windows + computing minutes_since_last_rain ...")
    all_df = label_rain_windows(all_df, segs_df)
    log.info(f"  in_rain_window=True : {all_df['in_rain_window'].sum():,} / {len(all_df):,}")
    log.info(f"  in_rain_window=False: {(~all_df['in_rain_window']).sum():,} / {len(all_df):,}")

    log.info(f"Finding dry blocks (min length = {cli.min_dry_minutes} min) ...")
    all_df = find_dry_blocks(all_df, cli.min_dry_minutes)
    n_dry_segs = int(all_df["dry_segment_id"].max() + 1) if (all_df["dry_segment_id"] >= 0).any() else 0
    log.info(f"  dry_segment count: {n_dry_segs}")
    log.info(f"  dry rows (in any segment): {(all_df['dry_segment_id'] >= 0).sum():,}")

    log.info("Per-dry-segment gate ffill ...")
    all_df = per_dry_segment_ffill(all_df, GATE_COLS)
    dry_mask = all_df["dry_segment_id"] >= 0
    gate_nan_pct = (all_df.loc[dry_mask, GATE_COLS].isna().sum() / dry_mask.sum() * 100).round(2)
    log.info(f"  gate NaN% in dry rows after ffill:")
    for c, v in gate_nan_pct.items():
        log.info(f"    {c:<28} {v}%")

    log.info("Building sliding windows ...")
    bundle = build_windows(all_df, args, x_cols, bool_x_cols, scaler_x, scaler_y, log)

    # ---- Per-checkpoint loop ----
    for label, ckpt_path, out_dir in runs_to_eval:
        _evaluate_one_checkpoint(
            label=label,
            ckpt_path=ckpt_path,
            out_dir=out_dir,
            args=args,
            device=device,
            bundle=bundle,
            scaler_y=scaler_y,
            cli=cli,
            log=log,
        )

    log.info("=" * 72)
    log.info("All checkpoints evaluated.")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
