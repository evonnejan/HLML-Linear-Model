"""投影片專用單格圖：錨定校正 (GT / raw / adj) + rain on/off，大字級乾淨版。

預設 segment 144、horizon t+15。標題帶全測試集的爆點數字。
用法:
    python slide_anchored_figure.py <run_dir> [--segment 144] [--horizon 15] [--outputs outputs]
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from compute_anchored_mse import load_run_args
from data_provider.Data_Factory import data_provider
from utils.metrics import CORR
from visualize_segment import load_segment_bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--segment", type=int, default=144)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--outputs", default="outputs")
    a = ap.parse_args()

    out_dir = os.path.join(a.run_dir, a.outputs)
    preds = np.load(os.path.join(out_dir, "pred.npy"))
    trues = np.load(os.path.join(out_dir, "true.npy"))

    args = load_run_args(a.run_dir)
    ds, _ = data_provider(args, "test")
    f_dim = -1 if args.features == "MS" else 0
    seq_len = int(args.seq_len)
    bounds = load_segment_bounds(args)

    # ---- 全測試集爆點數字 (raw vs adj vs persistence) ----
    x_last_all = ds.y_raw[ds.valid_starts.astype(int) + seq_len - 1]          # (N,)
    c_all = preds[:, 0, f_dim] - x_last_all
    adj_all = preds - c_all[:, None, None]
    raw_mse = np.mean((preds - trues) ** 2)
    adj_mse = np.mean((adj_all - trues) ** 2)
    persist = np.broadcast_to(x_last_all[:, None, None], preds.shape)
    per_mse = np.mean((persist - trues) ** 2)
    raw_corr = float(np.nanmean(CORR(preds, trues)))
    adj_corr = float(np.nanmean(CORR(adj_all, trues)))

    # ---- segment panel ----
    sid, k = a.segment, a.horizon
    idx = np.where(ds.window_segment_ids == sid)[0]
    starts = ds.valid_starts[idx].astype(int)
    r0, r1 = int(starts.min()), min(int(starts.max() + seq_len + preds.shape[1]), len(ds.dates))
    gt_t = pd.to_datetime(ds.dates[r0:r1])
    gt_v = ds.y_raw[r0:r1]

    tgt_t = pd.to_datetime(ds.dates[starts + seq_len + k - 1])
    raw_k = preds[idx, k - 1, f_dim]
    adj_k = raw_k - c_all[idx]
    rain_on, rain_off = bounds.get(sid, (None, None))

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(gt_t, gt_v, "-", color="black", lw=3.0, label="GroundTruth", zorder=4)
    ax.plot(tgt_t, raw_k, "--", color="#ff7f0e", lw=1.6, alpha=0.9, label="Pred raw", zorder=2)
    ax.plot(tgt_t, adj_k, "-", color="#1f77b4", lw=2.4, label="Pred adj (anchored)", zorder=3)
    if rain_on is not None:
        ax.axvline(rain_on, color="#2ca02c", ls="--", lw=1.8, label="SegmentStart (rain on)")
    if rain_off is not None:
        ax.axvline(rain_off, color="#d62728", ls="--", lw=1.8, label="SegmentEnd (rain off)")

    ax.set_title(f"Anchored Correction — Segment {sid} (HL01),  t+{k}", fontsize=19, pad=12)
    ax.set_xlabel("Target Time", fontsize=15)
    ax.set_ylabel("HL01 (mm)", fontsize=15)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(20); lbl.set_ha("right")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.95)

    headline = (f"Test set (all 33 segments):\n"
                f"raw MSE {raw_mse:,.0f}  →  adj MSE {adj_mse:,.0f}   (−{(1-adj_mse/raw_mse)*100:.0f}%)\n"
                f"Corr {raw_corr:.2f} → {adj_corr:.2f}   (persistence MSE {per_mse:,.0f})")
    ax.text(0.985, 0.97, headline, transform=ax.transAxes, ha="right", va="top",
            fontsize=13, family="monospace",
            bbox=dict(boxstyle="round", fc="#fff7e6", ec="#d9a441", alpha=0.95))

    fig.tight_layout()
    dst = os.path.join(a.run_dir, "anchored_metric", f"slide_seg{sid}_t{k}.png")
    fig.savefig(dst, dpi=200); plt.close(fig)
    print(f"headline: raw MSE {raw_mse:,.0f} -> adj {adj_mse:,.0f} | Corr {raw_corr:.3f}->{adj_corr:.3f}")
    print(f"已寫出: {dst}")


if __name__ == "__main__":
    main()
