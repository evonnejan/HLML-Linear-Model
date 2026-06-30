"""整段 segment 視覺化：固定提前量連續線（GT / raw / adj）。

每張圖 = 一個 segment，2x2 四格、每格一個 lead time k：
  - GT       : 該 segment 連續真實目標值 (粗綠實線)
  - raw  k   : 每個發射時刻 t 的 pred[t, k-1]      (細紅虛線, 半透明)
  - adj  k   : adj[t,k-1] = pred[t,k-1] - c_t       (中藍點線)
預測值畫在「目標時刻 t+k」, 與 GT 同一日曆時間對齊 → 同格垂直可直接比較。

x_last / GT 直接取 dataset 的 y_raw(原始尺度, 與已 inverse-transform 的 pred.npy 同尺度)。
設計見 docs/superpowers/specs/2026-06-03-segment-visualization-design.md。

用法:
    python visualize_segment.py <run_dir> [--segments 144,143,...|all]
        [--horizons 2,5,10,15] [--outputs outputs] [--out-subdir anchored_metric]
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

DEFAULT_SEGMENTS = "144,143,152,134,130"   # 代表集: 最偏/尖峰最大/最不偏/中位/最長
DEFAULT_HORIZONS = "2,5,10,15"


def load_segment_bounds(args):
    """從 run 的資料檔讀 segment_id -> (SegmentStart, SegmentEnd)，rain on/off 用。"""
    path = os.path.join(args.root_path, args.data_path)
    try:
        df = pd.read_csv(path, usecols=["segment_id", "SegmentStart", "SegmentEnd"])
    except (ValueError, FileNotFoundError):
        print("  [note] 資料檔無 SegmentStart/SegmentEnd 欄位，略過 rain on/off")
        return {}
    df = df.drop_duplicates("segment_id")
    return {int(r.segment_id): (pd.to_datetime(r.SegmentStart), pd.to_datetime(r.SegmentEnd))
            for r in df.itertuples()}


def plot_segment(sid, *, preds, trues, y_raw, dates, valid_starts, seg_ids,
                 seq_len, f_dim, horizons, target, dst_dir, bounds=None):
    idx = np.where(seg_ids == sid)[0]
    if len(idx) == 0:
        print(f"  [skip] segment {sid}: 無 window")
        return
    starts = valid_starts[idx].astype(int)
    r0, r1 = int(starts.min()), int(starts.max() + seq_len + preds.shape[1])
    r1 = min(r1, len(dates))

    gt_t = pd.to_datetime(dates[r0:r1])
    gt_v = y_raw[r0:r1]
    x_last = y_raw[starts + seq_len - 1]                 # (n,)
    c = preds[idx, 0, f_dim] - x_last                    # per-window offset
    mean_absc = float(np.mean(np.abs(c)))

    # 預先算各 k 的線與 MSE, 並收集 y 範圍
    panel = {}
    ys = [gt_v]
    for k in horizons:
        tgt_t = pd.to_datetime(dates[starts + seq_len + k - 1])
        raw_k = preds[idx, k - 1, f_dim]
        adj_k = raw_k - c
        gt_k = trues[idx, k - 1, f_dim]
        mse_raw = float(np.mean((raw_k - gt_k) ** 2))
        mse_adj = float(np.mean((adj_k - gt_k) ** 2))
        panel[k] = (tgt_t, raw_k, adj_k, mse_raw, mse_adj)
        ys += [raw_k, adj_k]
    ylo, yhi = float(np.min(np.concatenate(ys))), float(np.max(np.concatenate(ys)))
    pad = 0.05 * (yhi - ylo + 1e-9)
    ylim = (ylo - pad, yhi + pad)

    rain_on, rain_off = (bounds or {}).get(sid, (None, None))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    for j, (ax, k) in enumerate(zip(axes.ravel(), horizons)):
        tgt_t, raw_k, adj_k, mse_raw, mse_adj = panel[k]
        ax.plot(gt_t, gt_v, "-", color="#2ca02c", lw=2.4, label="GT", zorder=3)
        ax.plot(tgt_t, raw_k, "--", color="#d62728", lw=1.0, alpha=0.55, label="raw", zorder=1)
        ax.plot(tgt_t, adj_k, "-.", color="#1f77b4", lw=1.6, label="adj", zorder=2)
        if rain_on is not None:
            ax.axvline(rain_on, color="#2ca02c", ls="--", lw=1.3,
                       label="SegmentStart (rain on)" if j == 0 else None)
        if rain_off is not None:
            ax.axvline(rain_off, color="#d62728", ls="--", lw=1.3,
                       label="SegmentEnd (rain off)" if j == 0 else None)
        ax.set_title(f"t+{k}   raw MSE={mse_raw:,.0f}   adj MSE={mse_adj:,.0f}", fontsize=10)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30); lbl.set_ha("right")
    axes.ravel()[0].legend(loc="best", fontsize=8)

    d0, d1 = gt_t[0].strftime("%Y-%m-%d %H:%M"), gt_t[-1].strftime("%Y-%m-%d %H:%M")
    fig.suptitle(f"Segment {sid}  ({target})   {d0} ~ {d1}   "
                 f"n_windows={len(idx)}   mean|c|={mean_absc:.1f}", fontsize=13)
    fig.supylabel("water level"); fig.supxlabel("target time  (t+k)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"seg_{sid}.png")
    fig.savefig(dst, dpi=170); plt.close(fig)
    print(f"  segment {sid:>3}: windows={len(idx):>4}  mean|c|={mean_absc:6.1f}  -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--segments", default=DEFAULT_SEGMENTS, help="逗號分隔 id, 或 all")
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out-subdir", default="anchored_metric")
    a = ap.parse_args()

    out_dir = os.path.join(a.run_dir, a.outputs)
    preds = np.load(os.path.join(out_dir, "pred.npy"))
    trues = np.load(os.path.join(out_dir, "true.npy"))
    horizons = [int(x) for x in a.horizons.split(",")]
    if max(horizons) > preds.shape[1]:
        raise SystemExit(f"horizon {max(horizons)} > pred_len {preds.shape[1]}")

    args = load_run_args(a.run_dir)
    ds, _ = data_provider(args, "test")
    f_dim = -1 if args.features == "MS" else 0
    bounds = load_segment_bounds(args)

    seg_ids = ds.window_segment_ids
    if a.segments.strip().lower() == "all":
        seg_list = sorted(set(int(s) for s in seg_ids))
    else:
        seg_list = [int(s) for s in a.segments.split(",")]

    dst_dir = os.path.join(a.run_dir, a.out_subdir, "segments")
    print(f"畫 {len(seg_list)} 個 segment, horizons={horizons}")
    for sid in seg_list:
        plot_segment(sid, preds=preds, trues=trues, y_raw=ds.y_raw, dates=ds.dates,
                     valid_starts=ds.valid_starts, seg_ids=seg_ids, seq_len=args.seq_len,
                     f_dim=f_dim, horizons=horizons, target=args.target, dst_dir=dst_dir,
                     bounds=bounds)


if __name__ == "__main__":
    main()
