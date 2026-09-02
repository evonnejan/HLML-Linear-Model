"""drycut 切分結果視覺化：檢視 build_drycut_segments_meta.py 切出來的 segments。

產出（圖內文字全英文）:
- overview.png      : 全期間水位 + 雨量，核心綠色、buffer 粉紅色色帶
- duration_hist.png : segment duration（含 buffer）分布長條圖
- seg_XXX.png       : 每個 segment 一張，上=水位、下=雨量。
                      核心（SegmentStart~End）綠底、buffer（Win 超出核心的部分）粉紅底；
                      前後灰底是「被剔除的乾段」的顯示窗（固定 --context-hours，
                      只是取景範圍；實際相鄰乾段真實長度標在標題 gap 欄位）。

用法:
    python visualize_drycut_segments.py [--meta dataset/rain_segments_meta_drycut_L3h_buf0.csv]
        [--water-col HL01] [--context-hours 3] [--segments all|1,5,12]

輸出目錄: analysis/drycut_segments/<meta 檔名>/（已 gitignore）
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from build_drycut_segments_meta import DUR_BINS, DUR_LABELS

ALL_CSV = "dataset/all_minute_wide.csv"
CORE_COLOR = "tab:green"
BUFFER_COLOR = "hotpink"


def fmt_gap(hours: float) -> str:
    return "edge" if pd.isna(hours) else f"{hours:.1f}h"


def shade_segment(ax, r, *, alpha_core, alpha_buf):
    ax.axvspan(r["WinStart"], r["SegmentStart"], color=BUFFER_COLOR, alpha=alpha_buf, lw=0)
    ax.axvspan(r["SegmentEnd"], r["WinEnd"], color=BUFFER_COLOR, alpha=alpha_buf, lw=0)
    ax.axvspan(r["SegmentStart"], r["SegmentEnd"], color=CORE_COLOR, alpha=alpha_core, lw=0)


def plot_overview(df, meta, water_col, rain_col, out_path):
    ds = df.iloc[::10]  # 10-min 抽樣，全年線圖夠用
    fig, (ax_w, ax_r) = plt.subplots(
        2, 1, figsize=(18, 6), sharex=True, height_ratios=[3, 1])
    ax_w.plot(ds["date"], ds[water_col], lw=0.4, color="black")
    ax_r.fill_between(ds["date"], 0, ds[rain_col], color="steelblue", step="mid")
    for _, r in meta.iterrows():
        for ax in (ax_w, ax_r):
            shade_segment(ax, r, alpha_core=0.35, alpha_buf=0.45)
    ax_w.set_ylabel(water_col)
    ax_r.set_ylabel(rain_col)
    ax_w.set_title(f"Kept segments over full period ({len(meta)} segments; "
                   f"green = core, pink = buffer)")
    ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_duration_hist(meta, out_path):
    binned = pd.cut(meta["DurationMinutes"], bins=DUR_BINS, labels=DUR_LABELS)
    counts = binned.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=CORE_COLOR, alpha=0.8)
    ax.bar_label(bars)
    ax.set_xlabel("segment duration (incl. buffer)")
    ax.set_ylabel("# segments")
    med = meta["DurationMinutes"].median()
    ax.set_title(f"Segment duration distribution (n={len(meta)}, median={med:.0f}m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_segment(df, r, *, water_col, rain_col, context_min, out_path):
    lo = r["WinStart"] - pd.Timedelta(minutes=context_min)
    hi = r["WinEnd"] + pd.Timedelta(minutes=context_min)
    win = df.loc[(df["date"] >= lo) & (df["date"] <= hi)]
    if win.empty:
        return

    fig, (ax_w, ax_r) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[3, 1])
    ax_w.plot(win["date"], win[water_col], lw=0.9, color="black")
    ax_r.fill_between(win["date"], 0, win[rain_col], color="steelblue", step="mid")
    for ax in (ax_w, ax_r):
        ax.axvspan(lo, r["WinStart"], color="grey", alpha=0.18, lw=0)
        ax.axvspan(r["WinEnd"], hi, color="grey", alpha=0.18, lw=0)
        shade_segment(ax, r, alpha_core=0.12, alpha_buf=0.20)
        ax.axvline(r["WinStart"], color=CORE_COLOR, lw=1)
        ax.axvline(r["WinEnd"], color=CORE_COLOR, lw=1)
    ax_w.set_ylabel(water_col)
    ax_r.set_ylabel(rain_col)
    ax_w.set_title(
        f"seg {r['segment_id']:03d}  {r['WinStart']} ~ {r['WinEnd']}  "
        f"dur={r['DurationMinutes']}m wet={r['WetMinutes']}m  "
        f"dry gap before={fmt_gap(r['GapBeforeHours'])} after={fmt_gap(r['GapAfterHours'])}  "
        f"(grey = removed dry)")
    ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meta", default="dataset/rain_segments_meta_drycut_L3h_buf0.csv")
    parser.add_argument("--all-csv", default=ALL_CSV)
    parser.add_argument("--water-col", default="HL01")
    parser.add_argument("--rain-col", default="Past10Min")
    parser.add_argument("--context-hours", type=float, default=3)
    parser.add_argument("--segments", default="all", help="all 或逗號分隔 segment_id")
    args = parser.parse_args()

    meta = pd.read_csv(args.meta,
                       parse_dates=["SegmentStart", "SegmentEnd", "WinStart", "WinEnd"])
    meta = meta.sort_values("WinStart").reset_index(drop=True)
    # 相鄰被剔除乾段的真實長度（window 之間；首尾 segment 靠資料邊界的一側為 "edge"）
    meta["GapBeforeHours"] = (
        (meta["WinStart"] - meta["WinEnd"].shift(1)).dt.total_seconds() / 3600)
    meta["GapAfterHours"] = (
        (meta["WinStart"].shift(-1) - meta["WinEnd"]).dt.total_seconds() / 3600)

    df = pd.read_csv(args.all_csv, usecols=["date", args.water_col, args.rain_col],
                     parse_dates=["date"])

    out_dir = Path("analysis/drycut_segments") / Path(args.meta).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overview(df, meta, args.water_col, args.rain_col, out_dir / "overview.png")
    plot_duration_hist(meta, out_dir / "duration_hist.png")

    if args.segments != "all":
        wanted = {int(s) for s in args.segments.split(",")}
        meta = meta.loc[meta["segment_id"].isin(wanted)]
    context_min = int(args.context_hours * 60)
    for _, r in meta.iterrows():
        plot_segment(df, r, water_col=args.water_col, rain_col=args.rain_col,
                     context_min=context_min,
                     out_path=out_dir / f"seg_{r['segment_id']:03d}.png")

    print(f"輸出 {len(meta)} 張 segment 圖 + overview.png + duration_hist.png -> {out_dir}/")


if __name__ == "__main__":
    main()
