"""從沒下雨的部分出發的 segment 切分（7/2 會議）：drycut 規則。

偵測「確定長乾段」：連續 Past10Min==0 且持續 >= L 小時，從全資料剔除，
剩餘時段即為要用的 segments。與現行主動找雨（Past1Hr>=0.1 + gap 30min）方向相反：
短於 L 的停雨段（雨間空檔、退水）會被保留在 segment 內。

命名說明：保留下來的 segment 都含雨（非 dry），故輸出叫 rain_segments_meta，
後綴 drycut 標示它是由「剔除長乾段」規則產生，與現行 rain_segments_meta.csv 區別。

規則:
- 資料來源為本機 dataset/all_minute_wide.csv（不重抓 SQL）。
- NaN 不算「確定沒雨」，會中斷乾段計數（本資料無 NaN，pipeline 化時的保險）。
- SegmentStart/SegmentEnd = 核心（頭尾皆緊鄰被剔除的長乾段）；
  WinStart/WinEnd = 核心往兩側各延 --buffer-minutes B 進被剔除的乾段（沿用現行
  rain_segments_meta 的 Win = Segment ± buffer 慣例，於資料首尾截斷）。
  實際可用資料 = [WinStart, WinEnd]，DurationMinutes 亦以此計。
- 要求 L >= 2B，保證相鄰 window 不重疊。
- 不含 split 欄位：split 方式後續另定。

用法:
    python build_drycut_segments_meta.py [--l-hours 3] [--buffer-minutes 0]

輸出: dataset/rain_segments_meta_drycut_L{L}h_buf{B}.csv
（segment_id, SegmentStart, SegmentEnd, WinStart, WinEnd, DurationMinutes, WetMinutes）
"""
import argparse
from pathlib import Path

import pandas as pd

ALL_CSV = "dataset/all_minute_wide.csv"
RAIN_COL = "Past10Min"

# duration 分 bin：右閉 (right=True)，資料為 10 分鐘量化 → 前三個 bin 即恰為 10/20/30m
DUR_BINS = [0, 10, 20, 30, 60, 120, 240, 480, 960, 1920, float("inf")]
DUR_LABELS = ["10m", "20m", "30m", "30-60m", "1-2h", "2-4h", "4-8h", "8-16h", "16-32h", ">32h"]


def find_drycut_core_segments(df: pd.DataFrame, *, rain_col: str, l_minutes: int) -> pd.DataFrame:
    dry = df[rain_col].eq(0)  # NaN -> False：缺測不是「確定沒雨」
    run_id = (dry != dry.shift()).cumsum()
    run_len = df.groupby(run_id)[rain_col].transform("size")
    removed = dry & (run_len >= l_minutes)

    kept = ~removed
    seg_id = (kept != kept.shift()).cumsum()
    return (
        df.loc[kept]
        .groupby(seg_id[kept])
        .agg(
            SegmentStart=("date", "min"),
            SegmentEnd=("date", "max"),
            WetMinutes=(rain_col, lambda s: int((s > 0).sum())),
        )
        .reset_index(drop=True)
    )


def print_duration_stats(meta: pd.DataFrame) -> None:
    dur = meta["DurationMinutes"]
    binned = pd.cut(dur, bins=DUR_BINS, labels=DUR_LABELS)
    table = pd.DataFrame({
        "segments": binned.value_counts().sort_index(),
        "total_minutes": dur.groupby(binned, observed=False).sum().astype(int),
    })
    table["pct_of_kept"] = (table["total_minutes"] / dur.sum() * 100).round(1)
    print("\nduration 分布（每個 bin 幾個 segment）:")
    print(table.to_string())
    stats = dur.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1)
    print("\nduration 統計量（分鐘）:")
    print(stats.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--l-hours", type=float, default=3, help="長乾段門檻（小時）")
    parser.add_argument("--buffer-minutes", type=int, default=0,
                        help="核心兩側往被剔除乾段延伸的分鐘數（Win = Segment ± B）")
    parser.add_argument("--all-csv", default=ALL_CSV)
    parser.add_argument("--rain-col", default=RAIN_COL)
    parser.add_argument("--out", default=None, help="輸出路徑（預設依參數自動命名）")
    parser.add_argument("--emit-training-csv", action="store_true",
                        help="產完 meta 後接著組裝訓練 CSV（呼叫 build_training_csv_from_meta）。"
                             "若既有輸出的來源指紋未變則自動跳過，不會重複產出相同大檔。")
    parser.add_argument("--training-csv-out", default=None,
                        help="搭配 --emit-training-csv 使用；預設依 meta 檔名自動命名")
    args = parser.parse_args()

    l_minutes = int(args.l_hours * 60)
    buf = args.buffer_minutes
    if l_minutes < 2 * buf:
        raise ValueError(f"需要 L >= 2*buffer（L={l_minutes}m, buffer={buf}m），否則相鄰 window 會重疊。")
    out = args.out or f"dataset/rain_segments_meta_drycut_L{args.l_hours:g}h_buf{buf}.csv"

    df = pd.read_csv(args.all_csv, usecols=["date", args.rain_col], parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    dt = df["date"].diff().dt.total_seconds().dropna()
    if (dt != 60).any():
        raise ValueError("all_minute_wide 不是連續 1 分鐘網格，run 長度計算會失真。")

    meta = find_drycut_core_segments(df, rain_col=args.rain_col, l_minutes=l_minutes)
    if meta.empty:
        raise ValueError("沒有任何保留 segment，請檢查參數。")

    meta = meta.sort_values("SegmentStart").reset_index(drop=True)
    meta["segment_id"] = range(1, len(meta) + 1)
    meta["WinStart"] = (meta["SegmentStart"] - pd.Timedelta(minutes=buf)).clip(lower=df["date"].min())
    meta["WinEnd"] = (meta["SegmentEnd"] + pd.Timedelta(minutes=buf)).clip(upper=df["date"].max())
    meta["DurationMinutes"] = (
        (meta["WinEnd"] - meta["WinStart"]).dt.total_seconds().div(60).astype(int) + 1)
    meta = meta.loc[:, ["segment_id", "SegmentStart", "SegmentEnd", "WinStart", "WinEnd",
                        "DurationMinutes", "WetMinutes"]]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)

    total = len(df)
    kept_min = int(meta["DurationMinutes"].sum())
    print(f"輸出: {out}")
    print(f"參數: L={args.l_hours:g}h, buffer={buf}min, rain_col={args.rain_col}")
    print(f"segments={len(meta)}  保留 {kept_min:,}/{total:,} 分鐘 ({kept_min/total*100:.1f}%)")
    print_duration_stats(meta)

    if args.emit_training_csv:
        # 延後 import：不加此旗標時不必付組裝腳本的載入成本
        from build_training_csv_from_meta import build_training_csv

        print(f"\n{'-' * 60}\n組裝訓練 CSV\n{'-' * 60}")
        build_training_csv(
            out,
            all_csv=args.all_csv,
            out=args.training_csv_out,
            skip_if_current=True,
        )


if __name__ == "__main__":
    main()
