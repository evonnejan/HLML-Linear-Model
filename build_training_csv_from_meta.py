"""依 segment meta 從 all_minute_wide.csv 切出訓練 CSV。

補上 pipeline 的斷點：`build_drycut_segments_meta.py` 只產出 segment 目錄（時間範圍），
本腳本把它與逐分鐘寬表組裝成 `run.py` / `data_provider/Data_Loader.py` 可直接吃的訓練檔。

切法與欄位語意完全沿用 `Data_From_SQL_4.py:400-436`（舊 pipeline 的組裝迴圈），
差別只在資料來源是本機 all_minute_wide.csv 而非 SQL，故不需連資料庫。

meta 無關規則：任何含
  segment_id, SegmentStart, SegmentEnd, WinStart, WinEnd
的 meta 都適用（drycut 任意 L/buffer、舊版 rain_segments_meta.csv 亦可）。

重要語意:
- 取用範圍是 [WinStart, WinEnd]（含 buffer），不是核心。
- `isRain` 沿用舊定義 = date 落在 [SegmentStart, SegmentEnd] 內，
  亦即「在核心內(1) / 在 buffer 內(0)」的標記，**不是**降雨量判定。
  buffer=0 時全為 1（該欄失去資訊量）。
- gate 欄在 all_minute_wide 中仍有約 7% NaN：`Data_From_SQL_all.py` 刻意
  延後 within-segment ffill（因為當時尚無 segment）。本腳本在切段後補做，
  對齊 `merge_gate_data.py:83-86`，補完後降到約 3.5%（殘餘皆在段首）。

用法:
    python build_training_csv_from_meta.py \
        --meta dataset/rain_segments_meta_drycut_L3h_buf60.csv

輸出: dataset/train_<meta 檔名去掉 rain_segments_meta_ 前綴>.csv
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

ALL_CSV = "dataset/all_minute_wide.csv"

GATE_COLS = [
    "north_gate_opening_1",
    "north_gate_opening_2",
    "north_gate_opening_3",
    "north_gate_opening_4",
    "south_gate_opening_1",
    "south_gate_opening_2",
    "south_gate_opening_3",
]

SEGMENT_COLS = ["date", "SegmentStart", "SegmentEnd", "segment_id", "WinStart", "WinEnd", "isRain"]
META_REQUIRED = ["segment_id", "SegmentStart", "SegmentEnd", "WinStart", "WinEnd"]

RAIN_COL = "Past10Min"
MIN_SINCE_RAIN_COL = "min_since_rain"


def add_min_since_rain(wide: pd.DataFrame, rain_col: str = RAIN_COL) -> pd.DataFrame:
    """在**切段前**於全年寬表上計算「距上次降雨的分鐘數」。

    必須全域計算：若切段後才算，每段開頭會被迫從 0 重新計數，
    而 segment 開頭正是「剛下過雨」或「退水中」的狀態，重算會抹掉這個資訊。

    定義與 eval_dry.py 的 minutes_since_last_rain 一致：
    以 rain_col > 0 為「有雨」，該分鐘計為 0，其後逐分鐘累加。
    資料開頭尚未出現任何雨之前為 NaN（無從得知距離）。
    """
    if rain_col not in wide.columns:
        raise ValueError(f"寬表缺少 {rain_col}，無法計算 {MIN_SINCE_RAIN_COL}。")

    wet = wide[rain_col] > 0
    # 每個 wet 分鐘開一個新群組；群組內的序號即距上次降雨的分鐘數
    grp = wet.cumsum()
    wide = wide.copy()
    wide[MIN_SINCE_RAIN_COL] = wide.groupby(grp).cumcount().astype("float64")
    wide.loc[grp == 0, MIN_SINCE_RAIN_COL] = pd.NA  # 首次降雨之前無定義
    return wide


def slice_segments(wide: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """對每個 meta 列取 [WinStart, WinEnd] 的分鐘，貼上 segment 欄位。"""
    parts = []
    for _, row in meta.iterrows():
        mask = (wide["date"] >= row["WinStart"]) & (wide["date"] <= row["WinEnd"])
        sub = wide.loc[mask].copy()
        if sub.empty:
            print(f"  [WARN] segment_id={row['segment_id']} 在寬表中無資料，跳過。")
            continue

        sub["segment_id"] = row["segment_id"]
        sub["SegmentStart"] = row["SegmentStart"]
        sub["SegmentEnd"] = row["SegmentEnd"]
        sub["WinStart"] = row["WinStart"]
        sub["WinEnd"] = row["WinEnd"]
        # 沿用 Data_From_SQL_4.label_rain_minutes：核心內為 True、buffer 內為 False
        sub["isRain"] = (sub["date"] >= row["SegmentStart"]) & (sub["date"] <= row["SegmentEnd"])
        parts.append(sub)

    if not parts:
        raise ValueError("meta 涵蓋的時間範圍內沒有任何資料。")
    return pd.concat(parts, ignore_index=True)


def fill_gate_within_segment(df: pd.DataFrame) -> pd.DataFrame:
    """段內 forward-fill 後把負值夾到 0，對齊 merge_gate_data.py:83-96。"""
    present = [c for c in GATE_COLS if c in df.columns]
    if not present:
        print("  [WARN] 找不到任何 gate 欄位，略過 ffill。")
        return df
    df[present] = df.groupby("segment_id", sort=False)[present].transform(lambda col: col.ffill())
    df[present] = df[present].clip(lower=0)
    return df


def check_no_overlap(meta: pd.DataFrame, allow_overlap: bool = False) -> None:
    """檢查相鄰 window 是否重疊。

    重疊代表同一批分鐘同時屬於兩個 segment，若這兩段被分到不同 split 即為
    train/test leakage。drycut 以 `L >= 2*buffer` 從結構上排除；舊法
    (rain_segments_meta.csv) 沒有這個保證，實測有 32 對重疊 —— 要重現舊法
    當對照組時需以 allow_overlap=True 明示接受此缺陷。
    """
    m = meta.sort_values("WinStart").reset_index(drop=True)
    gap = (m["WinStart"].shift(-1) - m["WinEnd"]).dt.total_seconds().div(60)[:-1]
    n_bad = int((gap <= 0).sum())
    if n_bad:
        msg = (f"meta 有 {n_bad} 對 window 重疊，會造成 train/test leakage。"
               " 請確認產 meta 時滿足 L >= 2*buffer。")
        if not allow_overlap:
            raise ValueError(msg)
        shared = int(-gap[gap <= 0].sum())
        print(f"  [WARN] {msg}\n"
               f"         已用 --allow-overlap 明示接受（重疊共約 {shared:,} 分鐘）。"
               f" 此為舊法的已知缺陷，僅適用於重現舊法作為對照組。")
        return
    if len(gap):
        print(f"  重疊檢查 OK：0 對重疊，最小相鄰間隔 {gap.min():.0f} 分鐘")


def report(df: pd.DataFrame, meta: pd.DataFrame, seq_len: int, pred_len: int) -> None:
    need = seq_len + pred_len
    dur = df.groupby("segment_id").size()
    ok = int((dur >= need).sum())
    windows = int((dur - need + 1).clip(lower=0).sum())
    print(f"\n列數 {len(df):,} | segments {df['segment_id'].nunique()} / meta {len(meta)}")
    print(f"段長（分鐘）: min {dur.min()} | 中位 {int(dur.median())} | max {dur.max()}")
    print(f"isRain=1（核心）{int(df['isRain'].sum()):,} 列 | isRain=0（buffer）{int((~df['isRain']).sum()):,} 列")
    print(f"\n以 seq_len={seq_len}, pred_len={pred_len}（需 {need} 分鐘/window）計:")
    print(f"  可生 window 的段 {ok}/{len(dur)} | 訓練 windows {windows:,}")

    na = df.isna().sum()
    na = na[na > 0]
    print("\nNaN 報告（含 NaN 的 window 會在 Data_Loader 被丟棄）:")
    if na.empty:
        print("  無 NaN")
    else:
        for col, cnt in na.items():
            print(f"  {col:<26} {cnt:>8,}  ({cnt / len(df) * 100:5.2f}%)")


def default_out_path(meta_path: str) -> str:
    stem = Path(meta_path).stem
    for prefix in ("rain_segments_meta_", "segments_meta_", "rain_segments_meta"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):] or "old"
            break
    return f"dataset/train_{stem}.csv"


def _fingerprint(meta_path: str, all_csv: str, opts: dict) -> str:
    """meta 內容 + 寬表識別 + 產生選項的指紋，用來判斷既有輸出是否已是最新。"""
    h = hashlib.sha256()
    h.update(Path(meta_path).read_bytes())
    st = Path(all_csv).stat()
    h.update(f"{all_csv}|{st.st_size}|{int(st.st_mtime)}".encode())
    h.update(repr(sorted(opts.items())).encode())
    return h.hexdigest()


def build_training_csv(
    meta_path: str,
    *,
    all_csv: str = ALL_CSV,
    out: str | None = None,
    seq_len: int = 96,
    pred_len: int = 15,
    rain_col: str = RAIN_COL,
    add_msr: bool = True,
    skip_if_current: bool = False,
    allow_overlap: bool = False,
) -> str:
    """組裝訓練 CSV；回傳輸出路徑。可被其他腳本直接呼叫。

    skip_if_current=True 時，若既有輸出的來源指紋未變則跳過重算，
    避免同一組 meta 反覆產出一模一樣的大檔。
    """
    meta = pd.read_csv(meta_path, parse_dates=["SegmentStart", "SegmentEnd", "WinStart", "WinEnd"])
    missing = [c for c in META_REQUIRED if c not in meta.columns]
    if missing:
        raise ValueError(f"meta 缺少必要欄位: {missing}")

    out = out or default_out_path(meta_path)
    stamp = Path(out).with_suffix(".source.sha256")
    fp = _fingerprint(meta_path, all_csv,
                      {"rain_col": rain_col, "add_msr": add_msr, "allow_overlap": allow_overlap})

    if skip_if_current and Path(out).exists() and stamp.exists() and stamp.read_text().strip() == fp:
        print(f"[skip] {out} 已是最新（來源指紋未變），不重複產生。")
        return out

    print(f"meta   : {meta_path}（{len(meta)} 段）")
    print(f"寬表   : {all_csv}")
    check_no_overlap(meta, allow_overlap=allow_overlap)

    wide = pd.read_csv(all_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    print(f"  寬表 {len(wide):,} 列 x {len(wide.columns)} 欄，"
          f"範圍 {wide['date'].min()} ~ {wide['date'].max()}")

    if add_msr:
        wide = add_min_since_rain(wide, rain_col)
        msr = wide[MIN_SINCE_RAIN_COL]
        print(f"  已加 {MIN_SINCE_RAIN_COL}（全域計算）：中位 {msr.median():.0f} 分鐘、"
              f"max {msr.max():.0f}、首次降雨前 NaN {int(msr.isna().sum()):,} 列")

    df = slice_segments(wide, meta)
    df = fill_gate_within_segment(df)

    other = [c for c in df.columns if c not in SEGMENT_COLS]
    df = (df.loc[:, [*SEGMENT_COLS, *other]]
            .sort_values(["segment_id", "date"])
            .reset_index(drop=True))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    stamp.write_text(fp)
    print(f"\n輸出: {out}  ({len(df.columns)} 欄)")
    report(df, meta, seq_len, pred_len)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta", required=True, help="segment meta CSV（drycut 或舊版皆可）")
    parser.add_argument("--all-csv", default=ALL_CSV, help="逐分鐘寬表")
    parser.add_argument("--out", default=None, help="輸出路徑（預設依 meta 檔名自動命名）")
    parser.add_argument("--seq-len", type=int, default=96, help="僅用於報表估算 window 數")
    parser.add_argument("--pred-len", type=int, default=15, help="僅用於報表估算 window 數")
    parser.add_argument("--rain-col", default=RAIN_COL, help=f"計算 {MIN_SINCE_RAIN_COL} 用的雨量欄")
    parser.add_argument("--no-min-since-rain", action="store_true",
                        help=f"不產生 {MIN_SINCE_RAIN_COL} 欄")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="允許 meta 有重疊 window（僅用於重現舊法作為對照組）")
    parser.add_argument("--skip-if-current", action="store_true",
                        help="既有輸出的來源指紋未變時跳過重算")
    args = parser.parse_args()

    build_training_csv(
        args.meta,
        all_csv=args.all_csv,
        out=args.out,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        rain_col=args.rain_col,
        add_msr=not args.no_min_since_rain,
        skip_if_current=args.skip_if_current,
        allow_overlap=args.allow_overlap,
    )


if __name__ == "__main__":
    main()
