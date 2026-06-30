"""Build minute-level wide table covering water + rain + gate (whole period),
plus rain segments metadata with train/val/test split labels.

Outputs:
- dataset/all_minute_wide.csv        : 1-min wide table (water + rain + gate)
- dataset/rain_segments_meta.csv     : rain segment + window + split metadata

Reuses SQL client and loaders from Data_From_SQL_4 / Data_From_SQL_5 so the
underlying queries and resample/ffill behaviour match the existing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Data_From_SQL_4 import (
    SQLServerClient,
    build_rain_wide,
    build_water_wide,
    build_event_windows,
    load_rain_data,
    load_water_data,
    pick_rain_value_col,
    segment_intervals,
    PRE_WINDOW_MINUTES,
    POST_WINDOW_MINUTES,
    RAIN_THRESHOLD,
    RAIN_GAP_MINUTES,
    MIN_RAIN_SEGMENT_MINUTES,
)
from Data_From_SQL_5 import (
    load_gate_obs,
    load_mapping,
    validate_mapping,
    build_wide_table as build_gate_wide_raw,
)

ALL_MINUTE_OUTPUT_CSV = Path("dataset/all_minute_wide.csv")
SEGMENTS_META_OUTPUT_CSV = Path("dataset/rain_segments_meta.csv")

GATE_STALENESS_LIMIT = pd.Timedelta("5min")
DROP_COLS = ["StationId"]

GATE_TARGET_COLUMNS = [
    "開度_北側閘門(1)",
    "開度_北側閘門(2)",
    "開度_北側閘門(3)",
    "開度_北側閘門(4)",
    "開度_南側閘門(1)",
    "開度_南側閘門(2)",
    "開度_南側閘門(3)",
]
GATE_RENAME_MAP = {
    "開度_北側閘門(1)": "north_gate_opening_1",
    "開度_北側閘門(2)": "north_gate_opening_2",
    "開度_北側閘門(3)": "north_gate_opening_3",
    "開度_北側閘門(4)": "north_gate_opening_4",
    "開度_南側閘門(1)": "south_gate_opening_1",
    "開度_南側閘門(2)": "south_gate_opening_2",
    "開度_南側閘門(3)": "south_gate_opening_3",
}

# Match Data_Loader segment-based split: 70 / 10 / (rest), sorted by SegmentStart
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1


def build_gate_wide_irregular(gate_obs_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot gate obs to wide, filter to target gates, rename to English.

    Keep irregular observation timestamps — alignment is done by merge_asof
    in merge_all_sources with a staleness limit, matching merge_gate_data.py.
    """
    wide_raw = build_gate_wide_raw(gate_obs_df)

    missing = [column for column in GATE_TARGET_COLUMNS if column not in wide_raw.columns]
    if missing:
        raise ValueError(f"閘門 wide table 缺少欄位：{missing}")

    gate = wide_raw.loc[:, ["date", *GATE_TARGET_COLUMNS]].copy()
    gate["date"] = pd.to_datetime(gate["date"], errors="coerce")
    gate = gate.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    gate = gate.rename(columns=GATE_RENAME_MAP)
    return gate


def merge_all_sources(
    water_wide: pd.DataFrame,
    rain_wide: pd.DataFrame,
    gate_irregular: pd.DataFrame,
    staleness_limit: pd.Timedelta = GATE_STALENESS_LIMIT,
) -> pd.DataFrame:
    """Anchor on water_wide. Left-join rain (both already 1-min ffill).
    Gate uses the SAME row-wise approach as merge_gate_data.py:62-78 to
    keep preprocessing consistent with the training data:

      1. single row-wise merge_asof(direction='backward')
      2. backward_lag > staleness_limit → set all 7 gate cols to NaN
      3. clip negatives to 0

    The within-segment ffill step from merge_gate_data is intentionally
    NOT applied here, because all_minute_wide has no segments. eval_dry.py
    is responsible for per-(dry-)segment ffill at slicing time, mirroring
    merge_gate_data.py:83-86.
    """
    for name, df in [("water", water_wide), ("rain", rain_wide), ("gate", gate_irregular)]:
        if "date" not in df.columns:
            raise ValueError(f"{name} 缺少 date 欄位。")

    anchor = water_wide.copy()
    anchor["date"] = pd.to_datetime(anchor["date"])
    anchor = anchor.sort_values("date").reset_index(drop=True)

    rain = rain_wide.copy()
    rain["date"] = pd.to_datetime(rain["date"])
    merged = anchor.merge(rain, on="date", how="left")

    gate = gate_irregular.copy()
    gate["date"] = pd.to_datetime(gate["date"])
    gate = gate.sort_values("date").reset_index(drop=True)
    gate["_gate_obs_time"] = gate["date"]

    gate_cols = [GATE_RENAME_MAP[col] for col in GATE_TARGET_COLUMNS]

    merged = pd.merge_asof(
        merged.sort_values("date"),
        gate,
        on="date",
        direction="backward",
    )

    backward_lag = merged["date"] - merged["_gate_obs_time"]
    stale_mask = (backward_lag > staleness_limit) | merged["_gate_obs_time"].isna()
    merged.loc[stale_mask, gate_cols] = float("nan")

    merged[gate_cols] = merged[gate_cols].clip(lower=0)
    merged = merged.drop(columns=["_gate_obs_time"])

    drop_present = [col for col in DROP_COLS if col in merged.columns]
    if drop_present:
        merged = merged.drop(columns=drop_present)

    return merged.reset_index(drop=True)


def assign_split_to_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Apply 70/10/(rest) segment-based split, sorted by SegmentStart.

    Mirrors data_provider/Data_Loader.py:106-113.
    """
    if segments.empty:
        return segments.assign(split=pd.Series(dtype=str))

    ordered = segments.sort_values("SegmentStart").reset_index(drop=True)
    nseg = len(ordered)
    if nseg < 3:
        raise ValueError(f"Segment 數量不足以切分 train/val/test：nseg={nseg}")

    train_n = max(1, int(nseg * TRAIN_FRAC))
    val_n = max(1, int(nseg * VAL_FRAC))
    if train_n + val_n >= nseg:
        val_n = max(1, nseg - train_n - 1)

    splits = ["train"] * train_n + ["val"] * val_n + ["test"] * (nseg - train_n - val_n)
    ordered["split"] = splits
    return ordered


def build_segments_meta(rain_df: pd.DataFrame, rain_value_col: str) -> pd.DataFrame:
    print(f"使用雨量欄位偵測 segment：{rain_value_col}")
    segments = segment_intervals(
        rain_df,
        time_col="ObsTime",
        value_col=rain_value_col,
        threshold=RAIN_THRESHOLD,
        gap_minutes=RAIN_GAP_MINUTES,
        min_duration_minutes=MIN_RAIN_SEGMENT_MINUTES,
    )
    windows = build_event_windows(segments)
    if windows.empty:
        raise ValueError("沒有偵測到任何降雨 segment。")

    meta = assign_split_to_segments(windows)
    meta = meta.loc[:, ["segment_id", "SegmentStart", "SegmentEnd", "WinStart", "WinEnd", "split"]]
    return meta


def main() -> None:
    client = SQLServerClient()

    print("=== 1/4 載入水位資料 ===")
    water_df = load_water_data(client)
    water_wide = build_water_wide(water_df)
    print(f"水位 wide: rows={len(water_wide):,}, range={water_wide['date'].min()} ~ {water_wide['date'].max()}")

    print("\n=== 2/4 載入雨量資料 ===")
    rain_df, rain_value_col = load_rain_data(client)
    rain_wide = build_rain_wide(rain_df)
    print(f"雨量 wide: rows={len(rain_wide):,}, range={rain_wide['date'].min()} ~ {rain_wide['date'].max()}")

    print("\n=== 3/4 載入閘門資料 ===")
    mapping_df = load_mapping(client)
    validate_mapping(mapping_df)
    gate_obs_df = load_gate_obs(client)
    gate_obs_df = gate_obs_df.merge(mapping_df, on=["PqId", "FullName"], how="inner")
    gate_irregular = build_gate_wide_irregular(gate_obs_df)
    print(f"閘門 (irregular): rows={len(gate_irregular):,}, range={gate_irregular['date'].min()} ~ {gate_irregular['date'].max()}")

    print("\n=== 4/4 合併並輸出 (water-anchored, merge_asof gate w/ 5min staleness) ===")
    all_minute_wide = merge_all_sources(water_wide, rain_wide, gate_irregular)
    ALL_MINUTE_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_minute_wide.to_csv(ALL_MINUTE_OUTPUT_CSV, index=False)
    print(f"all_minute_wide 輸出：{ALL_MINUTE_OUTPUT_CSV}")
    print(f"  rows={len(all_minute_wide):,}, cols={len(all_minute_wide.columns):,}")
    print(f"  range={all_minute_wide['date'].min()} ~ {all_minute_wide['date'].max()}")
    print(f"  columns: {list(all_minute_wide.columns)}")

    print("\n  NaN 比例 (每欄):")
    nan_pct = (all_minute_wide.isna().sum() / len(all_minute_wide) * 100).round(2)
    for col, pct in nan_pct.items():
        if pct > 0:
            print(f"    {col:<30} {pct}%")

    segments_meta = build_segments_meta(rain_df, rain_value_col)
    SEGMENTS_META_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    segments_meta.to_csv(SEGMENTS_META_OUTPUT_CSV, index=False)
    print(f"\nrain_segments_meta 輸出：{SEGMENTS_META_OUTPUT_CSV}")
    print(f"  segments={len(segments_meta):,}")
    print(f"  split counts:\n{segments_meta['split'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
