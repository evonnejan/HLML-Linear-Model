"""Merge irregular gate-opening observations into the 1-minute water-level/rainfall dataset.

Output: dataset/water_level_rain_gate_all.csv
  - Same rows and datetime index as water_level_rain_all4.csv
  - 7 gate columns appended; negative readings clipped to 0 (no physical meaning)
  - Stale gate values (backward lag > STALENESS_LIMIT) set to NaN
  - Remaining per-column sensor NaN filled forward within each segment
  - Constant-valued helper columns (e.g. StationId) dropped from the output
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

WATER_LEVEL_CSV = Path("dataset/water_level_rain_all4.csv")
GATE_CSV = Path("dataset/wra_cogate_obs_wide_gate_opening.csv")
OUTPUT_CSV = Path("dataset/water_level_rain_gate_all.csv")

GATE_COLS = [
    "north_gate_opening_1",
    "north_gate_opening_2",
    "north_gate_opening_3",
    "north_gate_opening_4",
    "south_gate_opening_1",
    "south_gate_opening_2",
    "south_gate_opening_3",
]

SEGMENT_COL = "segment_id"
STALENESS_LIMIT = pd.Timedelta("5min")
DROP_COLS = ["StationId"]


def merge_gate(
    wl_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    staleness_limit: pd.Timedelta = STALENESS_LIMIT,
) -> pd.DataFrame:
    """Merge gate observations into the water-level anchor DataFrame.

    Args:
        wl_df: Water-level/rainfall anchor. Must have columns: 'date',
               SEGMENT_COL, 'SegmentStart'. Row count is preserved.
        gate_df: Gate observations with 'date' and GATE_COLS columns.
        staleness_limit: Maximum allowed backward lag. Rows exceeding this
                         get NaN gate values before within-segment ffill.

    Returns:
        Copy of wl_df with GATE_COLS appended.
    """
    wl = wl_df.copy()
    gate = gate_df.copy()

    wl["date"] = pd.to_datetime(wl["date"])
    gate["date"] = pd.to_datetime(gate["date"])
    wl["SegmentStart"] = pd.to_datetime(wl["SegmentStart"])

    # Carry the gate observation timestamp as a value column so we can
    # compute backward lag after the merge.
    gate["_gate_obs_time"] = gate["date"]

    wl_sorted = wl.sort_values("date").reset_index(drop=True)
    gate_sorted = gate.sort_values("date").reset_index(drop=True)

    merged = pd.merge_asof(
        wl_sorted,
        gate_sorted,
        on="date",
        direction="backward",
    )

    # ------------------------------------------------------------------ #
    #  Staleness reset                                                     #
    # ------------------------------------------------------------------ #
    backward_lag = merged["date"] - merged["_gate_obs_time"]
    stale_mask = (backward_lag > staleness_limit) | merged["_gate_obs_time"].isna()
    merged.loc[stale_mask, GATE_COLS] = float("nan")

    # ------------------------------------------------------------------ #
    #  Within-segment forward-fill                                         #
    # ------------------------------------------------------------------ #
    merged[GATE_COLS] = (
        merged.groupby(SEGMENT_COL, sort=False)[GATE_COLS]
        .transform(lambda col: col.ffill())
    )

    # ------------------------------------------------------------------ #
    #  Clip negative gate values to 0                                     #
    #  Negative gate openings have no physical meaning; treat them as     #
    #  fully-closed. NaN entries are preserved.                           #
    # ------------------------------------------------------------------ #
    merged[GATE_COLS] = merged[GATE_COLS].clip(lower=0)

    merged = merged.drop(columns=["_gate_obs_time"])
    return merged


def _build_report(result: pd.DataFrame) -> str:
    """Build a human-readable text report of gate-column NaN statistics."""
    total_rows = len(result)
    nan_counts = result[GATE_COLS].isna().sum()
    nan_pct = (nan_counts / total_rows * 100).round(2)

    # Per-segment NaN analysis
    has_any_nan = result[GATE_COLS].isna().any(axis=1)
    seg_total = result.groupby(SEGMENT_COL).size().rename("total_rows")
    seg_nan = result[has_any_nan].groupby(SEGMENT_COL).size().rename("nan_rows")
    seg_stats = seg_total.to_frame().join(seg_nan, how="left").fillna({"nan_rows": 0})
    seg_stats["nan_rows"] = seg_stats["nan_rows"].astype(int)
    seg_stats["nan_pct"] = (seg_stats["nan_rows"] / seg_stats["total_rows"] * 100).round(2)

    segments_with_nan = seg_stats[seg_stats["nan_rows"] > 0].sort_index()
    n_clean_segs = (seg_stats["nan_rows"] == 0).sum()

    lines = []
    lines.append("=" * 60)
    lines.append("Gate Data Merge Report")
    lines.append("=" * 60)
    lines.append(f"Total rows          : {total_rows:,}")
    lines.append(f"Total segments      : {len(seg_stats):,}")
    lines.append(f"Segments with no NaN: {n_clean_segs:,} of {len(seg_stats):,}")
    lines.append("")

    lines.append("--- Missing value % per gate column ---")
    for col in GATE_COLS:
        lines.append(f"  {col:<30} {nan_counts[col]:>6} NaN  ({nan_pct[col]:.2f}%)")
    lines.append("")

    lines.append("--- Segments containing NaN values ---")
    if len(segments_with_nan) == 0:
        lines.append("  (none)")
    else:
        lines.append(f"  {'segment_id':<12} {'total_rows':>10} {'nan_rows':>10} {'nan_%':>8}")
        lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8}")
        for seg_id, row in segments_with_nan.iterrows():
            lines.append(
                f"  {seg_id:<12} {row['total_rows']:>10,} {row['nan_rows']:>10,} {row['nan_pct']:>7.2f}%"
            )
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    print(f"Loading water-level data from: {WATER_LEVEL_CSV}")
    wl = pd.read_csv(WATER_LEVEL_CSV)

    print(f"Loading gate data from: {GATE_CSV}")
    gate = pd.read_csv(GATE_CSV)

    print(f"Merging ({len(wl):,} water-level rows × {len(gate):,} gate rows)...")
    result = merge_gate(wl, gate)

    assert len(result) == len(wl), (
        f"Row count mismatch: input={len(wl)}, output={len(result)}"
    )

    # Sort by segment_id first, then by date within each segment.
    result = result.sort_values([SEGMENT_COL, "date"]).reset_index(drop=True)

    # Drop helper columns that carry no model signal (constant across the dataset).
    drop_present = [c for c in DROP_COLS if c in result.columns]
    if drop_present:
        result = result.drop(columns=drop_present)
        print(f"Dropped constant helper columns: {drop_present}")

    # ------------------------------------------------------------------ #
    #  Console summary                                                     #
    # ------------------------------------------------------------------ #
    nan_counts = result[GATE_COLS].isna().sum()
    nan_pct = (nan_counts / len(result) * 100).round(1)
    has_any_nan = result[GATE_COLS].isna().any(axis=1)
    seg_nan_count = (result[has_any_nan].groupby(SEGMENT_COL).size() > 0).sum()
    total_segs = result[SEGMENT_COL].nunique()
    n_clean_segs = total_segs - seg_nan_count

    print("\n--- Merge validation ---")
    print(f"Output rows             : {len(result):,}  (matches input: ✓)")
    print(f"Segments with no NaN    : {n_clean_segs:,} of {total_segs:,}")
    print(f"\nGate column NaN counts after merge:")
    for col in GATE_COLS:
        print(f"  {col:<30} {nan_counts[col]:>6} NaN  ({nan_pct[col]}%)")

    # ------------------------------------------------------------------ #
    #  Text report                                                         #
    # ------------------------------------------------------------------ #
    report_text = _build_report(result)
    report_path = OUTPUT_CSV.with_suffix(".report.txt")
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to : {report_path}")
    print(report_text)

    # ------------------------------------------------------------------ #
    #  Save CSV                                                            #
    # ------------------------------------------------------------------ #
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"CSV saved to    : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
