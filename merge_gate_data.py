"""Merge irregular gate-opening observations into the 1-minute water-level/rainfall dataset.

Output: dataset/water_level_rain_gate_all.csv
  - Same rows and datetime index as water_level_rain_all4.csv
  - 7 gate columns appended
  - Stale gate values (backward lag > STALENESS_LIMIT) set to NaN
  - Remaining per-column sensor NaN filled forward within each segment
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

    merged = merged.drop(columns=["_gate_obs_time"])
    return merged
