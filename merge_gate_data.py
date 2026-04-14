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
