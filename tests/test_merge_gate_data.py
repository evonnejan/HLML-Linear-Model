"""Tests for merge_gate_data.py — uses synthetic DataFrames, no real CSV files."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from merge_gate_data import GATE_COLS, SEGMENT_COL, STALENESS_LIMIT, merge_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wl(dates, segment_ids, segment_starts=None):
    """Build a minimal water-level DataFrame."""
    if segment_starts is None:
        segment_starts = list(dates)
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        SEGMENT_COL: segment_ids,
        "SegmentStart": pd.to_datetime(segment_starts),
    })


def _make_gate(dates, values_per_col):
    """Build a minimal gate DataFrame. values_per_col: list of 7 lists."""
    data = {"date": pd.to_datetime(dates)}
    for col, vals in zip(GATE_COLS, values_per_col):
        data[col] = vals
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Task 2: Normal alignment
# ---------------------------------------------------------------------------

def test_normal_alignment_assigns_most_recent_gate_obs():
    """Each water-level row gets the gate obs that immediately preceded it."""
    wl = _make_wl(
        dates=["2024-01-01 10:00:00", "2024-01-01 10:01:00", "2024-01-01 10:02:00"],
        segment_ids=[1, 1, 1],
        segment_starts=["2024-01-01 10:00:00"] * 3,
    )
    gate = _make_gate(
        dates=["2024-01-01 09:59:30", "2024-01-01 10:01:30"],
        values_per_col=[[1.0, 2.0]] + [[0.0, 0.0]] * 6,
    )
    result = merge_gate(wl, gate)

    # 10:00 → nearest backward obs is 09:59:30 (lag=30s) → value 1.0
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01 10:00:00"), "north_gate_opening_1"].iloc[0] == 1.0
    # 10:01 → nearest backward obs is 09:59:30 (lag=90s, still < 5min) → value 1.0
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01 10:01:00"), "north_gate_opening_1"].iloc[0] == 1.0
    # 10:02 → nearest backward obs is 10:01:30 (lag=30s) → value 2.0
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01 10:02:00"), "north_gate_opening_1"].iloc[0] == 2.0


# ---------------------------------------------------------------------------
# Task 3: Staleness reset
# ---------------------------------------------------------------------------

def test_stale_gate_obs_becomes_nan():
    """Gate values are NaN when the nearest backward obs exceeds STALENESS_LIMIT."""
    wl = _make_wl(
        dates=["2024-01-01 10:00:00"],
        segment_ids=[1],
        segment_starts=["2024-01-01 09:00:00"],
    )
    # Gate obs is 16 minutes before the water-level row → exceeds 5-min limit
    gate = _make_gate(
        dates=["2024-01-01 09:44:00"],
        values_per_col=[[1.83]] + [[0.0]] * 6,
    )
    result = merge_gate(wl, gate, staleness_limit=pd.Timedelta("5min"))

    assert math.isnan(result["north_gate_opening_1"].iloc[0])


def test_fresh_gate_obs_is_not_reset():
    """Gate values are kept when backward lag is within the staleness limit."""
    wl = _make_wl(
        dates=["2024-01-01 10:00:00"],
        segment_ids=[1],
        segment_starts=["2024-01-01 09:00:00"],
    )
    # Gate obs is 3 minutes before the water-level row → within 5-min limit
    gate = _make_gate(
        dates=["2024-01-01 09:57:00"],
        values_per_col=[[1.83]] + [[0.0]] * 6,
    )
    result = merge_gate(wl, gate, staleness_limit=pd.Timedelta("5min"))

    assert result["north_gate_opening_1"].iloc[0] == pytest.approx(1.83)


# ---------------------------------------------------------------------------
# Task 4: Within-segment forward-fill
# ---------------------------------------------------------------------------

def test_within_segment_sensor_nan_is_ffilled():
    """Per-column sensor NaN within a segment is filled forward from the previous valid obs."""
    wl = _make_wl(
        dates=["2024-01-01 10:00:00", "2024-01-01 10:01:00", "2024-01-01 10:02:00"],
        segment_ids=[1, 1, 1],
        segment_starts=["2024-01-01 10:00:00"] * 3,
    )
    gate = _make_gate(
        dates=["2024-01-01 09:59:30", "2024-01-01 10:01:30"],
        # north_gate_opening_1: first obs=1.0, second obs=NaN (sensor outage)
        values_per_col=[[1.0, float("nan")]] + [[0.5, 0.5]] * 6,
    )
    result = merge_gate(wl, gate)

    # 10:02 → obs at 10:01:30 → north_1 = NaN from gate, ffill from 10:01 → 1.0
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01 10:02:00"), "north_gate_opening_1"].iloc[0] == pytest.approx(1.0)


def test_ffill_does_not_cross_segment_boundary():
    """Forward-fill from segment 1 does not propagate into segment 2."""
    wl = _make_wl(
        dates=[
            "2024-01-01 10:00:00",  # segment 1
            "2024-01-01 11:00:00",  # segment 2 — 60-min gap, stale gate
        ],
        segment_ids=[1, 2],
        segment_starts=["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
    )
    gate = _make_gate(
        dates=["2024-01-01 09:59:30"],  # only one obs, 60.5 min before segment 2 start
        values_per_col=[[1.83]] + [[0.0]] * 6,
    )
    result = merge_gate(wl, gate, staleness_limit=pd.Timedelta("5min"))

    # Segment 1: lag=30s → fresh → 1.83
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01 10:00:00"), "north_gate_opening_1"].iloc[0] == pytest.approx(1.83)
    # Segment 2: lag=60.5 min → stale → NaN; ffill within segment 2 only → still NaN
    assert math.isnan(result.loc[result["date"] == pd.Timestamp("2024-01-01 11:00:00"), "north_gate_opening_1"].iloc[0])


# ---------------------------------------------------------------------------
# Task 5: Structural guarantees
# ---------------------------------------------------------------------------

def test_row_count_is_preserved():
    """Output has exactly the same number of rows as the water-level input."""
    wl = _make_wl(
        dates=["2024-01-01 10:00:00", "2024-01-01 10:01:00", "2024-01-01 10:02:00"],
        segment_ids=[1, 1, 1],
        segment_starts=["2024-01-01 10:00:00"] * 3,
    )
    gate = _make_gate(
        dates=["2024-01-01 09:59:30"],
        values_per_col=[[1.0]] + [[0.0]] * 6,
    )
    result = merge_gate(wl, gate)
    assert len(result) == len(wl)


def test_no_future_gate_obs_assigned():
    """No gate observation in the future relative to the water-level row should be assigned."""
    wl = _make_wl(
        dates=["2024-01-01 10:00:00", "2024-01-01 10:01:00"],
        segment_ids=[1, 1],
        segment_starts=["2024-01-01 10:00:00"] * 2,
    )
    # Gate obs at 10:00:30 is AFTER the first wl row at 10:00:00
    gate = _make_gate(
        dates=["2024-01-01 09:59:00", "2024-01-01 10:00:30"],
        values_per_col=[[1.0, 2.0]] + [[0.0, 0.0]] * 6,
    )
    result = merge_gate(wl, gate)
    # 10:00:00 → only obs at 09:59:00 is valid (10:00:30 is future) → value=1.0
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01 10:00:00"), "north_gate_opening_1"].iloc[0] == pytest.approx(1.0)
    # 10:01:00 → obs at 10:00:30 is now valid (lag=30s) → value=2.0
    assert result.loc[result["date"] == pd.Timestamp("2024-01-01 10:01:00"), "north_gate_opening_1"].iloc[0] == pytest.approx(2.0)


def test_all_gate_cols_present_in_output():
    """All 7 gate columns appear in the output."""
    wl = _make_wl(["2024-01-01 10:00:00"], [1], ["2024-01-01 10:00:00"])
    gate = _make_gate(["2024-01-01 09:59:30"], [[1.0]] + [[0.0]] * 6)
    result = merge_gate(wl, gate)
    for col in GATE_COLS:
        assert col in result.columns
