"""Customer 13-feature input-contract checks; no training/model dependency."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project_stage1.inference import InputValidationError, validate_input
from project_stage1.inference.validation import SCHEMA_PATH

SCHEMA = json.loads(SCHEMA_PATH.read_text())
COLUMNS = SCHEMA["columns"]
ANCHOR = SCHEMA["anchor_column"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def history():
    dates = pd.date_range("2026-09-26 11:01:00", periods=60, freq="min").strftime("%Y-%m-%d %H:%M:%S")
    return pd.DataFrame({
        "date": dates,
        **{name: np.arange(60, dtype=float) for name in COLUMNS[1:]},
    })


def check(frame):
    return validate_input(frame)


def assert_code(frame, code):
    with pytest.raises(InputValidationError) as error:
        check(frame)
    assert error.value.code == code
    return error.value


def test_valid_reorders_columns_uses_final_hl01_anchor_and_does_not_modify_input(history):
    history = history[list(reversed(history.columns))]
    before = history.copy(deep=True)
    got = check(history)
    assert list(got.history.columns) == COLUMNS
    assert got.origin == pd.Timestamp("2026-09-26 12:00:00")
    assert got.anchor_value == 59.0
    assert got.summary() == {"valid": True, "rows": 60, "origin": "2026-09-26 12:00:00", "anchor_value": 59.0}
    pd.testing.assert_frame_equal(history, before)
    got.history.loc[0, "HL02"] = -100
    pd.testing.assert_frame_equal(history, before)


def test_manual_example_file_passes():
    example = pd.read_csv(ROOT / "project_stage1/examples/example_minute.csv").tail(60).reset_index(drop=True)
    got = check(example)
    assert got.origin == pd.Timestamp("2024-10-31 11:00:00") and got.anchor_value == 1450.0


@pytest.mark.parametrize("n", [0, 59, 61])
def test_wrong_window_length(history, n):
    frame = history.iloc[:n] if n <= 60 else pd.concat([history, history.iloc[-1:]])
    assert_code(frame, "ROW_COUNT")


def test_dropped_minute_with_60_rows_is_not_silently_filled(history):
    dates = pd.to_datetime(history.date)
    history.loc[30:, "date"] = (dates[30:] + pd.Timedelta(minutes=1)).dt.strftime("%Y-%m-%d %H:%M:%S")
    error = assert_code(history, "TIME_GAP")
    assert error.row == 30


def test_duplicate_timestamp(history):
    history.loc[20, "date"] = history.loc[19, "date"]
    assert_code(history, "DUPLICATE_TIMES")


def test_reversed_time(history):
    assert_code(history.iloc[::-1], "TIME_ORDER")


@pytest.mark.parametrize("time", [
    "2026-09-26T11:01:00+08:00", "2026-09-26T03:01:00Z",   # timezone suffix
    "2026-09-26T11:01:00", "2026/09/26 11:01:00",          # other layouts that a loose parser accepts
    "2026-9-26 11:01:00", "2026-09-26 11:01", " 2026-09-26 11:01:00",
])
def test_only_the_declared_timestamp_format_is_accepted(history, time):
    history.loc[0, "date"] = time
    error = assert_code(history, "TIMESTAMP_FORMAT")
    assert error.row == 0


def test_naive_datetime_values_are_accepted(history):
    history["date"] = pd.to_datetime(history.date)
    assert check(history).origin == pd.Timestamp("2026-09-26 12:00:00")


def test_timezone_aware_datetime_values_are_rejected(history):
    history["date"] = pd.to_datetime(history.date).dt.tz_localize("Asia/Taipei")
    assert_code(history, "TIMESTAMP_FORMAT")


def test_missing_timestamp(history):
    history.loc[5, "date"] = None
    assert_code(history, "TIMESTAMP_MISSING")


def test_subminute_is_not_rounded(history):
    history.loc[13, "date"] = "2026-09-26 11:14:01"
    assert_code(history, "MINUTE_ALIGNMENT")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, None])
def test_nonfinite_anchor_or_feature_is_rejected(history, value):
    for field in (ANCHOR, "HL03"):
        modified = history.copy()
        modified.loc[13, field] = value
        error = assert_code(modified, "NONFINITE_VALUE")
        assert error.field == field and error.row == 13


def test_nullable_missing_feature_is_rejected(history):
    history["HL03"] = history.HL03.astype("Float64")
    history.loc[13, "HL03"] = pd.NA
    assert_code(history, "NONFINITE_VALUE")


@pytest.mark.parametrize("field", ["Past10Min", "Past1Hr"])
@pytest.mark.parametrize("value", [-99.0, -98.0, -0.1])
def test_rainfall_sentinels_and_negative_amounts_rejected(history, field, value):
    history.loc[13, field] = value
    error = assert_code(history, "NEGATIVE_RAINFALL")
    assert error.field == field and error.row == 13


def test_negative_gate_is_rejected(history):
    history.loc[13, "north_gate_opening_1"] = -0.1
    error = assert_code(history, "NEGATIVE_GATE")
    assert error.field == "north_gate_opening_1" and error.row == 13


@pytest.mark.parametrize("field", ["HL01", "HL03"])
def test_negative_water_level_is_rejected(history, field):
    history.loc[13, field] = -1.0
    error = assert_code(history, "NEGATIVE_WATER_LEVEL")
    assert error.field == field and error.row == 13


@pytest.mark.parametrize("dtype", [str, bool, complex])
def test_numeric_types(history, dtype):
    history["HL03"] = history.HL03.astype(dtype)
    assert_code(history, "NUMERIC_TYPE")


def test_missing_extra_and_legacy_now_columns(history):
    assert_code(history.drop(columns="HL03"), "MISSING_COLUMNS")
    assert_code(history.assign(segment_id=1), "EXTRA_COLUMNS")
    assert_code(history.assign(Now=123.4), "EXTRA_COLUMNS")


def test_duplicate_column_names(history):
    assert_code(pd.concat([history, history[["HL03"]]], axis=1), "DUPLICATE_COLUMNS")


def test_model_overflow(history):
    history.loc[0, "HL03"] = 1e300
    assert_code(history, "NUMERIC_RANGE")


def test_import_does_not_load_torch():
    code = "import sys; from project_stage1.inference import validate_input; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
