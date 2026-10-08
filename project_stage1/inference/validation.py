"""預測前檢查一份整理好的 60 分鐘窗口。

這個模組不補值、不重新取樣、不標準化，也不做預測；
它無法只從數值判斷單位是否正確，或資料是否過舊。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

SCHEMA_PATH = Path(__file__).with_name("input_schema.json")

NEGATIVE_CODES = {
    "water_level": "NEGATIVE_WATER_LEVEL",
    "rainfall": "NEGATIVE_RAINFALL",
    "gate": "NEGATIVE_GATE",
}


class InputValidationError(ValueError):
    """檢查沒通過時丟出的錯誤，包含錯誤碼與出問題的欄位、列號。"""

    def __init__(self, code: str, message: str, *, field: str | None = None, row: int | None = None):
        self.code, self.field, self.row = code, field, row
        super().__init__(message)

    def as_dict(self):
        return {"code": self.code, "message": str(self), "field": self.field, "row": self.row}


@dataclass(frozen=True)
class ValidatedWindow:
    """通過檢查的窗口副本：欄位依 schema 排序，數值維持原始單位。"""

    history: pd.DataFrame
    anchor_value: float
    origin: pd.Timestamp

    def summary(self):
        return dict(valid=True, rows=len(self.history),
                    origin=self.origin.strftime("%Y-%m-%d %H:%M:%S"), anchor_value=self.anchor_value)


def _timestamp(value, fmt, field, row):
    if value is None or value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        raise InputValidationError("TIMESTAMP_MISSING", "Timestamp is missing", field=field, row=row)
    if isinstance(value, str):
        # strptime alone accepts unpadded fields such as "2026-9-26"; the round trip rejects them.
        try:
            t = pd.Timestamp(datetime.strptime(value, fmt))
        except ValueError as exc:
            raise InputValidationError("TIMESTAMP_FORMAT", "Write timestamps as YYYY-MM-DD HH:MM:SS (Taiwan time, no timezone)",
                                       field=field, row=row) from exc
        if t.strftime(fmt) != value:
            raise InputValidationError("TIMESTAMP_FORMAT", "Write timestamps as YYYY-MM-DD HH:MM:SS (Taiwan time, no timezone)",
                                       field=field, row=row)
    elif isinstance(value, datetime):
        t = pd.Timestamp(value)
        if t.tzinfo is not None:
            raise InputValidationError("TIMESTAMP_FORMAT", "Timestamp must not include a timezone or UTC offset", field=field, row=row)
    else:
        raise InputValidationError("TIMESTAMP_FORMAT", "Write timestamps as YYYY-MM-DD HH:MM:SS (Taiwan time, no timezone)",
                                   field=field, row=row)
    if t.second or t.microsecond or t.nanosecond:
        raise InputValidationError("MINUTE_ALIGNMENT", "Timestamp must be exactly on a minute boundary; do not round implicitly", field=field, row=row)
    return t


def validate_input(history: pd.DataFrame) -> ValidatedWindow:
    """檢查窗口；不通過時丟出 InputValidationError，通過時回傳整理後的副本。

    數值不標準化，時間維持台灣時間。列必須已經由舊到新排好，這裡不會補任何值。
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(history, pd.DataFrame):
        raise InputValidationError("INPUT_TYPE", "history must be a pandas DataFrame")
    if history.columns.has_duplicates:
        raise InputValidationError("DUPLICATE_COLUMNS", "Duplicate column names are not allowed")
    date_col = schema["timestamp_column"]
    columns = schema["columns"]
    missing = [c for c in columns if c not in history.columns]
    if missing:
        raise InputValidationError("MISSING_COLUMNS", f"Missing required columns: {missing}")
    extra = [c for c in history.columns if c not in columns]
    if extra:
        raise InputValidationError("EXTRA_COLUMNS", f"Unexpected columns: {extra}; pass only the declared columns")
    if len(history) != schema["seq_len"]:
        raise InputValidationError("ROW_COUNT", f"Expected exactly {schema['seq_len']} rows, got {len(history)}")
    fmt = schema["timestamp_format"]
    dates = pd.DatetimeIndex([_timestamp(t, fmt, date_col, i) for i, t in enumerate(history[date_col])])
    if dates.has_duplicates:
        raise InputValidationError("DUPLICATE_TIMES", "Duplicate minute timestamps", field=date_col)
    if not dates.is_monotonic_increasing:
        raise InputValidationError("TIME_ORDER", "Rows must be in strictly increasing time order", field=date_col)
    intervals = dates[1:] - dates[:-1]
    bad = np.flatnonzero(intervals != pd.Timedelta(seconds=schema["frequency_seconds"]))
    if len(bad):
        i = int(bad[0] + 1)
        raise InputValidationError("TIME_GAP", f"Expected one-minute spacing between rows {i-1} and {i}", field=date_col, row=i)
    negative_code = {name: NEGATIVE_CODES[group]
                     for group, names in schema["nonnegative_columns"].items() for name in names}
    for name in columns:
        if name == date_col:
            continue
        col = history[name]
        if not is_numeric_dtype(col.dtype) or is_bool_dtype(col.dtype) or is_complex_dtype(col.dtype):
            raise InputValidationError("NUMERIC_TYPE", "Column must be a real numeric column, not strings or booleans", field=name)
        values = col.to_numpy(dtype=np.float64, na_value=np.nan)
        bad = np.flatnonzero(~np.isfinite(values))
        if len(bad):
            raise InputValidationError("NONFINITE_VALUE", "Missing, NaN or infinite value; upstream preparation is required", field=name, row=int(bad[0]))
        if name in negative_code:
            negative = np.flatnonzero(values < 0)
            if len(negative):
                raise InputValidationError(negative_code[name], "Value must be nonnegative; prepare negative source values upstream",
                                           field=name, row=int(negative[0]))
        if np.any(np.abs(values) > np.finfo(np.float32).max):
            raise InputValidationError("NUMERIC_RANGE", "Value exceeds model float32 representable range", field=name)
    clean = history.loc[:, columns].copy(deep=True).reset_index(drop=True)
    clean[date_col] = dates
    return ValidatedWindow(clean, float(clean[schema["anchor_column"]].iloc[-1]), dates[-1])
