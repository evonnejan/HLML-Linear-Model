"""forecast_from_table.forecast() picks t from the minute table as documented."""
import pandas as pd
import pytest

import project_stage1.inference.forecast_from_table as fft
from project_stage1.inference import InputValidationError, Predictor

MINUTE = pd.read_csv(fft.EXAMPLE_MINUTE)
HL01 = dict(zip(pd.to_datetime(MINUTE["date"]), MINUTE["HL01"]))


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def at(clock):
    return f"2024-10-31 {clock}"


def mark_unmeasured(monkeypatch, *clocks):
    monkeypatch.setattr(fft, "read_hl01_changes", lambda start, end: {pd.Timestamp(at(c)) for c in clocks})


def test_latest_measured_minute_becomes_t(predictor):
    result = fft.forecast(predictor, now=at("11:05:42"))
    assert result["time"] == at("11:00:00") and result["pred"][0] == 1450.0 and len(result["pred"]) == 15


def test_steps_back_past_unmeasured_hl01(predictor, monkeypatch):
    mark_unmeasured(monkeypatch, "11:00:00", "10:59:00")
    result = fft.forecast(predictor, now=at("11:05:00"))
    assert result["time"] == at("10:58:00") and result["pred"][0] == HL01[pd.Timestamp(at("10:58:00"))]


def test_no_measured_minute_within_14_minutes(predictor, monkeypatch):
    mark_unmeasured(monkeypatch, *[f"10:{m}:00" for m in range(51, 60)], "11:00:00")
    with pytest.raises(fft.NoForecastTimeError):
        fft.forecast(predictor, now=at("11:05:00"))  # 10:51–11:00 all unmeasured; 10:50 is too old


def test_t_may_be_14_but_not_15_minutes_before_trigger(predictor):
    assert fft.forecast(predictor, now=at("11:14:00"))["time"] == at("11:00:00")
    with pytest.raises(fft.NoForecastTimeError):
        fft.forecast(predictor, now=at("11:15:00"))  # table stuck at 11:00


def test_given_t_is_used_without_age_check(predictor):
    result = fft.forecast(predictor, t=at("10:30:00"), now="2030-01-01 00:00:00")
    assert result["time"] == at("10:30:00") and result["pred"][0] == HL01[pd.Timestamp(at("10:30:00"))]


def test_given_t_with_unmeasured_hl01_does_not_step_back(predictor, monkeypatch):
    mark_unmeasured(monkeypatch, "10:30:00")
    with pytest.raises(fft.NoForecastTimeError, match="not a measured value"):
        fft.forecast(predictor, t=at("10:30:00"))


def test_given_t_missing_from_table(predictor):
    with pytest.raises(fft.NoForecastTimeError, match="not in the minute table"):
        fft.forecast(predictor, t=at("12:00:00"))


def test_given_t_without_59_earlier_rows_fails_validation(predictor):
    with pytest.raises(InputValidationError) as info:
        fft.forecast(predictor, t=at("10:00:00"))  # table starts at 09:31
    assert info.value.code == "ROW_COUNT"


def test_unordered_rows_are_sorted_by_time(predictor, monkeypatch):
    shuffled = MINUTE.sample(frac=1, random_state=0).reset_index(drop=True)
    monkeypatch.setattr(fft, "read_minute_table", lambda start, end: shuffled)
    expected = predictor.predict(MINUTE.tail(60).reset_index(drop=True))
    assert fft.forecast(predictor, now=at("11:05:00")) == expected


def test_main_appends_result(tmp_path, monkeypatch, capsys):
    output = tmp_path / "outputs" / "hl01_forecast.csv"
    monkeypatch.setattr(fft, "OUTPUT_CSV", output)
    assert fft.main() == 0 and fft.main() == 0
    saved = pd.read_csv(output)
    assert list(saved.columns) == ["time", *[f"pred_{k}" for k in range(1, 16)]]
    assert len(saved) == 2 and saved.loc[0, "time"] == at("11:00:00") and saved.loc[0, "pred_1"] == 1450.0


def test_main_returns_98_when_no_t(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fft, "OUTPUT_CSV", tmp_path / "outputs" / "hl01_forecast.csv")
    monkeypatch.setattr(fft, "read_hl01_changes", lambda start, end: set(pd.to_datetime(MINUTE["date"])))
    assert fft.main() == 98
    assert "No minute with a measured HL01" in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()


def test_main_returns_99_when_window_fails(tmp_path, monkeypatch, capsys):
    bad = MINUTE.copy()
    bad.loc[len(bad) - 1, "Past10Min"] = -99.0
    monkeypatch.setattr(fft, "OUTPUT_CSV", tmp_path / "outputs" / "hl01_forecast.csv")
    monkeypatch.setattr(fft, "read_minute_table", lambda start, end: bad)
    assert fft.main() == 99
    assert "NEGATIVE_RAINFALL" in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()
