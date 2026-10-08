"""The delivered inference/run_forecast.py runs end to end with the real model bundle."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

import project_stage1.inference.run_forecast as run_forecast

ROOT = Path(__file__).resolve().parents[1]


def test_example_runs_and_appends_one_row_per_run(tmp_path):
    output = tmp_path / "outputs" / "hl01_forecast.csv"
    # Same entry point as `python -m project_stage1.inference.run_forecast`, but writing to tmp_path
    # so the test never touches the delivered inference/outputs/ folder.
    code = (f"import project_stage1.inference.run_forecast as r; r.OUTPUT_CSV = __import__('pathlib').Path({str(output)!r}); "
            "raise SystemExit(r.main())")
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr
    lines = [line.split() for line in result.stdout.strip().splitlines()]
    assert lines[0] == ["time", "2024-10-31", "11:00:00"]
    assert [line[0] for line in lines[1:16]] == [f"pred_{k}" for k in range(1, 16)]
    assert float(lines[1][1]) == 1450.0
    saved = pd.read_csv(output)
    assert list(saved.columns) == ["time", *[f"pred_{k}" for k in range(1, 16)]]
    assert len(saved) == 2 and saved.loc[0, "time"] == "2024-10-31 11:00:00" and saved.loc[0, "pred_1"] == 1450.0


def test_rejected_window_returns_exit_code_99_and_saves_nothing(tmp_path, monkeypatch, capsys):
    path = tmp_path / "missing_one_row.csv"
    pd.read_csv(run_forecast.EXAMPLE_MINUTE).tail(59).to_csv(path, index=False)
    monkeypatch.setattr(run_forecast, "EXAMPLE_MINUTE", path)
    monkeypatch.setattr(run_forecast, "OUTPUT_CSV", tmp_path / "outputs" / "hl01_forecast.csv")
    assert run_forecast.main() == 99
    assert "ROW_COUNT" in capsys.readouterr().out
    assert not (tmp_path / "outputs").exists()
