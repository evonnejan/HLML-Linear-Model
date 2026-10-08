"""Predictor wiring: bundle export, scaling, anchored output and input rejection."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from project_stage1.inference import InputValidationError, Predictor
from project_stage1.inference.model import Model
from project_stage1.inference.predictor import anchor
from project_stage1.full_fit.export_model_bundle import export_bundle

ROOT = Path(__file__).resolve().parents[1]
BRANCH = ["HL02", "HL03", "HL04", "HL05", "HL06"]
EXOG = ["Past10Min", "Past1Hr", "north_gate_opening_1", "north_gate_opening_2", "north_gate_opening_3",
        "south_gate_opening_1", "south_gate_opening_2", "south_gate_opening_3"]
ARGS = dict(model="DLinearMix2", seq_len=60, pred_len=15, dlinear_kernel_size=25, flatten_fusion=True,
            fusion_hidden_dim=32, exog_emb_dim=16, dropout=0.1,
            input_col=",".join(BRANCH), exog_col=",".join(EXOG), input_cols=BRANCH, exog_cols=EXOG)


@pytest.fixture
def example():
    return pd.read_csv(ROOT / "project_stage1/examples/example_minute.csv").tail(60).reset_index(drop=True)


@pytest.fixture
def run_dir(tmp_path, example):
    """A finished-run folder laid out like the research runs (random weights)."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "run_args.json").write_text(json.dumps(ARGS))
    x = example[BRANCH + EXOG].to_numpy(dtype=float)
    np.savez(run / "scaler.npz", x_mean=x.mean(axis=0), x_scale=x.std(axis=0) + 1.0,
             y_mean=np.array([1400.0]), y_scale=np.array([120.0]), features=np.array(BRANCH + EXOG))
    torch.manual_seed(0)
    torch.save(Model(SimpleNamespace(**ARGS)).state_dict(), run / "checkpoints/checkpoint.pth")
    return run


@pytest.fixture
def predictor(run_dir, tmp_path):
    return Predictor(export_bundle(run_dir, tmp_path / "bundle"))


def test_bundle_contents(run_dir, tmp_path):
    bundle = export_bundle(run_dir, tmp_path / "bundle")
    assert sorted(p.name for p in bundle.iterdir()) == ["checkpoint.pth", "config.json", "scaler.npz"]
    config = json.loads((bundle / "config.json").read_text())
    assert config["branch_features"] == BRANCH and config["exog_features"] == EXOG


def test_predict_returns_t_and_15_anchored_values(predictor, example):
    result = predictor.predict(example)
    assert list(result) == ["time", "pred"]
    assert result["time"] == "2024-10-31 11:00:00"         # t, the last input row
    assert len(result["pred"]) == 15
    assert result["pred"][0] == 1450.0                     # first step equals HL01(t) exactly
    assert all(np.isfinite(result["pred"]))


def test_predict_matches_manual_pipeline(predictor, run_dir, example):
    """Scale -> model -> inverse scale -> anchored, done by hand from the run's own files."""
    with np.load(run_dir / "scaler.npz") as s:
        x = (example[BRANCH + EXOG].to_numpy(dtype=float) - s["x_mean"]) / s["x_scale"]
        model = Model(SimpleNamespace(**ARGS))
        model.load_state_dict(torch.load(run_dir / "checkpoints/checkpoint.pth"))
        model.eval()
        with torch.no_grad():
            raw = model(torch.tensor(x, dtype=torch.float32)[None]).numpy().ravel() * s["y_scale"][0] + s["y_mean"][0]
    expected = 1450.0 + raw - raw[0]
    got = np.array(predictor.predict(example)["pred"])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_hl01_history_does_not_change_the_prediction(predictor, example):
    """HL01 is not a model input; only its last value (the anchor) matters."""
    changed = example.copy()
    changed.loc[:58, "HL01"] = 0.0
    assert predictor.predict(changed) == predictor.predict(example)


def test_invalid_window_is_rejected_before_the_model(predictor, example):
    with pytest.raises(InputValidationError) as error:
        predictor.predict(example.iloc[:59])
    assert error.value.code == "ROW_COUNT"


def test_anchor_keeps_changes_and_moves_start():
    np.testing.assert_allclose(anchor(np.array([120.0, 122.0, 125.0]), 100.0), [100.0, 102.0, 105.0])


def test_scaler_feature_order_mismatch_is_refused(run_dir, tmp_path):
    bundle = export_bundle(run_dir, tmp_path / "bundle")
    with np.load(bundle / "scaler.npz") as s:
        arrays = dict(s)
    arrays["features"] = arrays["features"][::-1]
    np.savez(bundle / "scaler.npz", **arrays)
    with pytest.raises(ValueError, match="feature order"):
        Predictor(bundle)


def test_export_refuses_mismatched_run(run_dir, tmp_path):
    args = json.loads((run_dir / "run_args.json").read_text())
    args["input_cols"] = BRANCH[::-1]
    (run_dir / "run_args.json").write_text(json.dumps(args))
    with pytest.raises(ValueError, match="feature order"):
        export_bundle(run_dir, tmp_path / "bundle")
