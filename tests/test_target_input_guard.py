"""Target-input regression checks; no run.py import, training, or file writes."""
import ast
import fnmatch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from data_provider.Data_Loader import Dataset_Custom


ROOT = Path(__file__).resolve().parents[1]
RUN_TREE = ast.parse((ROOT / "run.py").read_text())


@pytest.fixture
def cli():
    # Execute only pure argument functions. The training entry point is not run.
    names = {"_parse_csv_cols", "_expand_col_patterns", "_validate_target_inputs",
             "_configure_mix_model_args"}
    nodes = [n for n in RUN_TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"fnmatch": fnmatch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "run.py", "exec"), namespace)
    return namespace


def args(**overrides):
    values = dict(model="DLinearMix2", target="HL01", input_col="HL02,HL03",
                  exog_col=None, branch_in=None, exog_in=None, mix_in=None)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("name", ["input_col", "exog_col"])
@pytest.mark.parametrize("value", ["HL01", "HL02, HL01 ", ["HL02", " HL01 "]])
def test_cli_rejects_explicit_target(cli, name, value):
    with pytest.raises(ValueError, match="target='HL01'.*" + name):
        cli["_configure_mix_model_args"](args(**{name: value}))


def test_cli_rejects_wildcard_after_header_expansion(cli):
    columns = cli["_expand_col_patterns"](["HL*"], ["date", "HL01", "HL02", "HL03"])
    with pytest.raises(ValueError, match="target='HL01'.*input_col"):
        cli["_configure_mix_model_args"](args(input_col=",".join(columns)))


def test_cli_checks_the_selected_target_not_a_hardcoded_name(cli):
    with pytest.raises(ValueError, match="target='HL03'"):
        cli["_configure_mix_model_args"](args(target="HL03"))


def test_main_preflight_rejects_target_before_constant_drop(cli):
    main = next(n for n in RUN_TREE.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    preflight = next(n for n in main.body if isinstance(n, ast.If)
                     and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                             and x.func.id == "_expand_col_args" for x in ast.walk(n)))
    # Restrict execution to the preflight statement block, never main or training.
    a = args(input_col="HL*")
    def expand(config):
        config.input_col = ",".join(cli["_expand_col_patterns"]([config.input_col], ["HL01", "HL02"]))
    drop = Mock(side_effect=AssertionError("target must be rejected before constant removal"))
    ns = dict(cli, args=a, _expand_col_args=expand, _drop_constant_columns=drop)
    with pytest.raises(ValueError, match="target='HL01'"):
        exec(compile(ast.Module(body=preflight.body, type_ignores=[]), "run.py", "exec"), ns)
    drop.assert_not_called()


@pytest.mark.parametrize("exog", [None, "Past10Min,gate"])
def test_cli_accepts_non_target_channels(cli, exog):
    a = args(exog_col=exog)
    cli["_configure_mix_model_args"](a)
    assert a.input_cols == ["HL02", "HL03"]
    assert a.exog_cols == ([] if exog is None else ["Past10Min", "gate"])
    assert a.enc_in == 2 + len(a.exog_cols)


@pytest.mark.parametrize("model", ["DLinearMix", "DLinearMix2"])
@pytest.mark.parametrize("name", ["input_col", "exog_col"])
@pytest.mark.parametrize("flag", ["train", "val", "test"])
def test_direct_dataset_rejects_target_before_reading(monkeypatch, model, name, flag):
    read = Mock(side_effect=AssertionError("invalid target inputs must not read data"))
    monkeypatch.setattr(pd, "read_csv", read)
    kwargs = dict(input_col="HL02", exog_col=None)
    kwargs[name] = [" HL01 "]
    with pytest.raises(ValueError, match="target='HL01'.*" + name):
        Dataset_Custom(root_path="unused", size=[3, 2, 2], flag=flag,
                       model_name=model, target="HL01", **kwargs)
    read.assert_not_called()


def test_dataset_keeps_target_labels_and_current_anchor(monkeypatch):
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=18, freq="min"),
                          "segment_id": np.repeat([1, 2, 3], 6),
                          "HL01": np.arange(18, dtype=float) + 100,
                          "HL02": np.arange(18, dtype=float) + 200})
    split = pd.DataFrame({"segment_id": [1, 2, 3], "split": ["train", "val", "test"]})
    monkeypatch.setattr(pd, "read_csv", lambda path, **kw:
                        split.copy() if str(path) == "synthetic-split.csv" else frame.copy())
    ds = Dataset_Custom(root_path="unused", size=[3, 2, 2], model_name="DLinearMix2",
                        target="HL01", input_col="HL02", segment_col="segment_id",
                        split_file="synthetic-split.csv", freq="min", timeenc=1)
    x, y, *_ = ds[0]
    assert ds.x_cols == ["HL02"]
    assert len(ds) == 2
    assert x.shape == (3, 1)
    assert y.shape == (4, 1)
    raw_y = ds.inverse_transform(y)
    np.testing.assert_allclose(raw_y[:, 0], [101, 102, 103, 104])
    assert raw_y[ds.label_len - 1, 0] == pytest.approx(102)
