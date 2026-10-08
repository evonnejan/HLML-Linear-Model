"""Training-search safeguards: test data cannot enter validation-only search."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from exp.exp_Main2 import Exp_Main
from scripts.run_delivery_sweep import score
from utils.tools import adjust_learning_rate


def test_validation_only_never_loads_test_and_reloads_best(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x[:, -1:, :]).expand(-1, 3, -1)

    requested = []
    class TinyExperiment(Exp_Main):
        def _build_model(self):
            return TinyModel()

        def _get_data(self, flag):
            requested.append(flag)
            if flag == "test":
                raise AssertionError("test dataset must not be read")
            x = torch.arange(48, dtype=torch.float32).reshape(8, 3, 2) / 10
            y = x[:, :, :1] * 0.5
            data = TensorDataset(x, y, torch.zeros_like(x), torch.zeros_like(y))
            return data, DataLoader(data, batch_size=4)

    args = SimpleNamespace(model="TinyModel", use_gpu=False, use_multi_gpu=False, use_amp=False,
        run_dir=str(tmp_path), learning_rate=0.001, criterion="mse", lradj="type1",
        train_only=False, validation_only=True, patience=2, train_epochs=3, batch_size=4,
        early_stop_metric="corr", features="S", pred_len=3, output_attention=False)
    exp = TinyExperiment(args)
    exp.train("tiny")
    assert requested == ["train", "val"]
    assert exp.best_epoch in [1, 2, 3]
    assert (tmp_path / "checkpoints/training_history.csv").exists()
    saved = torch.load(tmp_path / "checkpoints/checkpoint.pth", weights_only=True)
    for name, value in exp.model.state_dict().items():
        torch.testing.assert_close(value, saved[name])


def test_fixed_full_fit_uses_only_train_and_saves_final_epoch(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x[:, -1:, :]).expand(-1, 3, -1)

    requested = []

    class TinyExperiment(Exp_Main):
        def _build_model(self):
            return TinyModel()

        def _get_data(self, flag):
            requested.append(flag)
            if flag != "train":
                raise AssertionError("fixed full fit must not read validation or test data")
            x = torch.arange(48, dtype=torch.float32).reshape(8, 3, 2) / 10
            y = x[:, :, :1] * 0.5
            data = TensorDataset(x, y, torch.zeros_like(x), torch.zeros_like(y))
            return data, DataLoader(data, batch_size=4)

        def _run_train_epoch(self, train_loader, model_optim, criterion, scaler, epoch, total_epochs):
            # Make the saved state prove that it is the final epoch rather than
            # a checkpoint selected by the lowest training loss.
            with torch.no_grad():
                self.model.linear.weight.fill_(epoch + 1)
                self.model.linear.bias.fill_(epoch + 1)
            return float(total_epochs - epoch)

    args = SimpleNamespace(model="TinyModel", use_gpu=False, use_multi_gpu=False, use_amp=False,
        run_dir=str(tmp_path), learning_rate=0.001, criterion="mse", lradj="type1",
        train_only=True, validation_only=False, train_epochs=3, batch_size=4,
        features="S", pred_len=3, output_attention=False)
    exp = TinyExperiment(args)
    exp.train_fixed_epochs()

    assert requested == ["train"]
    assert exp.epochs_trained == exp.final_epoch == exp.best_epoch == 3
    assert [row["epoch"] for row in exp.training_history] == [1, 2, 3]
    assert (tmp_path / "checkpoints/training_history.csv").is_file()
    saved = torch.load(tmp_path / "checkpoints/checkpoint.pth", weights_only=True)
    assert torch.equal(saved["linear.weight"], torch.full_like(saved["linear.weight"], 3.0))
    for name, value in exp.model.state_dict().items():
        torch.testing.assert_close(value, saved[name])


def test_scoring_checks_shape_and_finite_values():
    true = np.arange(30, dtype=float).reshape(10, 3, 1)
    assert score(true, true)["rmse"] == 0
    with pytest.raises(ValueError, match="Invalid"):
        score(true[:-1], true)
    bad = true.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        score(bad, true)
    with pytest.raises(ValueError, match="correlation"):
        score(np.ones_like(true), true)


def test_exp_default_preserves_original_curve_and_accepts_matched_gamma():
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=.003)
    args = SimpleNamespace(lradj="exp", learning_rate=.003)
    for epoch in (1, 2, 8, 80):
        adjust_learning_rate(optimizer, epoch, args)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(max(.003 * .85 ** epoch, .00003))
    args.exp_decay_gamma = .92
    adjust_learning_rate(optimizer, 8, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.003 * .92 ** 8)


@pytest.mark.parametrize("warmup", [3, 8])
def test_matched_warmup_reaches_base_then_same_decay(warmup):
    optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=.0003)
    args = SimpleNamespace(lradj="warmup_exp", learning_rate=.003,
                           warmup_epochs=warmup, lr_decay_gamma=.85, lr_floor_ratio=.01)
    adjust_learning_rate(optimizer, warmup-1, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.003)
    adjust_learning_rate(optimizer, warmup, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.003*.85)
