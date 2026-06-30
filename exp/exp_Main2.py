"""
exp_Main2.py — Training / testing experiment class.

Output paths consumed by downstream scripts (do NOT rename or reformat):
  outputs/
    metrics.npy                       — [mae, mse, rmse, mape, mspe, rse]  float32
    pred.npy / true.npy               — shape (N, pred_len, 1), inverse-transformed to real scale
    metrics_horizon.csv               — columns: horizon, MSE, RMSE, MAE, Corr
    mse_horizon.png
    metrics_segment.csv               — columns: segment, num_windows, horizon, MSE, RMSE, MAE, Corr
    segment_horizon_rank.csv          — columns: segment, horizon, num_points, mse, corr
    segment_horizon_points.csv.gz     — columns: segment, window_idx, horizon, target_time,
                                                  true, pred, abs_err, sq_err
    meeting.csv                       — top/bottom summary of rank table
  checkpoints/
    checkpoint.pth
  run_overview.txt                    — human-readable run summary
  (run_args.json / run_cmd.txt are written by run.py, not here)
"""

import csv
import importlib
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_provider.Data_Factory import data_provider
from exp.exp_Basic import Exp_Basic
from models import DLinear, DLinearMix, DLinearMix2, Linear, NLinear
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:
    tqdm = importlib.import_module("tqdm").tqdm
except Exception:
    tqdm = None

# ---------------------------------------------------------------------------
# Supported models — register new models here.
# analyze_full_inference.py uses "Linear" in model_name to decide the forward
# signature, so this dict also serves as the canonical whitelist of valid names.
# ---------------------------------------------------------------------------
_MODEL_DICT = {
    "DLinear":    DLinear,
    "DLinearMix": DLinearMix,
    "NLinear":    NLinear,
    "Linear":     Linear,
    "DLinearMix2": DLinearMix2,
}


class Exp_Main(Exp_Basic):

    def __init__(self, args):
        super().__init__(args)
        self.use_amp = bool(
            getattr(self.args, "use_amp", False) and self.device.type == "cuda"
        )
        if getattr(self.args, "use_amp", False) and not self.use_amp:
            print("[WARNING] AMP is only effective on CUDA; current device is CPU/MPS, skipping AMP.")

    # ------------------------------------------------------------------ #
    #  Directory helpers                                                   #
    # ------------------------------------------------------------------ #

    def _checkpoint_dir(self) -> str:
        path = (
            os.path.join(self.args.run_dir, "checkpoints")
            if hasattr(self.args, "run_dir")
            else os.path.join(self.args.checkpoints, getattr(self.args, "setting", ""))
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _outputs_dir(self, subdir: str = "outputs") -> str:
        path = (
            os.path.join(self.args.run_dir, subdir)
            if hasattr(self.args, "run_dir")
            else os.path.join("./results", getattr(self.args, "setting", ""), subdir)
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _plots_dir(self) -> str:
        path = (
            os.path.join(self.args.run_dir, "plots")
            if hasattr(self.args, "run_dir")
            else os.path.join("./test_results", getattr(self.args, "setting", ""))
        )
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------ #
    #  Model / data / optimizer                                           #
    # ------------------------------------------------------------------ #

    def _build_model(self):
        if self.args.model not in _MODEL_DICT:
            raise ValueError(
                f"Unknown model '{self.args.model}'. "
                f"Supported models: {list(_MODEL_DICT.keys())}"
            )
        model = _MODEL_DICT[self.args.model].Model(self.args).float()

        if getattr(self.args, "use_multi_gpu", False) and self.device.type == "cuda":
            devices_text = str(getattr(self.args, "devices", str(getattr(self.args, "gpu", 0))))
            device_ids = [int(x.strip()) for x in devices_text.split(",") if x.strip()]
            model = nn.DataParallel(model, device_ids=device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        name = str(getattr(self.args, "criterion", "mse")).lower()
        if name == "huber":
            beta = float(getattr(self.args, "huber_beta", 1.0))
            print(f"  Loss: SmoothL1Loss(beta={beta})  [Huber — robust to outlier events]")
            return nn.SmoothL1Loss(beta=beta)
        print("  Loss: MSELoss")
        return nn.MSELoss()

    # ------------------------------------------------------------------ #
    #  Forward — single entry point shared by train / vali / test        #
    # ------------------------------------------------------------------ #

    def _forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass and return the raw output tensor."""
        if self.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(batch_x)
        return self.model(batch_x)

    def _slice_output(self, outputs: torch.Tensor, batch_y: torch.Tensor):
        """Trim outputs and targets to the last pred_len steps and the correct feature dim."""
        f_dim = -1 if self.args.features == "MS" else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        return outputs, batch_y

    def forward_batch(
        self,
        batch_x: torch.Tensor,
        batch_x_mark: torch.Tensor,
        dec_inp: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> torch.Tensor:
        """Unified forward pass for inference.

        All models currently in _MODEL_DICT use the simple (batch_x,) signature.
        The transformer-style path is kept here so that future non-linear models
        added to _MODEL_DICT with a different signature only require updating
        this method, not any analysis script.
        """
        if self.args.model in _MODEL_DICT:
            return self.model(batch_x)
        # Transformer-style path — reserved for future architectures.
        if getattr(self.args, "output_attention", False):
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def vali(self, vali_loader, criterion=None) -> dict:
        """Run the loader once and return {'mse', 'mae', 'corr'}.

        corr is the mean across horizons of Pearson correlation between pred
        and true at each horizon (horizons with zero variance are skipped).
        criterion is accepted for backward compatibility but unused.
        """
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, *_ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self._forward(batch_x)
                outputs, batch_y = self._slice_output(outputs, batch_y)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        self.model.train()
        if len(preds) == 0:
            return {"mse": float("nan"), "mae": float("nan"), "corr": float("nan")}
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        diff = preds - trues
        mse = float((diff ** 2).mean())
        mae = float(np.abs(diff).mean())
        # per-horizon corr -> mean (assumes shape [N, P, 1])
        p2 = preds.reshape(preds.shape[0], -1)
        t2 = trues.reshape(trues.shape[0], -1)
        corrs = []
        for h in range(p2.shape[1]):
            if p2[:, h].std() > 0 and t2[:, h].std() > 0:
                corrs.append(float(np.corrcoef(p2[:, h], t2[:, h])[0, 1]))
        mean_corr = float(np.mean(corrs)) if corrs else float("nan")
        return {"mse": mse, "mae": mae, "corr": mean_corr}

    @staticmethod
    def _select_score(metrics: dict, metric_name: str) -> float:
        """Convert a metric dict to a 'lower is better' scalar for EarlyStopping."""
        name = metric_name.lower()
        if name == "mse":
            return metrics["mse"]
        if name == "mae":
            return metrics["mae"]
        if name == "corr":
            # Higher corr is better; negate so EarlyStopping minimises.
            return -metrics["corr"]
        raise ValueError(f"Unknown early_stop_metric: {metric_name}")

    # ------------------------------------------------------------------ #
    #  Train                                                               #
    # ------------------------------------------------------------------ #

    def train(self, setting: str):
        train_data, train_loader = self._get_data("train")
        if not self.args.train_only:
            _, vali_loader = self._get_data("val")
            _, test_loader = self._get_data("test")

        checkpoint_path = self._checkpoint_dir()
        model_optim = self._select_optimizer()
        # For warm-up schedules, override optimizer init LR to warm-up start so
        # epoch 1 trains at 0.1*base instead of the optimizer's default (base).
        if str(getattr(self.args, 'lradj', '')).startswith('warmup'):
            warmup_start_lr = self.args.learning_rate * 0.1
            for pg in model_optim.param_groups:
                pg['lr'] = warmup_start_lr
            print(f'Warm-up: setting initial LR to {warmup_start_lr}')
        criterion = self._select_criterion()
        # Dual checkpoint: primary follows --early_stop_metric (writes 'checkpoint.pth',
        # preserves backward compat with downstream analysis scripts). Alt opportunistically
        # saves the best by the OTHER axis (MSE or Corr) — never triggers patience itself.
        metric_name = getattr(self.args, "early_stop_metric", "mse").lower()
        primary_metric = metric_name
        alt_metric = "corr" if metric_name in ("mse", "mae") else "mse"
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True,
            filename='checkpoint.pth',
        )
        alt_early_stopping = EarlyStopping(
            patience=10**9, verbose=False,
            filename='checkpoint_alt.pth',
        )
        self._primary_metric_name = primary_metric
        self._alt_metric_name = alt_metric
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        total_epochs = self.args.train_epochs
        epochs_trained = 0

        print(f"\n{'─'*60}")
        print(f"  Training start  |  model: {self.args.model}  |  device: {self.device}")
        print(f"  epochs={total_epochs}  batch={self.args.batch_size}  lr={self.args.learning_rate}")
        print(f"  Primary checkpoint: best-by-{primary_metric}  |  Alt checkpoint: best-by-{alt_metric}")
        print(f"{'─'*60}")

        for epoch in range(total_epochs):
            epochs_trained = epoch + 1
            train_loss = self._run_train_epoch(
                train_loader, model_optim, criterion, scaler, epoch, total_epochs
            )

            if not self.args.train_only:
                vali_metrics = self.vali(vali_loader)
                test_metrics = self.vali(test_loader)
                vali_score = self._select_score(vali_metrics, primary_metric)
                alt_score = self._select_score(vali_metrics, alt_metric)
                self._print_epoch_summary(
                    epoch, total_epochs, train_loss, vali_metrics, test_metrics, primary_metric
                )
                early_stopping(vali_score, self.model, checkpoint_path)
                alt_early_stopping(alt_score, self.model, checkpoint_path)
            else:
                self._print_epoch_summary(epoch, total_epochs, train_loss)
                early_stopping(train_loss, self.model, checkpoint_path)

            if early_stopping.early_stop:
                print(f"\n  Early stopping triggered (patience={self.args.patience} exhausted, primary={primary_metric}).")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        print(f"{'─'*60}")
        print(f"  Training done  |  epochs run: {epochs_trained}")
        print(f"{'─'*60}\n")

        best_model_path = os.path.join(checkpoint_path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        self.epochs_trained = epochs_trained
        return self.model

    def _run_train_epoch(
        self, train_loader, model_optim, criterion, scaler, epoch: int, total_epochs: int
    ) -> float:
        """Run one epoch and return the average training loss."""
        self.model.train()
        losses = []

        # Use tqdm with proper macOS configuration
        pbar = (
            tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"  Epoch {epoch+1:>3}/{total_epochs}",
                leave=False,
                ncols=80,
                file=sys.stderr,
                mininterval=0.1,  # Update every 100ms for smooth animation
                smoothing=0.5,  # Reduce smoothing for faster response
                disable=(tqdm is None),
            )
            if tqdm is not None
            else train_loader
        )

        for batch_x, batch_y, *_ in pbar:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            model_optim.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x)
                    outputs, batch_y_s = self._slice_output(outputs, batch_y)
                    # SmoothL1Loss on MPS requires contiguous tensors; slicing in
                    # _slice_output returns non-contiguous views.
                    loss = criterion(outputs.contiguous(), batch_y_s.contiguous())
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                outputs = self.model(batch_x)
                outputs, batch_y_s = self._slice_output(outputs, batch_y)
                loss = criterion(outputs.contiguous(), batch_y_s.contiguous())
                loss.backward()
                model_optim.step()

            loss_val = loss.item()
            losses.append(loss_val)

        return float(np.mean(losses))

    @staticmethod
    def _print_epoch_summary(
        epoch: int, total: int, train: float,
        vali=None, test=None, metric_name: str = "mse",
    ):
        parts = [f"Epoch {epoch+1:>3}/{total}", f"train_mse={train:.6f}"]
        if vali is not None:
            if isinstance(vali, dict):
                parts.append(
                    f"vali[mse={vali['mse']:.6f} mae={vali['mae']:.6f} corr={vali['corr']:+.4f}]"
                )
            else:
                parts.append(f"vali={vali:.6f}")
        if test is not None:
            if isinstance(test, dict):
                parts.append(
                    f"test[mse={test['mse']:.6f} mae={test['mae']:.6f} corr={test['corr']:+.4f}]"
                )
            else:
                parts.append(f"test={test:.6f}")
        if vali is not None and isinstance(vali, dict):
            parts.append(f"[saving by:{metric_name}]")
        print("  " + "  │  ".join(parts))

    # ------------------------------------------------------------------ #
    #  Test                                                                #
    # ------------------------------------------------------------------ #

    def test(self, setting: str, test: int = 0):
        """Evaluate trained model on test set.

        Iterates over available checkpoints (primary 'checkpoint.pth' and optional
        alt 'checkpoint_alt.pth'). Each checkpoint produces its own outputs/* dir
        and its own row in the summary CSV (with `checkpoint_type` field).
        """
        test_data, test_loader = self._get_data("test")
        ckpt_dir = self._checkpoint_dir()

        primary_metric = getattr(self, "_primary_metric_name",
                                 str(getattr(self.args, "early_stop_metric", "mse")).lower())
        alt_metric = getattr(self, "_alt_metric_name",
                             "corr" if primary_metric in ("mse", "mae") else "mse")

        ckpts_to_eval: list[tuple[str, str, str]] = []
        primary_ckpt = os.path.join(ckpt_dir, "checkpoint.pth")
        if os.path.exists(primary_ckpt):
            ckpts_to_eval.append((f"best_{primary_metric}", primary_ckpt, "outputs"))
        alt_ckpt = os.path.join(ckpt_dir, "checkpoint_alt.pth")
        if os.path.exists(alt_ckpt):
            ckpts_to_eval.append((f"best_{alt_metric}", alt_ckpt, "outputs_alt"))

        if not ckpts_to_eval:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

        for ckpt_label, ckpt_path, outputs_subdir in ckpts_to_eval:
            print(f"\n{'='*60}")
            print(f"  Testing checkpoint: {ckpt_label}  ({os.path.basename(ckpt_path)} → {outputs_subdir}/)")
            print(f"{'='*60}")
            self.model.load_state_dict(torch.load(ckpt_path))
            self._current_ckpt_type = ckpt_label
            self._do_test_eval(setting, test_data, test_loader, outputs_subdir)

    def _do_test_eval(self, setting: str, test_data, test_loader, outputs_subdir: str):
        """Inner test body: inference + metrics + saves for a single checkpoint."""
        # ---- diagnostic-only short-circuit: print params/FLOPs and return ----
        if getattr(self.args, "test_flop", False):
            try:
                batch_x, *_ = next(iter(test_loader))
            except StopIteration:
                print("  [test_flop] test loader is empty; nothing to measure.")
                return
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            return

        # ---- inference ----
        preds, trues, persists, segment_ids_all = self._run_inference(test_data, test_loader)

        # ---- inverse-transform to real scale ----
        if test_data.scale:
            shape = preds.shape
            preds    = test_data.inverse_transform(preds.reshape(-1, shape[-1])).reshape(shape)
            trues    = test_data.inverse_transform(trues.reshape(-1, shape[-1])).reshape(shape)
            persists = test_data.inverse_transform(persists.reshape(-1, shape[-1])).reshape(shape)

        outputs_dir = self._outputs_dir(subdir=outputs_subdir)

        # ---- global metrics ----
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        corr_arr = np.array(corr, dtype=float).reshape(-1)
        corr_mean = float(np.nanmean(corr_arr)) if corr_arr.size > 0 else float("nan")
        print(f"\n  [Test]  MSE={mse:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}  Corr(mean)={corr_mean:.6f}")

        # ---- persistence baseline + improvement vs model ----
        persist_mae, persist_mse, persist_rmse, _, _, _, persist_corr = metric(persists, trues)
        persist_corr_arr = np.array(persist_corr, dtype=float).reshape(-1)
        improve_mse_pct = (persist_mse - mse) / persist_mse * 100.0 if persist_mse > 0 else 0.0
        improve_mae_pct = (persist_mae - mae) / persist_mae * 100.0 if persist_mae > 0 else 0.0
        print(f"  [Persist] MSE={persist_mse:.6f}  MAE={persist_mae:.6f}")
        print(f"  [Improve] MSE: {improve_mse_pct:+.2f}%  MAE: {improve_mae_pct:+.2f}%  "
              f"(positive = model beats persistence)")
        self._persist_overall = {
            "persist_MSE": float(persist_mse),
            "persist_MAE": float(persist_mae),
            "persist_corr_mean": float(np.nanmean(persist_corr_arr)) if persist_corr_arr.size > 0 else float("nan"),
            "improve_MSE_pct": float(improve_mse_pct),
            "improve_MAE_pct": float(improve_mae_pct),
        }

        # ---- horizon-wise metrics ----
        self._save_horizon_metrics(preds, trues, corr_arr, outputs_dir)
        self._save_persistence_horizon(preds, trues, persists, outputs_dir)

        # ---- per-segment metrics (always produced; fallback to single segment when needed) ----
        if len(segment_ids_all) > 0:
            segment_ids = np.concatenate(segment_ids_all, axis=0)
            if len(segment_ids) != preds.shape[0]:
                print("  [WARNING] segment_ids length does not match preds; falling back to single segment (all zeros).")
                segment_ids = np.zeros(preds.shape[0], dtype=np.int64)
        else:
            print("  segment_col not set; falling back to single segment (all zeros).")
            segment_ids = np.zeros(preds.shape[0], dtype=np.int64)
        self._save_segment_metrics(preds, trues, segment_ids, test_data, outputs_dir)

        # ---- save core artifacts ----
        np.save(os.path.join(outputs_dir, "metrics.npy"),
                np.array([float(mae), float(mse), float(rmse),
                          float(mape), float(mspe), float(rse), corr_mean]))
        np.save(os.path.join(outputs_dir, "pred.npy"), preds)
        np.save(os.path.join(outputs_dir, "true.npy"), trues)
        # x_last (= 最後一個 input 時間點的目標值, 已 inverse-transform) 供事後計算
        # 錨定校正 MSE 用; 見 compute_anchored_mse.py。persists 沿 horizon 廣播,
        # 故每個 window 的 x_last = persists[:, 0, :]。
        np.save(os.path.join(outputs_dir, "persist.npy"), persists)

        # ---- summary CSV (cross-run comparison) ----
        self._append_summary_csv(setting, mae, mse, rmse, corr_arr, outputs_dir)

        # ---- run_overview.txt ----
        self._write_run_overview(outputs_dir)

    # ------------------------------------------------------------------ #
    #  Test helpers                                                        #
    # ------------------------------------------------------------------ #

    def _run_inference(self, test_data, test_loader):
        """Run full inference loop; return (preds, trues, persists, segment_ids_all).

        `persists` is the persistence baseline prediction (HL01 at the last input
        timestep, broadcast across all pred_len horizons). All three arrays share
        the same shape and ordering so per-horizon model-vs-persistence comparison
        is direct.
        """
        preds, trues, persists, segment_ids_all = [], [], [], []
        sample_ptr = 0
        label_len = int(self.args.label_len)
        pred_len  = int(self.args.pred_len)
        f_dim     = -1 if self.args.features == "MS" else 0
        if label_len < 1:
            raise ValueError(
                f"Persistence baseline requires label_len >= 1 (got {label_len}); "
                f"the last input timestep is read from batch_y[label_len-1]."
            )

        self.model.eval()
        with torch.no_grad():
            pbar = (
                tqdm(test_loader, total=len(test_loader), desc="  Inference",
                     leave=False, ncols=80, file=sys.stderr,
                     mininterval=0.1, smoothing=0.5,
                     disable=(tqdm is None))
                if tqdm is not None else test_loader
            )
            for batch_x, batch_y, *_ in pbar:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Persistence baseline (before slicing batch_y to pred horizons):
                # take target value at the last input timestep, broadcast across all
                # future horizons. Same feature-dim slicing as trues for alignment.
                last_input_target = batch_y[:, label_len - 1:label_len, f_dim:]
                persist = last_input_target.expand(-1, pred_len, -1).contiguous()

                outputs = self._forward(batch_x)
                outputs, batch_y = self._slice_output(outputs, batch_y)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                persists.append(persist.detach().cpu().numpy())
                preds.append(pred)
                trues.append(true)

                # track segment ids per sample
                batch_size_now = pred.shape[0]
                if (hasattr(test_data, "window_segment_ids")
                        and test_data.window_segment_ids is not None):
                    segment_ids_all.append(
                        test_data.window_segment_ids[sample_ptr: sample_ptr + batch_size_now]
                    )
                sample_ptr += batch_size_now

        return (
            np.concatenate(preds, axis=0),
            np.concatenate(trues, axis=0),
            np.concatenate(persists, axis=0),
            segment_ids_all,
        )

    def _save_horizon_metrics(
        self,
        preds: np.ndarray,
        trues: np.ndarray,
        corr_arr: np.ndarray,
        outputs_dir: str,
    ):
        """Compute and save horizon-wise metrics (CSV with MSE/RMSE/MAE/Corr + PNG of MSE)."""
        sq_err = np.square(preds - trues)
        abs_err = np.abs(preds - trues)
        if sq_err.ndim == 3:
            horizon_mse = sq_err.mean(axis=(0, 2))
            horizon_mae = abs_err.mean(axis=(0, 2))
        else:
            horizon_mse = sq_err.mean(axis=0)
            horizon_mae = abs_err.mean(axis=0)
        horizon_rmse = np.sqrt(horizon_mse)
        H = len(horizon_mse)

        corr_use = np.array(corr_arr, dtype=float).reshape(-1)
        if corr_use.size != H:
            corr_use = np.full(H, np.nan, dtype=float)

        df = pd.DataFrame({
            "horizon": np.arange(1, H + 1, dtype=np.int64),
            "MSE": horizon_mse.astype(float),
            "RMSE": horizon_rmse.astype(float),
            "MAE": horizon_mae.astype(float),
            "Corr": corr_use.astype(float),
        })
        df.to_csv(os.path.join(outputs_dir, "metrics_horizon.csv"), index=False, encoding="utf-8-sig")

        plt.figure(figsize=(8, 4))
        plt.plot(df["horizon"], df["MSE"], marker="o")
        plt.xlabel("Horizon (t+k)")
        plt.ylabel("MSE")
        plt.title("Horizon-wise MSE")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, "mse_horizon.png"), dpi=180)
        plt.close()

    def _save_persistence_horizon(
        self,
        preds: np.ndarray,
        trues: np.ndarray,
        persists: np.ndarray,
        outputs_dir: str,
    ):
        """Save per-horizon model vs persistence comparison.

        Columns: horizon, model_RMSE, persist_RMSE, RMSE_improve_pct,
                 model_MAE,  persist_MAE,  MAE_improve_pct,
                 model_Corr, persist_Corr
        Positive improve_pct means model beats persistence at that horizon.
        """
        def _per_h(arr_pred, arr_true):
            sq = np.square(arr_pred - arr_true)
            ab = np.abs(arr_pred - arr_true)
            if sq.ndim == 3:
                mse = sq.mean(axis=(0, 2))
                mae = ab.mean(axis=(0, 2))
            else:
                mse = sq.mean(axis=0)
                mae = ab.mean(axis=0)
            return mse, mae

        def _per_h_corr(arr_pred, arr_true):
            # Use feature dim 0 if present
            p = arr_pred[..., 0] if arr_pred.ndim == 3 else arr_pred
            t = arr_true[..., 0] if arr_true.ndim == 3 else arr_true
            H = p.shape[1]
            out = np.full(H, np.nan, dtype=float)
            for h in range(H):
                ph, th = p[:, h], t[:, h]
                if ph.std() == 0 or th.std() == 0:
                    continue
                out[h] = np.corrcoef(ph, th)[0, 1]
            return out

        m_mse, m_mae = _per_h(preds, trues)
        p_mse, p_mae = _per_h(persists, trues)
        m_corr = _per_h_corr(preds, trues)
        p_corr = _per_h_corr(persists, trues)

        with np.errstate(divide="ignore", invalid="ignore"):
            mse_imp = np.where(p_mse > 0, (p_mse - m_mse) / p_mse * 100.0, 0.0)
            mae_imp = np.where(p_mae > 0, (p_mae - m_mae) / p_mae * 100.0, 0.0)

        H = len(m_mse)
        df = pd.DataFrame({
            "horizon":         np.arange(1, H + 1, dtype=np.int64),
            "model_MSE":       m_mse.astype(float),
            "persist_MSE":     p_mse.astype(float),
            "MSE_improve_pct": mse_imp.astype(float),
            "model_MAE":       m_mae.astype(float),
            "persist_MAE":     p_mae.astype(float),
            "MAE_improve_pct": mae_imp.astype(float),
            "model_Corr":      m_corr.astype(float),
            "persist_Corr":    p_corr.astype(float),
        })
        df.to_csv(os.path.join(outputs_dir, "persistence_metrics.csv"),
                  index=False, encoding="utf-8-sig")

        # Companion plot: MSE improvement vs horizon (positive = model wins)
        plt.figure(figsize=(8, 4))
        plt.bar(df["horizon"], df["MSE_improve_pct"],
                color=["#d62728" if v < 0 else "#2ca02c" for v in df["MSE_improve_pct"]])
        plt.axhline(0, color="black", lw=0.8)
        plt.xlabel("Horizon (t+k)")
        plt.ylabel("MSE improvement over persistence (%)")
        plt.title("Model vs Persistence (positive = model wins)")
        plt.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, "persistence_improve.png"), dpi=180)
        plt.close()

        # Stash per-horizon improvements for summary CSV (h1 + last horizon)
        self._persist_h1_improve = float(mse_imp[0])
        self._persist_hN_improve = float(mse_imp[-1])

    def _save_segment_metrics(
        self,
        preds: np.ndarray,
        trues: np.ndarray,
        segment_ids: np.ndarray,
        test_data,
        outputs_dir: str,
    ):
        """
        Compute and save:
          - metrics_segment.csv
          - segment_horizon_rank.csv
          - segment_horizon_points.csv.gz
          - meeting.csv
        Column format is fully aligned with downstream scripts
        (analyze_full_inference / analyze_best_models_overview).
        """
        sq_err = np.square(preds - trues)
        abs_err = np.abs(preds - trues)

        # 2-D view (N, pred_len) used for per-point visualisation
        if preds.ndim == 3 and preds.shape[2] == 1:
            pred_viz = preds[:, :, 0]
            true_viz = trues[:, :, 0]
        elif preds.ndim == 3:
            pred_viz = preds.mean(axis=2)
            true_viz = trues.mean(axis=2)
        else:
            pred_viz = preds
            true_viz = trues

        start_indices = (
            test_data.valid_starts
            if test_data.valid_starts is not None
            else np.arange(pred_viz.shape[0], dtype=np.int64)
        )

        combined_rows = []
        points_rows = []
        rank_rows = []
        per_seg_mse: list[float] = []  # overall-MSE per segment, used for median / worst-decile aggregation

        for seg_id in pd.unique(segment_ids):
            mask = segment_ids == seg_id
            seg_sq = sq_err[mask]
            seg_abs = abs_err[mask]
            seg_pred = preds[mask]
            seg_true = trues[mask]

            # Helper: Pearson corr, NaN-safe (returns NaN when either side is constant or too short)
            def _corr(p_arr, t_arr):
                p = np.asarray(p_arr, dtype=float).ravel()
                t = np.asarray(t_arr, dtype=float).ravel()
                if p.size < 2 or t.size < 2 or p.std() == 0 or t.std() == 0:
                    return float("nan")
                return float(np.corrcoef(p, t)[0, 1])

            # overall (across all windows × all horizons within this segment)
            overall_mse = float(seg_sq.mean())
            overall_mae = float(seg_abs.mean())
            overall_corr = _corr(seg_pred, seg_true)
            per_seg_mse.append(overall_mse)
            combined_rows.append({
                "segment": seg_id,
                "num_windows": int(mask.sum()),
                "horizon": "all",
                "MSE": overall_mse,
                "RMSE": float(np.sqrt(overall_mse)),
                "MAE": overall_mae,
                "Corr": overall_corr,
            })

            # per-horizon
            if seg_sq.ndim == 3:
                seg_h_mse = seg_sq.mean(axis=(0, 2))
                seg_h_mae = seg_abs.mean(axis=(0, 2))
            else:
                seg_h_mse = seg_sq.mean(axis=0)
                seg_h_mae = seg_abs.mean(axis=0)
            for h_idx, (h_mse, h_mae) in enumerate(zip(seg_h_mse, seg_h_mae), start=1):
                if seg_pred.ndim == 3:
                    h_pred = seg_pred[:, h_idx - 1, 0]
                    h_true = seg_true[:, h_idx - 1, 0]
                else:
                    h_pred = seg_pred[:, h_idx - 1]
                    h_true = seg_true[:, h_idx - 1]
                combined_rows.append({
                    "segment": seg_id,
                    "num_windows": int(mask.sum()),
                    "horizon": int(h_idx),
                    "MSE": float(h_mse),
                    "RMSE": float(np.sqrt(h_mse)),
                    "MAE": float(h_mae),
                    "Corr": _corr(h_pred, h_true),
                })

            # points（for visualize / analyze_full_inference）
            for local_idx in np.where(mask)[0]:
                s_begin = int(start_indices[local_idx])
                for h_idx in range(self.args.pred_len):
                    t_idx = s_begin + self.args.seq_len + h_idx
                    if t_idx >= len(test_data.dates):
                        continue
                    y_t = float(true_viz[local_idx, h_idx])
                    y_p = float(pred_viz[local_idx, h_idx])
                    points_rows.append({
                        "segment": seg_id,
                        "window_idx": int(local_idx),
                        "horizon": int(h_idx + 1),
                        "target_time": pd.to_datetime(test_data.dates[t_idx]).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "true": y_t,
                        "pred": y_p,
                        "abs_err": abs(y_t - y_p),
                        "sq_err": (y_t - y_p) ** 2,
                    })

        # ---- metrics_segment.csv ----
        pd.DataFrame(combined_rows).to_csv(
            os.path.join(outputs_dir, "metrics_segment.csv"),
            index=False, encoding="utf-8-sig",
        )

        # ---- per-segment MSE aggregates (median / worst-decile / mean) ----
        # Stashed for summary CSV so heavy-event domination is visible at a glance.
        if per_seg_mse:
            arr = np.array(per_seg_mse, dtype=float)
            self._segment_mse_median = float(np.median(arr))
            self._segment_mse_worst_decile = float(np.percentile(arr, 90))
            self._segment_mse_mean = float(arr.mean())
            print(f"  [Segment]  n_segs={len(arr)}  MSE  median={self._segment_mse_median:.2f}"
                  f"  mean={self._segment_mse_mean:.2f}  worst-decile(p90)={self._segment_mse_worst_decile:.2f}")

        # ---- segment_horizon_points.csv.gz ----
        points_df = pd.DataFrame(points_rows)
        points_df.to_csv(
            os.path.join(outputs_dir, "segment_horizon_points.csv.gz"),
            index=False, encoding="utf-8-sig", compression="gzip",
        )

        # ---- segment_horizon_rank.csv ----
        for (seg_id, h_id), grp in points_df.groupby(["segment", "horizon"]):
            t_arr = grp["true"].to_numpy(dtype=float)
            p_arr = grp["pred"].to_numpy(dtype=float)
            corr_val = float(np.corrcoef(t_arr, p_arr)[0, 1]) if len(t_arr) >= 2 else np.nan
            rank_rows.append({
                "segment": seg_id,
                "horizon": int(h_id),
                "num_points": int(len(grp)),
                "mse": float(np.mean((p_arr - t_arr) ** 2)),
                "corr": corr_val,
            })

        rank_df = pd.DataFrame(rank_rows).sort_values(["mse", "corr"], ascending=[True, False])
        rank_df.to_csv(
            os.path.join(outputs_dir, "segment_horizon_rank.csv"),
            index=False, encoding="utf-8-sig",
        )

        # ---- meeting.csv ----
        meeting_rows = []
        meeting_rows += (rank_df.dropna(subset=["mse"])
                         .sort_values("mse", ascending=True).head(50)
                         .assign(category="best_mse").to_dict(orient="records"))
        meeting_rows += (rank_df.dropna(subset=["mse"])
                         .sort_values("mse", ascending=False).head(50)
                         .assign(category="worst_mse").to_dict(orient="records"))
        meeting_rows += (rank_df.dropna(subset=["corr"])
                         .sort_values("corr", ascending=False).head(10)
                         .assign(category="best_corr").to_dict(orient="records"))
        # "worst_corr" = least skill (|corr| closest to 0). A strongly negative
        # corr is "model learned the inverse pattern" — useful info but not the
        # same as "model has no signal". Ranking by absolute value catches the
        # truly skill-less segments.
        worst_corr_df = rank_df.dropna(subset=["corr"]).assign(
            abs_corr=lambda d: d["corr"].abs()
        ).sort_values("abs_corr", ascending=True).head(10).drop(columns=["abs_corr"])
        meeting_rows += worst_corr_df.assign(category="worst_corr").to_dict(orient="records")
        pd.DataFrame(meeting_rows).to_csv(
            os.path.join(outputs_dir, "meeting.csv"),
            index=False, encoding="utf-8-sig",
        )

    @staticmethod
    def _count_parameters(model: nn.Module) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _append_summary_csv(self, setting: str, mae, mse, rmse, corr_arr, outputs_dir: str):
        """Append a row to model-specific summary CSV for cross-run comparison."""
        csv_path = getattr(
            self.args, "summary_csv", os.path.join(os.getcwd(), "results_summary.csv")
        )
        # Do not call makedirs when dirname is empty (i.e. csv_path is just a filename in the current directory), to avoid FileNotFoundError.
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        corr_arr = np.array(corr_arr, dtype=float).reshape(-1)
        N = corr_arr.size
        corr_field_names = [f"corr_h{i}" for i in range(1, N + 1)]

        fieldnames = [
            "time", "setting", "model", "input_col", "exog_col", "output_col",
            "group",
            "seq_len", "pred_len", "stride", "kernel_size",
            "flatten_fusion", "exog_emb_dim", "fusion_hidden_dim", "dropout",
            "num_params",
            "epochs", "batch_size", "lr", "lradj",
            "warmup_epochs", "lr_decay_gamma", "lr_floor_ratio",
            "early_stop_metric", "patience", "seed",
            "criterion", "huber_beta", "checkpoint_type",
            "MSE", "RMSE", "MAE",
            "MSE_persegment_median", "MSE_persegment_mean", "MSE_persegment_worst_decile",
            "persist_MSE", "persist_MAE", "persist_corr_mean",
            "improve_MSE_pct", "improve_MAE_pct",
            "improve_h1_MSE_pct", "improve_hN_MSE_pct",
        ] + corr_field_names

        run_dir_abs = os.path.abspath(getattr(self.args, "run_dir", ""))

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "setting": run_dir_abs,
            "model": self.args.model,
            "input_col": getattr(self.args, "input_col", None) or self.args.target,
            "exog_col": getattr(self.args, "exog_col", None) or "",
            "output_col": self.args.target,
            "group": str(getattr(self.args, "segment_col", "") or ""),
            "seq_len": int(self.args.seq_len),
            "pred_len": int(self.args.pred_len),
            "stride": int(self.args.stride_train),
            "kernel_size": str(getattr(self.args, "dlinear_kernel_size", "")),
            "flatten_fusion": str(getattr(self.args, "flatten_fusion", "")),
            "exog_emb_dim": str(getattr(self.args, "exog_emb_dim", "")),
            "fusion_hidden_dim": str(getattr(self.args, "fusion_hidden_dim", "")),
            "dropout": str(getattr(self.args, "dropout", "")),
            "num_params": self._count_parameters(self.model),
            "epochs": int(getattr(self, "epochs_trained", self.args.train_epochs)),
            "batch_size": int(self.args.batch_size),
            "lr": float(self.args.learning_rate),
            "lradj": str(getattr(self.args, "lradj", "")),
            "warmup_epochs": str(getattr(self.args, "warmup_epochs", "")),
            "lr_decay_gamma": str(getattr(self.args, "lr_decay_gamma", "")),
            "lr_floor_ratio": str(getattr(self.args, "lr_floor_ratio", "")),
            "early_stop_metric": str(getattr(self.args, "early_stop_metric", "")),
            "patience": int(getattr(self.args, "patience", 0)),
            "seed": str(getattr(self.args, "seed", "")),
            "criterion": str(getattr(self.args, "criterion", "")),
            "huber_beta": str(getattr(self.args, "huber_beta", "")),
            "checkpoint_type": str(getattr(self, "_current_ckpt_type", "")),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MSE_persegment_median": float(getattr(self, "_segment_mse_median", float("nan"))),
            "MSE_persegment_mean":   float(getattr(self, "_segment_mse_mean",   float("nan"))),
            "MSE_persegment_worst_decile": float(getattr(self, "_segment_mse_worst_decile", float("nan"))),
            "persist_MSE": float(getattr(self, "_persist_overall", {}).get("persist_MSE", float("nan"))),
            "persist_MAE": float(getattr(self, "_persist_overall", {}).get("persist_MAE", float("nan"))),
            "persist_corr_mean": float(getattr(self, "_persist_overall", {}).get("persist_corr_mean", float("nan"))),
            "improve_MSE_pct": float(getattr(self, "_persist_overall", {}).get("improve_MSE_pct", float("nan"))),
            "improve_MAE_pct": float(getattr(self, "_persist_overall", {}).get("improve_MAE_pct", float("nan"))),
            "improve_h1_MSE_pct": float(getattr(self, "_persist_h1_improve", float("nan"))),
            "improve_hN_MSE_pct": float(getattr(self, "_persist_hN_improve", float("nan"))),
        }
        for i, c in enumerate(corr_arr, start=1):
            row[f"corr_h{i}"] = float(c)

        # Schema-safe append: if existing CSV has a different header (e.g. older
        # schema before new fields were added), migrate the file by re-writing
        # with the current header, padding old rows with empty values for
        # missing columns. This prevents silent column misalignment.
        existing_rows = []
        existing_header = None
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                try:
                    existing_header = next(reader)
                except StopIteration:
                    existing_header = None
                if existing_header is not None:
                    for parts in reader:
                        existing_rows.append(dict(zip(existing_header, parts)))

        header_matches = existing_header == fieldnames
        write_mode = "a" if header_matches else "w"
        with open(csv_path, write_mode, newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
            if not header_matches:
                writer.writeheader()
                for old in existing_rows:
                    writer.writerow({k: old.get(k, "") for k in fieldnames})
            writer.writerow(row)

    def _write_run_overview(self, outputs_dir: str):
        """Writing run_overview.txt（artificially generated summary, for use in summary.csv hyperlinks）。"""
        run_dir_abs = os.path.abspath(getattr(self.args, "run_dir", ""))
        num_params = self._count_parameters(self.model)
        lines = [
            f"run_dir:    {run_dir_abs}",
            f"device:     {self.device}",
            f"model:      {self.args.model}",
            f"num_params: {num_params:,}",
            f"input_col:  {getattr(self.args, 'input_col', None) or self.args.target}",
            f"exog_col:   {getattr(self.args, 'exog_col', None) or '(none)'}",
            f"target:     {self.args.target}",
            f"seq_len:    {self.args.seq_len}  pred_len: {self.args.pred_len}",
            f"kernel_size:{getattr(self.args, 'dlinear_kernel_size', '')}",
        ]
        if self.args.model == "DLinearMix2":
            lines += [
                f"flatten_fusion:   {getattr(self.args, 'flatten_fusion', '')}",
                f"exog_emb_dim:     {getattr(self.args, 'exog_emb_dim', '')}",
                f"fusion_hidden_dim:{getattr(self.args, 'fusion_hidden_dim', '')}",
                f"dropout:          {getattr(self.args, 'dropout', '')}",
            ]
        lines += [
            f"lradj:            {getattr(self.args, 'lradj', '')}",
            f"warmup_epochs:    {getattr(self.args, 'warmup_epochs', '')}",
            f"lr_decay_gamma:   {getattr(self.args, 'lr_decay_gamma', '')}",
            f"lr_floor_ratio:   {getattr(self.args, 'lr_floor_ratio', '')}",
            f"early_stop_metric:{getattr(self.args, 'early_stop_metric', '')}",
            f"patience:         {getattr(self.args, 'patience', '')}",
            f"seed:             {getattr(self.args, 'seed', '')}",
            f"criterion:        {getattr(self.args, 'criterion', '')}",
            f"huber_beta:       {getattr(self.args, 'huber_beta', '')}",
            f"checkpoint_type:  {getattr(self, '_current_ckpt_type', '')}",
        ]
        lines += [
            "",
            "outputs/",
            "  metrics.npy",
            "  pred.npy / true.npy",
            "  metrics_horizon.csv / mse_horizon.png",
            "  metrics_segment.csv",
            "  segment_horizon_rank.csv",
            "  segment_horizon_points.csv.gz",
            "  meeting.csv",
            "",
            f"outputs_abs: {os.path.abspath(outputs_dir)}",
        ]
        overview_path = os.path.join(run_dir_abs, "run_overview.txt")
        with open(overview_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ #
    #  Predict（保留原有介面，供 Dataset_Pred 使用）                      #
    # ------------------------------------------------------------------ #

    def predict(self, setting: str, load: bool = False):
        pred_data, pred_loader = self._get_data("pred")

        if load:
            ckpt = os.path.join(self._checkpoint_dir(), "checkpoint.pth")
            self.model.load_state_dict(torch.load(ckpt))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, *_ in pred_loader:
                batch_x = batch_x.float().to(self.device)
                pred = self._forward(batch_x).detach().cpu().numpy()
                preds.append(pred)

        preds = np.concatenate(preds, axis=0)
        if pred_data.scale:
            preds = pred_data.inverse_transform(preds)

        outputs_dir = self._outputs_dir()
        np.save(os.path.join(outputs_dir, "real_prediction.npy"), preds)
        pd.DataFrame(
            np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1),
            columns=pred_data.cols,
        ).to_csv(os.path.join(outputs_dir, "real_prediction.csv"), index=False)