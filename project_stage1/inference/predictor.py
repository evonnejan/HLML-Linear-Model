"""HL01 預測入口。

先載入一次 model bundle，之後每一輪把 60 列窗口交給 predict()：
檢查 → 取 13 欄 → X 標準化 → DLinearMix2 → Y 反標準化 → anchored。
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .model import Model
from .validation import SCHEMA_PATH, validate_input

DEFAULT_BUNDLE_DIR = Path(__file__).with_name("model_bundle")


def anchor(raw: np.ndarray, anchor_value: float) -> np.ndarray:
    """保留模型預測的變化量，把起點換成實測的 HL01(t)。"""
    return anchor_value + (raw - raw[0])


class Predictor:
    """保存同一個 model bundle 的模型、標準化參數與特徵順序。"""

    def __init__(self, bundle_dir: str | Path = DEFAULT_BUNDLE_DIR):
        bundle_dir = Path(bundle_dir)
        config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.features = [*config["branch_features"], *config["exog_features"]]
        self.pred_len = int(config["pred_len"])
        self.time_format = schema["timestamp_format"]

        # The bundle must fit the input contract, otherwise every prediction would be wrong silently.
        unknown = [name for name in self.features if name not in schema["columns"]]
        if unknown or schema["anchor_column"] in self.features or schema["timestamp_column"] in self.features:
            raise ValueError(f"Bundle features do not match input_schema.json columns: {self.features}")
        if int(config["seq_len"]) != int(schema["seq_len"]):
            raise ValueError(f"Bundle seq_len {config['seq_len']} differs from input_schema.json seq_len {schema['seq_len']}")

        with np.load(bundle_dir / "scaler.npz") as scaler:
            if [str(name) for name in scaler["features"]] != self.features:
                raise ValueError("scaler.npz feature order differs from config.json")
            self.x_mean = scaler["x_mean"].astype(np.float64)
            self.x_scale = scaler["x_scale"].astype(np.float64)
            self.y_mean = float(scaler["y_mean"][0])
            self.y_scale = float(scaler["y_scale"][0])

        self.model = Model(SimpleNamespace(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            input_col=config["branch_features"],
            exog_col=config["exog_features"],
            dlinear_kernel_size=config["dlinear_kernel_size"],
            flatten_fusion=config["flatten_fusion"],
            fusion_hidden_dim=config["fusion_hidden_dim"],
            exog_emb_dim=config["exog_emb_dim"],
            dropout=config["dropout"],
        ))
        state = torch.load(bundle_dir / "checkpoint.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, window: pd.DataFrame) -> dict:
        """回傳 {"time": t, "pred": [15 個 HL01 預測值]}。

        time 是這份 input 最後一列的時間（"YYYY-MM-DD HH:MM:SS"）；
        pred[0] 是 t+1 分鐘的預測，依序到 pred[14] 是 t+15 分鐘。
        窗口沒有通過檢查時丟出 InputValidationError，這一輪不會預測。
        """
        checked = validate_input(window)
        x = (checked.history[self.features].to_numpy(dtype=np.float64) - self.x_mean) / self.x_scale
        with torch.no_grad():
            out = self.model(torch.from_numpy(x).float().unsqueeze(0))   # [1, pred_len, 1]
        raw = out.numpy().reshape(-1).astype(np.float64) * self.y_scale + self.y_mean
        values = anchor(raw, checked.anchor_value)
        return {"time": checked.origin.strftime(self.time_format), "pred": [float(value) for value in values]}
