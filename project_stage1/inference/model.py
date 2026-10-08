# 請不要修改結構，否則訓練好的權重會無法載入。
import torch
import torch.nn as nn


def _parse_col_spec(col_spec):
    if col_spec is None:
        return []
    if isinstance(col_spec, str):
        return [item.strip() for item in col_spec.split(',') if item.strip()]
    if isinstance(col_spec, (list, tuple)):
        return [str(item).strip() for item in col_spec if str(item).strip()]
    text = str(col_spec).strip()
    return [text] if text else []


class moving_avg(nn.Module):
    """移動平均，用來取出時間序列的趨勢。輸入與輸出形狀：[B, L, C]。"""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """序列分解：回傳殘差（季節項）與移動平均（趨勢項）。"""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinearBranch(nn.Module):
    """單一通道的 DLinear 分支；Model 為每個水位通道各建一個。"""

    def __init__(self, seq_len, pred_len, kernel_size):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.decomposition = series_decomp(kernel_size)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [B, L, 1]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, 1, L]
        trend_init = trend_init.permute(0, 2, 1)        # [B, 1, L]

        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        out = seasonal_output + trend_output            # [B, 1, P]
        return out.permute(0, 2, 1)                    # [B, P, 1]


class ExogenousEncoder(nn.Module):
    """
    以 GRU 依時間順序讀取外生變數（雨量、閘門）的窗口，壓縮成一個固定長度的情境向量。

    雨量與閘門對水位的影響有時間延遲；GRU 逐步讀取整段窗口，
    最後的隱藏狀態保留了時間先後的資訊。

    輸入：x_exog [B, seq_len, exog_in]
    輸出：context [B, emb_dim]
    """

    def __init__(self, seq_len, exog_in, emb_dim=16, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.exog_in = int(exog_in)
        self.emb_dim = int(emb_dim)

        # GRU hidden_size == emb_dim: last hidden state is directly the context vector.
        self.gru = nn.GRU(
            input_size=self.exog_in,
            hidden_size=self.emb_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_exog):
        # x_exog: [B, L, E]
        _, h_n = self.gru(x_exog)   # h_n: [1, B, emb_dim]
        context = h_n.squeeze(0)    # [B, emb_dim]
        return self.dropout(context)


class HorizonWiseFusion(nn.Module):
    """
    對每一個預測步各自套用同一個 MLP。

    每一步的輸入：[各分支在該步的預測, 外生情境向量]
    每一步的輸出：該步的目標預測
    """

    def __init__(self, in_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        # z: [B, P, D]
        return self.net(z)                                   # [B, P, 1]


class FlattenFusion(nn.Module):
    """
    先把所有分支、所有預測步的結果攤平，再用一個 MLP 輸出整段預測。

    表達力較強，但參數較多。交付的模型使用這種融合方式（config.json 的 flatten_fusion 為 true）。
    """

    def __init__(self, in_dim, pred_len, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.pred_len = int(pred_len)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.pred_len),
        )

    def forward(self, z_flat):
        # z_flat: [B, D]
        out = self.net(z_flat)                               # [B, P]
        return out.unsqueeze(-1)                             # [B, P, 1]


class Model(nn.Module):
    """
    DLinearMix2：水位用 DLinear 分支、雨量與閘門用 GRU 編碼，再融合成單一目標（HL01）的預測。

    架構：
        每個水位通道 → 各自的 DLinear 分支 → 分支預測
        雨量、閘門    → ExogenousEncoder     → 情境向量
        [分支預測 + 情境向量] → 融合 MLP → 目標預測

    輸入 x 的形狀：[B, seq_len, branch_in + exog_in]
    - 前 branch_in 個通道是水位，每個通道一個 DLinear 分支
    - 後 exog_in 個通道是雨量與閘門
    通道順序必須和訓練時相同；交付時由 model_bundle/config.json 決定。

    設定欄位（除 seq_len、pred_len 外都可省略）：
    - enc_in 或 mix_in：輸入通道總數
    - branch_in：水位分支的通道數
    - exog_in：外生變數的通道數
    - dlinear_kernel_size：分解用的移動平均視窗，必須是奇數
    - flatten_fusion：True 為攤平後融合，False 為逐步融合
    - exog_emb_dim：GRU 隱藏層大小，也就是情境向量的長度
    - fusion_hidden_dim：融合 MLP 的隱藏層大小
    - dropout：外生編碼器與融合 MLP 的 dropout
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)

        kernel_size = int(getattr(configs, 'dlinear_kernel_size', 25))
        if kernel_size % 2 == 0:
            raise ValueError(
                f"DLinear kernel_size must be an odd number to ensure proper padding, got {kernel_size}"
            )

        input_cols = _parse_col_spec(getattr(configs, 'input_col', None))
        exog_cols = _parse_col_spec(getattr(configs, 'exog_col', None))

        default_branch = len(input_cols)
        default_exog = len(exog_cols)
        default_total = default_branch + default_exog

        raw_total_in = getattr(configs, 'mix_in', getattr(configs, 'enc_in', None))
        self.total_in = int(raw_total_in) if raw_total_in is not None else int(max(1, default_total))

        self.branch_in = int(getattr(configs, 'branch_in', default_branch if default_branch > 0 else self.total_in))
        self.exog_in = int(getattr(configs, 'exog_in', default_exog))

        if self.total_in <= 0:
            raise ValueError(f"total input channels must be positive, got {self.total_in}")
        if self.branch_in <= 0:
            raise ValueError(f"branch_in must be positive, got {self.branch_in}")
        if self.exog_in < 0:
            raise ValueError(f"exog_in must be >= 0, got {self.exog_in}")
        if self.branch_in + self.exog_in > self.total_in:
            raise ValueError(
                f"branch_in + exog_in cannot exceed total input channels: "
                f"{self.branch_in} + {self.exog_in} > {self.total_in}"
            )
        if self.branch_in + self.exog_in != self.total_in:
            raise ValueError(
                f"branch_in + exog_in must equal total input channels: "
                f"{self.branch_in} + {self.exog_in} != {self.total_in}"
            )

        self.flatten_fusion = bool(getattr(configs, 'flatten_fusion', False))
        self.dropout = float(getattr(configs, 'dropout', 0.1))
        self.exog_emb_dim = int(getattr(configs, 'exog_emb_dim', 16 if self.exog_in > 0 else 0))
        self.fusion_hidden_dim = int(getattr(configs, 'fusion_hidden_dim', 32))

        self.input_col = getattr(configs, 'input_col', None)
        self.exog_col = getattr(configs, 'exog_col', None)
        self.target = getattr(configs, 'target', None)

        # One DLinear branch per main input channel.
        self.branches = nn.ModuleList([
            DLinearBranch(self.seq_len, self.pred_len, kernel_size)
            for _ in range(self.branch_in)
        ])

        # Optional exogenous context branch (e.g. rainfall indicator sequence).
        if self.exog_in > 0:
            self.exog_encoder = ExogenousEncoder(
                seq_len=self.seq_len,
                exog_in=self.exog_in,
                emb_dim=self.exog_emb_dim,
                dropout=self.dropout,
            )
        else:
            self.exog_encoder = None
            self.exog_emb_dim = 0

        if self.flatten_fusion:
            flat_in_dim = self.branch_in * self.pred_len + self.exog_emb_dim
            flat_hidden = max(self.fusion_hidden_dim, 64)
            self.fusion = FlattenFusion(
                in_dim=flat_in_dim,
                pred_len=self.pred_len,
                hidden_dim=flat_hidden,
                dropout=self.dropout,
            )
        else:
            horizon_in_dim = self.branch_in + self.exog_emb_dim
            self.fusion = HorizonWiseFusion(
                in_dim=horizon_in_dim,
                hidden_dim=self.fusion_hidden_dim,
                dropout=self.dropout,
            )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, L, C], got ndim={x.ndim}")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got x.size(1)={x.size(1)}")
        # Channel count is validated below at the split point with a more informative message.

        # Split channels:
        # first branch_in channels -> main DLinear branches
        # next exog_in channels   -> exogenous context branch
        #
        # IMPORTANT: this model relies on an implicit channel ordering contract:
        #   x[:, :, :branch_in]                         -> main input channels
        #   x[:, :, branch_in : branch_in + exog_in]    -> exogenous channels
        #
        # This assertion catches any mismatch between declared counts and actual tensor width.
        expected_channels = self.branch_in + self.exog_in
        if x.size(2) != expected_channels:
            raise ValueError(
                f"Channel count mismatch: model expects exactly {expected_channels} channels "
                f"(branch_in={self.branch_in} + exog_in={self.exog_in}), "
                f"but x has {x.size(2)} channels. "
                f"Check that input_col and exog_col match the data loader column order."
            )
        x_main = x[:, :, :self.branch_in]   # [B, L, K]
        x_exog = x[:, :, self.branch_in:self.branch_in + self.exog_in] if self.exog_in > 0 else None

        # Branch-wise DLinear forecasts
        branch_preds = []
        for i, branch in enumerate(self.branches):
            xi = x_main[:, :, i:i + 1]                  # [B, L, 1]
            yi = branch(xi)                             # [B, P, 1]
            branch_preds.append(yi)
        branch_preds = torch.cat(branch_preds, dim=-1) # [B, P, K]

        # Exogenous context embedding
        exog_emb = self.exog_encoder(x_exog) if self.exog_encoder is not None else None

        if self.flatten_fusion:
            z = branch_preds.reshape(branch_preds.size(0), -1)   # [B, P*K]
            if exog_emb is not None:
                z = torch.cat([z, exog_emb], dim=-1)
            out = self.fusion(z)                                 # [B, P, 1]
        else:
            if exog_emb is not None:
                exog_rep = exog_emb.unsqueeze(1).expand(-1, self.pred_len, -1)  # [B, P, E]
                z = torch.cat([branch_preds, exog_rep], dim=-1)                  # [B, P, K+E]
            else:
                z = branch_preds                                                  # [B, P, K]
            out = self.fusion(z)                                                  # [B, P, 1]

        return out
