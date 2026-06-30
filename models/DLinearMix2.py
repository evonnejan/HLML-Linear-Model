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
    """
    Moving average block to highlight the trend of a time series.
    Input / output shape: [B, L, C]
    """

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
    """
    Series decomposition block.
    Returns seasonal(residual) and trend(moving average).
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinearBranch(nn.Module):
    """
    A single-channel DLinear branch.
    Each input channel gets its own branch in Model.
    """

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
    GRU-based encoder for exogenous sequences.

    Reads the past exogenous window step-by-step (preserving temporal order)
    and compresses it into a single context embedding.

    Design rationale:
    - Exogenous variables (rainfall, gate) have time-delayed effects on water level.
      e.g. rainfall accumulates upstream before reaching a sensor; gate opening
      drains water with a short but non-zero delay.
    - A GRU naturally captures both short-range (gate response) and long-range
      (upstream flow delay) temporal patterns within the lookback window.
    - The final hidden state summarises the entire past window in a fixed-size
      vector, without discarding temporal order (unlike flatten + statistics).

    Input:  x_exog [B, seq_len, exog_in]
    Output: context [B, emb_dim]
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
    Shared MLP applied to each forecast horizon separately.

    Input per horizon:
        [branch_pred_1, ..., branch_pred_K, exog_embedding]
    Output per horizon:
        target forecast for that horizon
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
    MLP that first flattens all branch forecasts across horizon, then outputs the full target forecast.

    This is more expressive, but also more parameter-heavy and can overfit more easily.
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
    DLinearMix2: branch-wise DLinear + exogenous-context fusion for single-target forecasting.

    ----------------------------------------------------------------------
    High-level architecture
    ----------------------------------------------------------------------
    Previous DLinearMix (earlier file):
        (input_cols + exog_cols) --linear early fusion--> one mixed series --> DLinear --> target forecast

    This DLinearMix2:
        each input_col --> its own DLinear branch --> branch forecasts
        exog_cols      --> exogenous encoder      --> context embedding
        [branch forecasts + exogenous context]    --> MLP fusion --> target forecast

    So compared with the previous model, this version:
    1) changes from EARLY FUSION to LATE/NEAR-LATE FUSION
    2) preserves each input channel's own temporal structure before fusion
    3) treats rainfall / exogenous signals as context instead of forcing them to behave like a target-like series
    4) allows two fusion styles controlled by a boolean flag:
       - flatten_fusion=False: horizon-wise shared fusion MLP (lighter, often stabler)
       - flatten_fusion=True : flatten all branch forecasts, then fuse with one larger MLP

    ----------------------------------------------------------------------
    Expected input layout
    ----------------------------------------------------------------------
    x shape: [B, seq_len, total_in]

    By default this model assumes:
    - the FIRST `branch_in` channels are input_cols (each gets its own DLinear branch)
    - the LAST `exog_in` channels are exogenous variables (e.g. rainfall binary indicator)

    Useful config fields (all optional except seq_len / pred_len):
    - enc_in or mix_in: total input channels
    - branch_in: number of main input channels used for DLinear branches
    - exog_in: number of exogenous channels
    - dlinear_kernel_size: decomposition moving-average kernel size (must be odd)
    - flatten_fusion: bool, switch between horizon-wise fusion and flatten fusion
    - exog_emb_dim: GRU hidden size = embedding size of exogenous encoder
    - fusion_hidden_dim: hidden size for fusion MLP
    - dropout: dropout used in exogenous encoder and fusion MLP
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
        # The data loader enforces this by building x_cols = input_cols + exog_cols in that order.
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
