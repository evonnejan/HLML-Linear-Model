import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
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
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    DLinear with channel mixer:
    (input_cols + exog_cols) -> single target channel -> DLinear forecast
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)

        kernel_size = getattr(configs, 'dlinear_kernel_size', 25)
        if kernel_size % 2 == 0:
            raise ValueError("DLinear kernel_size 必須是奇數，否則長度會對不齊")

        in_channels = int(getattr(configs, 'mix_in', getattr(configs, 'enc_in', 1)))
        if in_channels <= 0:
            raise ValueError(f"mix_in must be positive, got {in_channels}")

        self.input_col = getattr(configs, 'input_col', None)
        self.exog_col = getattr(configs, 'exog_col', None)
        self.target = getattr(configs, 'target', None)

        self.channel_mixer = nn.Linear(in_channels, 1)
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x_mixed = self.channel_mixer(x)
        seasonal_init, trend_init = self.decompsition(x_mixed)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        out = seasonal_output + trend_output
        return out.permute(0, 2, 1)
