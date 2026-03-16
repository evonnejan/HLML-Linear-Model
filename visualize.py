import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def short_label(setting: str) -> str:
    if setting.startswith("Linear"):
        return "Linear"
    if setting.startswith("NLinear"):
        return "NLinear"
    if setting.startswith("DLinear"):
        if "_ks" in setting:
            ks = setting.split("_ks")[-1]
            return f"DLinear ks={ks}"
        return "DLinear"
    return setting

def load_pred_true(results_dir, setting):
    folder = os.path.join(results_dir, setting)
    pred_path = os.path.join(folder, "pred.npy")
    true_path = os.path.join(folder, "true.npy")
    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        raise FileNotFoundError(f"Missing pred/true for setting={setting}: {folder}")

    pred = np.load(pred_path)
    true = np.load(true_path)
    pred = np.squeeze(pred)  # -> (N, pred_len)
    true = np.squeeze(true)

    if pred.ndim != 2 or true.ndim != 2:
        raise ValueError(f"Bad shape for {setting}: pred{pred.shape}, true{true.shape}")

    return pred, true

def compute_splits(n, seq_len, pred_len, train_only=False):
    num_train = int(n * (0.7 if not train_only else 1))
    num_test = int(n * 0.2)
    # test border1 per your Dataset_Custom
    border1 = n - num_test - seq_len
    # number of test samples per your Dataset_Custom: num_test - pred_len + 1
    N = num_test - pred_len + 1
    anchor_start = border1 + seq_len  # this is the first timestamp of y (horizon=1)
    anchor_indices = np.arange(anchor_start, anchor_start + N)
    return num_train, num_test, border1, N, anchor_indices

def set_time_ticks(ax, t_start, t_end, tick_hours=3):
    span_min = (t_end - t_start) / np.timedelta64(1, "m")
    if span_min <= 120:  # <= 2hr
        interval = 5 if span_min <= 60 else 10
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, tick_hours)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

def horizon_metrics(pred, true):
    err = pred - true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    bias = np.mean(err, axis=0)
    return mae, rmse, bias

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="series", choices=["series", "horizon_mae", "forecast"])
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--csv_path", default="./dataset/water_level_all.csv")
    ap.add_argument("--target", default="HL01")
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--pred_len", type=int, default=15)

    # for series plot
    ap.add_argument("--horizon", type=int, default=1, help="1..pred_len (used in series plot)")
    ap.add_argument("--day", default=None, help="YYYY-MM-DD (optional)")
    ap.add_argument("--window_minutes", type=int, default=None, help="e.g., 10/20/30/60 (series plot)")
    ap.add_argument("--tick_hours", type=int, default=3)

    # for forecast plot
    ap.add_argument("--time", default=None, help="HH:MM or HH:MM:SS (forecast anchor time; optional)")
    ap.add_argument("--anchor", type=int, default=None, help="test sample index (0..N-1), overrides day/time")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--show_range", action="store_true")

    args = ap.parse_args()

    # settings list (must match your results/<setting>/ folders)
    settings = [
        "Linear_custom_ftS_sl60_pl15",
        "NLinear_custom_ftS_sl60_pl15",
        "DLinear_custom_ftS_sl60_ll30_pl15_ks7",
        "DLinear_custom_ftS_sl60_ll30_pl15_ks13",
        "DLinear_custom_ftS_sl60_ll30_pl15_ks25",
        "DLinear_custom_ftS_sl60_ll30_pl15_ks49",
    ]

    # load CSV for time axis & raw target values
    df = pd.read_csv(args.csv_path)
    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV columns: {df.columns.tolist()}")

    dt_all = pd.to_datetime(df["date"])
    y_all = df[args.target].fillna(0).to_numpy()
    n = len(df)

    # load one pred/true to infer pred_len and sample count
    pred0, true0 = load_pred_true(args.results_dir, settings[0])
    pred_len = true0.shape[1]
    # trust run outputs; pred_len should match
    if args.pred_len != pred_len:
        # allow mismatch but warn by print
        print(f"[WARN] args.pred_len={args.pred_len} but files pred_len={pred_len}. Using files pred_len.")
    args.pred_len = pred_len

    # compute test mapping
    _, num_test, border1, N, anchor_indices = compute_splits(n, args.seq_len, args.pred_len)

    # quick range print
    if args.show_range:
        tmin = dt_all.iloc[anchor_indices[0]]
        tmax = dt_all.iloc[anchor_indices[-1] + (args.pred_len - 1)]
        print("Test anchor time range:", tmin, "to", tmax)
        print("N test samples:", N, "| num_test points:", num_test)
        return

    # -----------------------
    # plot: horizon_mae
    # -----------------------
    if args.plot == "horizon_mae":
        plt.figure(figsize=(10, 5))
        for s in settings:
            pred, true = load_pred_true(args.results_dir, s)
            mae, rmse, bias = horizon_metrics(pred, true)
            plt.plot(range(1, pred_len + 1), mae, label=short_label(s))
        plt.xlabel("Horizon (steps)")
        plt.ylabel("MAE (original scale)")
        plt.title("Horizon-wise MAE (test set)")
        plt.legend()
        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=200)
        else:
            plt.show()
        return

    # -----------------------
    # plot: forecast (ONE anchor, full 15-step)
    # -----------------------
    if args.plot == "forecast":
        rng = np.random.default_rng(args.seed)

        # choose sample index i in [0..N-1]
        if args.anchor is not None:
            i = int(args.anchor)
            if not (0 <= i < N):
                raise ValueError(f"--anchor must be in 0..{N-1}, got {i}")
        else:
            # filter by day/time if provided
            anchor_times = dt_all.iloc[anchor_indices].reset_index(drop=True)  # length N

            # choose day
            if args.day is None:
                days = anchor_times.dt.date.unique()
                day = rng.choice(days)
            else:
                day = pd.to_datetime(args.day).date()

            mask_day = (anchor_times.dt.date == day)
            cand = np.where(mask_day.to_numpy())[0]
            if len(cand) == 0:
                tmin = anchor_times.min()
                tmax = anchor_times.max()
                print(f"[ERROR] day={day} not in test anchors.")
                print("Test anchor day range:", tmin, "to", tmax)
                return

            # choose time (nearest) or random within day
            if args.time is not None:
                # build target datetime for nearest match
                # accept HH:MM or HH:MM:SS
                t_str = args.time.strip()
                # combine with day
                target_dt = pd.to_datetime(f"{day} {t_str}")
                diffs = np.abs((anchor_times.iloc[cand] - target_dt).to_numpy())
                j = int(np.argmin(diffs))
                i = int(cand[j])
            else:
                i = int(rng.choice(cand))

        # map to raw index
        anchor_raw = int(anchor_indices[i])  # index in original df where future starts (h=1)
        anchor_time = dt_all.iloc[anchor_raw]

        hist_start = anchor_raw - args.seq_len
        hist_end = anchor_raw - 1
        fut_start = anchor_raw
        fut_end = anchor_raw + args.pred_len - 1

        times_hist = dt_all.iloc[hist_start:anchor_raw]
        y_hist = y_all[hist_start:anchor_raw]
        times_fut = dt_all.iloc[fut_start:fut_end + 1]
        y_fut_true = y_all[fut_start:fut_end + 1]

        print(f"[Forecast anchor] sample_i={i}, anchor_time={anchor_time}")
        print(f"  history: {times_hist.iloc[0]} -> {times_hist.iloc[-1]} ({len(times_hist)} pts)")
        print(f"  future : {times_fut.iloc[0]} -> {times_fut.iloc[-1]} ({len(times_fut)} pts)")

        plt.figure(figsize=(12, 5))

        # plot history
        plt.plot(times_hist, y_hist, label="History (true)", linewidth=2.0)

        # plot true future
        plt.plot(times_fut, y_fut_true, label="Future (true)", linewidth=2.0)

        # plot model predictions (future only)
        for s in settings:
            pred, true = load_pred_true(args.results_dir, s)
            y_pred = pred[i, :]  # length pred_len
            plt.plot(times_fut, y_pred, label=short_label(s), alpha=0.9)

        # vertical line at anchor
        plt.axvline(anchor_time, linestyle="--", linewidth=1.0)

        ax = plt.gca()
        set_time_ticks(ax, times_hist.iloc[0].to_datetime64(), times_fut.iloc[-1].to_datetime64(), tick_hours=args.tick_hours)

        plt.title(f"Single Forecast @ {anchor_time.strftime('%Y-%m-%d %H:%M:%S')} | seq={args.seq_len} pred={args.pred_len}")
        plt.xlabel("Time")
        plt.ylabel("Water level")
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()

        if args.save:
            plt.savefig(args.save, dpi=200)
        else:
            plt.show()
        return

    # -----------------------
    # plot: series (rolling one horizon)
    # -----------------------
    # validate horizon
    if not (1 <= args.horizon <= pred_len):
        raise ValueError(f"--horizon must be in 1..{pred_len}")

    # build times for chosen horizon: anchor_times + (h-1)
    offset = args.horizon - 1
    times = dt_all.iloc[anchor_indices + offset].reset_index(drop=True)  # length N

    if args.show_range:
        print("Series time range:", times.min(), "to", times.max())
        return

    # choose day
    rng = np.random.default_rng(args.seed)
    if args.day is None:
        day = rng.choice(times.dt.date.unique())
    else:
        day = pd.to_datetime(args.day).date()

    mask = (times.dt.date == day).to_numpy()
    idx_all = np.where(mask)[0]
    if len(idx_all) == 0:
        print(f"[ERROR] day={day} not in series times for horizon={args.horizon}.")
        print("Series time range:", times.min(), "to", times.max())
        return

    # choose window
    if args.window_minutes is None or len(idx_all) <= args.window_minutes:
        idx = idx_all
    else:
        start_pos = int(rng.integers(0, len(idx_all) - args.window_minutes + 1))
        idx = idx_all[start_pos:start_pos + args.window_minutes]

    t_start = times.iloc[idx[0]]
    t_end = times.iloc[idx[-1]]
    print(f"[Series window] day={day}, horizon={args.horizon}, start={t_start}, end={t_end}, points={len(idx)}")

    # true and preds at this horizon
    y_true = true0[:, args.horizon - 1]

    plt.figure(figsize=(12, 5))
    plt.plot(times.iloc[idx], y_true[idx], label=f"True (h={args.horizon})", linewidth=2.0)

    for s in settings:
        pred, _ = load_pred_true(args.results_dir, s)
        y_pred = pred[:, args.horizon - 1]
        plt.plot(times.iloc[idx], y_pred[idx], label=short_label(s), alpha=0.9)

    ax = plt.gca()
    set_time_ticks(ax, t_start.to_datetime64(), t_end.to_datetime64(), tick_hours=args.tick_hours)

    plt.title(f"Rolling Series | day={day} | horizon={args.horizon} | window={len(idx)} min")
    plt.xlabel("Time")
    plt.ylabel("Water level")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
    else:
        plt.show()

if __name__ == "__main__":
    main()