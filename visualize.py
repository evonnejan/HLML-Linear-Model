import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from data_provider.Data_Loader import Dataset_Custom
from exp.exp_Main import Exp_Main


def _ensure_interactive_backend():
    backend = str(plt.get_backend()).lower()
    if "agg" not in backend:
        return

    for candidate in ["MacOSX", "TkAgg", "QtAgg"]:
        try:
            plt.switch_backend(candidate)
            return
        except Exception:
            continue


def get_latest_run(output_root: str) -> str:
    if not os.path.isdir(output_root):
        raise FileNotFoundError(f"output_root not found: {output_root}")
    dirs = [
        os.path.join(output_root, name)
        for name in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, name))
    ]
    if not dirs:
        raise FileNotFoundError(f"No run folders under: {output_root}")
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs[0]


def read_outputs(run_dir: str):
    outputs_dir = os.path.join(run_dir, "outputs")
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"outputs folder not found: {outputs_dir}")

    result = {
        "outputs_dir": outputs_dir,
        "mse_horizon": os.path.join(outputs_dir, "mse_horizon.csv"),
        "mse_segment_combined": os.path.join(outputs_dir, "mse_segment_combined.csv"),
        "points": os.path.join(outputs_dir, "segment_horizon_points.csv.gz"),
        "points_full": os.path.join(outputs_dir, "segment_horizon_points_full.csv.gz"),
        "rank": os.path.join(outputs_dir, "segment_horizon_rank.csv"),
        "meeting": os.path.join(outputs_dir, "meeting.csv"),
    }
    return result


def _points_has_segment_horizon(points_csv: str, segment, horizon: int) -> bool:
    if not os.path.exists(points_csv):
        return False
    try:
        df = pd.read_csv(points_csv, compression="gzip")
    except Exception:
        return False
    sub = df[(df["segment"].astype(str) == str(segment)) & (df["horizon"].astype(int) == int(horizon))]
    return len(sub) > 0


def _build_full_points_for_run(run_dir: str, out_points_csv: str):
    args_path = os.path.join(run_dir, "run_args.json")
    ckpt_path = os.path.join(run_dir, "checkpoints", "checkpoint.pth")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"run_args.json not found: {args_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    with open(args_path, "r", encoding="utf-8") as file:
        run_args = json.load(file)

    args = dict(run_args)
    args["run_dir"] = run_dir
    args["checkpoints"] = os.path.join(run_dir, "checkpoints")
    exp_args = SimpleNamespace(**args)

    exp = Exp_Main(exp_args)
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))

    timeenc = 0 if getattr(exp_args, "embed", "timeF") != "timeF" else 1
    dataset = Dataset_Custom(
        root_path=exp_args.root_path,
        data_path=exp_args.data_path,
        flag="train",
        size=[exp_args.seq_len, exp_args.label_len, exp_args.pred_len],
        features=exp_args.features,
        input_col=getattr(exp_args, "input_col", None),
        segment_col=getattr(exp_args, "segment_col", None),
        target=exp_args.target,
        stride=getattr(exp_args, "stride_eval", 1),
        scale=True,
        timeenc=timeenc,
        freq=exp_args.freq,
        train_only=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(getattr(exp_args, "batch_size", 32)),
        shuffle=False,
        num_workers=int(getattr(exp_args, "num_workers", 0)),
        drop_last=False,
    )

    preds = []
    trues = []

    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -exp_args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :exp_args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            if "Linear" in exp_args.model:
                outputs = exp.model(batch_x)
            else:
                if getattr(exp_args, "output_attention", False):
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if exp_args.features == "MS" else 0
            outputs = outputs[:, -exp_args.pred_len:, f_dim:]
            batch_y = batch_y[:, -exp_args.pred_len:, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    if getattr(dataset, "scale", False):
        shape_preds = preds.shape
        preds = dataset.inverse_transform(preds.reshape(-1, shape_preds[-1])).reshape(shape_preds)
        trues = dataset.inverse_transform(trues.reshape(-1, shape_preds[-1])).reshape(shape_preds)

    if preds.ndim == 3 and preds.shape[2] == 1:
        pred_viz = preds[:, :, 0]
        true_viz = trues[:, :, 0]
    elif preds.ndim == 3:
        pred_viz = preds.mean(axis=2)
        true_viz = trues.mean(axis=2)
    else:
        pred_viz = preds
        true_viz = trues

    if dataset.window_segment_ids is None or len(dataset.window_segment_ids) == 0:
        raise RuntimeError("No segment ids available for full-data points export")

    start_indices = dataset.valid_starts if dataset.valid_starts is not None else np.arange(pred_viz.shape[0], dtype=np.int64)
    segment_ids = dataset.window_segment_ids

    points_rows = []
    for local_idx in range(pred_viz.shape[0]):
        seg_id = segment_ids[local_idx]
        s_begin = int(start_indices[local_idx])
        for horizon_idx in range(int(exp_args.pred_len)):
            t_idx = s_begin + int(exp_args.seq_len) + horizon_idx
            if t_idx >= len(dataset.dates):
                continue
            y_t = float(true_viz[local_idx, horizon_idx])
            y_p = float(pred_viz[local_idx, horizon_idx])
            points_rows.append(
                {
                    "segment": seg_id,
                    "window_idx": int(local_idx),
                    "horizon": int(horizon_idx + 1),
                    "target_time": pd.to_datetime(dataset.dates[t_idx]).strftime("%Y-%m-%d %H:%M:%S"),
                    "true": y_t,
                    "pred": y_p,
                    "abs_err": abs(y_t - y_p),
                    "sq_err": (y_t - y_p) ** 2,
                }
            )

    pd.DataFrame(points_rows).to_csv(out_points_csv, index=False, encoding="utf-8-sig", compression="gzip")


def resolve_points_file(run_dir: str, files: dict, segment, horizon: int) -> str:
    test_points = files["points"]
    full_points = files["points_full"]

    if _points_has_segment_horizon(test_points, segment, horizon):
        return test_points

    if _points_has_segment_horizon(full_points, segment, horizon):
        return full_points

    _build_full_points_for_run(run_dir, full_points)
    if _points_has_segment_horizon(full_points, segment, horizon):
        return full_points

    raise ValueError(f"No data for segment={segment}, horizon={horizon} in test/full points")


def plot_horizon_mse(csv_path: str, out_path: str = None):
    df = pd.read_csv(csv_path)
    if out_path is None:
        _ensure_interactive_backend()
    plt.figure(figsize=(8, 4))
    plt.plot(df["horizon"], df["mse"], marker="o")
    plt.xlabel("Horizon (t+k)")
    plt.ylabel("MSE")
    plt.title("Horizon-wise MSE")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
    else:
        plt.show(block=True)
    plt.close()


def plot_segment_curve(points_csv: str, segment, horizon: int, out_path: str = None):
    df = pd.read_csv(points_csv, compression="gzip")
    sub = df[(df["segment"].astype(str) == str(segment)) & (df["horizon"].astype(int) == int(horizon))].copy()
    if len(sub) == 0:
        raise ValueError(f"No data for segment={segment}, horizon={horizon}")

    sub["target_time"] = pd.to_datetime(sub["target_time"])
    sub = sub.sort_values("target_time")

    if out_path is None:
        _ensure_interactive_backend()

    plt.figure(figsize=(12, 4.5))
    plt.plot(sub["target_time"], sub["true"], label="GroundTruth", linewidth=2)
    plt.plot(sub["target_time"], sub["pred"], label="Prediction", linewidth=2)
    mse = np.mean((sub["pred"].to_numpy() - sub["true"].to_numpy()) ** 2)
    corr = np.corrcoef(sub["pred"].to_numpy(), sub["true"].to_numpy())[0, 1] if len(sub) >= 2 else np.nan
    plt.title(f"segment={segment}, horizon=t+{horizon} | mse={mse:.4f}, corr={corr:.4f}")
    plt.xlabel("Target Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
    else:
        plt.show(block=True)
    plt.close()


def print_topk(rank_csv: str, k: int = 10):
    df = pd.read_csv(rank_csv)
    print("\n=== Top by lowest MSE ===")
    print(df.sort_values("mse", ascending=True).head(k).to_string(index=False))

    if "corr" in df.columns:
        df_corr = df.dropna(subset=["corr"]).sort_values("corr", ascending=False)
        print("\n=== Top by highest CORR ===")
        print(df_corr.head(k).to_string(index=False))


def print_meeting(meeting_csv: str):
    df = pd.read_csv(meeting_csv)
    for category in ["best_mse", "worst_mse", "best_corr", "worst_corr"]:
        sub = df[df["category"] == category]
        if len(sub) == 0:
            continue
        print(f"\n=== {category} ===")
        print(sub.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Visualization utility for runs/<setting>/outputs")
    parser.add_argument("--output_root", type=str, default="./runs")
    parser.add_argument("--run_dir", type=str, default=None, help="Specific run folder; default uses latest under output_root")
    parser.add_argument("--mode", type=str, default="topk", choices=["topk", "meeting", "horizon", "segment"])
    parser.add_argument("--segment", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else get_latest_run(args.output_root)
    files = read_outputs(run_dir)

    print(f"Using run_dir: {run_dir}")

    if args.mode == "topk":
        if not os.path.exists(files["rank"]):
            raise FileNotFoundError(f"rank file not found: {files['rank']}")
        print_topk(files["rank"], k=args.k)
        return

    if args.mode == "meeting":
        if not os.path.exists(files["meeting"]):
            raise FileNotFoundError(f"meeting file not found: {files['meeting']}")
        print_meeting(files["meeting"])
        return

    if args.mode == "horizon":
        if not os.path.exists(files["mse_horizon"]):
            raise FileNotFoundError(f"horizon csv not found: {files['mse_horizon']}")
        plot_horizon_mse(files["mse_horizon"], out_path=args.save)
        return

    if args.mode == "segment":
        if args.segment is None or args.horizon is None:
            raise ValueError("--mode segment requires --segment and --horizon")
        points_file = resolve_points_file(run_dir, files, segment=args.segment, horizon=args.horizon)
        plot_segment_curve(points_file, segment=args.segment, horizon=args.horizon, out_path=args.save)
        return


if __name__ == "__main__":
    main()
