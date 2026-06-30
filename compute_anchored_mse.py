"""事後計算「錨定到最後一個 input 值」的校正指標（方案 A，免重跑模型）。

對每個 window、每個目標特徵：
    c   = pred[0] - x_last          # 預測起點與最後觀測值的落差
    adj = pred - c                  # 整條平移，使 adj[0] == x_last
報出 raw / anchored / persistence 三組的 MSE, RMSE, MAE, Corr。

x_last（= 最後一個 input 時間點的目標值）不在 pred.npy/true.npy 裡，但完全來自
資料本身：優先讀 outputs/persist.npy（若存在），否則用 run_args.json 重建 test
dataset 取得（與 pred.npy 同順序，test loader shuffle=False）。pred/true 已是
inverse-transform 後的真實尺度，故 x_last 也做相同 inverse-transform。

用法:
    python compute_anchored_mse.py <run_dir> [--outputs outputs] [--out-subdir anchored_metric]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from data_provider.Data_Factory import data_provider
from utils.metrics import CORR


def load_run_args(run_dir):
    with open(os.path.join(run_dir, "run_args.json"), "r", encoding="utf-8") as f:
        return argparse.Namespace(**json.load(f))


def collect_x_last(args):
    """重建 test dataset，回傳 x_last[N,1,nf]（真實尺度）。順序同 pred.npy。"""
    test_data, test_loader = data_provider(args, "test")
    label_len = int(args.label_len)
    f_dim = -1 if args.features == "MS" else 0
    chunks = []
    for batch in test_loader:
        batch_y = batch[1].float()
        # seq_y[label_len-1] == 最後一個 input 時間點 (Data_Loader.py:446-450)
        chunks.append(batch_y[:, label_len - 1:label_len, f_dim:].numpy())
    x_last = np.concatenate(chunks, axis=0)  # (N,1,nf)
    if getattr(test_data, "scale", False):
        sh = x_last.shape
        x_last = test_data.inverse_transform(x_last.reshape(-1, sh[-1])).reshape(sh)
    return x_last


def four_metrics(pred, true):
    """回傳 (MSE, RMSE, MAE, Corr)。Corr 用 utils.metrics.CORR 逐 horizon 再取均值。"""
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    corr = float(np.nanmean(CORR(pred, true)))
    return mse, float(np.sqrt(mse)), mae, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--outputs", default="outputs", help="讀哪個 outputs 夾 (outputs / outputs_alt)")
    ap.add_argument("--out-subdir", default="anchored_metric", help="結果寫到的新子資料夾名")
    a = ap.parse_args()

    out_dir = os.path.join(a.run_dir, a.outputs)
    dst_dir = os.path.join(a.run_dir, a.out_subdir)
    os.makedirs(dst_dir, exist_ok=True)

    run_args = load_run_args(a.run_dir)
    preds = np.load(os.path.join(out_dir, "pred.npy"))
    trues = np.load(os.path.join(out_dir, "true.npy"))

    persist_npy = os.path.join(out_dir, "persist.npy")
    if os.path.exists(persist_npy):
        x_last = np.load(persist_npy)[:, 0:1, :]
        source = "persist.npy"
    else:
        x_last = collect_x_last(run_args)
        source = "重建 test dataset"

    if x_last.shape[0] != preds.shape[0]:
        raise SystemExit(f"window 數不符: x_last={x_last.shape[0]} vs pred={preds.shape[0]}")

    c = preds[:, 0:1, :] - x_last
    adj = preds - c                       # adj[:,0,:] == x_last
    persist = np.broadcast_to(x_last, preds.shape)

    rows = {
        "raw":         four_metrics(preds, trues),
        "anchored":    four_metrics(adj, trues),
        "persistence": four_metrics(persist, trues),
    }
    table = pd.DataFrame(
        [[k, *v] for k, v in rows.items()],
        columns=["method", "MSE", "RMSE", "MAE", "Corr"],
    )

    # 逐 horizon (raw vs anchored)
    def per_h(arr):
        return np.mean((arr - trues) ** 2, axis=(0, 2)), np.mean(np.abs(arr - trues), axis=(0, 2))

    def per_h_corr(arr):
        # 每個 horizon 跨 window 的 Pearson corr (用特徵 0, 同 pipeline 慣例)
        p, t = arr[..., 0], trues[..., 0]
        out = np.full(p.shape[1], np.nan)
        for h in range(p.shape[1]):
            ph, th = p[:, h], t[:, h]
            if ph.std() > 0 and th.std() > 0:
                out[h] = np.corrcoef(ph, th)[0, 1]
        return out

    raw_mse_h, raw_mae_h = per_h(preds)
    anc_mse_h, anc_mae_h = per_h(adj)
    horizon = pd.DataFrame({
        "horizon": np.arange(1, preds.shape[1] + 1),
        "raw_MSE": raw_mse_h, "anchored_MSE": anc_mse_h,
        "raw_MAE": raw_mae_h, "anchored_MAE": anc_mae_h,
        "raw_Corr": per_h_corr(preds), "anchored_Corr": per_h_corr(adj),
    })

    # 完整性 / 交叉驗證
    anchor_ok = bool(np.allclose(adj[:, 0, :], x_last[:, 0, :], atol=1e-6))
    persist_mse = rows["persistence"][0]
    cross = "n/a (無 persistence_metrics.csv)"
    pm_csv = os.path.join(out_dir, "persistence_metrics.csv")
    if os.path.exists(pm_csv):
        ref = pd.read_csv(pm_csv)["persist_MSE"].mean()
        cross = f"重建={persist_mse:.6f}  vs  pipeline 存檔={ref:.6f}  diff={abs(persist_mse-ref):.2e}"

    # 寫檔
    table.to_csv(os.path.join(dst_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    horizon.to_csv(os.path.join(dst_dir, "metrics_horizon.csv"), index=False, encoding="utf-8-sig")
    meta = {
        "run_dir": a.run_dir, "outputs": a.outputs, "x_last_source": source,
        "windows": int(preds.shape[0]), "pred_len": int(preds.shape[1]),
        "target": run_args.target, "features": run_args.features,
        "metrics": {k: dict(zip(["MSE", "RMSE", "MAE", "Corr"], v)) for k, v in rows.items()},
        "anchor_check_ok": anchor_ok, "persistence_cross_check": cross,
    }
    with open(os.path.join(dst_dir, "anchored_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 列印
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print(f"run     : {a.run_dir}")
    print(f"outputs : {a.outputs}   x_last 來源: {source}")
    print(f"windows : {preds.shape[0]}   pred_len={preds.shape[1]}   target={run_args.target}")
    print("-" * 60)
    print(table.to_string(index=False))
    print("-" * 60)
    print(f"[check] adj[:,0]==x_last : {anchor_ok}")
    print(f"[check] persistence 交叉驗證: {cross}")
    print(f"\n已寫出: {dst_dir}/  (metrics.csv, metrics_horizon.csv, anchored_metrics.json)")


if __name__ == "__main__":
    main()
