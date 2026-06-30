"""視覺化 GT / Pred / adj_Pred（錨定校正後預測）。

對選定的 window 畫 15 步預測：
  - 黑點(t=0): x_last = 最後一個 input 值(錨定點)
  - GT      : 真實未來 (true.npy)
  - Pred    : 模型原始預測 (pred.npy)
  - adj_Pred: 校正後 = Pred - (Pred[0] - x_last)，起點貼回 x_last

用法:
    python visualize_anchored.py <run_dir> [--outputs outputs] [--n 9]
        [--mode even|topbias|random] [--windows 0,100,500] [--seed 0]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_anchored_mse import load_run_args, collect_x_last


def pick_windows(c_abs, n, mode, explicit, seed):
    N = len(c_abs)
    if explicit:
        return [int(i) for i in explicit.split(",")]
    if mode == "topbias":                       # 偏移最大的 window(效果最明顯)
        return sorted(np.argsort(c_abs)[-n:].tolist())
    if mode == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(N, size=min(n, N), replace=False).tolist())
    return np.linspace(0, N - 1, num=min(n, N), dtype=int).tolist()  # even


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out-subdir", default="anchored_metric")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--mode", default="even", choices=["even", "topbias", "random"])
    ap.add_argument("--windows", default=None, help="指定 window 索引,逗號分隔(覆蓋 --mode)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = os.path.join(a.run_dir, a.outputs)
    preds = np.load(os.path.join(out_dir, "pred.npy"))   # (N,H,nf)
    trues = np.load(os.path.join(out_dir, "true.npy"))

    persist_npy = os.path.join(out_dir, "persist.npy")
    if os.path.exists(persist_npy):
        x_last = np.load(persist_npy)[:, 0:1, :]
    else:
        x_last = collect_x_last(load_run_args(a.run_dir))

    c = preds[:, 0:1, :] - x_last
    adj = preds - c

    f = 0                                        # 畫第一個目標特徵
    H = preds.shape[1]
    hx = np.arange(1, H + 1)
    c_abs = np.abs(c[:, 0, f])
    idxs = pick_windows(c_abs, a.n, a.mode, a.windows, a.seed)

    ncol = 3
    nrow = int(np.ceil(len(idxs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax, w in zip(axes.ravel(), idxs):
        xl = x_last[w, 0, f]
        ax.plot([0], [xl], "ko", ms=6, label="x_last (anchor)")
        ax.plot(np.r_[0, hx], np.r_[xl, trues[w, :, f]], "-", color="#2ca02c", lw=2, label="GT")
        ax.plot(hx, preds[w, :, f], "--", color="#d62728", lw=1.8, label="Pred")
        ax.plot(np.r_[0, hx], np.r_[xl, adj[w, :, f]], "-.", color="#1f77b4", lw=1.8, label="adj_Pred")
        ax.set_title(f"win {w}   offset c={c[w,0,f]:+.1f}", fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"GT vs Pred vs adj_Pred  ({a.mode}, target={load_run_args(a.run_dir).target})",
                 fontsize=12)
    fig.supxlabel("horizon (t+k min)")
    fig.supylabel("water level")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    dst_dir = os.path.join(a.run_dir, a.out_subdir)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"viz_{a.mode}.png")
    fig.savefig(dst, dpi=160)
    plt.close(fig)
    print(f"windows 畫了: {idxs}")
    print(f"已寫出: {dst}")


if __name__ == "__main__":
    main()
