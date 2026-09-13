"""實驗矩陣合表器：把 24 個 run 的結果彙整成可橫向比較的表。

## 為什麼 anchored 放在這裡算（而非訓練時）

anchored 是一個純函數，輸入全部來自已存下的 npy：

    x_last   = persist[:, 0, :]                    # persistence 沿 horizon 廣播
    anchored = pred - (pred[:, 0:1, :] - x_last[:, None, :])

放在事後的好處：公式要改時重跑幾秒即可（不必重訓 24 次）、訓練碼完全不動、
與既有的 `compute_anchored_mse.py` 共用同一套定義不會漂移。

## 輸出

    results_long.csv       tidy：一列 = (run × checkpoint × anchoring)，24×2×3 = 144 列
    summary_by_config.csv  對 fold 取 mean±std —— **這是拿來做判斷的表**
    results_wide.csv       給人看的樞紐表

fold 是**重複維度不是處理因子**：我們不問「fold 2 比 fold 1 好嗎」（各 fold 的
val 期間不同、難度本就不同），而是用它看同一設定穩不穩。因此 summary 裡
`corr_std` 與 `corr_mean` 同等重要——平均高但 std 大代表只是某個 fold 運氣好。

## 只有 test 有 anchored

`outputs/` 只存 test 的預測，沒有 val 的 npy，故 val 僅能取訓練過程的指標
（在 summary CSV 裡），算不了 anchored。headline 看 test。

用法:
    python collect_matrix.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("experiments")
MANIFEST = ROOT / "manifest.csv"
RESULTS = ROOT / "results"

CKPTS = [("best_corr", "outputs"), ("best_mse", "outputs_alt")]


def corr_per_horizon(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐 horizon 的 Pearson 相關（跨 window）。"""
    out = []
    for h in range(pred.shape[1]):
        p, t = pred[:, h, 0], true[:, h, 0]
        if p.std() == 0 or t.std() == 0:
            out.append(np.nan)
        else:
            out.append(float(np.corrcoef(p, t)[0, 1]))
    return np.array(out, dtype=float)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    ch = corr_per_horizon(pred, true)
    return {
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "corr": float(np.nanmean(ch)),
        "corr_h1": float(ch[0]),
        "corr_hN": float(ch[-1]),
    }


def per_segment_mse(pred, true, seg_ids) -> dict:
    """per-segment MSE 的中位數與 worst-decile。

    有效樣本單位是「降雨事件」而非 window（同段內 window 高度相關），
    故 pooled MSE 會被長颱風段主導；per-segment 統計才看得出泛化失敗。
    """
    if seg_ids is None:
        return {"seg_mse_median": np.nan, "seg_mse_p90": np.nan, "n_segments": np.nan}
    se = ((pred - true) ** 2).mean(axis=(1, 2))
    d = pd.DataFrame({"seg": seg_ids, "se": se}).groupby("seg")["se"].mean()
    return {"seg_mse_median": float(d.median()),
            "seg_mse_p90": float(d.quantile(0.90)),
            "n_segments": int(len(d))}


def load_seg_ids(outdir: Path, n: int):
    """從 metrics_segment.csv 之外的來源取每個 window 的 segment_id（若有）。"""
    f = outdir / "segment_horizon_points.csv.gz"
    if not f.exists():
        return None
    try:
        # 欄名是 'segment'（非 segment_id），逐 horizon 展開故需去重回 window 層級
        d = pd.read_csv(f, usecols=["window_idx", "segment"]).drop_duplicates("window_idx")
        if len(d) == n:
            return d.sort_values("window_idx")["segment"].to_numpy()
        print(f"  [WARN] {f.name}: window 數 {len(d)} 與 pred 的 {n} 不符，略過 per-segment 統計。")
    except Exception as e:
        print(f"  [WARN] 讀 {f.name} 失敗（{e}），略過 per-segment 統計。")
    return None


def collect(manifest_path: Path) -> pd.DataFrame:
    man = pd.read_csv(manifest_path, keep_default_na=False)
    done = man[man.status == "done"]
    if done.empty:
        raise SystemExit("manifest 裡沒有 status=done 的 run。")

    rows = []
    for _, r in done.iterrows():
        run_dir = Path(r.run_dir)
        if not run_dir.exists():
            print(f"  [WARN] {r.run_key}: 找不到 run_dir {run_dir}，跳過。")
            continue
        args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))

        # 防線：實驗的核心前提是 HL01 不在 input
        assert args["target"] not in (args.get("input_col") or "").split(","), \
            f"{r.run_key}: target 出現在 input_col"

        for ckpt, sub in CKPTS:
            outdir = run_dir / sub
            try:
                pred = np.load(outdir / "pred.npy")
                true = np.load(outdir / "true.npy")
                persist = np.load(outdir / "persist.npy")
            except FileNotFoundError as e:
                print(f"  [WARN] {r.run_key}/{sub}: {e.filename} 不存在，跳過。")
                continue

            x_last = persist[:, 0:1, :]
            seg_ids = load_seg_ids(outdir, len(pred))

            variants = {
                "raw": pred,
                "anchored": pred - (pred[:, 0:1, :] - x_last),
                "persistence": persist,
            }
            for mode, p in variants.items():
                rows.append({
                    "run_key": r.run_key, "dataset": r.dataset, "exog": r.exog,
                    "loss": r.loss, "fold": int(r.fold),
                    "checkpoint": ckpt, "anchoring": mode,
                    **metrics(p, true),
                    **per_segment_mse(p, true, seg_ids),
                    "n_windows": int(len(p)),
                    # 記錄「實際用到」的欄位：run.py 的 _drop_constant_columns 會砍常數欄，
                    # 兩個資料集砍掉的若不同，dataset 比較就不只一個變因。
                    "exog_in": args.get("exog_in"),
                    "exog_cols_actual": ",".join(args.get("exog_cols") or []),
                    "input_cols_actual": ",".join(args.get("input_cols") or []),
                    "seq_len": args.get("seq_len"), "epochs": args.get("train_epochs"),
                    "split_file": args.get("split_file"),
                    "duration_s": r.duration_s,
                })
    return pd.DataFrame(rows)


def add_improve(df: pd.DataFrame) -> pd.DataFrame:
    """加上相對 persistence 的改善百分比。

    單看 MSE 無法跨 fold 比較（各 fold 的 test 難度不同）；相對 persistence
    的改善幅度才是可比的量。
    """
    base = (df[df.anchoring == "persistence"]
            .set_index(["run_key", "checkpoint"])["mse"].rename("persist_mse"))
    df = df.join(base, on=["run_key", "checkpoint"])
    df["improve_mse_pct"] = (df.persist_mse - df.mse) / df.persist_mse * 100
    return df


def sanity(df: pd.DataFrame) -> None:
    print("\n--- 一致性檢查 ---")
    for ds, g in df.groupby("dataset"):
        n = g.exog_cols_actual[g.exog == "full"].nunique()
        print(f"  {ds}: exog=full 的實際欄位組合數 = {n} (應為 1)")
    combos = df.groupby(["dataset", "exog", "loss"]).fold.nunique()
    bad = combos[combos != 3]
    print(f"  每個 (dataset,exog,loss) 的 fold 數皆為 3: {'是' if bad.empty else f'否 → {dict(bad)}'}")
    full = df[df.exog == "full"].groupby("dataset").exog_in.unique()
    print(f"  exog=full 的 exog_in: {dict(full)}"
          f"  ← 兩資料集若不同，dataset 比較就不只一個變因")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()

    df = add_improve(collect(Path(args.manifest)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "results_long.csv", index=False)

    # fold 被 aggregate 掉：std 與 mean 同等重要
    summary = (df.groupby(["dataset", "exog", "loss", "checkpoint", "anchoring"])
                 .agg(corr_mean=("corr", "mean"), corr_std=("corr", "std"),
                      mse_mean=("mse", "mean"), mse_std=("mse", "std"),
                      improve_mean=("improve_mse_pct", "mean"),
                      improve_std=("improve_mse_pct", "std"),
                      seg_mse_median_mean=("seg_mse_median", "mean"),
                      n_folds=("fold", "count"))
                 .round(4).reset_index())
    summary.to_csv(out / "summary_by_config.csv", index=False)

    wide = (df[(df.anchoring == "anchored") & (df.checkpoint == "best_corr")]
            .pivot_table(index=["dataset", "exog", "loss"], columns="fold", values="corr")
            .round(4))
    wide.to_csv(out / "results_wide.csv")

    sanity(df)
    print(f"\n輸出: {out}/results_long.csv ({len(df)} 列)")
    print(f"      {out}/summary_by_config.csv ({len(summary)} 列)")
    print(f"      {out}/results_wide.csv")

    print("\n--- anchored / best_corr 的 headline ---")
    h = summary[(summary.anchoring == "anchored") & (summary.checkpoint == "best_corr")]
    print(h[["dataset", "exog", "loss", "corr_mean", "corr_std",
             "improve_mean", "improve_std", "n_folds"]].to_string(index=False))


if __name__ == "__main__":
    main()
