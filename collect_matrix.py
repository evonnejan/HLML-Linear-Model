"""實驗矩陣合表器：把各 run 的結果彙整成可橫向比較的表。

## anchored 為何在此事後計算

anchored 是純函數，輸入全部來自已存的 npy：

    x_last   = persist[:, 0, :]                    # persistence 沿 horizon 廣播
    anchored = pred - (pred[:, 0:1, :] - x_last)

放事後的好處：公式要改時重跑幾秒即可（不必重訓）、訓練碼完全不動。

## corr 的定義與範圍限制

本檔對常數 horizon（std==0）記 NaN 並排除，同時回報**有效 horizon 數** ——
與 `exp_Main2.vali()`（early stopping 用）的慣例一致。

但 `utils.metrics.CORR`（最終 test 報表、既有 anchored 工具）在分母加 1e-12，
常數 horizon 會算成 **0** 而非排除。**本版僅統一合表器內部的定義，尚未統一
所有入口**；兩者的數字不得混用或一起平均。

## eval_version

`EVAL_VERSION` 是人讀的標籤，`eval_code_hash` 是關鍵函式原始碼的雜湊。
兩者不同步時直接報錯 —— 這樣「改了公式忘了 bump 版本」不可能發生。
hash 範圍刻意只涵蓋定義數字的函式，改表格顯示不會誤觸發。

## 輸出

    results_long.csv       一列 = (run × checkpoint × anchoring)
    summary_by_config.csv  對 fold 取 mean±std —— 拿來做判斷的表
    results_wide.csv       樞紐表

fold 是重複維度不是處理因子：不問「fold 2 比 fold 1 好嗎」，而是看同一設定
穩不穩，故 `corr_std` 與 `corr_mean` 同等重要。

用法:
    python collect_matrix.py            # 要求結果完整，缺漏即失敗
    python collect_matrix.py --partial  # 允許部分結果，輸出明確標記
"""
import argparse
import hashlib
import inspect
import json
import logging
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from run_matrix import RUN_PARAM_KEYS, _norm, hash_params, read_manifest

ROOT = Path("experiments")
PLAN = ROOT / "plan.json"
MANIFEST = ROOT / "manifest.csv"
RESULTS = ROOT / "results"

EVAL_VERSION = "v1-corr-skip-constant"
ANCHORINGS = ["raw", "anchored", "persistence"]

log = logging.getLogger("collect")


class CollectError(RuntimeError):
    """必須拒收的情況。用明確例外而非 assert（assert 會被 python -O 移除）。"""


# ---------------------------------------------------------------- 指標
def corr_per_horizon(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """逐 horizon Pearson corr。常數序列記 NaN（非 0），並回報無效原因。"""
    out, reasons = [], []
    for h in range(pred.shape[1]):
        p, t = pred[:, h, 0], true[:, h, 0]
        if p.std() == 0 and t.std() == 0:
            out.append(np.nan); reasons.append(f"h{h+1}:pred&true皆常數")
        elif p.std() == 0:
            out.append(np.nan); reasons.append(f"h{h+1}:pred常數")
        elif t.std() == 0:
            out.append(np.nan); reasons.append(f"h{h+1}:true常數")
        else:
            out.append(float(np.corrcoef(p, t)[0, 1]))
    return np.array(out, dtype=float), reasons


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    ch, reasons = corr_per_horizon(pred, true)
    n_valid = int(np.isfinite(ch).sum())
    return {
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "corr": float(np.nanmean(ch)) if n_valid else float("nan"),
        "corr_h1": float(ch[0]), "corr_hN": float(ch[-1]),
        "corr_valid_horizons": n_valid, "corr_total_horizons": int(len(ch)),
        "corr_invalid_reasons": ";".join(reasons),
    }


def anchor_transform(pred: np.ndarray, persist: np.ndarray) -> np.ndarray:
    """整條平移使 pred[0] == x_last，保留形狀只修 level。"""
    return pred - (pred[:, 0:1, :] - persist[:, 0:1, :])


def per_segment_mse(pred, true, seg_ids) -> dict:
    """per-segment MSE 的中位數與 worst-decile。

    有效樣本單位是降雨事件而非 window（同段內 window 高度相關），pooled MSE
    會被長颱風段主導；per-segment 統計才看得出泛化失敗。
    """
    if seg_ids is None:
        return {"seg_mse_median": np.nan, "seg_mse_p90": np.nan, "n_segments": pd.NA}
    se = ((pred - true) ** 2).mean(axis=(1, 2))
    d = pd.DataFrame({"seg": seg_ids, "se": se}).groupby("seg")["se"].mean()
    return {"seg_mse_median": float(d.median()), "seg_mse_p90": float(d.quantile(0.90)),
            "n_segments": int(len(d))}


EVAL_FUNCS = [corr_per_horizon, metrics, anchor_transform, per_segment_mse]
EVAL_CODE_HASH = hashlib.sha256(
    "".join(inspect.getsource(f) for f in EVAL_FUNCS).encode()).hexdigest()[:8]

# 每次修改上列函式就必須同步 bump EVAL_VERSION 並更新此值。
EXPECTED_CODE_HASH = "2569904e"


def check_eval_version() -> None:
    if EVAL_CODE_HASH != EXPECTED_CODE_HASH:
        raise CollectError(
            f"評估邏輯已變更（code hash {EXPECTED_CODE_HASH} → {EVAL_CODE_HASH}）"
            f"但 EVAL_VERSION 仍為 {EVAL_VERSION!r}。\n"
            f"請 bump EVAL_VERSION 並把 EXPECTED_CODE_HASH 改為 {EVAL_CODE_HASH!r}；"
            f"舊版結果不可與新版混用或一起平均。")


# ---------------------------------------------------------------- 載入與核對
def load_seg_ids(outdir: Path, n: int):
    f = outdir / "segment_horizon_points.csv.gz"
    if not f.exists():
        return None
    try:
        # 欄名是 'segment'（非 segment_id），逐 horizon 展開故需去重回 window 層級
        d = pd.read_csv(f, usecols=["window_idx", "segment"]).drop_duplicates("window_idx")
        if len(d) == n:
            return d.sort_values("window_idx")["segment"].to_numpy()
        log.warning("%s: window 數 %d 與 pred 的 %d 不符，略過 per-segment 統計。", f.name, len(d), n)
    except Exception as e:
        log.warning("讀 %s 失敗（%s），略過 per-segment 統計。", f.name, e)
    return None


def checkpoint_dirs(args: dict) -> list[tuple[str, str]]:
    """checkpoint 名稱由 early_stop_metric 推導，對齊 exp_Main2.py:407-417。

    寫死 best_corr/best_mse 會在 early_stop_metric 改動或遇到舊 run 時標反。
    """
    primary = str(args.get("early_stop_metric", "mse")).lower()
    alt = "corr" if primary in ("mse", "mae") else "mse"
    return [(f"best_{primary}", "outputs"), (f"best_{alt}", "outputs_alt")]


def cross_check(row, args: dict, cell: dict) -> None:
    """manifest 標籤 / run_args 實際值 / plan 凍結設定 三方核對。"""
    p = cell["params"]
    actual = {k: _norm(args.get(k)) for k in RUN_PARAM_KEYS}
    # exog 比對實際欄位與順序，不只看 exog_in > 0
    actual["exog_col"] = _norm(",".join(args.get("exog_cols") or []) or None)
    expect = dict(p)
    expect["exog_col"] = _norm(
        ",".join(args.get("exog_cols") or []) or None) if p["exog_col"] is None and not args.get("exog_cols") else p["exog_col"]

    diffs = []
    for k in ["criterion", "fold", "data_path", "split_file", "seed", "seq_len",
              "pred_len", "target", "input_col", "early_stop_metric",
              "learning_rate", "train_epochs", "batch_size", "dropout"]:
        if actual.get(k) != p.get(k):
            diffs.append(f"{k}: run_args={actual.get(k)!r} vs plan={p.get(k)!r}")
    # exog：以實際欄位比對
    want_exog = p["exog_col"]
    got_exog = ",".join(args.get("exog_cols") or []) or None
    if want_exog is None and got_exog is not None:
        diffs.append(f"exog_col: 計畫為 none 但實際用了 {got_exog!r}")
    if want_exog is not None and got_exog is None:
        diffs.append(f"exog_col: 計畫為 {want_exog!r} 但實際沒有 exog")
    if args.get("target") in (args.get("input_cols") or []):
        diffs.append(f"target {args.get('target')} 出現在 input_cols（OI-01）")
    for k, v in cell["params"].items():
        if k in ("huber_beta", "dlinear_kernel_size", "lradj", "warmup_epochs",
                 "lr_decay_gamma", "lr_floor_ratio", "stride_train", "stride_eval",
                 "exog_emb_dim", "fusion_hidden_dim") and _norm(args.get(k)) != v:
            diffs.append(f"{k}(隱含預設): run_args={args.get(k)!r} vs plan={v!r}")

    if diffs:
        raise CollectError(f"[{row.run_key}] run_dir 的內容與計畫不符，拒收：\n    "
                           + "\n    ".join(diffs))


# ---------------------------------------------------------------- 收集
def collect(plan: dict, man: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows, notes = [], []
    for _, r in man[man.status == "done"].iterrows():
        run_dir = Path(r.run_dir)
        if not run_dir.exists():
            notes.append(f"{r.run_key}: run_dir 不存在 {run_dir}")
            continue
        args = json.loads((run_dir / "run_args.json").read_text(encoding="utf-8"))
        cell = plan["cells"][r.run_key]
        cross_check(r, args, cell)

        for ckpt, sub in checkpoint_dirs(args):
            outdir = run_dir / sub
            try:
                pred = np.load(outdir / "pred.npy")
                true = np.load(outdir / "true.npy")
                persist = np.load(outdir / "persist.npy")
            except FileNotFoundError as e:
                notes.append(f"{r.run_key}/{ckpt}: 缺少 {Path(e.filename).name}")
                continue

            if not (pred.shape == true.shape == persist.shape):
                raise CollectError(
                    f"[{r.run_key}/{ckpt}] shape 不一致：pred{pred.shape} "
                    f"true{true.shape} persist{persist.shape}（不可靠 broadcasting 掩蓋錯位）")
            for nm, a in [("pred", pred), ("true", true), ("persist", persist)]:
                if not np.isfinite(a).all():
                    raise CollectError(f"[{r.run_key}/{ckpt}] {nm} 含 NaN/inf，屬數值失敗，拒收。")

            seg_ids = load_seg_ids(outdir, len(pred))
            for mode, p in {"raw": pred,
                            "anchored": anchor_transform(pred, persist),
                            "persistence": persist}.items():
                m = metrics(p, true)
                if m["corr_valid_horizons"] < m["corr_total_horizons"]:
                    log.warning("%s/%s/%s: 僅 %d/%d 個 horizon 有有效 corr（%s）",
                                r.run_key, ckpt, mode, m["corr_valid_horizons"],
                                m["corr_total_horizons"], m["corr_invalid_reasons"])
                rows.append({
                    "run_key": r.run_key, "dataset": r.dataset, "exog": r.exog,
                    "loss": r.loss, "fold": int(r.fold), "checkpoint": ckpt,
                    "anchoring": mode, **m, **per_segment_mse(p, true, seg_ids),
                    "n_windows": int(len(p)),
                    "exog_in": args.get("exog_in"),
                    "exog_cols_actual": ",".join(args.get("exog_cols") or []),
                    "input_cols_actual": ",".join(args.get("input_cols") or []),
                    "seq_len": args.get("seq_len"),
                    "config_hash": r.config_hash, "plan_id": plan["plan_id"],
                    "eval_version": EVAL_VERSION, "eval_code_hash": EVAL_CODE_HASH,
                })
    return pd.DataFrame(rows), notes


def completeness(plan: dict, df: pd.DataFrame, partial: bool) -> None:
    """對照凍結 plan 的預期格線查缺漏與重複。

    必須先用 Counter 查重複再用 set 查缺漏：set() 會先把重複吃掉，
    導致 missing=0 且 extra=0 卻其實有同一格出現兩次。
    """
    ck_names = set()
    for _, g in df.groupby("run_key"):
        ck_names |= set(g.checkpoint.unique())
    expected = {(k, c, a) for k in plan["cells"] for c in sorted(ck_names) for a in ANCHORINGS}
    got = [(r.run_key, r.checkpoint, r.anchoring) for r in df.itertuples()]

    dups = {k: v for k, v in Counter(got).items() if v > 1}
    missing = expected - set(got)
    extra = set(got) - expected

    print(f"\n--- 完整性（預期 {len(expected)} 筆，實得 {len(got)} 筆）---")
    for label, items in [("重複", dups), ("缺漏", missing), ("非預期", extra)]:
        print(f"  {label}: {len(items)}" + (f"  例如 {sorted(items)[:3]}" if items else ""))
    if (dups or missing or extra) and not partial:
        raise CollectError("結果不完整或有重複。確認原因後再產正式報告；"
                           "只要看部分進度請加 --partial（輸出會標記為 partial）。")


def sanity(df: pd.DataFrame) -> None:
    print("\n--- 一致性 ---")
    for ds, g in df.groupby("dataset"):
        f = g[g.exog == "full"]
        if len(f):
            print(f"  {ds}: exog=full 的實際欄位組合數 = {f.exog_cols_actual.nunique()} (應為 1)"
                  f"，exog_in = {sorted(f.exog_in.unique())}")
    if df.eval_version.nunique() > 1:
        raise CollectError(f"表內混有多個 eval_version：{sorted(df.eval_version.unique())}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--plan", default=str(PLAN))
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--partial", action="store_true", help="允許不完整結果，輸出標記為 partial")
    args = ap.parse_args()

    check_eval_version()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    man = read_manifest(Path(args.manifest))
    df, notes = collect(plan, man)
    if df.empty:
        sys.exit("沒有可收集的結果（manifest 無 status=done 或 run_dir 皆不存在）。")
    for n in notes:
        log.warning("%s", n)

    completeness(plan, df, args.partial)
    sanity(df)

    # 相對 persistence 的改善率：各 fold 的 test 難度不同，絕對 MSE 不可跨 fold 比
    base = (df[df.anchoring == "persistence"]
            .set_index(["run_key", "checkpoint"])["mse"].rename("persist_mse"))
    df = df.join(base, on=["run_key", "checkpoint"])
    df["improve_mse_pct"] = (df.persist_mse - df.mse) / df.persist_mse * 100

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "_partial" if args.partial else ""
    df.to_csv(out / f"results_long{tag}.csv", index=False)

    full = df.corr_valid_horizons == df.corr_total_horizons
    summary = (df.assign(_full=full)
                 .groupby(["dataset", "exog", "loss", "checkpoint", "anchoring"])
                 .agg(corr_mean=("corr", "mean"), corr_std=("corr", "std"),
                      mse_mean=("mse", "mean"), mse_std=("mse", "std"),
                      improve_mean=("improve_mse_pct", "mean"),
                      improve_std=("improve_mse_pct", "std"),
                      seg_mse_median_mean=("seg_mse_median", "mean"),
                      n_folds_present=("fold", "nunique"),
                      n_folds_corr_valid=("corr", lambda s: int(s.notna().sum())),
                      n_folds_mse_valid=("mse", lambda s: int(s.notna().sum())),
                      n_folds_all_horizons=("_full", lambda s: int(s.sum())))
                 .round(4).reset_index())
    summary["n_folds_expected"] = len(plan["grid"]["folds"])
    summary["eval_version"] = EVAL_VERSION
    summary["eval_code_hash"] = EVAL_CODE_HASH
    summary.to_csv(out / f"summary_by_config{tag}.csv", index=False)

    wide = (df[(df.anchoring == "anchored") & (df.checkpoint.str.endswith("corr"))]
            .pivot_table(index=["dataset", "exog", "loss"], columns="fold",
                         values=["corr", "mse"]).round(4))
    wide = wide.assign(eval_version=EVAL_VERSION, eval_code_hash=EVAL_CODE_HASH)
    wide.to_csv(out / f"results_wide{tag}.csv")

    print(f"\n輸出: {out}/results_long{tag}.csv ({len(df)} 列)")
    print(f"      {out}/summary_by_config{tag}.csv ({len(summary)} 列)")
    print(f"      {out}/results_wide{tag}.csv")

    # 主表：corr 與 MSE 來自同一份 best_corr + anchored 預測。
    # 改善率不能取代 MSE —— baseline 10,000 改善 80% 仍是 2,000。
    head = summary[(summary.anchoring == "anchored")
                   & (summary.checkpoint.str.endswith("corr"))]
    print(f"\n{'='*100}")
    print(f"主表  best_corr + anchored   |   eval: {EVAL_VERSION} ({EVAL_CODE_HASH})"
          + ("   [PARTIAL]" if args.partial else ""))
    print("=" * 100)
    print(head[["dataset", "exog", "loss", "corr_mean", "corr_std",
                "mse_mean", "mse_std", "improve_mean", "improve_std",
                "n_folds_expected", "n_folds_present", "n_folds_corr_valid",
                "n_folds_all_horizons"]].to_string(index=False))
    print("\n註：corr 對常數 horizon 記 NaN 並排除（與 exp_Main2.vali 一致）；"
          "\n    utils.metrics.CORR 則算成 0，兩者數字不得混用或一起平均。")


if __name__ == "__main__":
    try:
        main()
    except CollectError as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)
