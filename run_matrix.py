"""實驗矩陣驅動腳本：dataset × exog × loss × fold 的因子實驗。

## 矩陣

    dataset(2)  drycut / 舊法        —— data_path 與 split_file 必須成對
    exog(2)     none / full          —— full 不含 isRain 與 min_since_rain
    loss(2)     mse / huber
    fold(3)     1 / 2 / 3            —— 重複維度，非處理因子；合表時取 mean±std
    seed        固定 42
    = 24 runs

## 兩階段設計

    階段一 plan  ：只產 manifest.csv（24 列，status=pending），不訓練
    階段二 run   ：執行 pending 的列，每跑完一列立刻寫回 manifest

manifest 存在的理由：
  1. run_dir 名稱**不含 loss 與 fold**（只有時間戳不同），24 個目錄光看名字
     分不出誰是誰，必須自己記錄對應關係。
  2. 支援續跑 —— 中斷後重跑會自動跳過 status=done 的列。
  3. 燒兩三個小時之前，可以先用 --dry-run 檢查 24 條指令是否正確。

## 安全檢查（跑之前就擋下來）

  - input_col 明列 HL02–06，**不可用 'HL*'**（會把 target HL01 餵進模型，見 OI-01）
  - split 檔的 sidecar 參數必須與本次設定一致（seq_len/pred_len/欄位集）
  - data_path 與 split_file 成對，避免用到別的資料集的切分

用法:
    python run_matrix.py plan                 # 產 manifest
    python run_matrix.py run --dry-run        # 印出 24 條指令，不執行
    python run_matrix.py run                  # 實際執行
    python run_matrix.py status               # 看進度
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_splits import check_sidecar

ROOT = Path("experiments")
MANIFEST = ROOT / "manifest.csv"
LOG_DIR = ROOT / "logs"
OUTPUT_ROOT = ROOT / "matrix_runs"

# ---------------------------------------------------------------- 因子定義
DATASETS = {
    "drycut": {"data_path": "train_drycut_L3h_buf60.csv",
               "split_file": "dataset/splits_train_drycut_L3h_buf60.csv"},
    "old":    {"data_path": "train_old.csv",
               "split_file": "dataset/splits_train_old.csv"},
}
EXOGS = {
    "none": None,
    "full": "Past10Min,Past1Hr,Now,*gate_opening*",
}
LOSSES = ["mse", "huber"]
FOLDS = [1, 2, 3]

# 固定不變的設定。input_col 明列，不可用 'HL*'：glob 會展開出 HL01（= target）。
CONST = {
    "model": "DLinearMix2",
    "data": "custom",
    "features": "S",
    "target": "HL01",
    "input_col": "HL02,HL03,HL04,HL05,HL06",
    "segment_col": "segment_id",
    "split_mode": "file",
    "seq_len": 60,
    "pred_len": 15,
    "label_len": 30,
    "batch_size": 64,
    "train_epochs": 80,
    "patience": 15,
    "learning_rate": 1e-3,
    "dropout": 0.1,
    "seed": 42,
    "early_stop_metric": "corr",
    "output_root": str(OUTPUT_ROOT),
}
FLAGS = ["--flatten_fusion"]

MANIFEST_COLS = ["run_key", "dataset", "exog", "loss", "fold", "status",
                 "run_dir", "started", "finished", "duration_s", "error"]


# ---------------------------------------------------------------- 前置檢查
def preflight() -> None:
    """跑之前把能靜態檢查的都擋下來，避免燒了兩小時才發現設定錯。"""
    problems = []

    if "HL01" in CONST["input_col"].split(","):
        problems.append("CONST.input_col 含 HL01（= target），違反核心原則（OI-01）。")

    for name, ds in DATASETS.items():
        csv = Path("dataset") / ds["data_path"]
        if not csv.exists():
            problems.append(f"[{name}] 找不到訓練 CSV：{csv}")
        if not Path(ds["split_file"]).exists():
            problems.append(f"[{name}] 找不到 split 檔：{ds['split_file']}")
            continue
        # sidecar 核對：split 的邊界是針對特定 seq_len/欄位集算的，對不上就會失真
        diffs = check_sidecar(
            ds["split_file"],
            seq_len=CONST["seq_len"], pred_len=CONST["pred_len"],
            target=CONST["target"], input_col=CONST["input_col"],
            exog_col=EXOGS["full"], segment_col=CONST["segment_col"],
        )
        for d in diffs:
            problems.append(f"[{name}] split sidecar 不一致 — {d}")

    if problems:
        print("前置檢查未通過：", file=sys.stderr)
        for p in problems:
            print(f"  ✗ {p}", file=sys.stderr)
        print("\n修正後再跑。split 檔可用下列指令重產：", file=sys.stderr)
        print(f"  python build_splits.py --data-path dataset/<csv> "
              f"--input-col '{CONST['input_col']}' --exog-col '{EXOGS['full']}' "
              f"--seq-len {CONST['seq_len']}", file=sys.stderr)
        sys.exit(1)
    print("前置檢查通過：資料集、split 檔、sidecar 參數皆一致。")


# ---------------------------------------------------------------- manifest
def build_manifest() -> pd.DataFrame:
    prev = {}
    if MANIFEST.exists():
        old = pd.read_csv(MANIFEST, keep_default_na=False)
        prev = {r.run_key: r._asdict() for r in old.itertuples(index=False)}

    rows = []
    for ds, ex, loss, fold in itertools.product(DATASETS, EXOGS, LOSSES, FOLDS):
        key = f"{ds}__{ex}__{loss}__f{fold}"
        if prev.get(key, {}).get("status") == "done":
            rows.append(prev[key])          # 已完成的原封不動保留，支援續跑
            continue
        rows.append({"run_key": key, "dataset": ds, "exog": ex, "loss": loss,
                     "fold": fold, "status": "pending", "run_dir": "",
                     "started": "", "finished": "", "duration_s": "", "error": ""})
    return pd.DataFrame(rows, columns=MANIFEST_COLS)


def save_manifest(df: pd.DataFrame) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)


# ---------------------------------------------------------------- 指令組裝
def build_cmd(row) -> list[str]:
    ds = DATASETS[row["dataset"]]
    cmd = [sys.executable, "run.py"]
    for k, v in CONST.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--data_path", ds["data_path"],
            "--split_file", ds["split_file"],
            "--fold", str(row["fold"]),
            "--criterion", row["loss"]]
    exog = EXOGS[row["exog"]]
    if exog is not None:                      # exog=none 時整個參數不給
        cmd += ["--exog_col", exog]
    cmd += FLAGS
    return cmd


def extract_run_dir(log_text: str) -> str:
    m = re.findall(r"run_dir='([^']+)'", log_text)
    return m[-1] if m else ""


# ---------------------------------------------------------------- 執行
def execute(dry_run: bool, only: str | None) -> None:
    df = pd.read_csv(MANIFEST, keep_default_na=False)
    todo = df[df.status != "done"]
    if only:
        todo = todo[todo.run_key.str.contains(only)]

    if dry_run:
        for _, row in todo.iterrows():
            print(f"\n# [{row.run_key}]")
            print("  " + " ".join(build_cmd(row)))
        print(f"\n共 {len(todo)} 條指令（--dry-run，未執行）。")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"待跑 {len(todo)} 組。")

    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        cmd = build_cmd(row)
        log_path = LOG_DIR / f"{row.run_key}.log"
        print(f"\n[{i}/{len(todo)}] {row.run_key}  {datetime.now():%H:%M:%S}")

        df.at[idx, "status"] = "running"
        df.at[idx, "started"] = datetime.now().astimezone().isoformat(timespec="seconds")
        save_manifest(df)                      # 每次都寫：中途 crash 也不丟進度

        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        dur = time.time() - t0
        log_text = log_path.read_text(encoding="utf-8", errors="replace")

        if proc.returncode == 0:
            df.at[idx, "run_dir"] = extract_run_dir(log_text)
            df.at[idx, "status"] = "done"
            df.at[idx, "error"] = ""
            print(f"    done  {dur/60:.1f} min")
        else:
            df.at[idx, "status"] = "failed"
            df.at[idx, "error"] = " | ".join(log_text.strip().splitlines()[-3:])[:500]
            print(f"    FAILED (rc={proc.returncode})  見 {log_path}")

        df.at[idx, "finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        df.at[idx, "duration_s"] = round(dur, 1)
        save_manifest(df)                      # 失敗也不中斷，繼續下一組

    status(df)


def status(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = pd.read_csv(MANIFEST, keep_default_na=False)
    counts = df.status.value_counts().to_dict()
    print(f"\n{'='*60}\n進度: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    done = df[df.status == "done"]
    if len(done) and (done.duration_s != "").any():
        secs = pd.to_numeric(done.duration_s, errors="coerce").dropna()
        print(f"已完成 {len(secs)} 組，單組平均 {secs.mean()/60:.1f} 分，累計 {secs.sum()/3600:.1f} 小時")
    failed = df[df.status == "failed"]
    if len(failed):
        print(f"\n失敗 {len(failed)} 組：")
        for _, r in failed.iterrows():
            print(f"  {r.run_key}: {r.error[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "run", "status"])
    ap.add_argument("--dry-run", action="store_true", help="只印指令不執行")
    ap.add_argument("--only", default=None, help="只跑 run_key 含此字串的組合")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if args.action == "plan":
        if not args.skip_preflight:
            preflight()
        df = build_manifest()
        save_manifest(df)
        print(f"manifest: {MANIFEST}（{len(df)} 組，"
              f"{int((df.status == 'pending').sum())} 待跑）")
        status(df)
    elif args.action == "run":
        if not MANIFEST.exists():
            sys.exit("找不到 manifest，請先跑：python run_matrix.py plan")
        if not args.skip_preflight and not args.dry_run:
            preflight()
        execute(args.dry_run, args.only)
    else:
        status()


if __name__ == "__main__":
    main()
