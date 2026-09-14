"""實驗矩陣驅動腳本：dataset × exog × loss × fold 的因子實驗。

## 矩陣

    dataset(2)  drycut / 舊法        —— data_path 與 split_file 必須成對
    exog(2)     none / full          —— full 不含 isRain 與 min_since_rain
    loss(2)     mse / huber
    fold(3)     1 / 2 / 3            —— 重複維度，非處理因子；合表時取 mean±std
    seed        固定 42
    = 24 runs

## 兩階段與凍結

    plan  產出 plan.json（設定快照）與 manifest.csv，不訓練
    run   **一律從 plan.json 讀參數**，不讀本模組的全域變數

凍結的理由：plan 之後若改了 CONST，未執行的 pending 也不該套用新值，
否則同一批實驗會混到兩種設定。因此執行期只信快照。

身份分兩層：
    plan_id          格線定義的 hash（有哪些格子）
    config_hash      逐 run 的完整參數 hash（這一格怎麼跑）——重用與否只看它

這讓失效範圍精準：改 learning_rate → 24 格全變；加一個 exog level → 既有
格子不變只新增格子；重產某個資料集 → 只有該資料集的格子失效。

## 安全檢查（燒時間之前就擋下）

  - input_col 明列 HL02–06，不可用 'HL*'（會展開出 target HL01，見 OI-01）
  - split sidecar 三層核對：資料內容 hash、split 自身 hash、分派有效性
  - 同群組（dataset,exog,loss）的三個 fold 狀態必須一致，不可只失效其中一折

用法:
    python run_matrix.py plan                  # 建立/更新計畫
    python run_matrix.py plan --accept-changes # 設定已變更，受影響者轉 pending
    python run_matrix.py plan --new            # 另開一批，舊結果完整保留
    python run_matrix.py run --dry-run         # 印出指令，不執行
    python run_matrix.py run                   # 實際執行
    python run_matrix.py status
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_splits import file_sha256, load_sidecar

ROOT = Path("experiments")
PLAN = ROOT / "plan.json"
MANIFEST = ROOT / "manifest.csv"
LOG_DIR = ROOT / "logs"
OUTPUT_ROOT = ROOT / "matrix_runs"
DATA_ROOT = Path("dataset")

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

# 固定設定。input_col 明列：'HL*' 會展開出 HL01（= target）。
CONST = {
    "model": "DLinearMix2", "data": "custom", "features": "S", "target": "HL01",
    "input_col": "HL02,HL03,HL04,HL05,HL06",
    "segment_col": "segment_id", "split_mode": "file",
    "seq_len": 60, "pred_len": 15, "label_len": 30,
    "batch_size": 64, "train_epochs": 80, "patience": 15,
    "learning_rate": 1e-3, "dropout": 0.1, "seed": 42,
    "early_stop_metric": "corr", "output_root": str(OUTPUT_ROOT),
}
FLAGS = ["--flatten_fusion"]

# run.py 未由本腳本指定、但會影響結果的預設值。納入 hash，並在合表時
# 驗證實際 run_args 與此一致 —— run.py 的預設若改動，該檢查會抓到。
IMPLICIT_DEFAULTS = {
    "huber_beta": 1.0, "dlinear_kernel_size": 25, "lradj": "type1",
    "warmup_epochs": 8, "lr_decay_gamma": 0.92, "lr_floor_ratio": 0.01,
    "stride_train": 1, "stride_eval": 1, "individual": True,
    "exog_emb_dim": 16, "fusion_hidden_dim": 32,
    "embed": "timeF", "freq": "min", "train_only": False, "use_amp": False,
}

# 定義一個 run 的參數白名單。plan 與 collect 共用同一份，確保兩邊
# 用相同欄位與正規化方式算 hash（不可拿整份含時間戳的 run_args 去比）。
RUN_PARAM_KEYS = sorted(
    set(CONST) - {"output_root"}
    | set(IMPLICIT_DEFAULTS)
    | {"data_path", "split_file", "fold", "criterion", "exog_col", "flatten_fusion"}
)

MANIFEST_DTYPES = {
    "run_key": "string", "dataset": "string", "exog": "string", "loss": "string",
    "status": "string", "run_dir": "string", "superseded_run_dir": "string",
    "started": "string", "finished": "string", "error": "string",
    "config_hash": "string", "fold": "Int64", "duration_s": "Float64",
}
MANIFEST_COLS = ["run_key", "dataset", "exog", "loss", "fold", "config_hash", "status",
                 "run_dir", "superseded_run_dir", "started", "finished",
                 "duration_s", "error"]
VALID_STATUS = {"pending", "running", "done", "failed"}


# ---------------------------------------------------------------- 雜湊
def _norm(v):
    """正規化：bool 先於 int 判斷（isinstance(True, int) 為真）。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return format(v, ".12g")
    if v is None:
        return None
    return str(v)


def run_params(dataset: str, exog: str, loss: str, fold: int, const: dict,
               implicit: dict) -> dict:
    """組出一個 run 的完整參數（plan 端）。"""
    ds = DATASETS[dataset] if dataset in DATASETS else None
    p = {**const, **implicit,
         "data_path": ds["data_path"], "split_file": ds["split_file"],
         "fold": fold, "criterion": loss, "exog_col": EXOGS[exog],
         "flatten_fusion": "--flatten_fusion" in FLAGS}
    return {k: _norm(p.get(k)) for k in RUN_PARAM_KEYS}


def hash_params(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def compute_plan() -> dict:
    """格線定義 + 每一格的完整參數，一起凍結。"""
    cells = {}
    for ds, ex, lo, fo in itertools.product(DATASETS, EXOGS, LOSSES, FOLDS):
        key = f"{ds}__{ex}__{lo}__f{fo}"
        params = run_params(ds, ex, lo, fo, CONST, IMPLICIT_DEFAULTS)
        cells[key] = {"dataset": ds, "exog": ex, "loss": lo, "fold": fo,
                      "params": params, "config_hash": hash_params(params)}
    grid = {"datasets": DATASETS, "exogs": EXOGS, "losses": LOSSES, "folds": FOLDS}
    plan_id = hashlib.sha256(
        json.dumps(sorted(cells), ensure_ascii=False).encode()).hexdigest()[:12]
    return {"plan_id": plan_id, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "grid": grid, "const": CONST, "implicit_defaults": IMPLICIT_DEFAULTS,
            "flags": FLAGS, "run_param_keys": RUN_PARAM_KEYS, "cells": cells}


# ---------------------------------------------------------------- manifest IO
def read_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    """讀 manifest 並套用型別。

    只讓數值欄接受空值：若對整張表用 na_values=[""]，run_dir/error 等會變成
    pd.NA，之後 Path(NA) / NA[:160] / bool(NA) 都會拋 TypeError。
    """
    df = pd.read_csv(path, dtype=MANIFEST_DTYPES, keep_default_na=False,
                     na_values={"fold": [""], "duration_s": [""]})
    for c in MANIFEST_COLS:
        if c not in df.columns:
            raise ValueError(f"manifest 缺少欄位 '{c}': {path}")
    bad = set(df.status.dropna()) - VALID_STATUS
    if bad:
        raise ValueError(f"manifest 有非法 status: {sorted(bad)}")
    if df.fold.isna().any():
        raise ValueError("manifest 有空的 fold")
    miss = df[(df.status == "done") & (df.run_dir.fillna("") == "")]
    if len(miss):
        raise ValueError(f"manifest 有 status=done 但 run_dir 為空的列: {list(miss.run_key)}")
    return df


def save_manifest(df: pd.DataFrame, path: Path = MANIFEST) -> None:
    """同目錄暫存檔 + os.replace，避免中斷時留下半截 CSV。

    這只保證單一驅動器下不留下不完整檔案，**不代表支援多程序同時寫 manifest**。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 前置檢查
def check_split_identity(name: str, ds: dict, plan_const: dict) -> list[str]:
    """sidecar 三層核對：資料身份、split 身份、分派有效性。"""
    problems = []
    split_csv = Path(ds["split_file"])
    data_csv = DATA_ROOT / ds["data_path"]
    if not data_csv.exists():
        return [f"[{name}] 找不到訓練 CSV：{data_csv}"]
    if not split_csv.exists():
        return [f"[{name}] 找不到 split 檔：{split_csv}"]

    side = load_sidecar(str(split_csv))
    if side is None:
        return [f"[{name}] 找不到 sidecar {split_csv.with_suffix('.json')}；請重產 split。"]

    # 第 1 層 —— 資料身份。路徑字串格式可能不同（'dataset/x.csv' vs 'x.csv'），
    # 一律以內容 hash 為準。
    if side.get("data_sha256") != file_sha256(data_csv):
        problems.append(f"[{name}] split 是用另一份資料產生的（data_sha256 不符）。")

    # 第 2 層 —— split 自身身份，防「新 CSV 配舊 sidecar」
    if side.get("split_sha256") != file_sha256(split_csv):
        problems.append(f"[{name}] split CSV 內容與 sidecar 記錄不符（split_sha256）。")

    # 純量參數
    for k, want in [("seq_len", plan_const["seq_len"]), ("pred_len", plan_const["pred_len"]),
                    ("target", plan_const["target"]), ("input_col", plan_const["input_col"]),
                    ("segment_col", plan_const["segment_col"]),
                    ("exog_col", EXOGS["full"])]:
        if str(side.get(k)) != str(want):
            problems.append(f"[{name}] sidecar.{k}={side.get(k)!r} 與計畫的 {want!r} 不符。")

    # 第 3 層 —— 分派有效性
    sp = pd.read_csv(split_csv, keep_default_na=False)
    seg = plan_const["segment_col"]
    if sp[seg].duplicated().any():
        problems.append(f"[{name}] split 檔有重複的 {seg}。")
    base = set(sp["split"].astype(str))
    if not base <= {"train", "val", "test"}:
        # base split 不可為空：空值只在 fold 欄代表「該折未使用」
        problems.append(f"[{name}] split 欄有非法值 {sorted(base - {'train','val','test'})}（不可為空）。")
    for f in plan_const["_folds"]:
        col = f"fold_{f}"
        if col not in sp.columns:
            problems.append(f"[{name}] split 檔缺少 {col}。")
            continue
        vals = set(sp[col].astype(str))
        if not vals <= {"train", "val", ""}:
            problems.append(f"[{name}] {col} 有非法值 {sorted(vals - {'train','val',''})}。")
        if (sp[col] == "val").sum() == 0:
            problems.append(f"[{name}] {col} 沒有任何 val segment。")
        # fold 的 train/val 不得與 base 的 test 重疊
        test_ids = set(sp.loc[sp["split"] == "test", seg])
        used = set(sp.loc[sp[col].isin(["train", "val"]), seg])
        if test_ids & used:
            problems.append(f"[{name}] {col} 使用了 base split 的 test segment（leakage）。")

    data_segs = set(pd.read_csv(data_csv, usecols=[seg])[seg])
    if data_segs != set(sp[seg]):
        problems.append(f"[{name}] split 涵蓋的 segment 與訓練 CSV 不一致"
                        f"（資料 {len(data_segs)} 段 vs split {len(sp)} 段）。")
    return problems


def preflight(plan: dict) -> None:
    problems = []
    const = dict(plan["const"], _folds=plan["grid"]["folds"])
    if const["target"] in const["input_col"].split(","):
        problems.append(f"input_col 含 target {const['target']}，違反核心原則（OI-01）。")
    for name, ds in plan["grid"]["datasets"].items():
        problems += check_split_identity(name, ds, const)
    if problems:
        print("前置檢查未通過：", file=sys.stderr)
        for p in problems:
            print(f"  ✗ {p}", file=sys.stderr)
        print("\nsplit 檔可用下列指令重產：", file=sys.stderr)
        print(f"  python build_splits.py --data-path dataset/<csv> "
              f"--input-col '{const['input_col']}' --exog-col '{EXOGS['full']}' "
              f"--seq-len {const['seq_len']}", file=sys.stderr)
        sys.exit(1)
    print("前置檢查通過：資料身份、split 身份、分派有效性皆一致。")


# ---------------------------------------------------------------- plan
def do_plan(accept_changes: bool, new_batch: bool) -> None:
    plan = compute_plan()
    preflight(plan)

    if new_batch and MANIFEST.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive = ROOT / f"archive-{stamp}"
        archive.mkdir(parents=True, exist_ok=True)
        for f in (MANIFEST, PLAN):
            if f.exists():
                shutil.move(str(f), str(archive / f.name))
        print(f"舊批次已封存到 {archive}/（結果目錄未刪除）")

    prev = read_manifest() if MANIFEST.exists() else None
    rows, stale = [], []
    for key, cell in plan["cells"].items():
        base = {"run_key": key, "dataset": cell["dataset"], "exog": cell["exog"],
                "loss": cell["loss"], "fold": cell["fold"],
                "config_hash": cell["config_hash"], "status": "pending",
                "run_dir": "", "superseded_run_dir": "", "started": "",
                "finished": "", "duration_s": pd.NA, "error": ""}
        old = None
        if prev is not None:
            hit = prev[prev.run_key == key]
            old = hit.iloc[0].to_dict() if len(hit) else None

        if old is None or old["status"] != "done":
            rows.append(base)
            continue
        if old["config_hash"] == cell["config_hash"]:
            rows.append({**base, **{k: old[k] for k in
                                    ["status", "run_dir", "superseded_run_dir",
                                     "started", "finished", "duration_s", "error"]}})
        else:
            stale.append(key)
            rows.append({**base, "superseded_run_dir": old["run_dir"] or ""})

    df = pd.DataFrame(rows, columns=MANIFEST_COLS).astype(
        {k: v for k, v in MANIFEST_DTYPES.items() if k in MANIFEST_COLS})

    # 群組一致性：同一 (dataset,exog,loss) 的三折會被平均，狀態必須同進同出
    grp = df.groupby(["dataset", "exog", "loss"]).status.nunique()
    mixed = grp[grp > 1]

    if stale and not accept_changes:
        print(f"\n✗ 偵測到 {len(stale)} 組已完成的 run 其設定與目前不符，拒絕覆寫。", file=sys.stderr)
        for k in stale[:8]:
            print(f"    {k}", file=sys.stderr)
        if len(stale) > 8:
            print(f"    ... 共 {len(stale)} 組", file=sys.stderr)
        print("\n  請擇一明確表態：", file=sys.stderr)
        print("    python run_matrix.py plan --accept-changes   # 受影響者轉 pending，其餘保留", file=sys.stderr)
        print("    python run_matrix.py plan --new              # 另開一批，舊結果完整保留", file=sys.stderr)
        sys.exit(1)

    save_manifest(df)
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"plan_id={plan['plan_id']}  →  {PLAN}")
    print(f"manifest: {MANIFEST}（{len(df)} 組，{int((df.status == 'pending').sum())} 待跑）")
    if stale:
        print(f"  已將 {len(stale)} 組設定變更者轉為 pending（舊 run_dir 保留於 superseded_run_dir）")
    if len(mixed):
        print(f"  [WARN] 下列群組內狀態不一致，平均時會混到不同設定：\n    {dict(mixed)}")
    status(df)


# ---------------------------------------------------------------- 執行
def build_cmd(cell: dict) -> list[str]:
    """一律從 plan 快照的參數組指令，不讀模組全域變數。"""
    p = cell["params"]
    cmd = [sys.executable, "run.py"]
    for k in ["model", "data", "features", "target", "input_col", "segment_col",
              "split_mode", "seq_len", "pred_len", "label_len", "batch_size",
              "train_epochs", "patience", "learning_rate", "dropout", "seed",
              "early_stop_metric", "data_path", "split_file", "fold", "criterion"]:
        cmd += [f"--{k}", str(p[k])]
    cmd += ["--output_root", str(OUTPUT_ROOT)]
    if p.get("exog_col") is not None:
        cmd += ["--exog_col", p["exog_col"]]
    if p.get("flatten_fusion"):
        cmd += ["--flatten_fusion"]
    return cmd


def extract_run_dir(log_text: str) -> str:
    m = re.findall(r"run_dir='([^']+)'", log_text)
    return m[-1] if m else ""


def execute(dry_run: bool, only: str | None) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    df = read_manifest()
    if (df.config_hash != pd.Series([plan["cells"][k]["config_hash"] for k in df.run_key],
                                    dtype="string")).any():
        sys.exit("manifest 與 plan.json 的 config_hash 不符，請重跑 plan。")

    # 凍結生效：執行一律用 plan.json 的參數。但若模組全域已被改動，使用者
    # 多半以為新值會生效 —— 明講出來，避免「我明明改了卻沒作用」的困惑。
    live = compute_plan()
    drifted = [k for k, c in live["cells"].items()
               if k in plan["cells"] and c["config_hash"] != plan["cells"][k]["config_hash"]]
    if drifted:
        print(f"[NOTE] run_matrix.py 目前的設定與 plan.json 不同（{len(drifted)} 格）。"
              f"\n       執行將**沿用 plan.json 的凍結參數**；要改用新設定請先跑 "
              f"`python run_matrix.py plan --accept-changes`。")

    todo = df[df.status != "done"]
    if only:
        todo = todo[todo.run_key.str.contains(only)]

    if dry_run:
        for _, row in todo.iterrows():
            print(f"\n# [{row.run_key}]  config_hash={row.config_hash}")
            print("  " + " ".join(build_cmd(plan["cells"][row.run_key])))
        print(f"\n共 {len(todo)} 條指令（--dry-run，未執行）。")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"plan_id={plan['plan_id']}  待跑 {len(todo)} 組。")

    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        cmd = build_cmd(plan["cells"][row.run_key])
        log_path = LOG_DIR / f"{row.run_key}.log"
        print(f"\n[{i}/{len(todo)}] {row.run_key}  {datetime.now():%H:%M:%S}")

        df.at[idx, "status"] = "running"
        df.at[idx, "started"] = datetime.now().astimezone().isoformat(timespec="seconds")
        save_manifest(df)

        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        dur = time.time() - t0
        log_text = log_path.read_text(encoding="utf-8", errors="replace")

        if proc.returncode == 0:
            rd = extract_run_dir(log_text)
            if not rd:
                df.at[idx, "status"] = "failed"
                df.at[idx, "error"] = "退出碼 0 但 log 中找不到 run_dir"
                print("    FAILED: 找不到 run_dir")
            else:
                df.at[idx, "run_dir"] = rd
                df.at[idx, "status"] = "done"
                df.at[idx, "error"] = ""
                print(f"    done  {dur/60:.1f} min")
        else:
            df.at[idx, "status"] = "failed"
            df.at[idx, "error"] = " | ".join(log_text.strip().splitlines()[-3:])[:500]
            print(f"    FAILED (rc={proc.returncode})  見 {log_path}")

        df.at[idx, "finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        df.at[idx, "duration_s"] = round(dur, 1)
        save_manifest(df)

    status(df)


def status(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = read_manifest()
    counts = df.status.value_counts().to_dict()
    print(f"\n{'='*60}\n進度: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    done = df[df.status == "done"]
    secs = done.duration_s.dropna()
    if len(secs):
        print(f"已完成 {len(secs)} 組，單組平均 {secs.mean()/60:.1f} 分，累計 {secs.sum()/3600:.2f} 小時")
    failed = df[df.status == "failed"]
    if len(failed):
        print(f"\n失敗 {len(failed)} 組：")
        for _, r in failed.iterrows():
            print(f"  {r.run_key}: {(r.error or '')[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "run", "status"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="只跑 run_key 含此字串的組合")
    ap.add_argument("--accept-changes", action="store_true",
                    help="設定已變更：受影響的已完成 run 轉 pending")
    ap.add_argument("--new", action="store_true", help="封存舊批次，另開一批")
    args = ap.parse_args()

    if args.action == "plan":
        do_plan(args.accept_changes, args.new)
    elif args.action == "run":
        if not PLAN.exists() or not MANIFEST.exists():
            sys.exit("找不到 plan.json 或 manifest.csv，請先跑：python run_matrix.py plan")
        execute(args.dry_run, args.only)
    else:
        status()


if __name__ == "__main__":
    main()
