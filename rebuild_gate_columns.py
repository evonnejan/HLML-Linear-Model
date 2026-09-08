"""以「逐欄 merge_asof」重建 all_minute_wide.csv 的閘門欄位（不需連 SQL）。

## 修正的問題

`Data_From_SQL_all.py:126-137` 用**整列** merge_asof 併閘門：

    merged = pd.merge_asof(merged, gate, on="date", direction="backward")

它對每一分鐘只抓「時間最接近的那**一列**」。但原始閘門寬表有 39% 的列是
**部分回報**（7 個欄位只有其中幾個有值），一旦抓到這種列，其餘欄位就變成
NaN —— 即使更早的列明明有那些欄位的值。逐欄的歷史資訊被整列邏輯丟掉了。

## 改法

每個閘門欄位各自做一次 merge_asof，且**只使用該欄有值的觀測**，各自套用
5 分鐘 staleness。等於為每個感測器維持獨立的「最後已知值」，互不干擾。

## 前提

資料已凍結（2024-08-13 ~ 2025-08-03，不再新增），本機 gate 觀測檔涵蓋完整
區間且與 all_minute_wide 現有值 100% 相符，故可安全地在本機重建。

用法:
    python rebuild_gate_columns.py            # 就地重建（自動備份）
    python rebuild_gate_columns.py --dry-run  # 只比較不寫檔
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ALL_CSV = "dataset/all_minute_wide.csv"
GATE_CSV = "dataset/wra_cogate_obs_wide_gate_opening.csv"
STALENESS = pd.Timedelta("5min")


def merge_gate_per_column(
    grid: pd.DataFrame, gate: pd.DataFrame, gate_cols: list[str], staleness: pd.Timedelta
) -> pd.DataFrame:
    """逐欄 backward merge_asof，每欄只用自己有值的觀測，各自套 staleness。"""
    out = grid[["date"]].copy()
    for col in gate_cols:
        src = gate.loc[gate[col].notna(), ["date", col]].copy()
        src["_obs_time"] = src["date"]
        m = pd.merge_asof(out[["date"]], src, on="date", direction="backward")
        stale = ((m["date"] - m["_obs_time"]) > staleness) | m["_obs_time"].isna()
        vals = m[col].to_numpy(dtype="float64").copy()
        vals[stale.to_numpy()] = np.nan
        # 負值無物理意義，視為全關（沿用 merge_gate_data.py:90-96）
        out[col] = np.clip(vals, 0, None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-csv", default=ALL_CSV)
    ap.add_argument("--gate-csv", default=GATE_CSV)
    ap.add_argument("--staleness-minutes", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="只比較差異，不寫檔")
    ap.add_argument("--no-backup", action="store_true", help="不建立 .gatev1.bak 備份")
    args = ap.parse_args()

    staleness = pd.Timedelta(minutes=args.staleness_minutes)
    wide = pd.read_csv(args.all_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    gate = pd.read_csv(args.gate_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    gate_cols = [c for c in gate.columns if c != "date" and c in wide.columns]
    if not gate_cols:
        raise ValueError("找不到共同的閘門欄位。")

    print(f"寬表  : {args.all_csv}（{len(wide):,} 列）")
    print(f"閘門  : {args.gate_csv}（{len(gate):,} 筆觀測）")
    print(f"欄位  : {len(gate_cols)} 個，staleness={staleness}")

    part = gate[gate_cols].notna().sum(axis=1)
    print(f"  原始觀測中 7 欄全有的列: {int((part == len(gate_cols)).sum()):,} "
          f"/ {len(gate):,}（{(part == len(gate_cols)).mean()*100:.0f}%）← 其餘為部分回報")

    new_gate = merge_gate_per_column(wide, gate, gate_cols, staleness)

    old_na = wide[gate_cols].isna()
    new_na = new_gate[gate_cols].isna()
    print(f"\n{'欄位':<26}{'修正前 NaN':>12}{'修正後 NaN':>12}")
    for c in gate_cols:
        print(f"{c:<26}{old_na[c].mean()*100:>11.2f}%{new_na[c].mean()*100:>11.2f}%")
    print(f"{'任一欄 NaN':<26}{old_na.any(axis=1).mean()*100:>11.2f}%{new_na.any(axis=1).mean()*100:>11.2f}%")

    both = (~old_na) & (~new_na)
    same = 0
    tot = 0
    for c in gate_cols:
        mask = both[c].to_numpy()
        tot += int(mask.sum())
        same += int(np.isclose(wide[c].to_numpy()[mask], new_gate[c].to_numpy()[mask]).sum())
    print(f"\n一致性檢查：兩邊皆有值的儲存格 {tot:,} 個，數值相同 {same/max(tot,1)*100:.2f}%"
          f"  {'OK' if same == tot else '← 不一致，請檢查'}")

    if args.dry_run:
        print("\n--dry-run：未寫檔。")
        return

    if not args.no_backup:
        bak = Path(args.all_csv).with_suffix(".gatev1.bak.csv")
        if not bak.exists():
            shutil.copy2(args.all_csv, bak)
            print(f"\n已備份原檔 → {bak}")
        else:
            print(f"\n備份已存在，保留不覆蓋 → {bak}")

    wide[gate_cols] = new_gate[gate_cols].to_numpy()
    wide.to_csv(args.all_csv, index=False)
    print(f"已重建閘門欄位並寫回 {args.all_csv}")
    print("\n下一步：重跑 build_training_csv_from_meta.py 與 build_splits.py，"
          "訓練 CSV 的來源指紋會因寬表 mtime 改變而自動失效。")


if __name__ == "__main__":
    main()
