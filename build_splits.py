"""Chronologically ordered, segment-wise splitting with rolling-origin / expanding-window CV.

依 segment 的「可用 forecasting window 數」決定 train/val/test 邊界，而非按段數或列數硬切。

設計優先序（高到低）:
    1. Segment integrity   — 每個 segment 是不可拆分的最小單位，整段只能屬於一個 partition
    2. Chronological order — 依 SegmentStart 排序，train 永遠早於 val，val 早於 test
    3. Sufficient eval data— 邊界候選需讓各 partition 都有足夠 window
    4. Approximate ratio   — 比例只求接近目標，不為湊比例拆段

為什麼用 window 數而非段數：segment 長度差距極大（130 ~ 2040 分鐘），
按段數切會讓實際訓練樣本數嚴重偏離目標比例。

window 數的計算完全對齊 `data_provider/Data_Loader.py`：
    - need = seq_len + pred_len
    - 只在同一 segment 的連續區塊內滑動，步長 = stride
    - 任一 window 的 [s, s+need) 範圍內若有 NaN（檢查欄位 = input_col + exog_col + target）
      則該 window 被丟棄
因此本腳本算出的數字與訓練時實際拿到的樣本數一致（同樣的 col/seq_len/stride 前提下）。

Rolling-origin expanding-window CV:
    dev = train + val 的所有 segment（即 test 之前的全部）。
    dev 依 window 數切成 (1 + n_folds) 塊：塊 0 為初始 train，塊 1..k 為各 fold 的 val。
    fold i:  train = 塊 0..i-1（逐 fold 擴張）, val = 塊 i
    每個 fold 的 val 期間嚴格晚於其 train 期間。

輸出 CSV（一列一 segment）:
    segment_id, SegmentStart, n_windows, n_rows, split, fold_1, ..., fold_k
    split  ∈ {train, val, test}      ← 單一切分，給一般訓練用
    fold_i ∈ {train, val, ""}        ← rolling-origin CV，給 model selection 用

用法:
    python build_splits.py --data-path dataset/train_drycut_L3h_buf60.csv
"""
import argparse
import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd


def expand_cols(patterns: str | None, available: list[str]) -> list[str]:
    """展開 glob（'HL*'、'*gate_opening*'），對齊 run.py:45-67。"""
    if not patterns:
        return []
    out: list[str] = []
    for p in [s.strip() for s in patterns.split(",") if s.strip()]:
        matches = [c for c in available if fnmatch.fnmatchcase(c, p)] if any(ch in p for ch in "*?[") else ([p] if p in available else [])
        if not matches:
            raise ValueError(f"欄位樣式 '{p}' 在 CSV 中找不到對應欄位。")
        for m in matches:
            if m not in out:
                out.append(m)
    return out


def count_windows_per_segment(
    df: pd.DataFrame, *, seg_col: str, check_cols: list[str], need: int, stride: int
) -> pd.DataFrame:
    """逐 segment 算可用 window 數（NaN-aware），對齊 Data_Loader 的 valid_starts 邏輯。"""
    rows = []
    for sid, s in df.groupby(seg_col, sort=False):
        n = len(s)
        bad = s[check_cols].isna().any(axis=1).to_numpy()
        csum = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])
        starts = range(0, max(0, n - need + 1), stride)
        n_win = sum(1 for i in starts if csum[i + need] - csum[i] == 0)
        rows.append(
            {
                seg_col: sid,
                "SegmentStart": pd.to_datetime(s["SegmentStart"].iloc[0]),
                "n_rows": n,
                "n_windows": int(n_win),
            }
        )
    return pd.DataFrame(rows).sort_values("SegmentStart").reset_index(drop=True)


def find_overlap_groups(win: pd.DataFrame, seg_col: str) -> list[list]:
    """找出 window 互相重疊的 segment 群組（連通分量）。

    重疊代表同一批分鐘同時屬於兩個 segment；若它們被分到不同 split 就是 leakage。
    drycut 以 `L >= 2*buffer` 從結構上排除重疊；舊法（rain_segments_meta.csv）的
    gap=30min 合併門檻與 ±60min 視窗互相矛盾，實測有 32 對重疊。
    同一群組的段必須留在同一個 partition。
    """
    if not {"WinStart", "WinEnd"}.issubset(win.columns):
        return []
    w = win.sort_values("WinStart").reset_index(drop=True)
    ids = w[seg_col].tolist()
    starts = w["WinStart"].tolist()
    ends = w["WinEnd"].tolist()

    groups: list[list] = []
    cur = [ids[0]]
    running_end = ends[0]
    for i in range(1, len(ids)):
        if starts[i] <= running_end:          # 與目前這串重疊 → 併入
            cur.append(ids[i])
            running_end = max(running_end, ends[i])
        else:
            groups.append(cur)
            cur = [ids[i]]
            running_end = ends[i]
    groups.append(cur)
    return [g for g in groups if len(g) > 1]


def blocked_boundaries(meta: pd.DataFrame, groups: list[list], seg_col: str) -> np.ndarray:
    """標記哪些邊界位置會把同一個重疊群組切開，不可作為 split 邊界。

    邊界 index i 的語意是「前 i 段歸前一個 partition」，因此 i 落在群組內部
    （即 meta 第 i-1 段與第 i 段同屬一群）就會拆散該群組。
    """
    n = len(meta)
    blocked = np.zeros(n + 1, dtype=bool)
    if not groups:
        return blocked
    pos = {sid: i for i, sid in enumerate(meta[seg_col])}
    for g in groups:
        idxs = sorted(pos[sid] for sid in g if sid in pos)
        for i in idxs[1:]:
            blocked[i] = True                  # 切在這裡會把群組拆開
    return blocked


def pick_boundary(cum: np.ndarray, total: int, target_frac: float, lo: int, hi: int,
                  blocked: np.ndarray | None = None) -> int:
    """在 [lo, hi] 內挑一個 segment 邊界，使累積 window 佔比最接近 target_frac。

    回傳的 index 意義是「前 idx 個 segment 歸前一個 partition」。
    blocked 標記的位置會拆散重疊群組，予以排除；若全被排除則退回不設限。
    """
    if hi <= lo:
        return lo
    cand = np.arange(lo, hi + 1)
    if blocked is not None:
        allowed = cand[~blocked[cand]]
        if len(allowed):
            cand = allowed
    target = total * target_frac
    err = np.abs(cum[cand] - target)
    return int(cand[int(np.argmin(err))])


def assign_split(meta: pd.DataFrame, ratios: tuple[float, float, float],
                 blocked: np.ndarray | None = None) -> pd.DataFrame:
    """依累積 window 數找 train/val/test 邊界，保持 segment 完整。"""
    w = meta["n_windows"].to_numpy()
    cum = np.concatenate([[0], np.cumsum(w)])  # cum[i] = 前 i 段的 window 總數
    total = int(cum[-1])
    n = len(meta)
    if total == 0:
        raise ValueError("所有 segment 的可用 window 數皆為 0，請檢查 seq_len/pred_len 或 NaN 狀況。")

    r_train, r_val, _ = ratios
    # 每個 partition 至少留 1 段
    b1 = pick_boundary(cum, total, r_train, 1, n - 2, blocked)
    b2 = pick_boundary(cum, total, r_train + r_val, b1 + 1, n - 1, blocked)

    split = np.array(["test"] * n, dtype=object)
    split[:b1] = "train"
    split[b1:b2] = "val"
    meta = meta.copy()
    meta["split"] = split
    return meta


def assign_folds(meta: pd.DataFrame, n_folds: int, init_frac: float,
                 blocked: np.ndarray | None = None) -> pd.DataFrame:
    """dev(=train+val) 內做 rolling-origin expanding-window：塊 0 起始 train，塊 1..k 各為一個 val。"""
    meta = meta.copy()
    for i in range(1, n_folds + 1):
        meta[f"fold_{i}"] = ""
    if n_folds <= 0:
        return meta

    dev = meta[meta["split"].isin(["train", "val"])]
    if len(dev) < n_folds + 1:
        print(f"  [WARN] dev 僅 {len(dev)} 段，不足以切 {n_folds} 個 fold，跳過 CV 欄位。")
        return meta

    idx = dev.index.to_numpy()
    w = dev["n_windows"].to_numpy()
    cum = np.concatenate([[0], np.cumsum(w)])
    total = int(cum[-1])

    # 依累積 window 找 (n_folds+1) 塊的邊界
    # dev 為 meta 的時間前綴，故可直接取 blocked 的前 len(idx)+1 個位置
    dev_blocked = blocked[: len(idx) + 1] if blocked is not None else None
    bounds = [pick_boundary(cum, total, init_frac, 1, len(idx) - n_folds, dev_blocked)]
    for i in range(1, n_folds):
        frac = init_frac + (1 - init_frac) * i / n_folds
        lo = bounds[-1] + 1
        hi = len(idx) - (n_folds - i)
        bounds.append(pick_boundary(cum, total, frac, lo, hi, dev_blocked))
    bounds.append(len(idx))  # 最後一塊吃到 dev 結尾

    for f in range(1, n_folds + 1):
        tr_end = bounds[f - 1]
        va_end = bounds[f]
        col = f"fold_{f}"
        meta.loc[idx[:tr_end], col] = "train"
        meta.loc[idx[tr_end:va_end], col] = "val"
    return meta


def report(meta: pd.DataFrame, n_folds: int, ratios: tuple[float, float, float]) -> None:
    total = meta["n_windows"].sum()
    print(f"\n{'='*74}\n單一切分（依 window 數，目標 {ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%}）\n{'='*74}")
    print(f"{'split':<8}{'segs':>6}{'windows':>10}{'實際比例':>12}{'期間':>44}")
    for name in ["train", "val", "test"]:
        sub = meta[meta["split"] == name]
        if sub.empty:
            continue
        rng = f"{sub.SegmentStart.min():%Y-%m-%d} ~ {sub.SegmentStart.max():%Y-%m-%d}"
        print(f"{name:<8}{len(sub):>6}{sub.n_windows.sum():>10,}{sub.n_windows.sum()/total:>11.1%}{rng:>44}")
    print(f"{'總計':<8}{len(meta):>6}{total:>10,}")

    if n_folds > 0 and f"fold_{n_folds}" in meta.columns:
        print(f"\n{'='*74}\nRolling-origin expanding-window CV（{n_folds} folds，dev 內）\n{'='*74}")
        print(f"{'fold':<8}{'train段':>8}{'train win':>11}{'val段':>7}{'val win':>10}{'val 期間':>32}")
        for f in range(1, n_folds + 1):
            col = f"fold_{f}"
            tr = meta[meta[col] == "train"]
            va = meta[meta[col] == "val"]
            rng = f"{va.SegmentStart.min():%Y-%m-%d} ~ {va.SegmentStart.max():%Y-%m-%d}" if not va.empty else "-"
            print(f"{f:<8}{len(tr):>8}{tr.n_windows.sum():>11,}{len(va):>7}{va.n_windows.sum():>10,}{rng:>32}")
        print("\n  每個 fold 的 val 期間皆嚴格晚於其 train 期間；train 隨 fold 擴張（expanding window）。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", required=True, help="訓練 CSV（需含 segment_id 與 SegmentStart）")
    ap.add_argument("--segment-col", default="segment_id")
    ap.add_argument("--target", default="HL01")
    ap.add_argument("--input-col", default="HL*")
    ap.add_argument("--exog-col", default="min_since_rain,Past10Min,Past1Hr,Now,*gate_opening*")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test 目標比例")
    ap.add_argument("--n-folds", type=int, default=3, help="rolling-origin fold 數（0 = 不產）")
    ap.add_argument("--init-train-frac", type=float, default=0.5,
                    help="dev 中作為第一個 fold 起始 train 的 window 佔比")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError(f"--ratios 需為三個和為 1 的數字，得到 {args.ratios}")

    df = pd.read_csv(args.data_path)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    for col in (args.segment_col, "SegmentStart", "date"):
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要欄位 '{col}'。")

    available = [c for c in df.columns if c != "date"]
    x_cols = expand_cols(args.input_col, available) + expand_cols(args.exog_col, available)
    check_cols = list(dict.fromkeys(x_cols + [args.target]))
    need = args.seq_len + args.pred_len

    print(f"資料      : {args.data_path}（{len(df):,} 列）")
    print(f"NaN 檢查欄: {len(check_cols)} 欄 = input+exog+target")
    print(f"window    : need={need} (seq_len={args.seq_len}+pred_len={args.pred_len}), stride={args.stride}")

    meta = count_windows_per_segment(
        df, seg_col=args.segment_col, check_cols=check_cols, need=need, stride=args.stride
    )
    dead = int((meta["n_windows"] == 0).sum())
    if dead:
        print(f"  [WARN] {dead} 段的可用 window 為 0（過短或整段含 NaN），仍會被分派 split 但不產生樣本。")

    # 重疊防護：window 互相重疊的 segment 必須留在同一 partition，否則共用的分鐘
    # 會同時出現在兩個 split（leakage）。drycut 因 L>=2*buffer 不會有重疊。
    blocked = None
    if {"WinStart", "WinEnd"}.issubset(df.columns):
        win = (df[[args.segment_col, "WinStart", "WinEnd"]]
               .drop_duplicates(subset=[args.segment_col]).copy())
        win["WinStart"] = pd.to_datetime(win["WinStart"])
        win["WinEnd"] = pd.to_datetime(win["WinEnd"])
        groups = find_overlap_groups(win, args.segment_col)
        if groups:
            n_seg = sum(len(g) for g in groups)
            print(f"  [重疊防護] 偵測到 {len(groups)} 個重疊群組（共 {n_seg} 段），"
                  f"將強制留在同一 partition 以避免 leakage。")
            blocked = blocked_boundaries(meta, groups, args.segment_col)
        else:
            print("  [重疊防護] 無重疊 segment。")

    meta = assign_split(meta, ratios, blocked)
    meta = assign_folds(meta, args.n_folds, args.init_train_frac, blocked)

    out = args.out or f"dataset/splits_{Path(args.data_path).stem}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    report(meta, args.n_folds, ratios)
    print(f"\n輸出: {out}")


if __name__ == "__main__":
    main()
