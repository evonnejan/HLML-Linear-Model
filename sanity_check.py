import os
import numpy as np
import pandas as pd
from data_provider.Data_Loader import Dataset_Custom

def get_segment_split(df, seg_col="segment_id"):
    # 依 SegmentStart 排序（若沒有 SegmentStart，就用每個 segment 的最小 date）
    if "SegmentStart" in df.columns:
        seg_info = df[[seg_col, "SegmentStart"]].drop_duplicates().copy()
        seg_info["SegmentStart"] = pd.to_datetime(seg_info["SegmentStart"])
        seg_ids = seg_info.sort_values("SegmentStart")[seg_col].tolist()
    else:
        tmp = df[[seg_col, "date"]].drop_duplicates().copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        seg_ids = (
            tmp.groupby(seg_col)["date"].min()
            .sort_values()
            .index.tolist()
        )

    n = len(seg_ids)
    train_n = max(1, int(n * 0.7))
    val_n = max(1, int(n * 0.1))
    if train_n + val_n >= n:
        val_n = max(1, n - train_n - 1)
    train_ids = seg_ids[:train_n]
    val_ids = seg_ids[train_n:train_n + val_n]
    test_ids = seg_ids[train_n + val_n:]
    return train_ids, val_ids, test_ids

def check_no_cross(df_cur, seg_col, valid_starts, need, num_checks=30, seed=0):
    seg = df_cur[seg_col].to_numpy()
    rng = np.random.default_rng(seed)
    if len(valid_starts) == 0:
        return True, "valid_starts is empty"

    picks = rng.integers(0, len(valid_starts), size=min(num_checks, len(valid_starts)))
    for j in picks:
        s = int(valid_starts[j])
        block = seg[s:s+need]
        if len(block) < need:
            return False, f"window too short at sample={j}, start={s}"
        if not np.all(block == block[0]):
            return False, f"crossed segment at sample={j}, start={s}"
    return True, "OK"

def main():
    root_path = "./dataset/"
    data_path = "water_level_all2.csv"
    seg_col = "segment_id"

    seq_len = 30      # 改 40 也可
    pred_len = 15
    label_len = min(30, seq_len)  # 確保 label_len <= seq_len

    x_col = "HL02"
    y_col = "HL01"

    df = pd.read_csv(os.path.join(root_path, data_path))
    train_ids, val_ids, test_ids = get_segment_split(df, seg_col=seg_col)

    print(f"[Segments] total={len(set(df[seg_col]))}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    need = seq_len + pred_len
    lengths = df.groupby(seg_col).size()

    def stats(ids, name):
        L = lengths.loc[ids].values
        windows = np.maximum(0, L - need + 1)
        print(f"[{name}] segments={len(ids)}, usable_segments={(windows>0).sum()}, windows={int(windows.sum())}")

    stats(train_ids, "train")
    stats(val_ids, "val")
    stats(test_ids, "test")

    for flag, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        ds = Dataset_Custom(
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            size=[seq_len, label_len, pred_len],
            features="S",
            input_col=x_col,
            target=y_col,
            segment_col=seg_col,      # 這會啟用不跨事件取樣（你已做）
            scale=True,
            timeenc=0,
            freq="min",
            train_only=False
        )

        # 取出該 split 的 df_cur（用同樣 ids 過濾），用來驗證 valid_starts 不跨事件
        df_cur = df[df[seg_col].isin(ids)].reset_index(drop=True)

        ok, msg = check_no_cross(df_cur, seg_col, ds.valid_starts, need, num_checks=30, seed=0)
        print(f"[Sanity:{flag}] len(ds)={len(ds)}, valid_starts={len(ds.valid_starts) if ds.valid_starts is not None else None}, no-cross={ok} ({msg})")

    print("Done.")

if __name__ == "__main__":
    main()