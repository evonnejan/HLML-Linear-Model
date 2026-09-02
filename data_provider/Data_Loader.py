import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

MIX_STYLE_MODELS = {'DLinearMix', 'DLinearMix2'}

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', input_col=None, exog_col=None, segment_col=None,
                 target='OT', stride=1, scale=True, timeenc=0, freq='h', train_only=False,
                 model_name=None, split_file=None, fold=None):
        # size [seq_len, label_len, pred_len] — required, no default.
        # label_len is currently unused by the linear-style models but kept
        # in the signature for backwards compatibility with the dataset API.
        if size is None:
            raise ValueError(
                "Dataset_Custom requires `size=[seq_len, label_len, pred_len]`; "
                "no default is provided to avoid silently mismatching CLI args."
            )
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.stride = max(1, int(stride))
        self.features = features
        self.segment_col = segment_col
        self.input_col = input_col
        self.exog_col = exog_col
        self.target = target
        self.model_name = model_name
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.split_file = split_file
        self.fold = fold

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _load_split_file(self, seg_col, seg_ids_sorted):
        """讀取 build_splits.py 產生的 segment-wise split 指派。

        - fold 為 None：使用 `split` 欄（train/val/test）。
        - fold=k     ：使用 `fold_k` 欄決定 train/val（rolling-origin expanding window），
                       test 仍取自 `split` 欄，確保 final hold-out 在所有 fold 間固定不變。
        回傳的 id 皆依 seg_ids_sorted 的時間順序排列。
        """
        split_df = pd.read_csv(self.split_file)
        split_df.columns = [c.lstrip('﻿') for c in split_df.columns]
        if seg_col not in split_df.columns:
            raise ValueError(f"split_file 缺少 '{seg_col}' 欄: {self.split_file}")
        if "split" not in split_df.columns:
            raise ValueError(f"split_file 缺少 'split' 欄: {self.split_file}")

        base = dict(zip(split_df[seg_col], split_df["split"].astype(str)))
        missing = [sid for sid in seg_ids_sorted if sid not in base]
        if missing:
            raise ValueError(
                f"split_file 未涵蓋 {len(missing)} 個 segment（例如 {missing[:5]}）: {self.split_file}"
            )

        assign = dict(base)
        if self.fold is not None:
            col = f"fold_{int(self.fold)}"
            if col not in split_df.columns:
                have = [c for c in split_df.columns if c.startswith("fold_")]
                raise ValueError(f"split_file 沒有 '{col}' 欄（可用: {have}）: {self.split_file}")
            fold_map = dict(zip(split_df[seg_col], split_df[col].fillna("").astype(str)))
            # test 固定沿用 base；dev 內的 train/val 由 fold 欄決定，空字串代表該 fold 不使用
            assign = {
                sid: ("test" if base[sid] == "test" else (fold_map.get(sid, "") or "unused"))
                for sid in seg_ids_sorted
            }

        train_ids = [sid for sid in seg_ids_sorted if assign[sid] == "train"]
        val_ids = [sid for sid in seg_ids_sorted if assign[sid] == "val"]
        test_ids = [sid for sid in seg_ids_sorted if assign[sid] == "test"]
        unused = [sid for sid in seg_ids_sorted if assign[sid] == "unused"]

        tag = f"fold_{self.fold}" if self.fold is not None else "split"
        print(
            f"[split_file] {os.path.basename(str(self.split_file))} ({tag}): "
            f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
            + (f" unused={len(unused)}" if unused else "")
        )
        return train_ids, val_ids, test_ids

    @staticmethod
    def _parse_col_spec(col_spec):
        if col_spec is None:
            return []
        if isinstance(col_spec, str):
            return [item.strip() for item in col_spec.split(',') if item.strip()]
        if isinstance(col_spec, (list, tuple)):
            return [str(item).strip() for item in col_spec if str(item).strip()]
        text = str(col_spec).strip()
        return [text] if text else []

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # =========================
        # 1) segment-based split (train/val/test by segment_id)
        # =========================
        seg_col = getattr(self, "segment_col", None)

        if seg_col is None and "segment_id" in df_raw.columns and self.set_type == 0:
            print(
                "[WARNING] CSV contains a 'segment_id' column but --segment_col is not set. "
                "Falling back to row-based train/val/test split, which may cross segment boundaries. "
                "Pass --segment_col segment_id to enable segment-aware splitting."
            )

        if seg_col is not None:
            if seg_col not in df_raw.columns:
                raise ValueError(f"segment_col={seg_col} not in columns: {df_raw.columns.tolist()}")

            # 依 SegmentStart 排序事件（若沒有 SegmentStart，就用每段最小 date）
            if "SegmentStart" in df_raw.columns:
                seg_info = df_raw[[seg_col, "SegmentStart"]].drop_duplicates().copy()
                seg_info["SegmentStart"] = pd.to_datetime(seg_info["SegmentStart"])
                seg_ids_sorted = seg_info.sort_values("SegmentStart")[seg_col].tolist()
            else:
                tmp = df_raw[[seg_col, "date"]].drop_duplicates().copy()
                tmp["date"] = pd.to_datetime(tmp["date"])
                seg_ids_sorted = (
                    tmp.groupby(seg_col)["date"].min()
                    .sort_values()
                    .index.tolist()
                )

            nseg = len(seg_ids_sorted)
            if nseg < 3:
                raise ValueError(f"Not enough segments for split: nseg={nseg}")

            # train_only: 用全部資料當 train（不做 val/test）
            if self.train_only:
                train_ids = seg_ids_sorted
                val_ids = []
                test_ids = []
            elif getattr(self, "split_file", None):
                # 外部指定的 segment-wise split（build_splits.py 產生）。
                # split 決策抽離到 loader 之外，換切法不需動訓練碼。
                train_ids, val_ids, test_ids = self._load_split_file(seg_col, seg_ids_sorted)
            else:
                train_n = max(1, int(nseg * 0.7))
                val_n = max(1, int(nseg * 0.1))
                if train_n + val_n >= nseg:
                    val_n = max(1, nseg - train_n - 1)

                train_ids = seg_ids_sorted[:train_n]
                val_ids = seg_ids_sorted[train_n:train_n + val_n]
                test_ids = seg_ids_sorted[train_n + val_n:]

            # helper: 讓 df_train/df_cur 內 segment 都是連續區塊（之後 valid_starts 才好算）
            seg_rank = {sid: i for i, sid in enumerate(seg_ids_sorted)}

            def _filter_and_sort(df, ids):
                if len(ids) == 0:
                    return df.iloc[0:0].copy()
                out = df[df[seg_col].isin(ids)].copy()
                out["_seg_rank"] = out[seg_col].map(seg_rank)
                out["date"] = pd.to_datetime(out["date"])
                out = out.sort_values(["_seg_rank", "date"]).drop(columns=["_seg_rank"]).reset_index(drop=True)
                return out

            # scaler 永遠用 train segments fit
            df_train = _filter_and_sort(df_raw, train_ids)

            # 依 flag 取當前 split
            if self.set_type == 0:      # train
                df_cur = _filter_and_sort(df_raw, train_ids)
            elif self.set_type == 1:    # val
                df_cur = _filter_and_sort(df_raw, val_ids)
            else:                        # test
                df_cur = _filter_and_sort(df_raw, test_ids)

            if len(df_cur) == 0:
                raise ValueError(f"Current split has 0 rows. flag={self.set_type}, train_only={self.train_only}")

            # segment split 模式：整個 df_cur 都是這個 split 的資料
            border1, border2 = 0, len(df_cur)

        else:
            # =========================
            # 2) fallback: old row-based split (keep for compatibility)
            # =========================
            df_cur = df_raw
            df_train = df_raw

            num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test

            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        # =========================
        # 3) column list (exclude date)
        # =========================
        cols = list(df_cur.columns)
        if "date" not in cols:
            raise ValueError("CSV must contain 'date' column")
        cols.remove("date")

        if self.features != "S":
            # M / MS: keep all columns (including target) except date
            # MS: enforce target as the last column (important if your exp selects last dim as target)
            if self.features == "MS":
                if self.target not in cols:
                    raise ValueError(f"target={self.target} not in columns: {df_cur.columns.tolist()}")
                cols = [c for c in cols if c != self.target] + [self.target]

        # =========================
        # 4) build df_x, df_y and scale using df_train only
        # =========================
        if self.model_name in MIX_STYLE_MODELS:
            input_cols = self._parse_col_spec(self.input_col)
            exog_cols = self._parse_col_spec(self.exog_col)

            if len(input_cols) == 0:
                raise ValueError(f"{self.model_name} requires input_col (comma-separated allowed), e.g., HL02,HL03")

            x_cols = []
            for column in input_cols + exog_cols:
                if column not in x_cols:
                    x_cols.append(column)

            for column in x_cols:
                if column not in df_cur.columns:
                    raise ValueError(f"column={column} not in columns: {df_cur.columns.tolist()}")
            if self.target not in df_cur.columns:
                raise ValueError(f"target={self.target} not in columns: {df_cur.columns.tolist()}")

            # Identify boolean indicator columns; they keep their 0/1 values and bypass the scaler.
            bool_x_cols = [c for c in x_cols if df_cur[c].dtype == bool]
            cont_x_cols = [c for c in x_cols if c not in bool_x_cols]
            cont_idx = [x_cols.index(c) for c in cont_x_cols]

            if self.set_type == 0:
                print(
                    f"[Dataset {self.model_name}] input_cols={input_cols}, exog_cols={exog_cols}, "
                    f"target={self.target}, scale={self.scale}, segment_split={seg_col is not None}, "
                    f"bool_passthrough={bool_x_cols}"
                )

            # Cast booleans to int8 so the resulting array is purely numeric.
            df_x_cur = df_cur[x_cols].copy()
            df_x_train = df_train[x_cols].copy()
            for c in bool_x_cols:
                df_x_cur[c] = df_x_cur[c].astype(np.int8)
                df_x_train[c] = df_x_train[c].astype(np.int8)

            df_y_cur_raw = df_cur[[self.target]]
            df_y_train_raw = df_train[[self.target]]

            # Determine which TRAIN rows contain NaN in any checked column;
            # those rows are excluded from scaler fit so statistics are not
            # polluted by fillna(0) values.
            nan_check_cols = list(x_cols)
            if self.target not in nan_check_cols:
                nan_check_cols.append(self.target)
            train_bad_rows = df_train[nan_check_cols].isna().any(axis=1).to_numpy()
            n_clean_train = int((~train_bad_rows).sum())
            if self.set_type == 0 and train_bad_rows.any():
                print(
                    f"[scaler fit] train: using {n_clean_train}/{len(train_bad_rows)} clean rows "
                    f"({train_bad_rows.sum()} NaN-bearing rows excluded from scaler fit)"
                )

            # Now apply fillna(0) for any rows that survive the window-level filter
            # but might still contain a NaN in another column not in nan_check_cols.
            df_x_cur = df_x_cur.fillna(0)
            df_x_train = df_x_train.fillna(0)
            df_y_cur = df_y_cur_raw.fillna(0)
            df_y_train = df_y_train_raw.fillna(0)

            if self.scale:
                self.scaler_y = StandardScaler()
                self.scaler_y.fit(df_y_train.values[~train_bad_rows])
                data_y_all = self.scaler_y.transform(df_y_cur.values)

                data_x_all = df_x_cur.values.astype(np.float32, copy=True)
                if len(cont_x_cols) > 0:
                    self.scaler_x = StandardScaler()
                    train_cont_vals = df_x_train[cont_x_cols].values[~train_bad_rows]
                    self.scaler_x.fit(train_cont_vals)
                    data_x_all[:, cont_idx] = self.scaler_x.transform(df_x_cur[cont_x_cols].values)
                else:
                    self.scaler_x = None

                self.scaler = self.scaler_y
            else:
                self.scaler_x = None
                self.scaler_y = None
                self.scaler = None
                data_x_all = df_x_cur.values.astype(np.float32, copy=True)
                data_y_all = df_y_cur.values

            self.x_cols = x_cols
            self.bool_x_cols = bool_x_cols

        elif self.features in ["M", "MS"]:
            train_bad_rows = df_train[cols].isna().any(axis=1).to_numpy()
            if self.set_type == 0 and train_bad_rows.any():
                print(
                    f"[scaler fit] train: using {(~train_bad_rows).sum()}/{len(train_bad_rows)} clean rows "
                    f"({train_bad_rows.sum()} NaN-bearing rows excluded from scaler fit)"
                )

            df_data_cur = df_cur[cols].fillna(0)
            df_data_train = df_train[cols].fillna(0)

            if self.scale:
                self.scaler = StandardScaler()
                self.scaler.fit(df_data_train.values[~train_bad_rows])
                data_all = self.scaler.transform(df_data_cur.values)
            else:
                self.scaler = None
                data_all = df_data_cur.values

            data_x_all = data_all
            data_y_all = data_all

        elif self.features == "S":
            # single-input single-output, allow cross-column mapping
            x_col = self.input_col if getattr(self, "input_col", None) else self.target
            y_col = self.target

            if x_col not in df_cur.columns:
                raise ValueError(f"input_col={x_col} not in columns: {df_cur.columns.tolist()}")
            if y_col not in df_cur.columns:
                raise ValueError(f"target={y_col} not in columns: {df_cur.columns.tolist()}")

            if self.set_type == 0:  # print only for train
                print(f"[Dataset S] x_col={x_col}, y_col={y_col}, scale={self.scale}, segment_split={seg_col is not None}")

            s_check_cols = [x_col, y_col] if x_col != y_col else [x_col]
            train_bad_rows = df_train[s_check_cols].isna().any(axis=1).to_numpy()
            if self.set_type == 0 and train_bad_rows.any():
                print(
                    f"[scaler fit] train: using {(~train_bad_rows).sum()}/{len(train_bad_rows)} clean rows "
                    f"({train_bad_rows.sum()} NaN-bearing rows excluded from scaler fit)"
                )

            df_x_cur = df_cur[[x_col]].fillna(0)
            df_y_cur = df_cur[[y_col]].fillna(0)

            df_x_train = df_train[[x_col]].fillna(0)
            df_y_train = df_train[[y_col]].fillna(0)

            if self.scale:
                self.scaler_x = StandardScaler()
                self.scaler_y = StandardScaler()

                self.scaler_x.fit(df_x_train.values[~train_bad_rows])
                self.scaler_y.fit(df_y_train.values[~train_bad_rows])

                data_x_all = self.scaler_x.transform(df_x_cur.values)
                data_y_all = self.scaler_y.transform(df_y_cur.values)

                # important: inverse_transform should map back to Y scale (HL01)
                self.scaler = self.scaler_y
            else:
                self.scaler_x = None
                self.scaler_y = None
                self.scaler = None
                data_x_all = df_x_cur.values
                data_y_all = df_y_cur.values

        else:
            raise ValueError(f"Unknown features type: {self.features}")

        # =========================
        # 5) time features / stamp (use df_cur, not df_raw)
        # =========================
        df_stamp = df_cur[["date"]].iloc[border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp["date"])

        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].apply(lambda r: r.month)
            df_stamp["day"] = df_stamp["date"].apply(lambda r: r.day)
            df_stamp["weekday"] = df_stamp["date"].apply(lambda r: r.weekday())
            df_stamp["hour"] = df_stamp["date"].apply(lambda r: r.hour)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # =========================
        # 6) final sliced arrays used by __getitem__
        # =========================
        self.data_x = data_x_all[border1:border2]
        self.data_y = data_y_all[border1:border2]
        self.data_stamp = data_stamp
        
        # --- for plotting / debugging ---
        self.dates = pd.to_datetime(df_cur["date"]).iloc[border1:border2].to_numpy()
        self.y_raw = df_cur[self.target].iloc[border1:border2].fillna(0).to_numpy()

        # 如果你也想記錄輸入欄位原始值（可選）
        if self.model_name in MIX_STYLE_MODELS:
            x_cols = getattr(self, "x_cols", self._parse_col_spec(self.input_col))
            if len(x_cols) > 0 and x_cols[0] in df_cur.columns:
                self.x_raw = df_cur[x_cols[0]].iloc[border1:border2].fillna(0).to_numpy()
            else:
                self.x_raw = df_cur[self.target].iloc[border1:border2].fillna(0).to_numpy()
        else:
            x_col = self.input_col if getattr(self, "input_col", None) else self.target
            self.x_raw = df_cur[x_col].iloc[border1:border2].fillna(0).to_numpy()

        # =========================
        # 7) build valid window start indices (segment-wise, on df_cur)
        # =========================
        # Determine NaN-check columns: any NaN inside a window's [s, s+seq_len+pred_len)
        # range will cause that window to be dropped, so the model never sees fillna(0)
        # values as either input or target.
        if self.model_name in MIX_STYLE_MODELS:
            nan_check_cols = list(self.x_cols)
            if self.target not in nan_check_cols:
                nan_check_cols.append(self.target)
        elif self.features in ("M", "MS"):
            nan_check_cols = list(cols)
        else:
            x_col_for_check = self.input_col if getattr(self, "input_col", None) else self.target
            nan_check_cols = [x_col_for_check, self.target] if x_col_for_check != self.target else [self.target]
        bad_rows = df_cur[nan_check_cols].iloc[border1:border2].isna().any(axis=1).to_numpy()

        self.valid_starts = None
        self.window_segment_ids = None
        need = self.seq_len + self.pred_len
        split_name = {0: "train", 1: "val", 2: "test"}[self.set_type]

        def _drop_nan_windows(valid_arr, bad_arr):
            """Return valid_arr with any window touching a NaN row removed."""
            if len(valid_arr) == 0 or not bad_arr.any():
                return valid_arr
            csum = np.concatenate([[0], np.cumsum(bad_arr.astype(np.int32))])
            window_bad = (csum[need:] - csum[:-need]) > 0
            keep_mask = ~window_bad[valid_arr]
            n_before = int(len(valid_arr))
            kept = valid_arr[keep_mask]
            dropped = n_before - int(len(kept))
            if dropped > 0:
                print(
                    f"[NaN window filter] {split_name}: dropped {dropped}/{n_before} windows "
                    f"({dropped / max(n_before, 1) * 100:.2f}%) due to NaN in {nan_check_cols}"
                )
            return kept

        if seg_col is not None:
            seg = df_cur[seg_col].iloc[border1:border2].to_numpy()
            n = len(seg)

            valid = []
            i = 0
            while i < n:
                j = i
                # find continuous block of same segment_id
                while j < n and seg[j] == seg[i]:
                    j += 1
                L = j - i
                max_start = L - need
                if max_start >= 0:
                    valid.extend(range(i, i + max_start + 1, self.stride))
                i = j

            valid = _drop_nan_windows(np.asarray(valid, dtype=np.int64), bad_rows)
            self.valid_starts = valid
            self.window_segment_ids = seg[self.valid_starts] if len(self.valid_starts) > 0 else np.array([], dtype=seg.dtype)
        else:
            # Row-based fallback split: enumerate every stride-th start in the current slice.
            n = border2 - border1
            if n >= need:
                valid = np.arange(0, n - need + 1, self.stride, dtype=np.int64)
                valid = _drop_nan_windows(valid, bad_rows)
            else:
                valid = np.asarray([], dtype=np.int64)
            self.valid_starts = valid

    def __getitem__(self, index):
        s_begin = int(self.valid_starts[index]) if self.valid_starts is not None else index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.valid_starts is not None:
            return len(self.valid_starts)
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        if hasattr(self, "scaler") and (self.scaler is not None):
            return self.scaler.inverse_transform(data)
        return data
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False):
        if size is None:
            raise ValueError("Dataset_Pred requires size=[seq_len, label_len, pred_len]")
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)