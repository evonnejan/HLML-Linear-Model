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

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv', input_col=None, exog_col=None, segment_col=None,
                 target='OT', stride=1, scale=True, timeenc=0, freq='h', train_only=False,
                 model_name=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
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

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

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
        if self.model_name == "DLinearMix":
            input_cols = self._parse_col_spec(self.input_col)
            exog_cols = self._parse_col_spec(self.exog_col)

            if len(input_cols) == 0:
                raise ValueError("DLinearMix requires input_col (comma-separated allowed), e.g., HL02,HL03")

            x_cols = []
            for column in input_cols + exog_cols:
                if column not in x_cols:
                    x_cols.append(column)

            for column in x_cols:
                if column not in df_cur.columns:
                    raise ValueError(f"column={column} not in columns: {df_cur.columns.tolist()}")
            if self.target not in df_cur.columns:
                raise ValueError(f"target={self.target} not in columns: {df_cur.columns.tolist()}")

            if self.set_type == 0:
                print(f"[Dataset DLinearMix] input_cols={input_cols}, exog_cols={exog_cols}, target={self.target}, scale={self.scale}, segment_split={seg_col is not None}")

            df_x_cur = df_cur[x_cols].fillna(0)
            df_y_cur = df_cur[[self.target]].fillna(0)

            df_x_train = df_train[x_cols].fillna(0)
            df_y_train = df_train[[self.target]].fillna(0)

            if self.scale:
                self.scaler_x = StandardScaler()
                self.scaler_y = StandardScaler()

                self.scaler_x.fit(df_x_train.values)
                self.scaler_y.fit(df_y_train.values)

                data_x_all = self.scaler_x.transform(df_x_cur.values)
                data_y_all = self.scaler_y.transform(df_y_cur.values)

                self.scaler = self.scaler_y
            else:
                self.scaler_x = None
                self.scaler_y = None
                self.scaler = None
                data_x_all = df_x_cur.values
                data_y_all = df_y_cur.values

            self.x_cols = x_cols

        elif self.features in ["M", "MS"]:
            df_data_cur = df_cur[cols].fillna(0)
            df_data_train = df_train[cols].fillna(0)

            if self.scale:
                self.scaler = StandardScaler()
                self.scaler.fit(df_data_train.values)
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

            df_x_cur = df_cur[[x_col]].fillna(0)
            df_y_cur = df_cur[[y_col]].fillna(0)

            df_x_train = df_train[[x_col]].fillna(0)
            df_y_train = df_train[[y_col]].fillna(0)

            if self.scale:
                self.scaler_x = StandardScaler()
                self.scaler_y = StandardScaler()

                self.scaler_x.fit(df_x_train.values)
                self.scaler_y.fit(df_y_train.values)

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
        if self.model_name == "DLinearMix":
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
        self.valid_starts = None
        self.window_segment_ids = None
        if seg_col is not None:
            seg = df_cur[seg_col].iloc[border1:border2].to_numpy()
            n = len(seg)
            need = self.seq_len + self.pred_len

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

            self.valid_starts = np.asarray(valid, dtype=np.int64)
            self.window_segment_ids = seg[self.valid_starts]

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
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
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