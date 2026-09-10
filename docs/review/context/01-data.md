# 01 — 資料

> **本檔是「受審宣稱」（Tier 1）。** 所有數字皆為 2026-09-10 實際執行指令取得，
> 未轉抄 `PROGRESS.md`。審查者應自行驗算；不符即為 finding。

---

## 1. 資料在哪裡

全部在 `dataset/`（**已 gitignore**，不在 GitHub repo 裡，只存在於本機）。
共 22 個 CSV，約 500MB。原始來源是一台 **SQL Server**，透過
`Data_From_SQL_4.py` 的 `SQLServerClient` 取數（密碼走 Windows 整合驗證或
`os.getenv("DB_PASSWORD")`，**程式中無寫死帳密**）。

⚠️ **資料已凍結。** 現有資料是 2026-06-02 那次取數的結果（閘門欄於 2026-09-02
在本機重建），涵蓋 **2024-08-13 16:01 ~ 2025-08-03 23:59，511,679 分鐘**，
中間無缺分鐘（`build_drycut_segments_meta.py:97-99` 會驗證 1 分鐘連續網格）。
要延長時間範圍必須回到有 SQL 連線的環境重跑 `Data_From_SQL_all.py`。

## 2. 資料譜系（lineage）

```
SQL Server
  ├─ 水位 ──────────────────────────┐
  ├─ 降雨 ──────────────────────────┤
  └─ 閘門 → wra_cogate_obs_long.csv │   (247MB, long: ObsTime/PqId/FullName/ObsValue)
              │                      │
              ├─ pivot ─────────► wra_cogate_obs_wide.csv        (20 欄, 中文欄名)
              │                      │
              └─ filter_wra_cogate_columns.py
                                 ► wra_cogate_obs_wide_gate_opening.csv (7 個開度欄)
                                     │
                    Data_From_SQL_all.py（逐欄 merge_asof, 5min staleness）
                                     ▼
                          dataset/all_minute_wide.csv   ★ 唯一的現行原始寬表
                                     │
        rebuild_gate_columns.py 就地重建閘門欄 → 備份 all_minute_wide.gatev1.bak.csv
                                     │
             ┌───────────────────────┴────────────────────────┐
             │                                                │
   build_drycut_segments_meta.py                    rain_segments_meta.csv
   （乾段反向切分 L/buffer 參數化）                     （舊法 meta，159 段，含 split 欄）
             ▼                                                │
   rain_segments_meta_drycut_L3h_buf60.csv                     │
             │                                                │
             └──────► build_training_csv_from_meta.py ◄────────┘
                              （切段 + isRain + min_since_rain + 段內 gate ffill）
                       ▼                              ▼
        train_drycut_L3h_buf60.csv          train_old.csv (--allow-overlap)
                       │                              │
                       └────── build_splits.py ───────┘
                       ▼                              ▼
     splits_train_drycut_L3h_buf60.csv     splits_train_old.csv
```

## 3. 檔案總表

狀態分四類：**現行**（pipeline 正在用）／**備份**／**中間產物**（現行 pipeline 的上游，不直接餵模型）／**歷史遺留**（已被取代，勿用）／**一次性分析**。

| 檔案 | 大小 | 列數 | 欄數 | 時間範圍 | 狀態 | 說明與差異 |
|---|---|---|---|---|---|---|
| `all_minute_wide.csv` | 66M | 511,679 | 23 | 2024-08-13 16:01 → 2025-08-03 23:59 | **現行** | **唯一的原始寬表。** 逐分鐘連續網格。欄：`date` + HL01–06 + 8 個降雨欄 + 7 個閘門開度欄。閘門欄已於 2026-09-02 用逐欄 merge_asof 重建。無 BOM。 |
| `all_minute_wide.gatev1.bak.csv` | 53M | 511,679 | 23 | 同上 | 備份 | 閘門欄重建**之前**的版本（整列 merge_asof 產物，閘門 NaN ~73%）。`rebuild_gate_columns.py` 自動建立。**僅供比對，勿用於訓練。** |
| `wra_cogate_obs_long.csv` | 247M | 2,911,880 | 4 | 2024-08-13 → 2025-08-03 | 中間產物 | 閘門原始長表 `ObsTime,PqId,FullName,ObsValue`，含毫秒。**最大的檔，審查時不要整檔載入。** |
| `wra_cogate_obs_wide.csv` | 38M | 646,921 | 20 | 同上 | 中間產物 | 上一檔 pivot 成寬表，欄名為中文（開度／全開／全關 × 南北 × 編號）。日期格式 `2024/8/13 00:00:09.830`。 |
| `wra_cogate_obs_wide_gate_opening.csv` | 18M | 283,872 | 8 | 同上 | 中間產物 | `filter_wra_cogate_columns.py` 只留 7 個**開度**欄並改英文名。全開/全關欄被捨棄。 |
| `wra_cogate_pqid_fullname_map.csv` | 4.0K | 19 | 2 | — | 中間產物 | PqId → 中文全名對照。 |
| `rain_segments_meta_drycut_L3h_buf60.csv` | 16K | **179** | 7 | — | **現行** | **正式的 drycut 切分 meta（L=3h, buffer=60min）。** 欄：`segment_id,SegmentStart,SegmentEnd,WinStart,WinEnd,DurationMinutes,WetMinutes`。**無 `split` 欄**（切分決策已抽離到 `build_splits.py`）。 |
| `rain_segments_meta_drycut_L3h_buf0.csv` | 16K | 179 | 7 | — | 一次性分析 | 同上但 buffer=0。段數相同、window 較短。用來證明 buf=0 不可用（僅 72 段可生訓練 window）。 |
| `rain_segments_meta.csv` | 16K | **159** | 6 | — | 歷史遺留 | **舊法 meta**，含 `split` 欄（切分綁在 meta 裡）。仍被用來產 `train_old.csv` 作為對照組。已知有 32 對 window 重疊。 |
| `train_drycut_L3h_buf60.csv` | 12M | 54,590 | 30 | 2024-08-14 15:50 → 2025-08-03 20:39 | **現行** | **主力訓練檔。** 179 段。段長 min 130 / 中位 200 / max 2040 列。**有 UTF-8 BOM。** |
| `train_old.csv` | 11M | 50,489 | 30 | 2024-08-14 15:40 → 2025-08-03 21:20 | **現行（對照組）** | 走**同一條**前處理路徑重現舊法切分，159 段。段長 min 181 / 中位 261 / max 1521。與 drycut 版的差異**只有切分法**，其餘欄位與處理完全相同。**有 BOM。** |
| `train_*.source.sha256` | 64B | — | — | — | **現行** | 來源指紋 sidecar（meta 內容 + 寬表 size/mtime + 選項的 sha256），供 `--emit-training-csv` 判斷是否需重產。 |
| `splits_train_drycut_L3h_buf60.csv` | 12K | 179 | 8 | — | **現行** | 欄：`segment_id,SegmentStart,n_rows,n_windows,split,fold_1,fold_2,fold_3`。 |
| `splits_train_old.csv` | 8.0K | 159 | 8 | — | **現行（對照組）** | 同上結構。 |
| `water_level_rain_gate_all.csv` | 11M | 50,489 | 29 | 2024-08-14 15:40 → 2025-08-03 21:20 | 歷史遺留 | 舊訓練檔。與 `train_old.csv` 的差別：**少 `min_since_rain` 欄**（29 vs 30 欄），且閘門欄是修正前的值。**勿用。** |
| `water_level_rain_all4.csv` | 9.1M | 50,489 | 23 | 同上 | 歷史遺留 | 再上一版：無閘門欄，多一個 `StationId` 欄。 |
| `water_level_all3.csv` | 7.0M | 50,489 | 13 | 同上 | 歷史遺留 | 再上一版：只有水位 + segment 欄 + `isRain`，無降雨無閘門。 |
| `water_level_all2.csv` | 3.0M | 31,409 | 10 | 2024-08-14 16:40 → 2025-08-03 20:20 | 歷史遺留 | **只有核心分鐘、無 buffer**（31,409 = `train_old.csv` 的 `isRain=True` 列數，交叉驗證一致）。 |
| `water_level_all.csv` | 27M | 511,679 | 7 | 2024-08-13 16:01 → 2025-08-03 23:59 | 歷史遺留 | 只有 `date` + HL01–06 的全年寬表，是 `all_minute_wide.csv` 的前身。 |
| `rain_outside_segments_minutes.csv` | 152K | 4,933 | 4 | — | 一次性分析 | `list_rain_outside_segments.py` 產出：落在 segment 之外卻有雨的分鐘。 |
| `rain_outside_segments_runs.csv` | 8.0K | 86 | 8 | — | 一次性分析 | 上一檔彙整成 86 個連續 run。 |

## 4. 欄位字典

### 4.1 水位（單位：公分，整數量化）

| 欄 | 說明 |
|---|---|
| `HL01` | **目標站。** 全年 NaN 0.01% |
| `HL02`–`HL06` | 鄰站（上游）。NaN 0.00–0.02% |

### 4.2 降雨（單位：mm）

`Past10Min, Past1Hr, Past3Hr, Past6Hr, Past12Hr, Past24Hr, Past2Day, Past3Day, Now` — 全年 NaN 0.00%。

**⚠️ 語意陷阱：**
- 觀測值以 **0.5mm 量化**（`Past1Hr >= 0.1` 實質等同 `>= 0.5`）。
- 現行 pipeline 的「有雨」判定用 **`Past1Hr`**（`build_drycut_segments_meta.py` 預設 `--rain-col`）。
  `Past1Hr` 有一小時的尾巴，會讓「雨中分鐘」相對官方的 `Past10Min` 定義**膨脹約 2.4 倍**；
  改用 `Past10Min` 會讓資料量少約 60%。完整比較見 `docs/rain_event_definition_comparison.md`。

### 4.3 閘門開度（7 欄）

`north_gate_opening_1..4`, `south_gate_opening_1..3`

| 位置 | 任一欄 NaN |
|---|---|
| `all_minute_wide.csv`（分鐘層級） | **7.06%**（逐欄 6.77–6.81%） |
| `train_drycut_L3h_buf60.csv`（段內 ffill 後） | **3.48%** |
| `train_old.csv`（段內 ffill 後） | **3.41%** |

**⚠️ 語意陷阱：** `all_minute_wide.csv` 的閘門欄**刻意不做 within-segment ffill**
（產該檔時還沒有 segment 可分組）。任何從寬表切段的程式都**必須自行補做**，
否則大量 window 會因 NaN 被丟棄。`build_training_csv_from_meta.py:99-108` 負責補。
負值被 clip 到 0（視為全關）。

### 4.4 切分與衍生欄（只存在於訓練 CSV）

| 欄 | 說明 |
|---|---|
| `segment_id` | 段編號，1..N，依 `SegmentStart` 時序遞增 |
| `SegmentStart` / `SegmentEnd` | **核心**時段（有雨的那段）起訖 |
| `WinStart` / `WinEnd` | 核心 ± buffer 後的**實際取值**範圍 |
| `isRain` | **⚠️ 不是降雨旗標。** 定義是 `SegmentStart <= date <= SegmentEnd`，實為「在核心(True) / 在 buffer(False)」的標記。**buffer=0 時此欄恆為 True、完全失去資訊量** —— 這是 buf=0 不可用的另一個理由。drycut 版 True 33,110 / False 21,480；舊法版 True 31,409 / False 19,080。 |
| `min_since_rain` | 距上次降雨的分鐘數。**在切段前於全年寬表全域計算**（`build_training_csv_from_meta.py:53-73`），避免每段從 0 重數而抹掉段首的退水狀態。drycut 版：min 0 / 中位 22 / **max 24,240（16.8 天）** / NaN 60 列（資料開頭首次降雨之前無定義）。 |

## 5. 前處理決策

### 5.1 閘門合併：逐欄 merge_asof（2026-09-02 修正）

原始閘門寬表約 **39% 的列是部分回報**（7 欄只有其中幾欄有值）。舊的**整列**
`merge_asof` 每分鐘只抓時間最近的那一列，一旦抓到部分回報列，其餘欄位就是 NaN，
即使更早的列有那些欄位的值。改為**逐欄** merge_asof（每欄只用自己有值的觀測，
各自套 5 分鐘 staleness）後，任一欄 NaN 由 **79.84% → 7.06%**。

實作：`Data_From_SQL_all.py:128-150`（新資料用）、`rebuild_gate_columns.py:39-54`（就地重建既有寬表用）。
一致性檢查宣稱：兩邊皆有值的 949,041 個儲存格數值 100% 相同。

### 5.2 drycut 反向切分（L=3h, buffer=60min）

不是「找出降雨事件」，而是**反過來剔除確定乾燥的長段**：

1. `Past1Hr == 0` 且連續長度 `>= L` 的 run 標記為 removed（`build_drycut_segments_meta.py:40-44`）。
   **NaN 不算「確定沒雨」**（`dry = df[rain_col].eq(0)`，NaN → False）。
2. 剩下的連通區塊即為核心 segment，取 `SegmentStart/SegmentEnd`。
3. 每段兩端各往外延伸 `buffer` 分鐘得到 `WinStart/WinEnd`（`build_drycut_segments_meta.py:106-107`），
   並 clip 在資料範圍內。

**參數選擇的依據：** buffer=60 是唯一讓 179 段 100% 可生訓練 window 的設定
（buf=0 僅 72 段、buf=30 僅 98 段）。`L >= 2*buffer` 的約束保證相鄰 window 不重疊
——這是**結構性**的防 leakage，舊法沒有這個保證。

### 5.3 可用段門檻

一段能不能產生訓練 window，門檻是 **`seq_len + pred_len`**，不是固定值。
seq_len=96 → 111 分鐘。（舊紀錄曾誤用 75min，那只是 `run.py` 預設 seq_len=60 下的巧合。）

## 6. 已知資料品質問題

1. **`min_since_rain` 最大值 24,240 分鐘（16.8 天）**：在一個以降雨事件為主的資料集裡出現 16.8 天沒下雨的分鐘，值得確認是否合理（可能來自枯水期被 buffer 擦到的邊緣）。
2. **`min_since_rain` 有 60–70 列 NaN**：資料開頭首次降雨之前無定義。這些列會讓對應 window 被 NaN filter 丟掉。
3. **訓練 CSV 有 UTF-8 BOM**（`train_*.csv`、`water_level_rain_gate_all.csv`、`wra_cogate_*.csv`），`all_minute_wide.csv` 沒有。`Data_Loader._load_split_file` 有明確處理 BOM（`data_provider/Data_Loader.py:63`），但主資料讀取路徑是否處理需確認。
4. **`HL04` / `HL05` 在訓練 CSV 仍有 0.01% NaN**：由 NaN window filter 處理，不影響正確性，但代表少數 window 被丟棄。
5. **閘門殘餘 3.48% NaN**：段首 ffill 無值可填者。
