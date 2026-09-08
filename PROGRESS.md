# PROGRESS.md

> Single source of truth across sessions. See `CLAUDE.md` for the rules that govern this file.
> 跨 session 的單一事實來源；維護規則見 `CLAUDE.md`。

---

## 0. Snapshot (rewritten each update / 每次覆寫)

- **Last updated:** 2026-09-02T18:20:00+08:00
- **Current goal:** 閘門前處理缺陷已修、兩個資料集（drycut / 舊法對照組）與 3-fold split 皆已備妥。下一步是跑實驗矩陣（36 runs），之後才進 roadmap #1 delta-target。
- **Status (one line):** 修正 `Data_From_SQL_all.py` 的整列 merge_asof 缺陷（閘門任一欄 NaN 79.8%→7.1%，兩邊皆有值處數值 100% 相同）；`dataset/train_drycut_L3h_buf60.csv`（179 段/33,381 win）與 `dataset/train_old.csv`（159 段/31,440 win）皆已產出，各配一份 3-fold split 檔。**尚未開始跑實驗矩陣。**
- **Next steps:**
  1. **實驗矩陣 36 runs**：資料集(2: drycut/舊法) × exog(3: isRain / 無 / min_since_rain) × loss(2: mse/huber) × fold(3) × seed(1)，約 4.2 小時。需新寫 (i) 因子驅動腳本 (ii) 把 anchored 指標併進同一張總表的彙整器。
  2. roadmap #1 **delta-target**，續接 #2–#6。完整版見 `docs/model_roadmap.md`。
- **Open questions / blockers:** 無 blocker。待決：split 方式（目前沿用 loader 內建 70/10/rest；隨機 per-segment 暫緩）；exog 路徑是否有效（待 #2 ablation）；**buffer=60 讓 5/20「後置 60min 混合動態」的疑慮重新浮現**（見 `docs/model_roadmap.md` 第 5 節）。資料來源為本機 `dataset/all_minute_wide.csv`（2026-06-02 產，2024-08-13 16:01~2025-08-03 23:59，511,679 分鐘），**未重抓 SQL**；要延長範圍需回 SQL 環境重跑 `Data_From_SQL_all.py`。
- **Must-know handoff points:**
  - **系統定位（2026-09-02 定案）**：**即時預警系統**；虛擬水位量測是更長遠目標。此定位決定 delta-target 可行（推論時有 HL01 當下值當錨）；若日後轉向虛擬量測，roadmap #1/#4 需重新設計。
  - **核心發現**：HL01 不在 input（input=HL02–06+exog）→ 無 level 錨 → 每 window 固定偏移；persistence 因此贏 raw 模型（raw MSE 20884 / anchored 1537 / persist 2430；Corr 0.819→0.988）。
  - **原則**：不放 HL01 自身歷史當 input（會自迴歸依賴）；用 HL01 最後值當外部錨可以。主指標 = **correlation**。
  - **L 與 buffer 的作用**：L 決定「多長的無雨算確定乾、要剔除」→ 控制切點與段數；buffer 決定「每段兩端往被剔除的乾段延伸多少」→ 保留退水尾巴並救活短段。約束 `L >= 2*buffer` 保證相鄰 window 不重疊（結構性避免 leakage，舊法沒有這個保證）。
  - **可用段門檻 = `seq_len + pred_len`，不是固定值**（舊紀錄誤用 75min）。seq_len=96 → 111min；`run.py` 預設 seq_len=60 → 75min。L=3h/buf=60 下 seq_len≤96 皆 179/179 可用，seq_len≥120 開始出現死段。
  - anchored 工具：`compute_anchored_mse.py`（raw/adj/persist 的 MSE/RMSE/MAE/Corr + 逐 horizon Corr）、`visualize_segment.py`、`slide_anchored_figure.py`、`visualize_anchored.py`；乾段 anchored 直接讀 `eval_dry/*/predictions.npz` 的 `persist`。
  - Repo：`github.com/evonnejan/HLML-Linear-Model`；DB 不可寫死帳密；大型輸出（含 `dataset/`、`analysis/`）已 gitignore。

---

## 1. Architecture & Key Decisions

| Date | Decision | Rationale | Alternatives considered |
|------|----------|-----------|-------------------------|
| 2026-06-30 | 沿用既有 GitHub repo 整理後 push（非另建新 repo） | repo 已存在且已連 origin，本地僅領先 9 commit | 另建新 repo（被否決：無需要） |
| 2026-06-30 | 用 `git rm --cached` 停止追蹤 1618 個 `test_results/` PDF | 移出版控但保留本機檔、不重寫歷史、操作安全 | 保持現狀（被否決：repo 臃腫）／filter-repo 重寫歷史（被否決：過重、需 force-push） |
| 2026-06-30 | backlog 依主題拆 5 個 commit | 歷史清楚、便於回溯 | 單一大 commit（被否決：歷史過粗） |
| 2026-06-30 | `docs/superpowers/`、`meeting_recap.txt`、`sh.txt` 不上傳 | 本地工具文件/私人草稿 | 上傳（依使用者意願否決） |
| 2026-06-30 | 新增 `CLAUDE.md` 為常駐指令、自動維護 `PROGRESS.md` | 跨 session 冷啟動接手 | 僅靠 auto-memory（不足以承載專案級進度） |
| 2026-06-30 | level bias 治法選 **delta-target**（非 NLinear/RevIN/後處理） | NLinear 錨到 input(鄰站)非 HL01；RevIN 需 HL01 歷史統計(違反原則)；後處理非 end-to-end。delta-target 錨對 HL01、只用最後值、架構不動 | NLinear（否決:錨錯通道）／RevIN（否決:需 HL01 歷史）／續用後處理 anchoring（否決:形狀沒被訓練優化） |
| 2026-09-02 | 閘門合併改**逐欄** merge_asof，並就地重建既有寬表 | 原始寬表 39% 的列是部分回報，整列比對會丟掉其他欄位的歷史值；資料已凍結故可在本機重建，不需回 SQL | 維持現狀（否決：白白損失 ~5% 訓練 window 且閘門特徵品質差）／當成實驗因子（否決：使用者指示先修正再實驗）／回 SQL 重抓（否決：不需要，本機資料已完整） |
| 2026-09-02 | rolling-origin fold 數定為 **3** | k=3 的 val 大小最平均（±4%）且每折含 28–38 個降雨事件；k=4/6 有折僅 15/9 段。有效樣本單位是事件而非 window | k=4（否決：一折僅 15 段）／k=5、k=6（否決：折內事件數過少，估計不穩） |
| 2026-09-02 | split 邊界依**可用 window 數**決定，且 split 決策抽離到 loader 之外（`--split_file`） | 段長差距 130~2040 分鐘，按段數切會使實際樣本比例嚴重偏離；抽離後換切法不需動訓練碼，且可版本化保存每次實驗用的切分 | 按段數切（否決：比例失真）／改寫 loader 內建邏輯（否決：每換一種切法就要動訓練碼）／隨機 per-segment split（暫緩：違反 forecasting 的時序因果） |
| 2026-09-02 | `isRain` 改用 `min_since_rain`，且在**切段前全域計算** | `isRain` 實為核心/buffer 標記，會把切分結構洩漏給模型；`min_since_rain` 直接描述退水階段、有物理意義。切段後才算會讓每段從 0 重數，抹掉段首的「剛下過雨」狀態 | 保留 isRain（否決：洩漏切分結構）／直接拿掉不補（保留為實驗矩陣的對照組） |
| 2026-09-02 | 系統定位＝**即時預警**，虛擬水位量測列為長遠目標 | 即時預警推論時有 HL01 當下值可當錨 → delta-target 可行；虛擬量測沒有 HL01，該路不通（`meeting_recap.txt` 5/20 已註明） | 直接做虛擬量測（否決：會封死 roadmap #1/#4） |
| 2026-09-02 | drycut 定 **L=3h, buffer=60min** | buf=60 是唯一讓 179 段 100% 可生訓練 window 的設定（buf=0 僅 72 段、buf=30 僅 98 段），且 34,900 win 已超過舊法 32,999；L 在 buf=60 下不敏感，選 3 保留最多段數與最細事件粒度 | buf=90（否決：多出的 1 萬 win 是純乾 padding）／L=4~6（否決：段數更少、無額外好處）／另立小碎段丟棄規則（否決：buf=60 後問題自動消失） |
| 2026-06-30 | 不放 HL01 自身歷史當 input；主指標用 correlation | 放 HL01 歷史會自迴歸過度依賴(試過)；level 可校正、變化形式才是學的重點 | 加 HL01 自迴歸（否決） |

---

## 2. File / Component Map

- `run.py` — 訓練/評估進入點（CLI argparse；`--model`, `--data`, `--seq_len`, `--pred_len`, `--features`, `--input_col`, `--exog_col` 等）。
- `Data_From_SQL_4.py` — 核心 SQL client（`SQLServerClient`，密碼來自 `os.getenv("DB_PASSWORD")`/Docker），含降雨事件切窗、segment 等 loader。
- `Data_From_SQL_3.py / _5.py / _all.py` — 沿用 `_4` 的 client 產生 water/rain/gate 寬表與含 train/val/test split 的 rain segment metadata。
- `filter_wra_cogate_columns.py` — 篩選 WraCoGate 欄位。
- `merge_gate_data.py` — 閘門資料合併（對齊 / staleness reset / segment ffill）。
- `data_provider/` — `Data_Loader.py`、`Data_Factory.py` 資料載入。
- `models/` — `DLinear.py`、`DLinearMix.py`、`DLinearMix2.py`（線性 + 外生變數融合）；`__init__.py` 註冊模型。
- `exp/` — `exp_Main.py`、`exp_Main2.py` 訓練/評估流程；`exp_Basic.py` 基底。
- `analyze_*.py` — best-model overview、full inference(+lag)、rain-outside-segments、anchored MSE 等分析。
- `compute_anchored_mse.py` / `list_rain_outside_segments.py` — anchored MSE 計算與降雨外 segment 分析。
- `eval_dry.py` + `scripts/draw_eval_dry_diagrams.py` — 乾期評估與圖表。
- `rebuild_gate_columns.py` — 以逐欄 merge_asof 就地重建 `all_minute_wide.csv` 的閘門欄（不需 SQL）；含 dry-run 與一致性檢查，會自動備份原檔為 `.gatev1.bak.csv`。
- `build_splits.py` — **segment-wise 時序切分 + rolling-origin expanding-window CV**。依各段可用 window 數（NaN-aware，對齊 Data_Loader）找 train/val/test 邊界，不拆段；輸出 `split` 與 `fold_k` 欄供 `run.py --split_file/--fold` 使用。
- `build_training_csv_from_meta.py` — **meta → 訓練 CSV 組裝器**（pipeline 斷點的補丁）。任何含 segment_id/SegmentStart/SegmentEnd/WinStart/WinEnd 的 meta 皆適用；切法沿用 `Data_From_SQL_4.py:400-436`，並補做 gate 段內 ffill。
- `analyze_dry_runs.py` / `build_drycut_segments_meta.py` / `visualize_drycut_segments.py` — 乾段反向切分法 drycut（7/2 起）：可行性統計、meta 產生（L/buffer 參數化、無 split）、切分結果檢視圖（英文、附 dry gap 標註）。
- `visualize.py` / `model_visualize.py` / `visualize_anchored.py` / `visualize_segment.py` / `slide_anchored_figure.py` — 視覺化。
- `utils/` — `metrics.py`、`tools.py`、`timefeatures.py`。
- `Source_Code/` — DLinear 原始參考碼（LTSF-Linear，Apache-2.0）：`DLinear/Linear/NLinear.py`、`exp_*`、`data_*`。
- `run_dlinearmix2_sweep*.sh` — 掃參腳本（base / criterion / noHL01 變體）。
- `tests/test_merge_gate_data.py` — `merge_gate_data` 單元測試。
- `technical_manual.md` / `technical_manual_xml.md` — 技術手冊。
- `docs/model_roadmap.md` — **模型改進 roadmap 完整版**（六項的理由/否決方案/限制、系統定位、volatility 對策）。摘要見第 4 節。
- `docs/` — 降雨事件定義比較、work summary、figures（注意：`docs/superpowers/` 已 gitignore）。

---

## 3. Changelog (newest-first, append-only / 新到舊，只 append)

### 2026-09-02T18:20:00+08:00 — 修正閘門整列 merge_asof 缺陷、產出舊法對照組、fold 定為 3
- **Trigger:** 使用者追問閘門 ffill 是否有改善空間；並指示「閘門要修正、修完再開始實驗」、fold 用 3。
- **What changed:**
  - **修正 `Data_From_SQL_all.py:126-148` 的閘門合併**：由整列 `merge_asof` 改為**逐欄** `merge_asof`（每欄只用自己有值的觀測，各自套 5 分鐘 staleness）。
  - 新增 `rebuild_gate_columns.py`：把同樣的修正套用到既有的 `all_minute_wide.csv`（資料已凍結，不需回 SQL），含 `--dry-run` 與數值一致性檢查，並自動備份原檔為 `dataset/all_minute_wide.gatev1.bak.csv`。
  - `build_training_csv_from_meta.py` 新增 `--allow-overlap`：舊 meta 有 32 對 window 重疊會被重疊檢查擋下，重現舊法當對照組時需明示接受。
  - `build_splits.py` 預設 `--n-folds` 由 4 改為 **3**。
- **Why（根因分析）:** 原始閘門寬表有 **39% 的列是部分回報**（7 欄只有其中幾欄有值）。整列 merge_asof 對每分鐘只抓「時間最近的那一列」，一旦抓到部分回報列，其餘欄位即為 NaN —— 即使更早的列有那些欄位的值。逐欄的歷史被整列邏輯丟掉了。
- **Files touched:** `Data_From_SQL_all.py`、`rebuild_gate_columns.py`(新增)、`build_training_csv_from_meta.py`、`build_splits.py`、`dataset/all_minute_wide.csv`(閘門欄重建)、`dataset/train_drycut_L3h_buf60.csv`(重產)、`dataset/train_old.csv`(新增)、`dataset/splits_*.csv`。**未動模型/訓練碼。**
- **Commands run:** `python rebuild_gate_columns.py --dry-run` → `python rebuild_gate_columns.py`；`build_training_csv_from_meta.py` 兩份 meta 各一次（舊法加 `--allow-overlap`）；`build_splits.py` 兩份各一次。
- **Result/verification:**
  - 閘門 NaN（分鐘層級）：任一欄 **79.84% → 7.06%**；逐欄皆由 ~73% 降至 ~6.8%。
  - **一致性檢查：兩邊皆有值的 949,041 個儲存格，數值 100.00% 相同** → 確認只是補回原本被丟掉的值，未竄改任何既有資料。
  - 訓練 CSV 的閘門 NaN 4.60% → 3.48%；drycut 可用 windows 32,786 → **33,381**（+595）。
  - 舊法對照組 `dataset/train_old.csv`：159 段 / 50,489 列 / **31,440 windows**；段長 min 181、中位 261、max 1521。重疊 32 對（約 1,610 分鐘）已明示接受。
  - 3-fold split（drycut）：val windows 4,294 / 4,641 / 4,627（極平均，±4%），val 段數 38/32/28。
- **fold 數選 3 的依據（實測 3/4/5/6）:** k=3 的 val 大小最平均且每折有 28–38 個降雨事件；k=4 有一折僅 15 段、k=6 有兩折僅 9 段。有效樣本單位是**事件**而非 window（同段內 window 高度相關），故事件數過少的折估計不穩。
- **關於 73% 落差的完整解釋:** all_minute_wide 舊值的閘門 NaN 為 73%，但以本機閘門 CSV 模擬整列 merge_asof 只得 20%。經比對：兩者數值 100% 相同（同源、無資料錯亂），且 90.7% 的 NaN 分鐘在本機檔中其實有 5 分鐘內的觀測。推論為 2026-06-02 那次 SQL 取數回傳的**逐欄覆蓋率遠低於**本機 4/8 的 CSV，與整列 merge 缺陷疊加後放大到 73%。此推論無法在無 SQL 環境下驗證，但無論成因為何，逐欄重建的結果皆已驗證正確。
- **Follow-ups:** 實驗矩陣 36 runs（見 Snapshot next steps 1）。

### 2026-09-02T17:10:00+08:00 — split 框架改版（segment-wise + rolling-origin CV）、min_since_rain、meta 一鍵產訓練檔
- **Trigger:** 使用者提出 train/val/test split 設計（segment integrity → chronological order → sufficient eval data → approximate ratio），並要求 `min_since_rain` 取代 `isRain`、以及 meta 產生器加一鍵開關。
- **What changed:**
  - 新增 `build_splits.py`：依各段**可用 forecasting window 數**（NaN-aware，計算方式對齊 `Data_Loader` 的 valid_starts）找 train/val/test 邊界，segment 不拆；並在 dev(=train+val) 內產 rolling-origin **expanding-window** folds（塊 0 為起始 train，塊 1..k 依序為各 fold 的 val，train 逐 fold 擴張）。
  - `Data_Loader` 新增 `split_file` / `fold` 參數與 `_load_split_file()`：**split 決策自 loader 抽離**，換切法不需再動訓練碼。未給 `split_file` 時行為與過去完全相同（向後相容）。
  - `Data_Factory` 透傳；`run.py` 新增 `--split_file`、`--fold`。
  - `build_training_csv_from_meta.py`：新增 `min_since_rain`（**切段前於全年寬表全域計算**，避免每段開頭被迫從 0 重算而抹掉退水狀態）；重構出可被外部呼叫的 `build_training_csv()`。
  - `build_drycut_segments_meta.py`：新增 `--emit-training-csv`，內部呼叫上述函式；以 **來源指紋（meta 內容 + 寬表 size/mtime + 選項）的 sha256 sidecar** 判斷是否已是最新，相同即跳過，避免重複產出相同大檔。
- **Files touched:** `build_splits.py`(新增)、`data_provider/Data_Loader.py`、`data_provider/Data_Factory.py`、`run.py`、`build_training_csv_from_meta.py`、`build_drycut_segments_meta.py`、`dataset/train_drycut_L3h_buf60.csv`(重產, 30 欄)、`dataset/splits_train_drycut_L3h_buf60.csv`(新增)。
- **Commands run:** `python build_training_csv_from_meta.py --meta ...buf60.csv`、`python build_splits.py --data-path dataset/train_drycut_L3h_buf60.csv`、Dataset_Custom 直接實例化的三組對照測試、`run.py --split_file ... --fold 2 --train_epochs 1` 端到端測試。
- **Result/verification:**
  - 單一切分：train 137 段/23,151 win (70.6%)、val 27/4,043 (12.3%)、test 15/5,592 (17.1%)。比例受 segment 完整性約束，符合「approximate」設計。
  - 4 個 rolling-origin fold 的 val 期間依序為 2024-12-24→2025-02-19、→2025-04-23、→2025-05-19、→2025-07-11，train 由 66 段/13,892 win 擴張到 142 段/23,827 win。
  - **關鍵驗證：loader 實際產生的 window 數與 `build_splits.py` 的計算完全一致**（split: 23,151/4,043/5,592；fold_1: 13,892/3,089/5,592；fold_4: 23,827/3,367/5,592），且 **test 在所有 fold 間固定為 5,592**。
  - 未給 `--split_file` 時 loader 走原路徑（19,803/4,024/8,959），向後相容確認。
  - `--emit-training-csv` 指紋機制：同參數重跑正確跳過、換 L 參數正確重產。
- **設計決策:** split 以 **window 數**而非段數決定邊界 —— segment 長度差距達 130~2040 分鐘，按段數切會讓實際樣本數嚴重偏離目標比例。
- **Follow-ups:** 實驗矩陣（見 Snapshot next steps 1）。

### 2026-09-02T16:05:00+08:00 — 補上 pipeline 斷點：meta → 訓練 CSV 組裝器
- **Trigger:** 使用者詢問「產完 meta 之後要怎麼訓練」，確認 pipeline 中間缺一支腳本。
- **What changed:** 新增 `build_training_csv_from_meta.py`，把 segment meta 與 `dataset/all_minute_wide.csv` 組裝成 `run.py` 可直接使用的訓練 CSV。並以定案參數產出 `dataset/train_drycut_L3h_buf60.csv`。
- **Why:** `build_drycut_segments_meta.py` 只產出 segment 目錄（時間範圍、無數值），`Data_Loader` 要的是帶 `segment_id`/`SegmentStart` 的逐分鐘訓練檔；舊 pipeline 的對應邏輯綁在需要連 SQL 的 `Data_From_SQL_4.py` 裡，無法重用。
- **Files touched:** `build_training_csv_from_meta.py`(新增)、`dataset/train_drycut_L3h_buf60.csv`(產出, gitignored)、`PROGRESS.md`。**未動模型/訓練碼。**
- **Commands run:** `python build_training_csv_from_meta.py --meta dataset/rain_segments_meta_drycut_L3h_buf60.csv`
- **Result/verification:**
  - 54,590 列 / 179 段（meta 179 段全數成功）/ 段長 min 130m、中位 200m、max 2040m。
  - **29 欄且欄位順序與舊 `water_level_rain_gate_all.csv` 完全一致**（程式比對確認），loader 讀法不變。
  - seq_len=96/pred_len=15 → 179/179 段可用、**34,900 windows**，與定案時的預估完全吻合。
  - `isRain` 核心 33,110 列 / buffer 21,480 列；核心數與 buf=0 版的保留分鐘數 33,110 完全一致 → 交叉驗證通過。
  - gate NaN 經段內 ffill 由 **73% 降至 4.4%**（殘餘為段首 ffill 無法填補者）；HL 欄殘餘 NaN 僅 13 列。
- **重要語意釐清（易誤解）:**
  - **`isRain` 不是降雨旗標**。其定義（`Data_From_SQL_4.label_rain_minutes`）是 `date ∈ [SegmentStart, SegmentEnd]`，實為「在核心(1) / 在 buffer(0)」的標記。**buffer=0 時該欄恆為 1、完全失去資訊量** —— 這是 buf=0 不可用的另一個理由。
  - all_minute_wide 的 gate 欄約 73% 是 NaN，因為 `Data_From_SQL_all.py` 刻意把 within-segment ffill 延後給下游（該檔當時無 segment 可分組）。任何從寬表切段的程式都必須自行補做，否則絕大多數 window 會因 NaN 被丟棄。
- **Follow-ups:** 跑 baseline 訓練並與舊資料集對照；之後進 roadmap #1。

### 2026-09-02T15:33:47+08:00 — drycut 參數定案(L=3h/buf=60)、產正式 meta、更正可用段門檻、系統定位定案
- **Trigger:** 使用者長期休息後回歸，要求完整複盤；複盤中定案 buffer/L 與系統定位。
- **What changed:**
  - **定案 `L=3h, buffer=60min`**，跑出正式 meta `dataset/rain_segments_meta_drycut_L3h_buf60.csv`（覆蓋 7/2 的同名樣式預覽檔）。
  - **定案系統定位＝即時預警系統**（虛擬水位量測為更長遠目標）→ 確認 roadmap #1 delta-target 可行（推論時有 HL01 當下值當錨）。
  - 新增 `docs/model_roadmap.md`：六項改進的完整版（理由／否決方案／限制／volatility 對策）。
  - **更正**：舊紀錄用「75min 可用門檻」判斷段夠不夠長是錯的；正確門檻是 `seq_len + pred_len`（seq_len=96→111min，`run.py` 預設 seq_len=60→75min，75 這個數字只是預設值下的巧合）。
- **Why:** 兩個月空窗後需重建脈絡；重算數據後發現 buffer=0 不可用（60% 段生不出 window），且門檻公式記錯會高估可用段數。
- **Files touched:** `dataset/rain_segments_meta_drycut_L3h_buf60.csv`(重產)、`docs/model_roadmap.md`(新增)、`PROGRESS.md`。**未動任何模型/訓練碼。**
- **Commands run:** `python build_drycut_segments_meta.py --l-hours 3 --buffer-minutes 60`；另以 scratchpad 腳本重算 L×buffer 網格與 seq_len 敏感度、驗證 window 重疊。
- **Result/verification:**
  - 正式 meta：**179 段 / 54,590 分鐘 / 10.7%**；duration min=130m、中位 200m、max 2040m；**無任何段 <2h**。
  - **重疊驗證：0 對重疊，最小相鄰間隔 61 分鐘**，與 `L-2B = 180-120 = 60` 的理論下界吻合 → `L>=2B` 約束確實成立。
  - L×buffer 網格（可生 window 的段數 / 訓練 windows，seq_len=96）：buf=0 → 72/179 段、21,600 win；buf=30 → 98/179、26,960；**buf=60 → 179/179、34,900**；buf=90 → 179/179、45,640。舊法對照 159 段、32,999 win。
  - → **buffer=60 使 100% 段可用，「57 段 10 分鐘碎段」問題自動消失，不需額外丟棄規則**；且訓練量已超過舊法。
  - seq_len 敏感度（L=3h/buf=60）：seq_len 30/60/96 皆 179/179 可用（46,714 / 41,344 / 34,900 win）；seq_len=120 起降為 122 段，180 → 94 段。
- **Follow-ups:**
  - **pipeline 斷點**：目前**沒有**腳本把 meta + `all_minute_wide.csv` 組成訓練 CSV，需新寫（見 Snapshot next steps 1）。
  - buffer=60 使 5/20「後置 60min 可能混合上升/衰退/平靜動態」的疑慮重新浮現，待 delta-target 後以 per-segment 指標檢查。
  - 7/25 的兩張 drawio（水位因子圖、pipeline 架構圖）用途未明，待使用者說明後補記。

### 2026-07-02T17:00:00+08:00 — duration bin 改細 + buffer 改 Win 慣例並以粉紅色呈現
- **Trigger:** 使用者要求（bin 改 10m/20m/30m/30-60m/…；buffer 用粉紅色）。
- **What changed:** `build_drycut_segments_meta.py`：bin 改右閉 `[0,10,20,30,60,120,240,480,960,1920,inf]`；meta schema 改回現行慣例 **SegmentStart/End=含雨核心、WinStart/WinEnd=核心±buffer（資料首尾截斷）**，DurationMinutes 以 Win 計，加 `L>=2*buffer` 防重疊檢查。`visualize_drycut_segments.py`：核心綠、buffer 粉紅（hotpink）、灰=剔除乾段；bin 常數改由 build 腳本 import；gap 改以 Win 邊界計。
- **Commands run:** build+viz buf=0 全量重產；buf=60 產預覽（`dataset/rain_segments_meta_drycut_L3h_buf60.csv` + `analysis/drycut_segments/.../seg_003,046`）。
- **Result/verification:** buf=0 數字不變（179 段/33,110 分鐘）；新 bin 顯示 57 段恰為 10m（孤立單筆雨測）。buf=60 預覽驗證：seg_046 dur 1920→2040m、gap 6.2/21.4→4.2/19.4h（兩側各吃 1h，正確）；seg_003 10m 核心→130m（跨過 75min 可用門檻，展示 buffer 拯救小碎段）。
- **Follow-ups:** buf=60 只是樣式預覽，正式 buffer 值仍待使用者定案。

### 2026-07-02T16:40:24+08:00 — 依使用者回饋改名 drycut、移除 split、補 duration 統計、圖全英文
- **Trigger:** 使用者七點回饋（命名誤導、split 先不用、要 duration 分布、圖不用中文、context 顯示窗說明）。
- **What changed:** `build_dry_segments_meta.py`/`visualize_dry_segments.py` 改名為 `build_drycut_segments_meta.py`/`visualize_drycut_segments.py`（保留的 segment 都含雨、非 dry，輸出改叫 `rain_segments_meta_drycut_*`）；meta 移除 split 欄（split 方式後續另定）；build 腳本加 duration 分 bin 計數＋統計量輸出；viz 全英文、新增 `duration_hist.png`、每段標題加上相鄰被剔除乾段的真實長度（gap before/after，資料邊界顯示 edge）；舊名檔案與輸出已刪除。
- **Files touched:** `build_drycut_segments_meta.py`、`visualize_drycut_segments.py`（新增）；`build_dry_segments_meta.py`、`visualize_dry_segments.py`、`dataset/dry_segments_meta_L3h_buf0.csv`、`analysis/dry_segments/`（刪除）。
- **Commands run:** `python build_drycut_segments_meta.py --l-hours 3 --buffer-minutes 0`、`python visualize_drycut_segments.py`。
- **Result/verification:** 179 段/33,110 分鐘不變；duration 分布：<15m 57 段（多為 10min 孤立陣雨）、中位 80m、max 1920m（32h）；>=4h 的 40 段就佔保留分鐘的 73%。圖抽查 OK（seg_046 標註 gap before=6.2h after=21.4h）。
- **Follow-ups:** 澄清：圖上灰色前後 ±3h 只是顯示窗（`--context-hours`），與 L/buffer 無關；實際相鄰乾段 ≥L 且通常更長，真實長度見標題。

### 2026-07-02T16:35:00+08:00 — 產出 L=3h buf=0 乾段切分 meta + 檢視圖
- **Trigger:** 使用者定案「L 先用 3、buffer=0，先看資料長相；buffer 基本確定要加但之後再定」。
- **What changed:** 新增 `build_dry_segments_meta.py`（L/buffer 參數化，重用 `Data_From_SQL_all.assign_split_to_segments` 的 70/10/rest split；NaN 不算確定乾；buffer 邊緣純乾碎片自動丟棄）與 `visualize_dry_segments.py`（全年 overview + 每段兩層圖：上 HL01 下 Past10Min，segment 綠底、前後 ±3h 被剔除脈絡灰底；CJK 字型 fallback）。
- **Files touched:** `build_dry_segments_meta.py`、`visualize_dry_segments.py`（皆新增）。
- **Commands run:** `python build_dry_segments_meta.py --l-hours 3 --buffer-minutes 0`、`python visualize_dry_segments.py --meta dataset/dry_segments_meta_L3h_buf0.csv`。
- **Result/verification:** `dataset/dry_segments_meta_L3h_buf0.csv`：179 段、33,110 分鐘（6.5%）、split train125/val17/test37，與 `analyze_dry_runs.py` 統計完全吻合。圖輸出 `analysis/dry_segments/dry_segments_meta_L3h_buf0/`（gitignored）。抽查：seg_046（最長 32h，雨間乾檔正確保留、但 segment 結束時退水被切掉→支持 buffer）、seg_003（孤立 10min 陣雨→超短碎段，共 85 段 <75min）、overview 顯示 test 又集中 7-8 月颱風季。
- **Follow-ups:** 使用者看圖 → 定 buffer/L/小段處理 → 重產正式 meta → 接 `data_provider/`。

### 2026-07-02T16:06:12+08:00 — 乾段反向切分法：可行性統計（analyze_dry_runs.py，未改 pipeline）
- **Trigger:** 使用者提出新切分想法（7/2 會議「從沒下雨的部分出發」）；經 AskUserQuestion 選「先看統計再定邊界」。
- **What changed:** 新增 `analyze_dry_runs.py`（read-only 統計腳本）：以 Past10Min==0 連續 ≥L 小時偵測長乾段，對 L=3/4/5/6h 算剔除/保留量、殘餘 segment 分佈、與現行 Past1Hr 版 rain windows 的重疊。另以 inline script 補算 ±60min buffer 版本。
- **Why:** 新法反向定義資料（剔除確定乾，其餘全留）：避開 Past1Hr 1 小時尾巴灌水、保留兩場雨之間 <L 的停雨段（退水動態）。先量化再定 L 與邊界規則。
- **Files touched:** `analyze_dry_runs.py`（新增）。
- **Commands run:** `python analyze_dry_runs.py`、inline buffer 計算。
- **Result/verification（資料 2024-08-13~2025-08-03，511,679 分鐘，97% 乾）:**
  - 雨分鐘保留率各 L 皆 100%（sanity check 通過）。
  - 不留 buffer：L=4h → 161 seg、36,780 分鐘（7.2%），但 p25≈12min，可用（≥75min）僅 91 seg。
  - 留 ±60min buffer：L=4h → 163 seg、56,220 分鐘（11.0%），161 seg 可用，中位長 3.5h。
  - 對照現行（48,847 分鐘、159 seg）：現行視窗內有 13k–18k 分鐘落在長乾段內（Past1Hr 尾巴+buffer），新法（無 buffer 版）會剔除；另新增 2k–8.5k 分鐘視窗外資料（雨間停雨段）。
- **Follow-ups:** 使用者定 L 與 buffer 規則 → 改 pipeline 產新 segments meta。注意：缺測不可視為「確定乾」（本份資料無 NaN，但 pipeline 化時要處理）。

### 2026-06-30T16:00:40+08:00 — anchored 診斷 + 模型改進 roadmap（討論，未改碼）
- **Trigger:** 使用者要求記錄 roadmap + 研究方向確立（scope change）。
- **What changed:** 完成 level-bias 診斷與一系列分析/視覺化工具；與使用者討論並定下模型改進方向。**未改動訓練碼**。
- **Why:** 找出 raw MSE 遠輸 persistence 的原因（per-window level bias），並規劃治本路線。
- **Files touched（皆新增分析/視覺化，未動模型/訓練）:** `compute_anchored_mse.py`（含逐 horizon Corr）、`visualize_anchored.py`、`visualize_segment.py`（含 rain on/off）、`slide_anchored_figure.py`、`analyze_rain_outside_segments.py`、`list_rain_outside_segments.py`、`docs/superpowers/specs/2026-06-03-segment-visualization-design.md`；`exp/exp_Main2.py` 僅 +1 行存 `persist.npy`。
- **Result/verification:** raw MSE 20884 / anchored 1537 / persistence 2430，Corr 0.819→0.988；persistence 交叉驗證 diff<0.01%。乾段:adj 砍 ~96% 誤差但仍輸 persistence（模型在乾段亂動）。根因:HL01 不在 input。
- **Follow-ups:** 見第 4 節 roadmap；下一步 delta-target。
- **Memory:** auto-memory 新增 `anchored-mse-finding`、`model-improvement-roadmap`。

### 2026-06-30T14:20:02+08:00 — commit & push PROGRESS.md
- **Trigger:** 使用者要求（先 commit PROGRESS.md）+ 隨後使用者 push。
- **What changed:** commit `PROGRESS.md` 第一版（`59f1a5a`）並 push 到 origin；本筆同時更新 Snapshot 與 origin 同步點。
- **Why:** 把進度追蹤檔納入版控、保持 Snapshot 與遠端一致。
- **Files touched:** `PROGRESS.md`。
- **Commands run:** `git add PROGRESS.md`、`git commit`、（使用者）`git push origin main`、`git fetch`、`git rev-list --left-right --count`。
- **Result/verification:** `origin/main == main == 59f1a5a`，0 ahead / 0 behind。
- **Follow-ups:** 決定是否追蹤 `CLAUDE.md`（目前未追蹤）。

### 2026-06-30T14:06:38+08:00 — Stage 3: 初始化 PROGRESS.md
- **Trigger:** 使用者要求（階段三）。
- **What changed:** 建立本檔第一版，盤點 git log / status / 目錄結構 / requirements 後填入第 0–5 節。
- **Why:** 提供跨 session 冷啟動接手的單一事實來源。
- **Files touched:** `PROGRESS.md`（新增）。
- **Commands run:** `git log --oneline`、`git status`、`git ls-files`、`cat requirements.txt`、`date +%Y-%m-%dT%H:%M:%S%z`。
- **Result/verification:** 工作區乾淨；`origin/main` 與本地 `main` 同步於 `df511c1`。
- **Follow-ups:** 詢問是否 commit & push `CLAUDE.md` + `PROGRESS.md`。

### 2026-06-30 — Stage 2: 新增 CLAUDE.md 常駐指令
- **Trigger:** 使用者要求（階段二）。
- **What changed:** 新增雙語 `CLAUDE.md`，定義自動維護 `PROGRESS.md` 的觸發時機、寫作要求、六節結構與專案速覽。
- **Why:** 讓後續每個 session 在開發同時維護進度。
- **Files touched:** `CLAUDE.md`（新增，未 commit）。
- **Result/verification:** 草稿經使用者確認改為雙語後寫入。

### 2026-06-30 — Stage 1: GitHub 整理與首次 push（9 commits）
- **Trigger:** 使用者要求（階段一）。
- **What changed:**
  - 重整 `.gitignore`（venv、`__pycache__`/`.pyc`、`.DS_Store`、`.claude/`、`node_modules`、各輸出目錄、私人草稿、預防性 secrets）。
  - `git rm --cached` 停止追蹤：`test_results/` 下 1618 PDF、14 個 `.pyc`、`logs/` 6 個 log、`docs/superpowers/`（皆保留本機）。
  - 刪除 `.codex_presentation_hlml_0603/`（6.1M，含 node_modules 的簡報子專案）。
  - backlog 依主題拆成 5 個 commit（chore 清理 / SQL 管線 / 模型訓練 / 分析視覺化 / 文件）。
- **Why:** 移除版控垃圾、保護無敏感資料、建立清楚提交歷史。
- **Files touched:** 見上述 5 個 commit；`.gitignore`。
- **Commands run:** `git rm -r --cached ...`、`git add`、`git commit`、（使用者）`git push origin main`。
- **Result/verification:** 敏感資料掃描乾淨（無寫死帳密）；push 後 `origin/main == main == df511c1`，0 ahead / 0 behind。
- **Follow-ups:** 階段二、三。

### Pre-existing (before this session)
- `08763e7` feat: add NaN report, segment-clean count, sort by segment_id then date
- `2bdef96` feat: add merge_gate_data main() and produce water_level_rain_gate_all.csv
- `af55c69` feat: implement merge_gate with alignment, staleness reset, segment ffill, and tests
- `898885c` feat: add merge_gate_data skeleton and test file

---

## 4. Known Issues & TODOs

- [ ] 詢問並（視意願）commit & push `CLAUDE.md` + `PROGRESS.md`。
- [ ] （可選）為 `Source_Code/` 補來源/授權 NOTICE（LTSF-Linear, Apache-2.0）。
- [x] drycut 切分參數定案 `L=3h, buffer=60min`；正式 meta 已產（2026-09-02）。
- [x] 系統定位定案＝即時預警（虛擬量測為長遠目標），delta-target 可行（2026-09-02）。
- [x] 寫「meta → 訓練 CSV」組裝腳本（2026-09-02，`build_training_csv_from_meta.py`）。
- [x] train/val/test split 框架改版（2026-09-02，`build_splits.py` + `--split_file`/`--fold`）。
- [x] `isRain` 改用 `min_since_rain`（2026-09-02，全域計算後切段）。
- [x] 修正閘門整列 merge_asof 缺陷（2026-09-02，改逐欄；NaN 79.8%→7.1%）。
- [x] 產出舊法對照組訓練 CSV `dataset/train_old.csv`（2026-09-02）。
- [x] fold 數定為 **3**（2026-09-02）。
- [ ] **實驗矩陣 36 runs**：因子驅動腳本 + anchored/non-anchored 合併總表。
- [ ] 用實驗矩陣取得 drycut vs 舊法的對照結果。
- [ ] 產 buf=60 正式版檢視圖並抽查（`visualize_drycut_segments.py --meta ...buf60.csv`）。
- [ ] **模型改進 roadmap（依序，完整版見 `docs/model_roadmap.md`）：**
  - [ ] **delta-target**（下一步）：target=`HL01_future − x_last`，推論加回；架構不動，與 DLinear 合併。
  - [ ] exog ablation：拿掉 exog 看 corr/MSE 變化，確認 exog 路徑是否有效。
  - [ ] exog/GRU horizon-aware：現況單一 [B,16] context broadcast 給全 15 horizon，無法表達雨延遲；改逐 horizon（cross-attention：GRU 逐步輸出 + 每 horizon 可學 query）。
  - [ ] branch-NLinear（偏離變體）：branch 減自己最後值、不加回 → 全偏離空間，配 delta-target。
  - [ ] 加乾段(dry windows)訓練：現只用降雨事件視窗 → 乾段 OOD 亂動；加平衡子集教「無驅動→不動」。
  - [ ] （可選）rain 與 gate 分開 encode。
- [ ] 資料切分（meeting_recap L72）：每 segment 隨機 split、整段不拆。**暫緩**——若 delta/正規化消掉 level 軸分布差異則可不做；要做須在 **127 個 union 合併單位**上分（±60min 緩衝致 32 對 segment 重疊、1642 共用 row → 防 leakage），test 為永久 hold-out 須挑代表性 seed。
- [ ] requirements.txt 與本機 venv 版本確認一致（venv 為 Python 3.14.3）。

---

## 5. Environment & Setup

- **Language/runtime:** Python 3.14.3（本機 `.venv/`；另有 `.HLML_Linear_venv/`）。
- **Install deps:** `pip install -r requirements.txt`（含 `torch==2.10.0`, `torchvision`, `pandas==3.0.1`, `numpy==2.4.3`, `scikit-learn`, `SQLAlchemy==2.0.49`, `pyodbc==5.3.0`, `matplotlib`, `tqdm`）。
- **Train (example):**
  ```bash
  python run.py --model DLinearMix2 --data custom \
    --data_path water_level_rain_gate_all.csv \
    --segment_col segment_id --features S --target HL01 \
    --input_col 'HL*' \
    --exog_col 'isRain,Past10Min,Past1Hr,Now,*gate_opening*' \
    --seq_len 96 --pred_len 15 --label_len 30 \
    --batch_size 64 --train_epochs 80 --patience 15 \
    --learning_rate 1e-3 --dropout 0.1 --early_stop_metric corr
  ```
- **Sweeps:** `./run_dlinearmix2_sweep.sh`（base）、`_criterion.sh`、`_noHL01.sh`（在 Apple Silicon MPS 上跑，單回合約 7 分鐘）。
- **Tests:** `pytest tests/test_merge_gate_data.py`（無 pytest 設定檔，直接指定路徑）。
- **DB:** SQL Server；連線用 Windows 整合驗證或 `DB_PASSWORD` 環境變數（亦可從 Docker 容器 `MSSQL_SA_PASSWORD` 讀取）。**勿寫死帳密。**
- **Gotchas:**
  - 大型輸出目錄已 gitignore，產出檔不要 `git add -f`。
  - macOS 會產生 `.DS_Store`（已 ignore）。
  - 資料切分需確保同一 segment 的 windows 不跨 split（見 TODO）。
