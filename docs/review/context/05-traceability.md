# 05 — 意圖 ↔ 實作對照表

> **這是本次審查的核心。** 每一條是「我宣稱要做的事」，配上「程式實際做的地方」。
> 審查者的工作是逐條比對，判定**相符 / 不符 / 無法判定**。
>
> **出處分兩級：** `[有出處]` = 可指到 `PROGRESS.md` / `meeting_recap.txt` / commit 的明確記載；
> `[推斷]` = 我從程式碼或上下文推斷的意圖，**可能是錯的，請一併質疑**。
>
> `file:line` 為 2026-09-10 實際讀取程式碼所得。程式改動後會漂移。

---

## A. 建模原則

### T-01 ⚠️ 不放 HL01 自身歷史當 input
- **為什麼：** 試過會造成自迴歸過度依賴；用 HL01 最後值當外部錨則可以。`[有出處]` `PROGRESS.md` 第 1 節 2026-06-30 決策列
- **實作位置：** **沒有任何強制機制。** `run.py:44-66` `_expand_col_patterns` 對 CSV 欄名做 glob 展開；`run.py:100-116` `_expand_col_args` 套用之。`data_provider/Data_Loader.py:247-250` 建 `x_cols = input_cols + exog_cols`。唯一的相關驗證是 `run.py:169-174`（input_col 與 exog_col 不可重疊）——**不含「input_col 不可包含 target」**。
- **我聲稱的驗證：** 無。
- **請審查者確認：**
  1. `--input_col 'HL*'` 是否確實展開成 `HL01,HL02,...,HL06`（含目標站）？
  2. `run_dlinearmix2_sweep.sh:25` 用的正是 `--input_col 'HL*'` 搭配 `--target HL01`；
     而 `run_dlinearmix2_sweep_noHL01.sh:25` 與 `_criterion.sh:28` 用明列的 `HL02..HL06`。
     「noHL01」這個檔名是否反證了 base sweep **確實**把 HL01 餵進去了？
  3. `PROGRESS.md` 第 5 節的標準訓練範例也用 `'HL*'`——照抄的人會直接違反本原則。
  4. 這是 bug、還是刻意保留的實驗變體？若是後者，為何沒有記載？
- **⚠️ 不要與 anchored 混淆：** anchored / persistence 需要的「最後一個 input 時刻的 HL01 值」
  走的是 **target/label 路徑**（`batch_y`），不是 `input_col`。把 HL01 排除在 `input_col`
  之外**不會**讓 anchored 失效。詳見 `02-method-eval.md` §3.3。
  本條要問的是另一件事：HL01 的**整條歷史**是否被當成 branch 通道餵進模型。

### T-02 主指標用 correlation，不用絕對誤差
- **為什麼：** level 可事後校正，變化形式才是要學的。`[有出處]` 同上決策列
- **實作位置：** `exp/exp_Main2.py:184-221`（`vali` 回傳 mse/mae/corr）、`exp/exp_Main2.py:223-237`（`_select_score` 把 corr 取負以統一成越小越好）、`run.py:--early_stop_metric`
- **請審查者確認：** corr 被定義為「每個 horizon 跨 window 的 Pearson，再對 horizon 取平均」。這是否真的衡量了「變化形式」？在有固定 level 偏移的情況下，跨 window 的 pooled corr 會不會主要反映 window 之間的水位差異而非段內動態？

### T-03 系統定位＝即時預警，故 delta-target 可行
- **為什麼：** 推論時有 HL01 當下讀數可當錨。`[有出處]` `meeting_recap.txt` 5/20、`docs/model_roadmap.md` §2
- **實作位置：** **尚未實作**（roadmap #1）。
- **請審查者確認：** 這個定位是否在論文中站得住腳？「用目標站當下值當錨」是否會讓貢獻退化成「預測殘差」而削弱新穎性？

## B. 資料取得與閘門處理

### T-04 不在程式中寫死任何 DB 帳密
- **為什麼：** 安全。`[有出處]` `CLAUDE.md`
- **實作位置：** `Data_From_SQL_4.py` 的 `SQLServerClient`（Windows 整合驗證或 `os.getenv("DB_PASSWORD")`）
- **請審查者確認：** 全 repo grep 是否真的沒有明文密碼、連線字串、伺服器位址外洩？

### T-05 閘門合併必須逐欄，不可整列
- **為什麼：** 原始寬表 39% 的列是部分回報，整列 merge_asof 會丟掉其他欄位的歷史值。`[有出處]` `PROGRESS.md` 2026-09-02T18:20
- **實作位置：** `Data_From_SQL_all.py:128-150`（新資料）、`rebuild_gate_columns.py:39-54`（就地重建）
- **我聲稱的驗證：** 任一欄 NaN 79.84% → 7.06%；本次實測 7.06% ✔
- **請審查者確認：** 兩處實作是否**行為完全一致**（staleness、clip、NaN 處理）？若不一致，用哪一支產的資料會不同。

### T-06 閘門 staleness 上限 5 分鐘
- **為什麼：** 過舊的觀測不應視為當前狀態。`[有出處]` `meeting_recap.txt` 4/15
- **實作位置：** `Data_From_SQL_all.py:143-146`、`rebuild_gate_columns.py:47-49`
- **請審查者確認：** 5 分鐘的依據是什麼？閘門回報頻率是否支持這個門檻？

### T-07 閘門負值視為全關
- **為什麼：** 負開度無物理意義。`[推斷]`（程式註解指向 `merge_gate_data.py:90-96`）
- **實作位置：** `rebuild_gate_columns.py:53`、`Data_From_SQL_all.py:150`、`build_training_csv_from_meta.py:106`
- **請審查者確認：** 負值出現的頻率與成因是否被調查過？直接 clip 會不會掩蓋感測器問題？

### T-08 寬表刻意不做 gate ffill，延後給下游
- **為什麼：** 產寬表時還沒有 segment 可分組。`[有出處]` `PROGRESS.md` 2026-09-02T16:05
- **實作位置：** 缺 ffill 在 `Data_From_SQL_all.py`；補做在 `build_training_csv_from_meta.py:99-108`
- **請審查者確認：** 任何**其他**從 `all_minute_wide.csv` 切段的程式是否都記得補做？漏做會讓絕大多數 window 因 NaN 被丟棄且不易察覺。

## C. 切分（drycut）

### T-09 用「剔除確定乾段」而非「找出降雨事件」
- **為什麼：** 從沒下雨的部分出發。`[有出處]` `meeting_recap.txt` 7/2
- **實作位置：** `build_drycut_segments_meta.py:39-57`
- **請審查者確認：** run-length 的計算（`dry != dry.shift()).cumsum()`）是否正確處理了資料頭尾？

### T-10 NaN 不算「確定沒雨」
- **為什麼：** 缺測不等於無雨。`[有出處]` 程式註解 `build_drycut_segments_meta.py:40`
- **實作位置：** `build_drycut_segments_meta.py:40` `dry = df[rain_col].eq(0)`（NaN → False）
- **請審查者確認：** 降雨欄實測 NaN 為 0.00%，所以此保護目前沒有實際作用——但邏輯是對的。確認無誤即可。

### T-11 `L >= 2*buffer` 保證相鄰 window 不重疊
- **為什麼：** 結構性避免 leakage，舊法沒有這個保證。`[有出處]` `PROGRESS.md` Snapshot handoff
- **實作位置：** buffer 外擴在 `build_drycut_segments_meta.py:106-107`；重疊檢查在 `build_training_csv_from_meta.py:110-134`
- **✅ 已確認（2026-09-13）：** 約束**有**被強制——`build_drycut_segments_meta.py:91-92` 在算出 `l_minutes` 與 `buf` 後立刻 `raise ValueError`。下游 `build_training_csv_from_meta.py:110-134` 的 `check_no_overlap` 為第二道防線。
- **請審查者確認：** 兩道防線的條件是否等價？`--allow-overlap` 繞過第二道時，第一道是否仍然有效？

### T-12 buffer 用來保留退水尾巴並救活短段
- **為什麼：** buf=0 僅 72 段可生 window、buf=30 僅 98 段、buf=60 全部 179 段可用。`[有出處]` `PROGRESS.md` 第 1 節 2026-09-02 決策列
- **實作位置：** `build_drycut_segments_meta.py:106-107`
- **請審查者確認：** buffer=60 讓每段兩端各含 60 分鐘的乾期。5/20 meeting 曾擔心「後置 60min 同時混雜上升動態、衰退動態、平靜狀態，模型難以掌握」。這個疑慮在 buf=60 定案後**重新成立**卻未被處理。

### T-13 可用段門檻 = `seq_len + pred_len`
- **為什麼：** 不是固定值；舊紀錄誤用 75min。`[有出處]` `PROGRESS.md` Snapshot handoff
- **實作位置：** `data_provider/Data_Loader.py:456`（`need = self.seq_len + self.pred_len`）、`build_splits.py:65`
- **請審查者確認：** 兩處的 `need` 定義是否一致。

## D. 衍生欄位

### T-14 `min_since_rain` 必須在切段前全域計算
- **為什麼：** 切段後才算會讓每段從 0 重數，抹掉段首「剛下過雨」的狀態。`[有出處]` `PROGRESS.md` 第 1 節 2026-09-02 決策列
- **實作位置：** `build_training_csv_from_meta.py:53-73`（`add_min_since_rain`），呼叫順序需確認在 `slice_segments`（75-97）之前
- **我聲稱的驗證：** drycut 版 max = 24,240 分鐘（16.8 天）→ 確實跨越了段界，代表是全域計算。
- **請審查者確認：** (1) 呼叫順序確實是「先算後切」嗎？(2) 16.8 天的值合理嗎？

### T-15 `min_since_rain` 取代 `isRain` 當 exog
- **為什麼：** `isRain` 實為核心/buffer 標記，會把**切分結構**洩漏給模型；`min_since_rain` 直接描述退水階段、有物理意義。`[有出處]` `PROGRESS.md` 第 1 節 2026-09-02 決策列
- **實作位置：** 欄位產生於 `build_training_csv_from_meta.py:53-73`；`build_splits.py:239` 的預設 exog 已改為 `min_since_rain,...`
- **請審查者確認：** ⚠️ **`run_dlinearmix2_sweep*.sh` 三支的 `--exog_col` 都還是 `isRain,...`**，與 `build_splits.py` 的預設不一致。這代表 (a) sweep 腳本尚未更新，且 (b) **算 window 數時用的 NaN-check 欄位與訓練時實際用的欄位不同** → window 數可能對不上（`min_since_rain` 有 60-70 列 NaN，`isRain` 沒有）。

### T-16 `isRain` 保留為實驗矩陣的對照組
- **為什麼：** 要用實驗證明 `min_since_rain` 較好，而非直接拿掉。`[有出處]` 同上
- **實作位置：** 訓練 CSV 同時保留 `isRain` 與 `min_since_rain` 兩欄（實測 30 欄含兩者）
- **請審查者確認：** 若 `isRain` 真的洩漏切分結構，把它當對照組跑出來的結果該怎麼詮釋？

## E. Split 與 leakage

### T-17 split 決策抽離到 loader 之外
- **為什麼：** 換切法不需動訓練碼，且可版本化保存每次實驗用的切分。`[有出處]` `PROGRESS.md` 第 1 節 2026-09-02 決策列
- **實作位置：** `build_splits.py` 產檔；`data_provider/Data_Loader.py:54-99` `_load_split_file` 消費；`run.py:--split_file/--fold`
- **請審查者確認：** loader 在未給 split_file 時的回退路徑是否仍存在？回退是否會安靜發生？

### T-18 split 邊界依「可用 window 數」而非段數
- **為什麼：** 段長差距 130~2040 分鐘，按段數切會使實際樣本比例嚴重偏離。`[有出處]` 同上
- **實作位置：** `build_splits.py:58-78`（`count_windows_per_segment`）、`build_splits.py:128-145`（`pick_boundary`）
- **請審查者確認：** `count_windows_per_segment` 的 NaN-aware 計數是否**逐行對齊** `Data_Loader` 的 `_drop_nan_windows`（`data_provider/Data_Loader.py:457-472`）？兩者用的 `check_cols` 是否相同？（見 T-15 的疑慮）

### T-19 segment 不拆、不跨 split
- **為什麼：** 同一段的 window 高度相關，散落在不同 split 即為 leakage。`[有出處]` `meeting_recap.txt` 4/15
- **實作位置：** `build_splits.py:147-168`（`assign_split` 以 segment 為單位）
- **請審查者確認：** 有沒有任何路徑會讓同一 `segment_id` 出現在兩個 split？

### T-20 重疊的 segment 必須留在同一 partition
- **為什麼：** 重疊代表同一批分鐘同時屬於兩段；分到不同 split 就是 leakage。`[有出處]` `build_splits.py:80-89` docstring
- **實作位置：** `build_splits.py:80-108`（`find_overlap_groups` 連通分量）、`build_splits.py:110-126`（`blocked_boundaries`）、`build_splits.py:139-142`（被 blocked 時的退回邏輯）
- **✅ 已實測（2026-09-13）：** `rain_segments_meta.csv` 有 **24 組重疊群組、涵蓋 56 段**；對照 `splits_train_old.csv` 的 `split` 與 `fold_1..3`，**跨 partition 的群組 = 0**，fallback 未被觸發。
- **請審查者確認：** `build_splits.py:134-140` 的 silent fallback（`allowed` 全空時退回含被禁刀口的完整候選）在什麼參數組合下會被觸發？是否該改成警告或 raise？

### T-21 test 在所有 fold 間固定不變
- **為什麼：** final hold-out 必須穩定，否則 fold 之間不可比。`[有出處]` `data_provider/Data_Loader.py:56-60` docstring
- **實作位置：** `data_provider/Data_Loader.py:83-86`
- **我聲稱的驗證：** `PROGRESS.md` 稱 test 在所有 fold 固定為 5,592 windows。
- **請審查者確認：** 驗算之。

### T-22 rolling-origin 用 expanding window，不是 sliding
- **為什麼：** 時序因果，train 只能往前擴張。`[有出處]` `PROGRESS.md` 2026-09-02T17:10
- **實作位置：** `build_splits.py:170-208`（`assign_folds`）
- **請審查者確認：** fold_k 的 train 是否確實**包含**前一 fold 的 val 期間？有無時間倒錯？

### T-23 fold 數 = 3
- **為什麼：** k=3 的 val 大小最平均且每折含 28–38 個降雨事件；k=4 有一折僅 15 段、k=6 有兩折僅 9 段。有效樣本單位是**事件**而非 window。`[有出處]` `PROGRESS.md` 第 1 節 2026-09-02 決策列
- **實作位置：** `build_splits.py:244` 預設 `--n-folds 3`
- **請審查者確認：** 「有效樣本單位是事件」這個論點是否成立？3 折對論文而言是否足夠？

### T-24 `--split_mode` 必填，不得靜默回退
- **為什麼：** 忘了帶而用到另一種切法，會讓整批實驗不可比又難察覺。`[有出處]` commit `a475fce`、`run.py:69-75` docstring
- **實作位置：** `run.py:69-98`（`_validate_split_args`）
- **請審查者確認：** 所有進入點都會經過這個驗證嗎？（例如直接實例化 `Dataset_Custom` 的程式碼路徑）

### T-25 隨機 per-segment split 暫緩
- **為什麼：** 違反 forecasting 的時序因果；若 delta/正規化消掉 level 軸分布差異則可不做。`[有出處]` `PROGRESS.md` 第 4 節、`meeting_recap.txt` 4/15 與 6/22
- **實作位置：** 未實作。
- **請審查者確認：** meeting 兩次提出、兩次擱置。「train/val/test 分布不同導致 scaler bias」這個原始問題是否已被其他手段解決？若沒有，暫緩是否合理？

## F. 訓練與評估

### T-26 window 內任一 NaN 即丟棄該 window
- **為什麼：** 模型不應把 `fillna(0)` 的假值當成輸入或目標。`[有出處]` `data_provider/Data_Loader.py:436-439` 註解
- **實作位置：** `data_provider/Data_Loader.py:441-452`（`bad_rows`）、`457-472`（`_drop_nan_windows`）
- **請審查者確認：** `nan_check_cols` 是否涵蓋所有真正會進模型的欄位（含 target）？前綴和的邊界（`csum[need:] - csum[:-need]`）是否正確？

### T-27 window 不跨 segment
- **為什麼：** 跨段的 window 會混合不相干的時間。`[推斷]`（程式行為）
- **實作位置：** `data_provider/Data_Loader.py:474-489`（在同一 `segment_id` 的連續區塊內列舉起點）
- **請審查者確認：** 若同一 `segment_id` 在檔案中**不連續**出現（例如排序被打亂），這個「連續區塊」邏輯會把它切成兩段處理。資料是否保證按時間排序？

### T-28 scaler 只 fit 在 train
- **為什麼：** 避免 val/test 資訊洩漏。`[推斷]`（標準做法）
- **實作位置：** `data_provider/Data_Loader.py:307-311`
- **請審查者確認：** 是否確實只用 train slice fit？`_drop_constant_columns`（`run.py:119-156`）另外用 70% 的規則自行切了一次 train——**這個規則與實際 split 檔可能不一致**，會不會用到 val/test 的資料來判斷常數欄？

### T-29 常數欄自動丟棄
- **為什麼：** std=0 的欄位經 StandardScaler 後變成常數 0，浪費模型容量。`[推斷]`（程式 docstring）
- **實作位置：** `run.py:119-156`
- **請審查者確認：** 見 T-28。另外，被丟掉的欄位有記錄下來供事後追溯嗎？

### T-30 Persistence baseline 必須一起報
- **為什麼：** 沒有 baseline 的水位預測沒有意義。`[有出處]` `meeting_recap.txt` 5/20
- **實作位置：** `exp/exp_Main2.py:617-694`
- **請審查者確認：** persistence 的定義（`t+k = t`）是否用了**未經 scaler 還原**的值？與模型指標是否在同一尺度上比較？

### T-31 雙 checkpoint：同時存 best-by-MSE 與 best-by-Corr
- **為什麼：** MSE 與 Corr 有 trade-off。`[有出處]` `meeting_recap.txt` 5/20
- **實作位置：** `exp/exp_Main2.py:239-316`（`train`）
- **請審查者確認：** 兩個 checkpoint 是否都有被後續評估使用？還是實際上只用了其中一個？

### T-32 Huber loss 作為對極端事件 robust 的選項
- **為什麼：** 降雨強度不均。`[有出處]` `meeting_recap.txt` 5/20
- **實作位置：** `exp/exp_Main2.py:132-140`
- **請審查者確認：** `huber_beta` 預設 1.0 是在**標準化後**的尺度上。這個 beta 對水位資料而言是否有意義？

### T-33 模型通道順序契約
- **為什麼：** 模型靠位置切分 branch 與 exog。`[有出處]` `models/DLinearMix2.py:303-312` 註解
- **實作位置：** 契約由 `data_provider/Data_Loader.py:247-250` 建立；模型端只驗**數量**（`models/DLinearMix2.py:316-323`）
- **請審查者確認：** 順序錯但數量對的情況不會被抓到。是否值得加一個欄名層級的驗證？
