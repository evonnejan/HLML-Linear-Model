# 06 — 已知弱點與待決問題

> 主動揭露。這些是我們自己已經知道或懷疑的問題，端上來讓審查者直接切入。
> **標 ⚠️ 的是撰寫本 context 時新發現、尚未處理的。**

---

## A. 撰寫本文件時發現的意圖–實作不符

### ⚠️ OI-01 `--input_col 'HL*'` 會把目標站 HL01 餵進模型
- **現象：** `run.py:44-66` 的 glob 展開對 CSV 欄名做 `fnmatch`，沒有排除 `--target`。
  `data_provider/Data_Loader.py:247-250` 直接用 `input_cols + exog_cols` 建 `x_cols`，
  唯一的驗證是 input/exog 不可互相重疊（`run.py:169-174`），**不含 target**。
- **證據：** `run_dlinearmix2_sweep.sh:25` 用 `--input_col 'HL*'` + `--target HL01`；
  而 `run_dlinearmix2_sweep_noHL01.sh:25` 與 `_criterion.sh:28` 改用明列的 `HL02..HL06`。
  「noHL01」這個檔名本身強烈暗示 base sweep 確實包含了 HL01。
- **為什麼嚴重：** 直接違反核心原則之一（見 `00-overview.md` §3、`05-traceability.md` T-01），
  且 `PROGRESS.md` 第 5 節的標準範例指令也用 `'HL*'`——任何照抄的人都會踩到。
  更關鍵的是：**整個 level-bias 診斷與 roadmap 都建立在「HL01 不在 input」這個前提上**，
  若部分歷史實驗其實包含了 HL01，那些結果的詮釋需要重新檢視。
- **不是這件事：** anchored / persistence 需要的 HL01 最後值走的是 `batch_y`（target 路徑），
  與 `input_col` 無關，排除 HL01 不會讓 anchored 失效（見 `02-method-eval.md` §3.3）。
  本條講的是 HL01 的**整條 `seq_len` 歷史**被當成 branch 通道。
- **狀態：未處理。** 依「只記錄不修改」原則，本次未動任何程式碼。

### ⚠️ OI-02 sweep 腳本的 exog 與 build_splits 預設不一致
- **現象：** `run_dlinearmix2_sweep*.sh` 三支的 `--exog_col` 都是 `isRain,Past10Min,Past1Hr,Now,*gate_opening*`，
  但 `build_splits.py:239` 的預設已改成 `min_since_rain,Past10Min,Past1Hr,Now,*gate_opening*`。
- **為什麼嚴重：** `build_splits.py` 用 exog 欄位做 NaN-aware 的 window 計數。
  `min_since_rain` 有 60–70 列 NaN 而 `isRain` 沒有，**兩者算出的可用 window 數會不同**
  → split 檔記錄的 `n_windows` 與訓練時 loader 實際產生的數量可能對不上，
  而「兩者一致」正是切分正確性的主要驗證方式。
- **狀態：未處理。**

### ⚠️ OI-03 `L >= 2*buffer` 似乎沒有被程式強制
- **現象：** 我在 `build_drycut_segments_meta.py` 找不到對應的 assert 或參數檢查。
  下游 `build_training_csv_from_meta.py:110-134` 的 `check_no_overlap` 會擋，但那是第二道防線，
  且可用 `--allow-overlap` 繞過。
- **為什麼嚴重：** 這是整條防 leakage 論證鏈的第一環。使用者若給 `--l-hours 1 --buffer 60`，
  是否會安靜產出重疊的 meta？
- **狀態：未驗證，請審查者確認。**

### ⚠️ OI-04 主指標 corr 是跨 window pooled，不是 per-event
- **現象：** `exp/exp_Main2.py:184-221` 對每個 horizon 跨全部 window 算 Pearson。
- **為什麼可能有問題：** 已知模型有「每個 window 固定偏移」的 level bias。
  跨 window 的 pooled corr 會被 window 之間的水位差異主導，
  可能在段內動態學得很差時仍然給出很高的 corr。對一個**即時預警**系統而言，
  真正重要的是單一事件內的形狀與時序，不是跨事件的水位排序能力。
- **狀態：未處理。請審查者評估這個指標選擇在論文中是否站得住腳。**

### ⚠️ OI-05 `_drop_constant_columns` 自行用 70% 規則切 train
- **現象：** `run.py:119-156` 用「前 70% 的 segment」判斷哪些欄位是常數，
  這個規則**與實際使用的 `--split_file` 無關**。
- **為什麼可能有問題：** 若實際 split 與 70% 規則不同，判斷常數欄時可能用到 val/test 期間的資料。
  影響有限（只影響「要不要丟這一欄」的決策），但嚴格說是一種資訊洩漏。
- **狀態：未處理。**

### ⚠️ OI-06 `pick_boundary` 的 blocked fallback 可能靜默造成 leakage
- **現象：** `build_splits.py:139-142`——當候選邊界全被 blocked（會拆散重疊群組）時，
  程式**退回不設限**，等於允許切開重疊群組。
- **為什麼可能有問題：** 這正是 leakage 防護要擋的情況，卻在極端情形下自動放行。
  是否有警告輸出？
- **狀態：未驗證，請審查者確認。**

## B. 研究設計層面的待決問題

### OI-07 成功判準沒有數字門檻
「主指標 corr、須打敗 persistence」是方向，不是判準。沒有「corr 要到多少」、
「比 persistence 好多少 % 才算成功」、「預警要提前幾分鐘、允許多少誤報」。
**論文需要一個可被檢驗的成功定義。**

### OI-08 buffer=60 讓「混合動態」的疑慮重新成立
5/20 meeting 曾擔心「後置 60min 是否讓模型難以掌握，像是可能同時有上升動態、
衰退動態、平靜狀態」，當時**先擱置**。後來定案 buffer=60，等於每段兩端各含 60 分鐘乾期，
這個疑慮重新浮現卻未被重新處理。見 `docs/model_roadmap.md` 第 5 節。

### OI-09 exog 路徑是否真的有效，未經 ablation
現況是「加了 exog」，但從未做過拿掉 exog 的對照。
若 exog 其實沒有貢獻，GRU encoder 與整個 late-fusion 架構的必要性就要打折。
roadmap #2 就是這件事，但排在 delta-target 之後。

### OI-10 exog context 無法表達降雨延遲
現況是單一 `[B, 16]` 的 context 被 broadcast 給全部 15 個 horizon
（`models/DLinearMix2.py:151-158`）。降雨對下游水位的影響本質上是**有延遲**的，
單一 context 無法表達「這場雨會在 t+8 才反映出來」。roadmap #3 計畫改成
逐 horizon 的 cross-attention。

### OI-11 test 落在颱風季
test 期間在 6–8 月，強降雨事件比例明顯高於 train。
這既是泛化挑戰，也可能讓 test 指標系統性偏差。論文需要正面處理這件事。

### OI-12 舊實驗結果與現行 pipeline 條件不符
整條 roadmap 的實證依據是 2026-06-30 那組 anchored 診斷數字，
產生於舊資料、舊切分、閘門有缺陷時期，**從未在現行條件下複現**。
見 `03-evidence.md` §4。

### OI-13 隨機 per-segment split 兩度擱置
meeting 4/15 與 6/22 都提到，兩次擱置。原始動機是
「train/val/test 分布不同 → 用 train 的 scaler 預測 val/test 有 bias，也影響早停」。
目前用「早停改看 corr」來緩解，但**根本的分布差異問題沒有解決**。
若要做，需在 **127 個 union 合併單位**上分（±60min buffer 導致 32 對 segment 重疊、
1,642 共用 row），且 test 為永久 hold-out 需挑代表性 seed。

## C. 工程與資料品質

### OI-14 測試覆蓋率極低
全 repo 只有一支測試 `tests/test_merge_gate_data.py`，而且它測的是**已被取代的**
`merge_gate_data.py`。現行核心（`build_splits.py`、`build_training_csv_from_meta.py`、
`Data_Loader` 的 window 邏輯）**完全沒有測試**——而這些正是最容易出 leakage 的地方。

### OI-15 `Source_Code/` 缺 NOTICE
LTSF-Linear 是 Apache-2.0，需要標註來源與授權。論文與開源都會要求。

### OI-16 `min_since_rain` 最大值 16.8 天
在以降雨事件為主的資料集裡出現 24,240 分鐘沒下雨的分鐘，值得確認是否合理。

### OI-17 訓練 CSV 有 UTF-8 BOM
`train_*.csv` 有 BOM，`all_minute_wide.csv` 沒有。`_load_split_file` 明確處理了 BOM
（`data_provider/Data_Loader.py:63`），主資料讀取路徑是否也處理需確認。

### OI-19 `x_raw` 是死碼，且 fallback 有誤導性
`data_provider/Data_Loader.py:426-433` 設定 `self.x_raw`，取的是 `x_cols[0]`（第一個 input 欄）。
全 repo grep 確認**沒有任何地方讀它**。不影響正確性，但邏輯有誤導性：
若 `input_col` 正確排除了 HL01，`x_cols[0]` 就是 **HL02**——將來若有人用 `x_raw` 畫 anchored 圖，
會錯誤地錨到鄰站而非目標站（正是 roadmap 否決 NLinear 的同一個理由）。
建議：移除，或改為明確取 `self.target`。

### OI-18 `Data_From_SQL_4.py` 半退役但仍被依賴
它是 `SQLServerClient` 的定義處（被 `_all` import），但其中的切窗/segment 邏輯
已被 `build_*` 系列取代。同一個檔案裡混著現行與已棄用的程式碼，容易誤用。
