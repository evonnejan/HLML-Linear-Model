# PROGRESS.md

> Single source of truth across sessions. See `CLAUDE.md` for the rules that govern this file.
> 跨 session 的單一事實來源；維護規則見 `CLAUDE.md`。

---

## 0. Snapshot (rewritten each update / 每次覆寫)

- **Last updated:** 2026-09-13T18:02:39+08:00
- **Current goal:** 審查機制已建置完成，**下一步是執行第一次完整 review（審查者：Codex）**，清掉 Blocker 後才啟動 36-run 實驗矩陣。
- **Status (one line):** 資料前處理全部就緒且 0 leakage；**審查機制 `docs/review/` 已全部產出、六個疑點已逐一查證釐清，可交 Codex 執行第一次 review**。查證結果：OI-03 **撤回**（誤報，檢查存在於 `build_drycut_segments_meta.py:91-92`）、OI-06 **降級**（實測 0 leakage，fallback 未觸發）、OI-04 **修正**（per-segment corr 已存在，待決的是 headline 指標選哪個）、OI-01/02/05 維持。**關鍵 anchored 數字經稽核未受污染。尚未開始跑實驗矩陣。**
- **Next steps:**
  1. **第一次完整 review**（Codex）→ 產出 `docs/review/reports/YYYY-MM-DD-r01/report.md`。啟動方式見 `docs/review/README.md`「審查者：從這裡開始」。
  2. **清掉報告中的 Blocker**，特別是 OI-01（`--input_col 'HL*'` 把 HL01 餵進模型）。
  3. **才啟動實驗矩陣 36 runs**：資料集(2: drycut/舊法) × exog(3: isRain / 無 / min_since_rain) × loss(2: mse/huber) × fold(3) × seed(1)，約 4.2 小時。需新寫 (i) 因子驅動腳本 (ii) 把 anchored 指標併進同一張總表的彙整器。
  4. roadmap #1 **delta-target**，續接 #2–#6。完整版見 `docs/model_roadmap.md`。
- **Open questions / blockers:** 無 blocker。待決：成功判準**尚無數字門檻**（僅「主指標 corr、須打敗 persistence」），已列為 review 的重點質疑項；split 方式（隨機 per-segment 暫緩）；exog 路徑是否有效（待 #2 ablation）；**buffer=60 讓 5/20「後置 60min 混合動態」的疑慮重新浮現**（見 `docs/model_roadmap.md` 第 5 節）。資料來源為本機 `dataset/all_minute_wide.csv`（2026-06-02 產、2026-09-02 重建閘門欄，2024-08-13 16:01~2025-08-03 23:59，511,679 分鐘），**未重抓 SQL**；要延長範圍需回 SQL 環境重跑 `Data_From_SQL_all.py`。
- **Must-know handoff points:**
  - **審查機制（2026-09-10 定案，spec 見 `docs/review/2026-09-10-research-review-system-design.md`）**：核心命題是**意圖 ↔ 實作一致性**（「我宣稱要做的」vs「程式實際做的」），外加 bug／優化／方向三軸。材料分三級：**Tier 0 事實**（程式碼、`dataset/` 實際內容，唯一權威）、**Tier 1 受審宣稱**（`docs/review/context/*`）、**Tier 2 背景**（本檔、`docs/model_roadmap.md`、`meeting_recap.txt` 等）；Tier 1/2 與 Tier 0 不符即為 finding。審查者**只出報告、不改任何檔案、不得跑訓練**。
  - **順序已定案：先 review、後實驗矩陣。** 理由：矩陣約 4.2 小時且吃 `train_drycut_L3h_buf60.csv` / `splits_*.csv`，若 review 抓到資料譜系／欄位語意／leakage 層級問題，先跑的實驗整批作廢。review 報告即矩陣的 go/no-go 依據。
  - **⚠️ run 稽核結果（2026-09-13）**：全 repo 116 個 run 中 **13 個的 `input_col` 含 HL01**（10 個為 2026-05-18 base sweep、3 個為 09-02/09-10 煙霧測試）；**但 6/3 的 anchored 分析用的是 2026-05-19 的 run，`input_col = HL02..HL06`，未受污染** → `raw 20884 / anchored 1537 / persist 2430` 仍然有效。
  - **⚠️ 已證實的意圖–實作不符（OI-01）**：`run.py:44-66` 的 glob 展開**不排除 target**，`Data_Loader:247-250` 也沒有把關（唯一驗證是 input/exog 不可互相重疊，`run.py:169-174`）。`run_dlinearmix2_sweep.sh:25` 用 `--input_col 'HL*'` + `--target HL01` → **HL01 被當成 branch input**，直接違反核心原則；而 `run_dlinearmix2_sweep_noHL01.sh:25` / `_criterion.sh:28` 用明列的 `HL02..HL06`。「noHL01」這個檔名本身即為佐證。本檔第 5 節的範例指令也用 `'HL*'`，照抄即中招。**依「只記錄不修改」原則未動程式碼**，完整記載見 `docs/review/context/06-open-issues.md`。
  - **系統定位（2026-09-02 定案）**：**即時預警系統**；虛擬水位量測是更長遠目標。此定位決定 delta-target 可行（推論時有 HL01 當下值當錨）；若日後轉向虛擬量測，roadmap #1/#4 需重新設計。
  - **核心發現**：HL01 不在 input（input=HL02–06+exog）→ 無 level 錨 → 每 window 固定偏移；persistence 因此贏 raw 模型（raw MSE 20884 / anchored 1537 / persist 2430；Corr 0.819→0.988）。**注意：此組數字產生於 2026-06-30 的舊資料、舊切分、閘門有缺陷時期，不可當現況證據——現行 pipeline 至今無任何模型結果。**
  - **原則**：不放 HL01 自身歷史當 input（會自迴歸依賴）；用 HL01 最後值當外部錨可以。主指標 = **correlation**。
  - **L 與 buffer 的作用**：L 決定「多長的無雨算確定乾、要剔除」→ 控制切點與段數；buffer 決定「每段兩端往被剔除的乾段延伸多少」→ 保留退水尾巴並救活短段。約束 `L >= 2*buffer` 保證相鄰 window 不重疊（結構性避免 leakage，舊法沒有這個保證）。
  - **可用段門檻 = `seq_len + pred_len`，不是固定值**（舊紀錄誤用 75min）。seq_len=96 → 111min；`run.py` 預設 seq_len=60 → 75min。L=3h/buf=60 下 seq_len≤96 皆 179/179 可用，seq_len≥120 開始出現死段。
  - anchored 工具：`compute_anchored_mse.py`（raw/adj/persist 的 MSE/RMSE/MAE/Corr + 逐 horizon Corr）、`visualize_segment.py`、`slide_anchored_figure.py`、`visualize_anchored.py`；乾段 anchored 直接讀 `eval_dry/*/predictions.npz` 的 `persist`。
  - **交付注意**：`dataset/`、`.claude/`、`docs/superpowers/`、`meeting_recap.txt` 皆已 gitignore。審查者需要讀資料，因此**必須交本機資料夾，不能只給 GitHub repo**。
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
| 2026-09-10 | 新增 `--split_mode` 且**無預設值**，設了 `--segment_col` 就必須明講 | 選用參數漏帶會靜默回退到另一種切法，36 runs 的實驗批次會整批不可比且難察覺；改成必填後漏帶直接報錯 | 維持選用+警告（否決：警告會被 log 淹沒）／直接把 file 設為預設（否決：破壞向後相容，且仍是隱性選擇） |
| 2026-09-10 | split 邊界加入**重疊防護**（重疊 segment 強制同 partition） | 舊法 gap=30min 與 ±60min 視窗矛盾造成 32 對重疊，其中 1 對跨 split（11 分鐘 leakage）。防護讓任何 meta 都有保證，且不需改動資料本身 | 不處理（否決：控制組帶已知 leakage）／裁掉重疊分鐘（否決：會動到資料、損失 buffer）／合併成 union 單位（否決：等效但改變 segment 定義，較侵入） |
| 2026-09-02 | 閘門合併改**逐欄** merge_asof，並就地重建既有寬表 | 原始寬表 39% 的列是部分回報，整列比對會丟掉其他欄位的歷史值；資料已凍結故可在本機重建，不需回 SQL | 維持現狀（否決：白白損失 ~5% 訓練 window 且閘門特徵品質差）／當成實驗因子（否決：使用者指示先修正再實驗）／回 SQL 重抓（否決：不需要，本機資料已完整） |
| 2026-09-02 | rolling-origin fold 數定為 **3** | k=3 的 val 大小最平均（±4%）且每折含 28–38 個降雨事件；k=4/6 有折僅 15/9 段。有效樣本單位是事件而非 window | k=4（否決：一折僅 15 段）／k=5、k=6（否決：折內事件數過少，估計不穩） |
| 2026-09-02 | split 邊界依**可用 window 數**決定，且 split 決策抽離到 loader 之外（`--split_file`） | 段長差距 130~2040 分鐘，按段數切會使實際樣本比例嚴重偏離；抽離後換切法不需動訓練碼，且可版本化保存每次實驗用的切分 | 按段數切（否決：比例失真）／改寫 loader 內建邏輯（否決：每換一種切法就要動訓練碼）／隨機 per-segment split（暫緩：違反 forecasting 的時序因果） |
| 2026-09-02 | `isRain` 改用 `min_since_rain`，且在**切段前全域計算** | `isRain` 實為核心/buffer 標記，會把切分結構洩漏給模型；`min_since_rain` 直接描述退水階段、有物理意義。切段後才算會讓每段從 0 重數，抹掉段首的「剛下過雨」狀態 | 保留 isRain（否決：洩漏切分結構）／直接拿掉不補（保留為實驗矩陣的對照組） |
| 2026-09-02 | 系統定位＝**即時預警**，虛擬水位量測列為長遠目標 | 即時預警推論時有 HL01 當下值可當錨 → delta-target 可行；虛擬量測沒有 HL01，該路不通（`meeting_recap.txt` 5/20 已註明） | 直接做虛擬量測（否決：會封死 roadmap #1/#4） |
| 2026-09-02 | drycut 定 **L=3h, buffer=60min** | buf=60 是唯一讓 179 段 100% 可生訓練 window 的設定（buf=0 僅 72 段、buf=30 僅 98 段），且 34,900 win 已超過舊法 32,999；L 在 buf=60 下不敏感，選 3 保留最多段數與最細事件粒度 | buf=90（否決：多出的 1 萬 win 是純乾 padding）／L=4~6（否決：段數更少、無額外好處）／另立小碎段丟棄規則（否決：buf=60 後問題自動消失） |
| 2026-06-30 | 不放 HL01 自身歷史當 input；主指標用 correlation | 放 HL01 歷史會自迴歸過度依賴(試過)；level 可校正、變化形式才是學的重點 | 加 HL01 自迴歸（否決） |
| 2026-09-10 | 建立可重複執行的研究審查機制，採**方案 B**（平台無關的純 Markdown 核心 + Claude Code slash command 薄封裝），保留升級至多 subagent 分工（方案 C）的路徑 | 價值全在 context/protocol/報告格式三者，且審查者是 **Codex**，核心必須平台無關；slash command 僅約 15 行的本地便利 | 方案 A 純 Markdown（否決：本地重跑不便）／方案 C 多 subagent 分工（暫緩：尚未跑過一次，不知瓶頸何在，屬未驗證的複雜度） |
| 2026-09-10 | 審查文件放 `docs/review/`，且 `context/`、`protocol/` 各自拆多檔 | `docs/superpowers/` 已 gitignore，spec 會遺失版控；`01-data.md` 需寫厚而不拖垮其他章；三個 block 各自成檔，升級 C 時「一檔配一 agent」零重寫 | 沿用 `docs/superpowers/specs/`（否決：不進版控）／單一大檔 CONTEXT.md（否決：資料節過厚、不利升級 C） |
| 2026-09-10 | **先做第一次 review，再跑 36-run 實驗矩陣** | 矩陣約 4.2 小時且吃 `train_drycut_L3h_buf60.csv` / `splits_*.csv`；若 review 抓到資料譜系／欄位語意／leakage 層級問題，先跑的實驗整批作廢。review 報告即矩陣的 go/no-go 依據 | 先跑實驗再 review（否決：可能白跑 4.2 小時） |
| 2026-09-10 | 審查材料分三級（Tier 0 事實 / Tier 1 受審宣稱 / Tier 2 背景），衝突即為 finding；審查者只出報告、不改檔、不跑訓練 | 直接支撐核心命題「意圖 ↔ 實作一致性」；若讓審查者順手修改則破壞證據、看不到原始狀態 | 讓審查者自行判斷材料權威性（否決：會把我們的宣稱當事實）／允許審查者順手修正（否決：破壞證據） |

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
- `docs/review/` — **研究審查機制**（2026-09-10 定案，**機制檔案尚未實作**）。`2026-09-10-research-review-system-design.md` 為 spec；預計產出 `README.md`、`context/00~06`（受審脈絡：目標／資料／方法與評估／證據台帳／程式碼地圖／意圖-實作對照表／已知弱點）、`protocol/`（`PROTOCOL.md` + block-D 資料 / block-C 程式碼 / block-M 方法與方向）、`reports/`（每次 review 一個資料夾）。

---

## 3. Changelog (newest-first, append-only / 新到舊，只 append)

### 2026-09-13T18:02:39+08:00 — 六個疑點逐一查證：撤回 1、降級 1、修正 1，並完成 run 稽核
- **Trigger:** 使用者逐項追問六個疑點，並質疑「anchored 應該需要用到 HL01」。
- **What changed:** 只改 `docs/review/context/` 與本檔，**未動任何程式碼或資料**。
- **查證結果:**
  - **OI-03 撤回（誤報）**：`L >= 2*buffer` 的檢查**確實存在**於 `build_drycut_segments_meta.py:91-92`。誤報成因是撰寫 context 時讀了該檔 39–58 與 95–120 行，**跳過 89–94**。保留紀錄以提醒 `context/` 屬 Tier 1 宣稱、會有錯。
  - **OI-06 降級**：實測 `rain_segments_meta.csv` 有 **24 組重疊群組、涵蓋 56 段**；對照 `splits_train_old.csv` 的 `split` 與 `fold_1..3`，**跨 partition 群組 = 0** → silent fallback 未被觸發，現行切分無 leakage。改列為潛在風險（建議 `allowed` 全空時警告或 raise）。
  - **OI-04 修正**：**per-segment corr 已存在**（`exp/exp_Main2.py:696-880` 的 `_save_segment_metrics` → `metrics_segment.csv`，每段含 `Corr`，分 `horizon="all"` 與 per-horizon）。原本寫「需補 per-event corr」是錯的。真正待決的是 **headline 主指標要用跨 window pooled 還是 per-segment 彙總**（彙總方式可沿用 5/20 對 MSE 決定的四種）。
  - **OI-01 維持**，並補上意圖判讀與修法建議：`sweep.sh` vs `_noHL01.sh` 只差 `--input_col`（`HL*` vs 明列）與 `--seed`（無 vs 42）。傾向「base 的 `HL*` 是無意的」（base 註解只提 seq_len×fusion、noHL01 特別標 "WITHOUT HL01"、base 未設 seed、5/19 後全改明列），**但意圖無記載，需使用者確認**。建議修法：偵測 `target ∈ input_col ∪ exog_col` 即報錯並印出展開後欄位，以 `--allow_target_in_input` 放行（因 `runs_sanity_seg/Linear_HL01-to-HL01` 是刻意的 sanity run）。
  - **OI-02 維持**，成因確認為 `min_since_rain` 是後加欄位、sweep 只更新了 `--split_mode` 沒更新 `--exog_col`。待改：`run_dlinearmix2_sweep.sh:26`、`_noHL01.sh:26`、`_criterion.sh:29`。
  - **OI-05 維持。**
- **✅ run 稽核（重要）:** 掃全部 `run_args.json`——`HL02..HL06` 27 runs、`HL02,HL03` 25 runs、**`HL01,HL02..HL06` 13 runs**、單站 DLinear 45 runs。13 個含 HL01 者集中在 2026-05-18（base sweep）與 09-02/09-10（煙霧測試）。**6/3 的 anchored 分析用的是 2026-05-19 17:56 的 run，`input_col = HL02..HL06`、`criterion=huber`、`early_stop_metric=mse`、9,412 test windows、`anchor_check_ok: true` → 未受污染。**
- **另釐清:** anchored / persistence 的 `x_last` 來自 **`batch_y`（target 路徑）**：`Data_Loader.py:277` → `507-509` → `exp_Main2.py:547` → `compute_anchored_mse.py:40-42`，完全獨立於 `batch_x`/`input_col`。把 HL01 排除在 `--input_col` 之外**不會**讓 anchored 失效。
- **Files touched:** `docs/review/context/02-method-eval.md`、`03-evidence.md`、`05-traceability.md`、`06-open-issues.md`、`PROGRESS.md`。
- **Commands run:** 掃 `run_args.json`、讀 `anchored_metrics.json`、讀兩支 sweep 腳本、`grep` `build_drycut_segments_meta.py` 的 buffer 檢查、以 pandas 重算重疊群組並比對 `splits_train_old.csv`。**未執行訓練、未改動程式或資料。**
- **Follow-ups:** 交 Codex 執行第一次 review → 清 Blocker → 才跑 36-run 實驗矩陣。

### 2026-09-10T19:30:24+08:00 — 建置研究審查機制 `docs/review/`，並發現 6 個意圖–實作疑點
- **Trigger:** 使用者指示「先幫我把會用到的東西補齊」。
- **What changed:** 依 spec 產出審查機制全部檔案，**未修改任何程式碼或資料**。
  - `docs/review/README.md`（入口與閱讀順序）
  - `docs/review/context/00-overview.md` ~ `06-open-issues.md`（7 檔，受審宣稱 Tier 1）
  - `docs/review/protocol/PROTOCOL.md` + `block-D-data.md` / `block-C-code.md` / `block-M-method.md`
  - `docs/review/reports/TEMPLATE.md`（固定 schema，跨次可比對）
  - `.claude/commands/research-review.md`（slash command）
  - `.gitignore`：`.claude/` → `.claude/*` + `!.claude/commands/`（讓觸發器進版控，其餘本機狀態照舊忽略）
  - `CLAUDE.md`：新增「Research review context」節（context/ 的維護時機、與本檔的分工、審查者硬性限制）
- **Why:** 讓外部 agent（Codex）能對「意圖 ↔ 實作一致性 / bug / 優化 / 方向」做可重複、可跨次比對的審查。
- **作業原則（已遵守）:** ①只記錄不修改——寫 context 時發現的問題全部列入 `06-open-issues.md`，未動任何程式碼 ②`file:line` 逐支實際讀 code 取得，未轉抄本檔 ③資料統計實跑取得，未轉抄本檔。
- **Commands run:** 資料盤點（`wc -l` / `head` / `tail` / `od` 逐檔）、pandas 統計（NaN 率、段數、段長、`isRain` 與 `min_since_rain` 分布）、程式碼閱讀（`sed -n` / `grep -n`）。**未執行訓練，未改動任何程式或資料。**
- **Result/verification（本次實測，與既有宣稱一致）:**
  - `all_minute_wide.csv` 511,679 列 / 23 欄；閘門任一欄 NaN **7.06%**（逐欄 6.77–6.81%）
  - `train_drycut_L3h_buf60.csv` 54,590 列 / **179 段** / 30 欄；段長 min 130 / 中位 200 / max 2,040；閘門 NaN **3.48%**；`isRain` True 33,110 / False 21,480
  - `train_old.csv` 50,489 列 / **159 段** / 30 欄；`isRain` True **31,409**（= `water_level_all2.csv` 列數，交叉驗證通過）
  - `min_since_rain`（drycut）：0 ~ **24,240** 分鐘，NaN 60 列
- **⚠️ 發現的 6 個疑點（皆未處理，留給 review 判定）:**
  - **OI-01** `--input_col 'HL*'` 會把目標站 HL01 餵進模型（`run.py:44-66` 不排除 target；`run_dlinearmix2_sweep.sh:25` 實際使用）——直接違反核心原則
  - **OI-02** sweep 腳本的 `--exog_col` 仍是 `isRain,...`，但 `build_splits.py:239` 預設已改 `min_since_rain,...` → 算 window 數與訓練時的 NaN-check 欄位不同，window 數可能對不上
  - **OI-03** `L >= 2*buffer` 似乎沒有在 `build_drycut_segments_meta.py` 被強制
  - **OI-04** 主指標 corr 是跨 window pooled（`exp/exp_Main2.py:184-221`），在有 level bias 時可能不反映段內動態
  - **OI-05** `run.py:119-156` 用自己的 70% 規則切 train 判斷常數欄，與實際 `--split_file` 無關
  - **OI-06** `build_splits.py:139-142` 候選邊界全被 blocked 時會退回不設限，可能靜默切開重疊群組
- **Files touched:** `docs/review/`（13 檔新增）、`.claude/commands/research-review.md`(新增)、`.gitignore`、`CLAUDE.md`、`PROGRESS.md`。
- **Follow-ups:** 執行第一次 review（Codex）→ 清 Blocker → 才跑 36-run 實驗矩陣。

### 2026-09-10T18:52:00+08:00 — 定案「可重複執行的研究審查機制」spec（未實作任何機制檔案）
- **Trigger:** 使用者要求把研究主題／目標／進度／核心程式碼整理成文件，交給另一個 agent（**Codex**）審查「我有沒有做錯、有沒有在往目標前進」；討論中擴充為一套**可重複執行**的審查機制。
- **What changed:** 新增 `docs/review/2026-09-10-research-review-system-design.md`（設計 spec，238 行）。**未寫任何機制檔案、未動任何程式碼或資料。**
- **Why:** 本檔是 append-only 開發流水帳，重心在「做了什麼」而非「為什麼相信這是對的」，且舊條目可能已被後續更正推翻；專案無成文的成功判準（僅「主指標 corr、須打敗 persistence」，無數字門檻）；根目錄 40+ 支 `.py` 與 `dataset/` 22 個檔混雜現行／一次性／歷史遺留／外來參考碼，外部審查者無法分辨、可能誤審死碼；使用者自己存疑之處散落各文件未集中。
- **使用者需求（重點）:** ①「我想知道的是我想做的事情，我現行的程式是否有符合我的要求去做出來」→ 核心命題定為**意圖 ↔ 實作一致性** ②程式是否有 bug、有無可優化之處 ③ data 要完整仔細解釋、說明資料在哪、彼此有什麼不同 ④ 核心程式碼（現在與之後持續使用的）一定要 review 過 ⑤ 每次完整 review 都產出報告，且這套要**可重複使用** ⑥ 終點是**可投稿論文**，中途要能展示成果，尺度從嚴 ⑦ 審查者可讀 code 與資料、可跑抽查指令，但**不做實驗（不跑訓練）**。
- **主要設計決策:** 見第 1 節新增的四列（方案 B／`docs/review/` 拆多檔／先 review 後實驗矩陣／材料三級分層）。
- **Files touched:** `docs/review/2026-09-10-research-review-system-design.md`(新增)、`PROGRESS.md`。
- **Commands run:** 僅唯讀探索（`ls -la dataset/`、`grep -n input_col run.py data_provider/Data_Loader.py`、`cat .gitignore`、`git log/status` 等）。**未執行任何會改動程式碼或資料的指令。**
- **Result/verification:** spec 檔已寫出（238 行）。機制本身尚未實作，故無功能可驗證。
- **探索中發現的待查疑點（已寫入 spec，留給 review）:**
  - 本檔第 5 節訓練範例為 `--input_col 'HL*' --target HL01`，而 `run.py:70` 有萬用字元展開邏輯；字面上 `HL*` 會把 HL01 展進 input，**與「不放 HL01 自身歷史當 input」原則衝突**。尚未確認是否另有排除機制。
  - `dataset/` 共 22 個檔（含 259MB 的 `wra_cogate_obs_long.csv`），現行／備份／中間產物／歷史遺留混雜，無成文說明。
  - `.gitignore` 忽略了 `dataset/`、`.claude/`、`docs/superpowers/`、`meeting_recap.txt` → 交付審查必須給**本機資料夾**而非 GitHub repo；且 slash command 需改 `.gitignore` 才能進版控。
- **過程中的一次自我修正:** 首次準備更新本檔時，用的是 session 起始（HEAD `eecd2a9`）的內容；實際檔案在此期間已因今日三個 commit（`3c6a701`/`e367621`/`a475fce`）更新。已重讀後 rebase，未覆蓋新內容。
- **Follow-ups:** 依 spec 產出 `docs/review/` 全部檔案 → 第一次 review（Codex）→ 清 Blocker → 才跑 36-run 實驗矩陣。

### 2026-09-10T18:30:00+08:00 — 新增 --split_mode，split 方式改為必須明確指定
- **Trigger:** 使用者提議「改成 --split_mode，這個參數一定要給，要用新 split 就再給 --split_file」。
- **What changed:**
  - `run.py` 新增 `--split_mode {file,builtin}`，**刻意無預設值**；新增 `_validate_split_args()` 在 parse 後做交叉檢查。
  - 規則：設了 `--segment_col` 就**必須**給 `--split_mode`；`file` 必須配 `--split_file`（且檔案須存在）；`builtin` 不可配 `--split_file` 或 `--fold`；沒有 `--segment_col` 時三個參數都不可給。
  - `Data_Loader` 的內建切法訊息由 `[WARNING]` 改為 `[split builtin]`（現在是明確選擇，不再是意外回退）。
  - 三支既有 sweep 腳本補上 `--split_mode builtin` 以維持原行為。
- **Why:** 先前 `--split_file` 是選用參數，忘了帶會**靜默**回退到內建切法，訓練照跑、結果照出。實驗矩陣要跑 36 runs，漏帶會讓結果不可比且極難察覺。改成必填後，忘記帶就直接報錯。
- **Files touched:** `run.py`、`data_provider/Data_Loader.py`、`run_dlinearmix2_sweep.sh`、`run_dlinearmix2_sweep_criterion.sh`、`run_dlinearmix2_sweep_noHL01.sh`。
- **Result/verification:** 六種錯誤路徑全部正確攔截（缺 split_mode／file 缺 split_file／builtin 配 split_file／builtin 配 fold／split_file 檔案不存在／無 segment_col 卻給 split_mode）。兩條合法路徑正常：`builtin` → train 20,173；`file --fold 2` → `[split_file] (fold_2): train=104 val=32 test=15 unused=28`、train **18,413**，與 `build_splits.py` 的 fold_2 數字完全一致。
- **Follow-ups:** 實驗矩陣 36 runs；驅動腳本一律用 `--split_mode file --split_file ... --fold k`。

### 2026-09-10T16:20:00+08:00 — 未給 --split_file 時加警告；釐清「重疊 vs leakage vs 重複」三者不同
- **Trigger:** 使用者追問「不給 split_file 會不會一直用到舊的內建切法」「說有重疊為何四種組合都 0 leakage」。
- **What changed:** `data_provider/Data_Loader.py` 在走內建 70/10/rest 分支時印出警告，說明該路徑忽略 window 數且**沒有重疊防護**，並提示改用 `--split_file`。僅在 `flag='train'` 印一次。
- **Why:** `--split_file` 是選用參數，忘了帶就會靜默回退到舊切法 —— 實驗矩陣要跑 36 runs，靜默回退會讓整批結果作廢且難以察覺。
- **Files touched:** `data_provider/Data_Loader.py`。
- **Result/verification:** 不給 `--split_file` → 警告出現、走內建切法（train 22,741 win）；給了 → 無警告、`[split_file] ... train=116 val=24 test=19`（train 23,606 win）。
- **四種組合的 leakage 實測（重要澄清）:**
  | | 舊 split（loader 內建，依段數） | 新 split（build_splits.py） |
  |---|---|---|
  | 舊 segmentation（32 對重疊） | 0 對 | 防護前 **1 對/11 分鐘**；防護後 0 對 |
  | drycut（0 重疊） | 0 對 | 0 對 |
  - **舊 split 的「安全」是巧合**：它依段數切（159 → 111/15/33），切點剛好沒落在任何重疊對之間；它對重疊完全無感，換資料/比例/seq_len 都可能踩到。
  - 新 split 依 **window 數**找邊界，切點不同，剛好踩到 seg 141(val)/142(test)。防護後 seg 141 被推入 test，兩段同組。
- **三個概念必須分清（先前敘述不夠精確）:**
  1. **重疊**：`train_old.csv` 有 32 對 segment 的視窗重疊 —— 這是 **segmentation 的固有屬性**，split 消不掉。
  2. **leakage**：重疊的兩段被分到不同 partition 才算。目前四種組合皆 0。
  3. **重複**：重疊的分鐘**仍在同一 partition 內出現兩次**。實測 `train_old.csv` 有 **1,642 個分鐘重複（3,284 列，6.50%）**，`train_drycut_L3h_buf60.csv` 為 **0**。這不是 leakage，但那些分鐘在訓練中被雙倍加權，是舊法殘留的失真，**重疊防護不會修正它**（要修得改 segmentation，而 drycut 正是解法）。
- **Follow-ups:** 實驗矩陣 36 runs；驅動腳本務必每個 run 都帶 `--split_file`。

### 2026-09-10T15:40:00+08:00 — build_splits 新增重疊防護；補上閘門與 leakage 的量化細節
- **Trigger:** 使用者追問「segment 重疊該怎麼處理」「再撈 SQL 是否能拿到更完整資料」。
- **What changed:** `build_splits.py` 新增 `find_overlap_groups()` 與 `blocked_boundaries()`：偵測 window 互相重疊的 segment 連通群組，並在挑 split/fold 邊界時排除會拆散群組的位置，強制同群組留在同一 partition。`pick_boundary`/`assign_split`/`assign_folds` 皆加上 `blocked` 參數。
- **Why:** 舊法的 `RAIN_GAP_MINUTES=30`（間隔 <30min 才合併）與 `PRE/POST_WINDOW_MINUTES=60`（各延 ±60min）互相矛盾 —— 間隔 40~120 分鐘的兩場雨不會被合併，但延伸後視窗重疊。實測 32 對重疊、共 1,610 分鐘。
- **Files touched:** `build_splits.py`、`dataset/splits_train_old.csv`(重產)、`dataset/splits_train_drycut_L3h_buf60.csv`(重產，內容不變)。
- **Commands run:** `python build_splits.py --data-path dataset/train_old.csv`、`--data-path dataset/train_drycut_L3h_buf60.csv`；另以獨立腳本驗證 leakage。
- **Result/verification:**
  - drycut：偵測到 **0 個重疊群組**（L>=2*buffer 的結構保證），split 與 fold 數字完全不變。
  - 舊法：偵測到 **24 個重疊群組（共 56 段）**；套用防護後 train/val/test = 116/24/19 段、22,047/4,554/4,839 win（70.1%/14.5%/15.4%，比防護前更接近目標比例）。
  - **leakage 驗證：舊法 32 對重疊中，跨 partition 者由 1 對降為 0；fold_1/2/3 亦皆為 0。**
- **修正先前偏重的說法:** 先前稱「32 對重疊會造成 leakage」過重。實測防護前只有 **1 對**跨 split（seg 141 val ↔ seg 142 test，核心間隔 110 分鐘，共用 **11 分鐘**），其餘 31 對都落在同一 partition 內、本就無害。量級不足以扭曲結論，但防護仍值得做，因為它讓任何 meta 都有保證。
- **關於「再撈 SQL 能否更完整」的結論（本輪查證）:**
  - 逐欄合併後的殘餘 NaN 6.77%（34,657 分鐘）中，**85.5% 集中在 16 個 >=1 小時的區塊**，65.8% 只來自 2 段長斷訊：2025-03-27~04-07（11.3 天）與 2024-09-07~09-11（4.6 天）。這是**感測器真的沒有回報**，重撈不會生出資料。
  - 部分回報**不是時間戳對齊問題**：把時間戳 round 到 1s/10s/1min，7 欄全有的比例皆維持 61.2% 不變 → 裝置本來就在不同時間點回報，本機 CSV 已忠實反映來源。
  - 大斷訊多落在乾段（已被 drycut 剔除），故訓練資料的 gate NaN 僅 3.48%，遠低於 6.77%。
  - **結論：重撈 SQL 預期不會改善**，除非資料庫端事後有回補那兩段長斷訊（僅能在有 SQL 環境時查證）。
- **Follow-ups:** 實驗矩陣 36 runs。

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
- [x] 定案研究審查機制 spec（2026-09-10，`docs/review/2026-09-10-research-review-system-design.md`）。
- [x] 依 spec 產出 `docs/review/` 全部檔案（2026-09-10，13 檔）。
- [x] 改 `.gitignore` 並新增 `.claude/commands/research-review.md`（2026-09-10）。
- [x] `CLAUDE.md` 增補「Research review context」節（2026-09-10）。
- [ ] **第一次完整 review**（審查者：Codex）→ `docs/review/reports/YYYY-MM-DD-r01/report.md`。
- [ ] 清掉第一次 review 報告中的 Blocker。
- [ ] **修 OI-01：`--input_col 'HL*'` 會把 HL01 餵進模型**（已證實，`run.py:44-66` 不排除 target；`run_dlinearmix2_sweep.sh:25` 實際使用）。待 review 判定嚴重度後處理。
- [ ] 處理 OI-02（sweep 的 `--exog_col` 仍為 `isRain`，未同步 `min_since_rain`）與 OI-05。
- [x] 查證 OI-03（**撤回，誤報**）、OI-06（**降級**，實測 0 leakage）、OI-04（**修正**，per-segment corr 已存在）（2026-09-13）。
- [ ] **決定 headline 主指標**：跨 window pooled corr vs per-segment 彙總（median / worst-decile）；early stopping 是否跟著改。
- [x] 稽核全 repo run 的 `input_col`，確認 anchored 分析未受 HL01 污染（2026-09-13）。
- [ ] 補上成功判準的數字門檻（目前僅「主指標 corr、須打敗 persistence」，無門檻）。
- [ ] **實驗矩陣 36 runs（待第一次 review 的 Blocker 清完才啟動）**：因子驅動腳本 + anchored/non-anchored 合併總表。驅動腳本一律用 `--split_mode file --split_file ... --fold k`（`--split_mode` 現為必填，漏帶會直接報錯）。
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
