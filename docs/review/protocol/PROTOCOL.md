# PROTOCOL — 審查任務書

> 這份文件定義**怎麼做一次審查**。內容近乎不動；隨專案演進的是 `../context/`。
> 啟動 prompt 刻意精簡，因為所有規則都在這裡。

---

## §0 審查者設定

**角色：** 嚴格的論文 reviewer ＋ 資深工程師。
本專案的終點是一篇**可投稿的論文**，請以「這份工作能不能通過同儕審查」為尺，**從嚴**。

**你要回答四件事：**
1. **意圖 ↔ 實作是否一致**（核心命題）
2. 程式是否有 **bug**
3. 有沒有可**優化**之處
4. **方向**是否正確——這些工作有沒有在往可投稿論文前進

**硬性限制：**
- 可讀程式碼、可讀 `dataset/`、可執行抽查與統計指令
- **不得執行訓練**（`run.py`、`run_*_sweep.sh`，或任何寫入 `checkpoints/`、`runs/`、`test_results/` 的指令）
- **不得修改任何檔案**，唯一例外是本次報告資料夾
- **不得 `git commit` / `git push`**

**大檔注意：** `dataset/wra_cogate_obs_long.csv` 247MB、`all_minute_wide.csv` 66MB。
一律用 `head` / `sed -n` / `awk` / `wc -l`，或 pandas 的 `nrows=` / `usecols=` 分塊讀。
**絕對不要整檔載入記憶體。**

## §0.5 執行紀律

- 不要在對話中複述讀到的內容或做進度回報，**直接寫進報告**——輸出全花在報告上。
- 每個 block 開始前，先列出你將驗證的 **5–10 個具體宣稱**（寫進報告，不是只在腦中想）。
- **完成一個 block 立刻把該段寫入 `report.md`**，不要全部做完才寫；中斷時不致整批白做。
- 某項驗證若需跑訓練、需 SQL 連線、或明顯超過合理時間，標記為
  **「未驗證（原因）」**後繼續，不要卡住。
- 只審 `../context/04-code-map.md` 標為**現行核心（MUST-REVIEW）**者；
  標為一次性分析／已棄用／外來參考碼的**不要花時間**。
- 沒有問題就寫「無」。**不得為湊數而提建議。**

## §1 材料分級

| 層級 | 材料 | 規則 |
|---|---|---|
| **Tier 0 事實** | 程式碼、`dataset/` 的實際內容 | **唯一權威。** |
| **Tier 1 受審宣稱** | `../context/*` | 與 Tier 0 不符 → **finding** |
| **Tier 2 背景** | `PROGRESS.md`、`docs/model_roadmap.md`、`meeting_recap.txt`（私人草稿）、`technical_manual.md`、`docs/rain_event_definition_comparison.md`、`docs/work_summary_*.md` | 提供脈絡與意圖；與 Tier 0/1 矛盾 → **finding** |

⚠️ `PROGRESS.md` 是 **append-only** 的開發流水帳，舊條目可能已被後續更正推翻
（例如舊紀錄的「75min 可用門檻」後來被更正為 `seq_len + pred_len`）。
讀它是為了理解**意圖**，不是為了取得事實。

**核心命題：** Tier 1/2 是「他說他要做的」，Tier 0 是「他實際做的」。**兩者對不上就是發現。**

## §2 前置檢查

1. 確認 `../context/` 是否過期：比對 `git log` 最新 commit 與 `context/` 各檔的最後更新日期。
   若程式碼在 context 之後有改動，`05-traceability.md` 的 `file:line` 可能已漂移——
   **在報告開頭標記，並以實際程式碼為準**。
2. **先讀 `../reports/` 中最新一份報告**（若有），以便產出 delta。

## §3 執行三個 block

依序：**Block D（資料）→ Block C（程式碼）→ Block M（方法與方向）**。
各自的指示見：

- `block-D-data.md`
- `block-C-code.md`
- `block-M-method.md`

## §4 報告格式

**輸出路徑：** `../reports/<實際今天日期 YYYY-MM-DD>-r<NN>/report.md`
`NN` 為當日第幾次（`r01`、`r02`…），依既有資料夾推算。
**不得使用 `2026-XX-XX` 之類的佔位符**，也不得把 `rNN` 寫死——否則會覆蓋前一次報告，delta 就斷了。
佐證（你跑過的指令與輸出）放同資料夾的 `evidence/`。

格式依 `../reports/TEMPLATE.md`。每則發現的固定欄位：

| 欄位 | 說明 |
|---|---|
| `ID` | `D-01` / `C-07` / `M-03`。**同一則跨次 review 沿用首次 ID** |
| `區塊` | D / C / M |
| `子類` | 如 C1 意圖一致性、C2 bug、C3 優化 |
| `嚴重度` | Blocker / Major / Minor / Nit |
| `位置` | `file:line` 或 `dataset/` 檔名 |
| `現象` | 觀察到什麼 |
| `為什麼是問題` | 對正確性／論文可投稿性的具體影響 |
| `建議` | 怎麼修 |
| `信心度` | 高 / 中 / 低 |

**嚴重度定義（避免灌水）：**
- **Blocker** — 會使實驗結果或論文結論無效：leakage、標籤錯誤、指標算錯、意圖與實作根本不符
- **Major** — 顯著影響結論可信度，或審稿人一定會問
- **Minor** — 正確性無虞但有實質改善空間
- **Nit** — 風格、命名、可讀性

**每則 finding 必須附 `file:line` 或你實際跑過的指令。** 沒有證據的推測標「信心度：低」。

**報告開頭必含：**
- 各嚴重度數量
- **與前一份報告的 delta**：新增 / 已解決 / 仍存在
- **方向判定**：前進 / 停滯 / 偏離 ＋ 理由

## §5 升級到多 agent 分工的條件

目前是單一 agent 跑完三個 block。出現下列任一情況時，改成一個 block 配一個 subagent
（三個 block 檔已刻意寫成可獨立抽出）：

- 單次 review 因 context 長度而漏審 MUST-REVIEW 清單中的檔案
- Block C 的 findings 數量多到無法在一次 pass 內查證
- `dataset/` 增長到抽查本身就佔滿預算
- 需要對同一份程式碼做多個獨立視角的交叉驗證（例如兩個 agent 各自審 leakage，比對結論）

升級時：`PROTOCOL.md` 保持不變（共用規則），三個 block 檔各自作為一個 subagent 的任務書，
最後由主 agent 依 §4 彙整成單一 `report.md`。
