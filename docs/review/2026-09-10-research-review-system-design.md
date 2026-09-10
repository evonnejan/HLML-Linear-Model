# Spec — 可重複執行的研究審查機制 / Research Review System

- **狀態:** 設計定案（2026-09-10T18:46:15+08:00），**尚未實作**
- **提出者:** 使用者
- **審查者平台:** **Codex**（非 Claude Code，故核心必須為平台無關的純 Markdown）
- **相關文件:** `CLAUDE.md`、`PROGRESS.md`、`docs/model_roadmap.md`

---

## 1. 問題陳述

本專案的研究終點是**一篇可投稿的論文**（中途需能展示階段成果），判準從嚴。
使用者需要一個外部 agent 定期審查：

1. **意圖 ↔ 實作是否一致** — 「我宣稱要做的事」與「程式實際做的事」是否相符（核心命題）
2. **程式是否有 bug、有無可優化之處**
3. **方向是否正確** — 這些工作有沒有在往可投稿論文前進

現況障礙：

- `PROGRESS.md` 是 append-only 開發流水帳，重心在「做了什麼」，不在「為什麼相信這是對的」；且舊條目可能已被後續更正推翻。
- 專案無成文的成功判準（僅有「即時預警定位」與「主指標 corr」，無數字門檻）。
- 根目錄 40+ 支 `.py` 與 `dataset/` 22 個檔混雜現行、一次性、歷史遺留與外來參考碼，外部審查者無法分辨，可能誤審死碼。
- 使用者自己存疑之處（exog 是否有效、buffer=60 混合動態、split 暫緩是否正確等）散落各文件，未集中。

---

## 2. 目標與非目標

### 目標

- 產出一套**可重複執行**的審查機制：每次執行都產出格式一致、可跨次比對的報告。
- 審查脈絡（context）與審查程序（protocol）分離，前者隨專案演進更新，後者近乎不動。
- 核心為**平台無關的純 Markdown**，任何 agent（Codex / Claude / 其他）或人都能執行。
- 保留升級為多 subagent 分工審查（方案 C）的路徑，升級時不需重寫。

### 非目標

- 不做自動化評分或 CI 整合。
- 審查者**不修改任何檔案**，只產出報告。
- 本 spec 不涵蓋 roadmap 各項模型改進的實作。

---

## 3. 關鍵決策

| 決策 | 理由 | 否決的替代方案 |
|---|---|---|
| 採方案 **B**（純 Markdown 核心 + Claude Code slash command 薄封裝），保留升級至 C 的路徑 | 價值全在 context/protocol/報告格式三者，且必須平台無關（審查者是 Codex）；slash command 僅為本地便利，約 15 行 | 方案 A 純 Markdown（否決：本地重跑不便）／方案 C 多 subagent 分工（暫緩：尚未跑過一次，不知瓶頸何在，屬未驗證的複雜度） |
| 文件放 `docs/review/`，**不放** `docs/superpowers/specs/` | `docs/superpowers/` 已被 gitignore，spec 與機制文件會遺失版控 | 沿用 superpowers 慣例路徑（否決：不進版控） |
| `context/` 與 `protocol/` 各自拆成子資料夾多檔 | `01-data.md` 需寫厚而不拖垮其他章；三個 block 各自成檔，升級 C 時「一檔配一 agent」零重寫 | 單一大檔 CONTEXT.md（否決：資料節過厚、且不利升級 C） |
| **先做第一次 review，再跑 36-run 實驗矩陣** | 實驗矩陣約 4.2 小時，吃的是 `train_drycut_L3h_buf60.csv` 與 `splits_*.csv`；若 review 抓到資料譜系／欄位語意／leakage 層級問題，先跑的實驗整批作廢。review 報告即實驗矩陣的 go/no-go 依據 | 先跑實驗再 review（否決：可能白跑 4.2 小時） |
| 材料分三級（Tier 0 事實 / Tier 1 受審宣稱 / Tier 2 背景），衝突即為 finding | 直接支撐核心命題：Tier 1/2 是「我說我要做的」，Tier 0 是「我實際做的」 | 讓審查者自行判斷材料權威性（否決：會把我們的宣稱當事實，失去審查意義） |
| 審查者**只記錄不修改**；撰寫 context 時發現的不一致亦只記錄 | 修掉了審查就失去意義，也讓使用者看不到原始狀態 | 允許審查者順手修正（否決：破壞證據） |
| Block M 依證據狀態自動切換審查對象 | 第一次 review 時現行 pipeline 尚無任何模型結果，M 只能審設計與計畫；往後有結果時需審結果。機制須重複使用故不能寫死 | 固定只審結果（否決：第一次無從審起）／固定只審設計（否決：往後失效） |
| 改 `.gitignore`：`.claude/` → `.claude/*` + `!.claude/commands/` | slash command 需進版控隨 repo 走，其餘本機狀態照舊忽略 | 維持現狀（否決：觸發器不進版控）／把觸發器放別處再 symlink（否決：過度設計） |

---

## 4. 檔案結構

```
docs/review/
├── README.md                  入口：這是什麼、怎麼跑一次 review、閱讀順序
├── context/                   審查脈絡（受審宣稱，Tier 1）
│   ├── 00-overview.md         用法 + 研究問題與目標
│   ├── 01-data.md             資料：總表 / 譜系 / 欄位字典 / 前處理 / 品質問題
│   ├── 02-method-eval.md      方法與模型 + 評估協議
│   ├── 03-evidence.md         已驗證 vs 未驗證（證據台帳）
│   ├── 04-code-map.md         程式碼地圖（含 MUST-REVIEW 清單）
│   ├── 05-traceability.md     意圖 ↔ 實作對照表
│   └── 06-open-issues.md      已知弱點與待決問題
├── protocol/                  審查程序（近乎不動）
│   ├── PROTOCOL.md            主任務書：審查者設定 / 材料分級 / 前置檢查 / 報告格式 / 升級條件
│   ├── block-D-data.md        資料審查
│   ├── block-C-code.md        程式碼審查
│   └── block-M-method.md      方法與方向審查
└── reports/
    ├── TEMPLATE.md
    └── YYYY-MM-DD-rNN/
        ├── report.md
        └── evidence/          審查者跑過的抽查指令與輸出

.claude/commands/research-review.md   Claude Code 端的一鍵觸發器（約 15 行）
```

**代價（已知並接受）:** 拆檔後無法「整份貼給 agent」。緩解：`README.md` 明訂閱讀順序；必要時另產合併版單檔。

---

## 5. `context/` 內容規格

### 00-overview.md
- 這套文件是什麼、給誰看、怎麼用
- 研究問題：預測標的 HL01 未來 15 分鐘（pred_len=15）
- 系統定位：**即時預警**（虛擬水位量測為長遠目標）
- 成功判準：主指標 correlation、須打敗 persistence baseline。**目前無數字門檻——須明確標示為缺口並請審查者評估其論文層級的嚴重性**
- 明確的非目標
- 終點：可投稿論文；中途需能展示階段成果

### 01-data.md（刻意加厚）
1. **取數路徑** — SQL Server → 哪支程式 → 哪個檔
2. **資料檔案總表** — `dataset/` 22 個檔逐一列出：路徑／大小／產生它的程式與指令／時間範圍／列數欄數／狀態（現行 / 備份 / 中間產物 / 歷史遺留 / 一次性分析）／**與其他檔的具體差異**
3. **資料譜系圖** — `wra_cogate_obs_long` → `_wide` → `_wide_gate_opening` → `all_minute_wide` → `train_*` → `splits_*`
4. **欄位字典** — 每欄定義、單位、量化粒度、缺值率、**語意陷阱**：
   - `isRain` 實為「核心(1)/buffer(0)」標記，非降雨旗標；buffer=0 時恆為 1
   - `Past1Hr` 尾巴使 isRain 相對官方 `Past10Min` 膨脹約 2.4x
   - gate 欄在 `all_minute_wide.csv` 中約 73% NaN，係刻意將 within-segment ffill 延後給下游
   - `min_since_rain` 必須在切段前於全域寬表計算
5. **前處理決策與參數** — drycut L=3h/buf=60、閘門逐欄 merge_asof（5 分鐘 staleness）、段內 ffill
6. **已知資料品質問題**

> **原則：所有統計（時間範圍、列數、欄位、缺值率）以實際執行指令取得，不得轉抄 `PROGRESS.md`。**

### 02-method-eval.md
- DLinearMix2 架構；與 LTSF-Linear（`Source_Code/`, Apache-2.0）的關係與授權
- 兩條原則：不放 HL01 自身歷史當 input（會自迴歸依賴）；主指標看 correlation
- 評估協議：指標定義、persistence baseline、anchored 診斷的定義與意義
- 3-fold rolling-origin expanding-window CV（fold=3 的實測依據）
- **防 leakage 完整論證鏈**：`L >= 2*buffer` → 相鄰 window 不重疊 → segment 不跨 split

### 03-evidence.md
- **已驗證**：每個數字附出處與**當時的資料／切分／程式版本條件**
- **未驗證**：現行 pipeline（drycut + 新 split + 修好的閘門）**尚無任何模型結果**；36-run 實驗矩陣未跑
- **明確警告**：2026-06-30 那組 anchored 數字（raw MSE 20,884 / anchored 1,537 / persist 2,430 / Corr 0.819→0.988）產生於舊資料、舊切分、閘門有缺陷時期，**不可當作現況證據**

### 04-code-map.md
每支檔標一級：**現行核心（MUST-REVIEW）** / 現行輔助 / 一次性分析 / 已棄用 / 外來參考碼。
現行核心至少涵蓋：`Data_From_SQL_all.py`、`rebuild_gate_columns.py`、`build_drycut_segments_meta.py`、`build_training_csv_from_meta.py`、`build_splits.py`、`data_provider/Data_Loader.py`、`data_provider/Data_Factory.py`、`models/DLinearMix2.py`、`exp/exp_Main2.py`、`run.py`。
附 end-to-end 重現指令鏈（寬表 → meta → 訓練 CSV → split → 訓練）。

### 05-traceability.md — 意圖 ↔ 實作對照表（主體，25–35 條）
每列欄位：

| 意圖（我的要求） | 為什麼 | 實作位置（file:line） | 我聲稱的驗證 | 出處與信心度 | 請審查者確認什麼 |

- **出處與信心度**分兩級：「明確有出處」（指出 `PROGRESS.md` / `meeting_recap.txt` / commit 的哪一行）與「我從程式推斷」。
- **實作位置必須實際讀 code 取得，不得轉抄 `PROGRESS.md`。**
- 已知待查範例：`PROGRESS.md` 的訓練範例為 `--input_col 'HL*' --target HL01`，而 `run.py:70` 有萬用字元展開；字面上會把 HL01 展進 input，與「不放 HL01 自身歷史」原則衝突。須實際確認是否另有排除機制。

### 06-open-issues.md
主動揭露，至少涵蓋：buffer=60 的「後置 60min 混合動態」疑慮；exog 路徑是否有效（未做 ablation）；per-segment 隨機 split 暫緩是否正確；閘門 73% NaN 落差成因無法在無 SQL 環境驗證；split 比例受 segment 完整性約束而偏離目標值；test 落在 6–8 月颱風季導致強降雨事件比例偏高；**以及撰寫 context 過程中自行發現的意圖–實作不符**。

---

## 6. `protocol/` 內容規格

### PROTOCOL.md

**§0 審查者設定**
- 角色：嚴格的論文 reviewer ＋ 資深工程師，以「可投稿」為尺，從嚴。
- 硬性限制：
  - 可讀程式碼、可讀 `dataset/`、可執行抽查與統計指令
  - **不得執行訓練**
  - **不得修改任何檔案**（只產出報告）
- 大檔注意：`wra_cogate_obs_long.csv` 259MB、`all_minute_wide.csv` 69MB。抽查請用 `head` / `awk` / `wc` / 分塊讀取，勿整檔載入記憶體。

**§1 材料分級**

| 層級 | 材料 | 規則 |
|---|---|---|
| **Tier 0 事實** | 程式碼、`dataset/` 實際內容 | 唯一權威 |
| **Tier 1 受審宣稱** | `docs/review/context/*` | 與 Tier 0 不符 → finding |
| **Tier 2 背景** | `PROGRESS.md`、`docs/model_roadmap.md`、`meeting_recap.txt`（私人草稿）、`technical_manual.md`、`docs/rain_event_definition_comparison.md`、`docs/work_summary_*.md` | 提供脈絡與意圖；與 Tier 0/1 矛盾 → finding。注意 `PROGRESS.md` 為 append-only，舊條目可能已被後續更正推翻 |

**§2 前置檢查**
- 確認 `context/` 是否過期（比對 git HEAD 與相關檔 mtime），過期則於報告標記。
- **先讀 `reports/` 中最新一份報告**，以便產出 delta。

**§3 執行三個 block** — 見 `block-D/C/M`。

**§4 報告格式** — 見下節。

**§5 升級至方案 C 的條件與做法** — 何時該拆 subagent、怎麼拆（一個 block 檔配一個 agent）。

### block-D-data.md — 資料審查
資料譜系是否成立；context 宣稱的統計是否驗得出來；欄位語意是否與程式一致；缺值處理是否合理；有無隱性 leakage；歷史遺留檔是否被誤用於現行 pipeline。

### block-C-code.md — 程式碼審查（對 MUST-REVIEW 清單）
- **C1 意圖一致性** — 程式做的是不是 context 宣稱的事
- **C2 正確性 / bug** — 邊界條件、缺值、索引、時間對齊、靜默失敗、型別
- **C3 優化與可維護性** — 效能瓶頸、重複邏輯、耦合、可測試性

### block-M-method.md — 方法與方向審查（論文尺度）
依證據狀態自動切換：

| 證據狀態 | 審查對象 |
|---|---|
| **無模型結果**（第一次 review 即為此狀態） | 設計與計畫：評估協議是否嚴謹、36-run 因子矩陣切得對不對、baseline/ablation 規劃是否完整、這條路徑是否通向可投稿論文 |
| **有模型結果**（往後） | 上述 ＋ 結論是否被證據支撐、實驗是否真的回答了它宣稱的問題 |

---

## 7. 報告格式（`reports/TEMPLATE.md`）

**每則發現的固定 schema：**

| 欄位 | 說明 |
|---|---|
| `ID` | `D-01` / `C-07` / `M-03`，同一則跨次 review 沿用首次 ID |
| `區塊` | D（資料）/ C（程式碼）/ M（方法與方向） |
| `子類` | 如 C1 意圖一致性、C2 bug、C3 優化 |
| `嚴重度` | Blocker / Major / Minor / Nit |
| `位置` | `file:line` 或 `dataset/` 檔名 |
| `現象` | 觀察到什麼 |
| `為什麼是問題` | 對正確性／論文可投稿性的具體影響 |
| `建議` | 怎麼修 |
| `審查者信心度` | 高 / 中 / 低 |

**報告開頭必含：**
- Blocker / Major / Minor / Nit 各自數量
- **與前一份報告的 delta**：新增 / 已解決 / 仍存在
- **方向判定**：前進 / 停滯 / 偏離，附理由

---

## 8. 與 `PROGRESS.md` 的分工

- `PROGRESS.md` = **開發流水帳**，append-only changelog，回答「做過什麼」
- `docs/review/context/` = **靜態審查脈絡**，回答「現況是什麼、為什麼這樣、程式在哪」
- `CLAUDE.md` 增補一條規則：`context/` 須在每次 review 前確認時效

---

## 9. 執行順序

1. 依本 spec 產出 `docs/review/` 全部檔案（context 需實際讀 code 與跑資料統計）
2. **第一次完整 review**（審查者：Codex）
3. 清掉 review 報告中的 Blocker
4. **才啟動 36-run 實驗矩陣**（資料集 2 × exog 3 × loss 2 × fold 3，約 4.2 小時）

---

## 10. 三條作業原則

1. **只記錄、不修改。** 撰寫 context 時若發現意圖與實作不符，列入 `06-open-issues.md`，不動程式碼。審查者同樣只出報告。
2. **實作位置以實際 code 為準。** 逐支讀完現行 pipeline 才填 `file:line`。
3. **資料統計以實際執行指令為準。** 22 個檔的時間範圍／列數／欄位／缺值率一律實跑取得。
