# CLAUDE.md — Persistent Project Instructions / 常駐專案指令

Persistent instructions for Claude Code in this project.
本檔為 Claude Code 在本專案的常駐指令。

**Priority / 優先級:** the user's explicit in-the-moment instructions first, then this file, then system defaults.
最高優先為使用者當下的明確指示，其次本檔，再次系統預設。

---

## Core Duty: Auto-maintain PROGRESS.md / 核心職責：自動維護 PROGRESS.md

Maintain `./PROGRESS.md` **while** developing, as the single source of truth across sessions. Assume the reader has zero memory of this project and must be able to cold-start from PROGRESS.md alone.

開發的「同時」維護專案根目錄的 `./PROGRESS.md`，作為跨 session 的單一事實來源。假設讀者對本專案毫無記憶，必須能僅靠 PROGRESS.md 冷啟動接手。

### When to update / 觸發時機

Update if ANY applies; when unsure, lean towards recording.
符合任一就更新；不確定時傾向於記。

- **On explicit user request.** / 使用者**明確要求**時。
- **Major changes / 重大變動:** create/delete/heavily modify files, add/remove dependencies, feature start/finish, architecture decisions, bug found or fixed, test-status change, config/env change, direction or scope change, a blocker appears.
  建立/刪除/大幅修改檔案、加減依賴、功能起訖、架構決策、發現或修復 bug、測試狀態變化、設定/環境變動、方向或範圍改變、出現 blocker。
- **Natural intervals / 自然間隔:** end of each session, roughly every 5–10 meaningful actions, before & after any destructive operation.
  每個 session 結束、約每 5–10 個有意義動作後、任何破壞性操作前後。

### Writing requirements / 寫作要求

- Timestamps in **ISO 8601** (e.g. `2026-06-30T14:05:00+08:00`).
  時間一律用 **ISO 8601**。
- Be concrete: real file paths, exact commands, verbatim error messages, verification results.
  內容具體：檔案路徑、實際指令、錯誤訊息原文、驗證結果。
- **Only Section 0 (Snapshot) is rewritten each time. Section 3 (Changelog) is append-only, newest-first, never overwritten.**
  **只有 Snapshot（第 0 節）每次重寫；Changelog（第 3 節）只 append、新到舊、不覆寫。**
- After each update, tell the user in 1–2 lines what was recorded and why.
  每次更新後，在對話中用一兩行告訴使用者「記了什麼、為什麼」。

### PROGRESS.md structure (six fixed sections) / 結構（固定六節）

- **0. Snapshot (top, rewritten each time / 置頂，每次覆寫):** last updated (ISO 8601), current goal, one-line status, next steps, open questions / blockers, must-know handoff points.
  last updated、current goal、一句話 status、next steps、open questions / blockers、handoff 必知重點。
- **1. Architecture & Key Decisions:** decision, rationale, date, alternatives considered.
  決策、理由、日期、考慮過的替代方案。
- **2. File / Component Map:** responsibility of each important file/module.
  每個重要檔案/模組的職責。
- **3. Changelog (newest-first, append-only / 新到舊，append 不覆寫):** each entry has trigger (user request / auto-detected / interval checkpoint), what changed, why, files touched, commands run, result/verification, follow-ups.
  每筆含 trigger、what changed、why、files touched、commands run、result/驗證、follow-ups。
- **4. Known Issues & TODOs:** checkbox list; mark date when done.
  checkbox 清單，完成時標日期。
- **5. Environment & Setup:** how to build/run/test, env vars, versions, gotchas.
  如何 build/run/test、env vars、版本、雷點。

---

## Project at a glance / 專案速覽

- Water-level / rainfall time-series forecasting using DLinear-family linear models (LTSF-Linear).
  水位/降雨時序預測，基於 DLinear 系列線性模型（LTSF-Linear）。
- Pipeline: `Data_From_SQL_*.py` (extract) → `data_provider/` (load) → `models/` (DLinear / DLinearMix*) → `exp/exp_Main*.py` (train) → `run.py` (entry point) → `analyze_*` / `*visualize*` (analysis & plotting).
  流程：`Data_From_SQL_*.py` 取數 → `data_provider/` 載入 → `models/` → `exp/exp_Main*.py` 訓練 → `run.py` 進入點 → `analyze_*` / `*visualize*` 分析與作圖。
- DB access uses Windows integrated auth or `os.getenv("DB_PASSWORD")`. **Never hardcode any credentials in code.**
  資料庫連線用 Windows 整合驗證或 `os.getenv("DB_PASSWORD")`；**勿在程式中寫死任何帳密**。
- Large/regenerable outputs (`dataset/`, `checkpoints/`, `runs*/`, `test_results/`, `logs/`, `analysis/`) are gitignored — do not re-add them to version control.
  大型/可重生輸出已 gitignore，勿加回版控。
