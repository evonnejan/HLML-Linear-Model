# PROGRESS.md

> Single source of truth across sessions. See `CLAUDE.md` for the rules that govern this file.
> 跨 session 的單一事實來源；維護規則見 `CLAUDE.md`。

---

## 0. Snapshot (rewritten each update / 每次覆寫)

- **Last updated:** 2026-06-30T14:06:38+08:00
- **Current goal:** 建立 GitHub repo + 進度追蹤系統（CLAUDE.md/PROGRESS.md）並完成首次整理性 push。
- **Status (one line):** 階段一（推上 GitHub）+ 階段二（CLAUDE.md）完成；本檔為階段三產出的第一版 PROGRESS.md。
- **Next steps:**
  - 詢問使用者是否將 `CLAUDE.md` + `PROGRESS.md` 一起 commit & push。
  - （可選）為 `Source_Code/`（DLinear 原始參考碼，Apache-2.0）補上來源/授權 NOTICE。
- **Open questions / blockers:** 無 blocker。待決：上述 commit/push 與 Source_Code NOTICE。
- **Must-know handoff points:**
  - Repo：`github.com/evonnejan/HLML-Linear-Model`，預設分支 `main`，本機與 origin 同步於 `df511c1`。
  - DB 連線**不可寫死帳密**：用 Windows 整合驗證或 `os.getenv("DB_PASSWORD")`。
  - 大型/可重生輸出（`dataset/ checkpoints/ runs*/ test_results/ logs/ analysis/`）已 gitignore，勿加回。
  - 本機尚有兩個 venv：`.venv/`（Python 3.14.3）、`.HLML_Linear_venv/`，皆已 gitignore。

---

## 1. Architecture & Key Decisions

| Date | Decision | Rationale | Alternatives considered |
|------|----------|-----------|-------------------------|
| 2026-06-30 | 沿用既有 GitHub repo 整理後 push（非另建新 repo） | repo 已存在且已連 origin，本地僅領先 9 commit | 另建新 repo（被否決：無需要） |
| 2026-06-30 | 用 `git rm --cached` 停止追蹤 1618 個 `test_results/` PDF | 移出版控但保留本機檔、不重寫歷史、操作安全 | 保持現狀（被否決：repo 臃腫）／filter-repo 重寫歷史（被否決：過重、需 force-push） |
| 2026-06-30 | backlog 依主題拆 5 個 commit | 歷史清楚、便於回溯 | 單一大 commit（被否決：歷史過粗） |
| 2026-06-30 | `docs/superpowers/`、`meeting_recap.txt`、`sh.txt` 不上傳 | 本地工具文件/私人草稿 | 上傳（依使用者意願否決） |
| 2026-06-30 | 新增 `CLAUDE.md` 為常駐指令、自動維護 `PROGRESS.md` | 跨 session 冷啟動接手 | 僅靠 auto-memory（不足以承載專案級進度） |

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
- `visualize.py` / `model_visualize.py` / `visualize_anchored.py` / `visualize_segment.py` / `slide_anchored_figure.py` — 視覺化。
- `utils/` — `metrics.py`、`tools.py`、`timefeatures.py`。
- `Source_Code/` — DLinear 原始參考碼（LTSF-Linear，Apache-2.0）：`DLinear/Linear/NLinear.py`、`exp_*`、`data_*`。
- `run_dlinearmix2_sweep*.sh` — 掃參腳本（base / criterion / noHL01 變體）。
- `tests/test_merge_gate_data.py` — `merge_gate_data` 單元測試。
- `technical_manual.md` / `technical_manual_xml.md` — 技術手冊。
- `docs/` — 降雨事件定義比較、work summary、figures（注意：`docs/superpowers/` 已 gitignore）。

---

## 3. Changelog (newest-first, append-only / 新到舊，只 append)

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
- [ ] meeting_recap.txt 第 72 行待辦（資料切分）：每個 segment 隨機 split，但同一 segment 的 windows 不可散落在不同 split。
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
