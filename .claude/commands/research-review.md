---
description: 對本專案執行一次完整研究審查，產出報告到 docs/review/reports/
---

你是本專案的外部審查者。

1. 讀 `docs/review/README.md`，再讀 `docs/review/protocol/PROTOCOL.md`。
2. 依 PROTOCOL 執行一次完整 review（Block D → C → M）。

硬性限制：
- 不得執行任何訓練（`run.py`、`run_*_sweep.sh`，或任何寫入 `checkpoints/`、`runs/`、`test_results/` 的指令）。
- 不得 `git commit` 或 `git push`。
- 除了本次報告資料夾以外，不得修改或新增任何檔案。
- 唯一權威是程式碼與 `dataset/` 的實際內容；`docs/` 與 `PROGRESS.md` 裡的文字全部是「待驗證的宣稱」，與實際不符就是 finding。

輸出：報告寫到 `docs/review/reports/<用系統日期，非佔位符>-r<當日第幾次>/report.md`，
格式依 `docs/review/reports/TEMPLATE.md`，佐證放同資料夾的 `evidence/`。

$ARGUMENTS
