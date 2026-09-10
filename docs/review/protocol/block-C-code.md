# Block C — 程式碼審查

> 前置：已讀 `PROTOCOL.md` 與 `../context/04-code-map.md`、`../context/05-traceability.md`。
> 產出：`report.md` 的 Block C 段落，findings 編號 `C-NN`。
> **只審 `04-code-map.md` §1 的「現行核心（MUST-REVIEW）」清單。**

---

## C0 先列出你要驗的東西

在報告裡先寫下你打算查證的 5–10 個具體項目，再逐一執行。

## C1 意圖一致性（最高優先）

**這是整次審查的核心。** 逐條走過 `../context/05-traceability.md` 的 T-01 ~ T-33，
對每一條判定：

| 判定 | 意義 |
|---|---|
| **相符** | 程式確實做了宣稱的事 |
| **不符** | 程式做的事與宣稱不同 → **finding**，通常至少 Major |
| **無法判定** | 需要跑實驗或缺少資訊 → 標「未驗證（原因）」 |
| **宣稱本身有問題** | 意圖描述模糊或自相矛盾 → **finding** |

每條 T-NN 的「請審查者確認」欄位已寫明該查什麼。**不要只回答「相符」——附上你看的 `file:line`。**

特別注意標 ⚠️ 的條目（T-01、T-11、T-15、T-20），以及 `../context/06-open-issues.md` §A
列出的六個已知疑點（OI-01 ~ OI-06）——那些是撰寫 context 時發現但**未處理**的，
請獨立查證並判定嚴重度（不要因為對方已自承就降級）。

## C2 正確性 / bug

對 MUST-REVIEW 清單逐支審，重點：

- **邊界條件** — 空 segment、單列 segment、長度剛好等於 `seq_len + pred_len` 的段、
  資料頭尾的 run-length、`clip(lower=..., upper=...)` 的邊界
- **缺值** — NaN 傳播、`fillna(0)` 有沒有混進訓練或評估、NaN 在布林運算中的行為
- **索引** — 前綴和的 off-by-one（`data_provider/Data_Loader.py:457-472`）、
  `iloc` vs `loc`、`border1/border2` 的閉開區間
- **時間對齊** — `merge_asof` 的 direction 與 staleness、時區、排序假設
  （`data_provider/Data_Loader.py:474-489` 假設同 segment 的列連續且已排序）
- **靜默失敗** — 例外被吞掉、fallback 悄悄改變行為
  （例如 `build_splits.py:139-142` 的 blocked 退回）、`try/except: pass`
- **型別** — 布林欄位進 scaler、`int` vs `float` 的除法、`astype` 截斷
- **可重現性** — 隨機種子是否被設定與記錄

## C3 優化與可維護性

- **效能** — 明顯的 O(n²)、逐列 `iterrows()`（`build_training_csv_from_meta.py:75-97` 有一個）、
  不必要的整檔載入、重複計算
- **重複邏輯** — 同一段邏輯在多處各寫一份而可能漂移
  （例如 window 計數同時存在於 `build_splits.py:58-78` 與 `data_provider/Data_Loader.py:457-489`；
  閘門合併同時存在於 `Data_From_SQL_all.py` 與 `rebuild_gate_columns.py`）
- **耦合** — 隱性契約（模型的通道順序假設）、跨檔案的魔術字串
- **可測試性** — 現行核心幾乎沒有測試（見 OI-14）。指出**最該補測試的 3 個地方**，並說明理由

## C4 你不需要做的事

- 不要審 `04-code-map.md` §3–§5 的檔案（一次性分析、已棄用、外來參考碼）
- 不要重寫程式碼、不要提供完整 patch——**指出問題與建議方向即可**
- 不要修任何東西
