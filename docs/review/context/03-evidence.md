# 03 — 證據台帳：已驗證 vs 未驗證

> **這是本次 review 最需要嚴格檢視的一節。**
> 規則：任何數字都必須附「出處」與「產生它的條件」。條件不同的數字不可互相比較。

---

## 1. ⚠️ 最重要的一句話

**現行 pipeline（drycut 切分 + 新 split 框架 + 修正後的閘門欄）至今沒有跑過任何模型實驗。**

`checkpoints/`、`runs/`、`test_results/` 裡的所有結果都產生於 2026-09-02 之前，
用的是**舊資料、舊切分、有缺陷的閘門欄**。它們**不能**當作現況的證據。

## 2. 已驗證（資料層，2026-09-10 實際執行取得）

| 宣稱 | 數值 | 驗證方式 |
|---|---|---|
| 原始寬表涵蓋範圍 | 511,679 列，2024-08-13 16:01 → 2025-08-03 23:59 | `wc -l` + `head`/`tail` |
| 閘門任一欄 NaN（分鐘層級） | **7.06%** | pandas `isna().any(axis=1).mean()` |
| 閘門逐欄 NaN | 6.77–6.81% | pandas `isna().mean()` |
| drycut 訓練檔 | 54,590 列 / **179 段** / 30 欄 | pandas `shape` + `nunique()` |
| drycut 段長 | min 130 / 中位 200 / max 2,040 列 | `groupby(segment_id).size()` |
| drycut 閘門 NaN（ffill 後） | **3.48%** | pandas |
| drycut `isRain` 分布 | True 33,110 / False 21,480 | `value_counts()` |
| 舊法對照組 | 50,489 列 / **159 段** / 30 欄 | 同上 |
| 舊法 `isRain` True | 31,409（= `water_level_all2.csv` 列數，交叉驗證一致） | 同上 |
| `min_since_rain` 範圍 | 0 ~ 24,240 分鐘，NaN 60 列（drycut） | pandas |

## 3. 已驗證（前一個 session 宣稱，本次未重驗）

出處為 `PROGRESS.md` changelog，**屬 Tier 2，需審查者自行驗算**：

| 宣稱 | 數值 | 出處 |
|---|---|---|
| 閘門修正前後一致性 | 兩邊皆有值的 949,041 個儲存格，數值 100% 相同 | `PROGRESS.md` 2026-09-02T18:20 |
| 閘門 NaN 改善 | 79.84% → 7.06% | 同上 |
| drycut 可用 windows | 33,381（seq_len=96 / pred_len=15） | 同上 |
| 舊法可用 windows | 31,440 | 同上 |
| 3-fold val windows | 4,294 / 4,641 / 4,627（±4%） | 同上 |
| 3-fold val 段數 | 38 / 32 / 28 | 同上 |
| loader 實際 window 數 == build_splits 計算值 | split 23,151/4,043/5,592 | `PROGRESS.md` 2026-09-02T17:10 |
| 舊法 window 重疊 | 32 對，約 1,610 分鐘 | `PROGRESS.md` 2026-09-02T18:20 |

## 4. ⚠️ 舊條件下的模型結果（不可當現況證據）

**產生條件：2026-06-30，舊資料、舊切分、閘門欄有缺陷、目標站 HL01。**

| 指標 | raw 模型 | anchored（平移到 input 最後值） | persistence |
|---|---|---|---|
| MSE | 20,884 | **1,537** | 2,430 |
| Corr | 0.819 | **0.988** | — |

結論（同樣受限於上述條件）：模型的**形狀**學得很好（Corr 0.988），但**絕對水位**錯得離譜
（raw MSE 是 anchored 的 13.6 倍）。根因是 HL01 不在 input，沒有 level 錨，
每個 window 產生固定偏移，連 persistence 都贏它。乾段更糟：anchored 砍掉約 96% 誤差後
**仍然輸給 persistence**（模型在乾段亂動）。

**✅ 污染檢查（2026-09-13）：** 該 run 的 `run_args.json` 顯示
`input_col = HL02,HL03,HL04,HL05,HL06`——**HL01 不在 input**，符合原則 1，數字未受 OI-01 影響。
run 訓練於 2026-05-19 17:56，anchored 計算於 2026-06-03 23:09，test windows 9,412，
`anchor_check_ok: true`，persistence 交叉驗證 diff 1.14e-01。
全 repo 116 個 run 中有 13 個的 `input_col` 含 HL01（10 個為 2026-05-18 的 base sweep、
3 個為 2026-09-02/09-10 的煙霧測試），**但都不是這次 anchored 分析所用的 run**。

> ⚠️ 這組數字是整個研究方向（delta-target、branch-NLinear、乾段訓練）的**唯一實證依據**。
> 若它在新 pipeline 下不成立，roadmap 的優先順序需要重排。
> **請審查者評估：把整條 roadmap 建立在一組未在現行條件下複現的數字上，風險有多大。**

## 5. 未驗證 / 尚無證據

| 項目 | 狀態 |
|---|---|
| 現行 pipeline 的任何模型指標 | **完全沒有** |
| drycut vs 舊法哪個好 | 未知（36-run 矩陣的主要目的之一） |
| exog 路徑是否真的有效 | 未知（需 ablation，roadmap #2） |
| `min_since_rain` 是否優於 `isRain` | 未知（36-run 矩陣因子之一） |
| Huber 是否優於 MSE | 未知（36-run 矩陣因子之一） |
| delta-target 是否有效 | 未實作 |
| 閘門欄修正對模型指標的影響 | 未知（只驗證了資料層的 NaN 改善） |
| 2026-06-30 那組 anchored 數字在現行 pipeline 下是否複現 | **未驗證** |

## 6. 給審查者的提示

Block M 因此分成兩半：

- **可以審的**：評估協議是否嚴謹、36-run 因子矩陣切得對不對、baseline 與 ablation 規劃
  是否完整、防 leakage 論證鏈是否成立、這條路徑是否通向可投稿論文。
- **審不了的**：任何關於「模型表現如何」的結論——因為現行條件下還沒有結果。
  請不要用第 4 節的舊數字去評價現況。
