# 模型改進 Roadmap / Model Improvement Roadmap

> 定於 2026-06-30，2026-09-02 補完細節與定位。
> 摘要版見 `PROGRESS.md` 第 4 節；本檔為完整版（含理由、否決方案、限制）。

---

## 0. 起因：level bias 診斷

2026-06-30 對 HL01 模型做 anchored 診斷，數字如下：

| 指標 | raw 模型 | anchored（平移到 input 最後值） | persistence |
|---|---|---|---|
| MSE | 20,884 | **1,537** | 2,430 |
| Corr | 0.819 | **0.988** | — |

`anchored` 的定義（保留形狀、只修 level）：

```
c   = pred[0] - x_last     # 預測起點與最後觀測值的落差
adj = pred - c             # 整條平移，使 adj[0] == x_last
```

**結論：模型的形狀學得極好（Corr 0.988），但絕對水位錯得離譜（raw MSE 是 anchored 的 13.6 倍）。**

**根因：HL01（目標站）不在 input 裡**（input 是 HL02–06 + exog）。模型看不到目標站當前水位 →
沒有 level 錨 → 每個 window 產生固定偏移 → 連 persistence 都贏它。

乾段更糟：anchored 砍掉約 96% 誤差後**仍然輸給 persistence**（模型在乾段亂動）。

---

## 1. 兩條原則

1. **不放 HL01 自身歷史當 input**（試過，會自迴歸過度依賴）。
   用 HL01 **最後值**當外部錨則可以。
2. **主指標看 correlation**。level 可以事後校正，變化形式才是要學的。

---

## 2. 系統定位（2026-09-02 定案）

**當前定位：即時預警系統（real-time warning）。虛擬水位量測是更長遠的目標。**

這個定位直接決定 roadmap #1 可不可行：

| 定位 | 推論時是否有 HL01 當下值 | delta-target |
|---|---|---|
| **即時預警**（當前） | 有（就是現在的感測讀數） | ✅ 可用 |
| 虛擬水位量測（長遠） | 無（該點根本沒感測器） | ❌ 不可用 |

出處：`meeting_recap.txt` 5/20「(即時預警系統可用 / 虛擬水位量測不可用)」。

**待未來轉向虛擬量測時，#1 與 #4 需重新設計**（屆時沒有錨可用，得改走
鄰站空間關係推估的路線）。

---

## 3. 六項改進（依序）

### ① delta-target ← 下一步

訓練目標從絕對水位改成相對變化量：

```
anchor    = HL01[s_end - 1]                              # input 最後一分鐘的 HL01
target[k] = HL01[s_end - 1 + k] - anchor     k = 1..15   # 訓練時預測「變化量」
推論時      pred_abs[k] = model_out[k] + anchor          # 加回去還原
```

**為何排第一**：把 anchored 診斷從「事後補救」升級成「端到端訓練」。
現行 `compute_anchored_mse.py` 是事後平移，形狀沒有被訓練優化過；
delta-target 讓模型直接在偏離空間學習。**架構完全不動**，只改 target 定義，
成本最低、預期收益最大。

**否決的替代方案**：

| 方案 | 否決理由 |
|---|---|
| NLinear | 它減的是 input 自己的最後值，但 input 是 HL02–06 → 錨到鄰站，錯通道 |
| RevIN | 需要 HL01 的歷史統計量 → 違反原則 1 |
| 續用後處理 anchoring | 非 end-to-end，形狀沒被優化 |

**歷史**：5/20、6/3、6/22 三次會議都提過「用 delta 做」，都「先擱置」，
6/30 才正式排進第一位。

### ② exog ablation

把 exog 整條拿掉重訓一次，比較 corr/MSE。

**為何**：③ 是大改，改之前得先知道 exog 路徑**現在到底有沒有貢獻**。
若拿掉幾乎沒差，③ 的優先度要重評，甚至該先查為什麼沒效。
便宜的體檢，擋在昂貴的改動前面。

### ③ exog / GRU horizon-aware

**現況**（`models/DLinearMix2.py`）：

```
x_exog [B, 96, E] → GRU → 只取最後 hidden state → context [B, 16]
                                                       ↓
branch_preds [B, 15, K] ─ flatten [B, 15*K] ─ concat ─→ FlattenFusion → [B, 15, 1]
```

**問題**：那個 `[B,16]` context 是**單一向量，broadcast 給全部 15 個 horizon**。
但降雨到水位有時間延遲——雨對 `t+1` 和對 `t+15` 的影響在物理上就不同。
單一 context 無法表達這個差異，15 個 horizon 拿到完全相同的外生資訊。

**改法**：GRU 改為輸出**所有時間步**（不只最後 hidden state），
每個 horizon 配一個**可學的 query**，用 cross-attention 去 attend 那條序列。
如此 `t+15` 可以自己學會去看 90 分鐘前的那場雨。

註：4/15 會議的「exogenous encoder 改 GRU」已完成，現行就是 GRU。本項是下一階段。

### ④ branch-NLinear（偏離變體）

每個 DLinear branch 先減掉自己的最後值，**但不加回去**，讓 branch 輸出停留在偏離空間。

**為何**：配合 ①。當 target 已是 delta，branch 還輸出絕對值的話，
融合層得同時處理兩種尺度。統一到偏離空間比較乾淨。

### ⑤ 加乾段（dry windows）訓練

現在只用降雨事件視窗訓練，加入一個平衡的乾段子集。

**為何**：乾段對模型是 OOD（訓練時沒看過）→ 模型在乾段亂動
（anchored 砍 96% 誤差仍輸 persistence）。要教會它「沒有驅動就不要動」。

**與 drycut 的關係**：drycut 保留了**雨中的停頓**（退水），已部分改善；
但**完全乾期**仍未涵蓋，本項仍需要。

### ⑥（可選）rain 與 gate 分開 encode

目前雨和閘門塞在同一個 exog 張量共用一個 GRU，但兩者時間尺度差很多
（閘門反應快、上游流達慢）。分開 encode 可能較好。

---

## 4. 暫緩項：per-segment 隨機 split

6/22 會議提出（「每個 segment 隨機 split，但不要讓一個 segment 的 windows
散落在不同 split」），6/30 決定**暫緩**。

- **暫緩理由**：若 delta-target 消掉 level 軸的分布差異，這件事可能就不必做。
- **舊法的難點**：±60min buffer 造成 **32 對 segment 重疊、1642 個共用 row**，
  必須在 **127 個 union 合併單位**上切分才不會 leakage。
- **drycut 已結構性解決此難點**：`L >= 2*buffer` 約束保證相鄰 window 不重疊
  （L=3h/buf=60 實測 0 重疊、最小間隔 61 分鐘）。若日後要做隨機 split，
  可直接在 179 個獨立 segment 上做，不需要 union 合併。
- test 是永久 hold-out，要做須挑有代表性的 seed。

**相關背景**（4/15 會議）：train/valid/test 分布不同 → 用 train 的 scaler
預測 val/test 有 bias，也影響早停。目前緩解是 `--early_stop_metric corr`。

---

## 5. 劇烈變動（volatility）的處理

切分方法只決定**用哪些資料**，不改變**資料本身的動態**。水位劇烈變動是
要預測的訊號本身，不是要消除的雜訊。已有／規劃中的對策：

| 層面 | 對策 | 狀態 |
|---|---|---|
| 目標參數化 | delta-target：把 level 與變化解耦 | 待做（#1） |
| 損失函數 | `--criterion Huber`：降低極端樣本主導梯度 | 已實作（5/20） |
| 早停準則 | `--early_stop_metric corr`：不被 level bias 綁架 | 已實作（5/20） |
| 評估指標 | window-weighted / per-segment mean / median / worst-decile 四種 MSE，避免颱風段主導 | 已實作（5/20） |
| 時間延遲 | horizon-aware cross-attention：讓不同 horizon 看不同時間尺度的雨 | 待做（#3） |
| 資料涵蓋 | drycut 保留退水段；加乾段訓練 | 部分完成／待做（#5） |

**已知未解的疑慮**（5/20 會議「先擱置」）：
> 後置 60min 是否讓模型難以掌握，像是可能同時有上升動態、衰退動態、平靜狀態

buffer=60 的定案**正好把這 60 分鐘後置段加回來**，此疑慮重新浮現。
處理順序：先做 #1（delta-target），再用 per-segment 指標檢查
「buffer 段內的 horizon 表現是否明顯較差」，若是則考慮調降 buffer 或
對 buffer 段加權。
