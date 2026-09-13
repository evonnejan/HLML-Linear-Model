# 02 — 方法與評估協議

> **本檔是「受審宣稱」（Tier 1）。** 與程式碼不符即為 finding。

---

## 1. 模型：DLinearMix2

實作：`models/DLinearMix2.py`（`Model` 類別於 `models/DLinearMix2.py:167`）。

### 1.1 架構

```
每個 input_col ──► 自己的 DLinear branch ──► branch forecasts [B, P, K]
exog_cols      ──► ExogenousEncoder(GRU)  ──► context embedding [B, E]
                          │
        [branch forecasts + exog context] ──► MLP fusion ──► [B, P, 1]
```

相對於前一版 `DLinearMix`（early fusion：把所有欄位線性混成一條序列再丟 DLinear），
這版改成 **late/near-late fusion**，保留每個輸入通道自己的時序結構，
並把降雨/閘門當作 **context** 而不是強迫它們表現得像目標序列。

### 1.2 兩種融合方式

由 `flatten_fusion` 布林旗標控制（`models/DLinearMix2.py:144-165`）：

| 設定 | 行為 |
|---|---|
| `flatten_fusion=False` | horizon-wise 共享 MLP：exog embedding 被 `expand` 廣播給全部 15 個 horizon |
| `flatten_fusion=True` | 把所有 branch forecasts 攤平，配一個較大的 MLP |

5/20 meeting 記錄：實測 `flatten_fusion=True` 表現較好。

### 1.3 通道順序契約（隱性，需注意）

模型假設 `x[:, :, :branch_in]` 是主輸入、`x[:, :, branch_in:branch_in+exog_in]` 是外生。
這個順序由 `Data_Loader` 建 `x_cols = input_cols + exog_cols` 保證
（`data_provider/Data_Loader.py:247-250`），模型端只用通道**數量**做驗證
（`models/DLinearMix2.py:316-323`）——數量對但順序錯的話**不會被抓到**。

### 1.4 與 LTSF-Linear 的關係

`Source_Code/` 是 LTSF-Linear 的原始參考碼（Apache-2.0），包含 `DLinear.py` / `Linear.py` /
`NLinear.py` / `exp_*` / `data_*`。**不是本專案的執行路徑**，僅供對照。
`models/DLinear.py` 等為本專案改寫版。
⚠️ 目前 repo 內**沒有 NOTICE 檔標註來源與授權**，論文與開源都需要補。

## 2. 訓練設定

- Optimizer：Adam（`exp/exp_Main2.py:129-130`）
- Loss：`--criterion mse`（`nn.MSELoss`）或 `huber`（`nn.SmoothL1Loss(beta)`），
  實作於 `exp/exp_Main2.py:132-140`。Huber 的動機是降雨強度不均、對極端事件 robust。
- Early stopping：`--early_stop_metric` 可選 `mse` / `mae` / `corr`。
  corr 會被取負號以統一成「越小越好」（`exp/exp_Main2.py:223-237`）。
- 裝置：Apple Silicon MPS，單回合約 7 分鐘。

## 3. 評估協議

### 3.1 主指標：correlation

`vali()` 的定義（`exp/exp_Main2.py:184-221`）：
把所有 window 的預測攤成 `[N, P]`，**對每個 horizon h 跨 N 個 window 算 Pearson 相關**，
再對 15 個 horizon 取平均。零變異的 horizon 會被跳過。

> ⚠️ 這是 **跨 window pooled** 的相關係數，不是 per-window 或 per-event 的形狀相關。
> 在有 level bias 的情況下，pooled corr 會被 window 之間的水位差異主導。
> 見 `06-open-issues.md` OI-04。
>
> **注意：per-segment 的 corr 其實已經存在**——`exp/exp_Main2.py:696-880` 的
> `_save_segment_metrics` 產出 `metrics_segment.csv`，每段都有 `Corr`
> （`horizon="all"` 與 per-horizon 兩種列）。只是它沒被當成主指標，也沒有驅動 early stopping。
> 待決的是「headline 該用哪一個」，不是「要不要新增功能」。

### 3.2 Persistence baseline

`predict t+k = t`（拿輸入最後一個值當所有 horizon 的預測）。
實作於 `exp/exp_Main2.py:617-694`，輸出 `persist_MSE` / `persist_MAE` /
`persist_corr_mean` / `improve_MSE_pct` / `improve_MAE_pct` /
`improve_h1_MSE_pct` / `improve_hN_MSE_pct`。

### 3.3 Anchored 診斷

因為 HL01 不在 input，模型沒有 level 錨，每個 window 會有固定偏移。
anchored 版本保留形狀、只修 level：

```
c   = pred[0] - x_last      # 預測起點與最後觀測值的落差
adj = pred - c              # 整條平移，使 adj[0] == x_last
```

工具：`compute_anchored_mse.py`（raw / adj / persist 的 MSE/RMSE/MAE/Corr + 逐 horizon Corr）、
`visualize_anchored.py`、`visualize_segment.py`、`slide_anchored_figure.py`。

**⚠️ `x_last` 從哪裡來（常見誤解，務必看清楚）**

anchored 與 persistence 都需要「最後一個 input 時刻的 HL01 值」。這個值**不是**從
`--input_col` 來的，而是走 **target/label 那條路**：

```
Data_Loader.py:277      data_y = df_cur[[self.target]]          ← 只含 HL01
Data_Loader.py:507-509  r_begin = s_end - label_len
                        seq_y   = data_y[r_begin:r_end]
                        → seq_y[label_len-1] == data_y[s_end-1]  ← 最後一個 input 時刻的 HL01
exp/exp_Main2.py:547    last_input_target = batch_y[:, label_len-1:label_len, f_dim:]
compute_anchored_mse.py:40-42   同一個表達式（重建 test dataset 後取）
```

`batch_y` 完全獨立於 `batch_x` / `input_col`。**因此把 HL01 排除在 `--input_col` 之外，
不會讓 anchored 或 persistence 失效**——`run_dlinearmix2_sweep_noHL01.sh` 能產出 anchored
結果即為佐證。

這正是原則 1 的界線所在：

| | 原則**禁止** | 原則**允許** |
|---|---|---|
| 用法 | HL01 整條 `seq_len` 歷史當 branch 通道 | HL01 在 `s_end-1` 的**單一最後值**當錨 |
| 資料路徑 | `batch_x` ← `input_col` | `batch_y` ← `target` |
| 後果 | 自迴歸過度依賴 | 只修 level，不影響模型學形狀 |

`exp/exp_Main2.py:525-529` 會強制 `label_len >= 1`，否則 persistence 取不到最後一個 input 時刻。

### 3.4 多種 MSE 彙總（因降雨強度不均）

meeting 5/20 決議的四種：window-weighted、per-segment 取平均、per-segment 中位數、
worst-decile per-segment。實作於 `exp/exp_Main2.py:696` 起的 `_save_segment_metrics`。

### 3.5 分解報表

- per-horizon（t+1 … t+15）的 MSE / Corr
- per-segment 的 MSE / Corr
- 雙 checkpoint：同時保存 best-by-MSE 與 best-by-Corr

## 4. 資料切分與交叉驗證

### 4.1 三層防 leakage 論證鏈

1. **`L >= 2*buffer`** → 相鄰 segment 的 `[WinStart, WinEnd]` 不會重疊
   （drycut 結構性保證；舊法沒有，實測 32 對重疊）
2. **window 不跨段** → `Data_Loader` 只在同一 `segment_id` 的連續區塊內列舉 window 起點
   （`data_provider/Data_Loader.py:474-489`）
3. **segment 不跨 split** → `build_splits.py` 以 segment 為最小單位切分，且用
   `find_overlap_groups`（`build_splits.py:80-108`）找出重疊群組，
   `blocked_boundaries`（`build_splits.py:110-126`）禁止把同一群組切開

### 4.2 切分方式

`build_splits.py` 依**各段可用 window 數**（NaN-aware，對齊 `Data_Loader` 的
`valid_starts` 邏輯，見 `build_splits.py:58-78`）決定邊界，**不是按段數切**——
因為段長差距達 130~2040 分鐘，按段數切會讓實際樣本比例嚴重偏離。
目標比例預設 `0.70,0.15,0.15`（`build_splits.py:243`）。

### 4.3 Rolling-origin expanding-window CV

在 dev（=train+val）內產 **3 個 fold**：塊 0 為起始 train，塊 1..k 依序當各 fold 的 val，
train 逐 fold 擴張（`build_splits.py:170-208`）。**test 在所有 fold 間固定不變**
（`data_provider/Data_Loader.py:83-86` 明確保證 test 沿用 `split` 欄）。

fold=3 的依據：k=3 的 val 大小最平均（±4%）且每折含 28–38 個降雨事件；
k=4 有一折僅 15 段、k=6 有兩折僅 9 段。**有效樣本單位是「事件」而非 window**
（同段內 window 高度相關），故事件數過少的折估計不穩。

### 4.4 `--split_mode` 為必填（2026-09-10）

`run.py:69-99` 的 `_validate_split_args` 強制：一旦給了 `--segment_col`，就必須明講
`--split_mode file`（用 split 檔）或 `builtin`（舊的依段數 70/10/rest，無重疊防護），
忘了帶會直接報錯。動機是避免靜默回退到另一種切法，導致整批實驗不可比又難察覺。

## 5. 計畫中的實驗矩陣（尚未執行）

**36 runs** = 資料集(2: drycut / 舊法) × exog(3: `isRain` / 無 / `min_since_rain`)
× loss(2: mse / huber) × fold(3) × seed(1)，約 4.2 小時。

需要新寫兩支工具：(i) 因子驅動腳本 (ii) 把 anchored 指標併進同一張總表的彙整器。

依決策，**本次 review 的 Blocker 清完才啟動**。
