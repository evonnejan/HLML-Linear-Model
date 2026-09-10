# 04 — 程式碼地圖

> **分級的用意：** repo 根目錄有 40+ 支 `.py`，混雜現行、一次性、歷史遺留與外來參考碼。
> 審查者**只需審「現行核心」**；其餘不要花時間，更不要照著已棄用的程式提建議。

---

## 1. 現行核心（MUST-REVIEW）

依資料流順序。**這些是現在與之後會持續使用的程式碼，必須逐支審過。**

| 檔案 | 行數 | 職責 | 重點看什麼 |
|---|---|---|---|
| `Data_From_SQL_all.py` | 244 | SQL → `all_minute_wide.csv`。逐欄 merge_asof + 5min staleness | `merge_all_sources`（89-156）的閘門合併正確性；`build_segments_meta`（180） |
| `rebuild_gate_columns.py` | 120 | 就地重建既有寬表的閘門欄（不需 SQL），含 dry-run 與一致性檢查、自動備份 | `merge_gate_per_column`（39-54）是否與 `Data_From_SQL_all.py` 的邏輯**完全一致** |
| `build_drycut_segments_meta.py` | 138 | 乾段反向切分 → segment meta（L/buffer 參數化，不含 split） | `find_drycut_core_segments`（39-57）的 run-length 邏輯與 NaN 處理；buffer clip（106-107）；`L >= 2*buffer` 是否真的被強制 |
| `build_training_csv_from_meta.py` | 266 | meta + 寬表 → 訓練 CSV。含 `min_since_rain`、`isRain`、段內 gate ffill、重疊檢查、來源指紋 | `add_min_since_rain`（53-73）是否真的在切段前；`slice_segments`（75-97）邊界是否閉區間；`fill_gate_within_segment`（99-108）；`check_no_overlap`（110-134） |
| `build_splits.py` | 304 | segment-wise 時序切分 + rolling-origin expanding-window CV | `count_windows_per_segment`（58-78）是否真的對齊 loader；`find_overlap_groups`（80-108）連通分量正確性；`blocked_boundaries`（110-126）；`assign_folds`（170-208） |
| `data_provider/Data_Loader.py` | 629 | 資料載入、scaler、window 列舉、split 指派 | `_load_split_file`（54-99）；window 列舉與 NaN filter（436-503）；scaler 是否只 fit 在 train（307-311） |
| `data_provider/Data_Factory.py` | 68 | loader 工廠，透傳參數 | 參數是否完整透傳，有無遺漏 |
| `run.py` | 506 | CLI 進入點 | `_expand_col_patterns`（44-66）／`_expand_col_args`（100-116）**是否會把 target 展進 input**；`_validate_split_args`（69-98）；`_drop_constant_columns`（119-156）；`_configure_mix_model_args`（158-）|
| `models/DLinearMix2.py` | 347 | 主力模型：branch DLinear + GRU exog encoder + MLP fusion | 通道順序契約（295-323）；兩種 fusion（120-165）；exog broadcast 是否能表達雨延遲 |
| `exp/exp_Main2.py` | 1070 | 訓練/評估流程 | `vali`（184-221）corr 定義；`_select_score`（223-237）；early stopping 與雙 checkpoint；`_save_persistence_horizon`（617-694）；`_save_segment_metrics`（696-） |
| `exp/exp_Basic.py` | — | 基底類別 | 裝置選擇 |
| `utils/metrics.py` | — | 指標實作 | 與 `exp_Main2` 內的計算是否一致 |

## 2. 現行輔助（會用到，但非資料流主幹）

| 檔案 | 職責 |
|---|---|
| `compute_anchored_mse.py` | anchored 診斷：raw/adj/persist 的 MSE/RMSE/MAE/Corr + 逐 horizon Corr |
| `eval_dry.py` + `scripts/draw_eval_dry_diagrams.py` | 乾期評估與圖表 |
| `visualize_segment.py` / `visualize_anchored.py` / `slide_anchored_figure.py` | anchored 相關視覺化 |
| `visualize_drycut_segments.py` | drycut 切分結果檢視圖 |
| `run_dlinearmix2_sweep*.sh` | 掃參腳本（base / criterion / noHL01 三個變體） |
| `tests/test_merge_gate_data.py` | 目前 repo 內**唯一**的單元測試 |

## 3. 一次性分析（跑過就沒再用，不需審）

`analyze_dry_runs.py`、`analyze_rain_outside_segments.py`、`list_rain_outside_segments.py`、
`analyze_best_models_overview.py`、`analyze_full_inference.py`、`analyze_full_inference_lag.py`、
`sanity_check.py`

## 4. 歷史遺留（已被取代，勿依此提建議）

| 檔案 | 被誰取代 |
|---|---|
| `Data_From_SQL.py` / `_2.py` / `_3.py` / `_5.py` | `Data_From_SQL_all.py` |
| `Data_From_SQL_4.py` | ⚠️ **半退役**：仍是 `SQLServerClient` 的定義處，被 `_all` import；但其中的切窗/segment 邏輯已被 `build_*` 系列取代 |
| `merge_gate_data.py` | 閘門合併邏輯已移入 `Data_From_SQL_all.py` / `rebuild_gate_columns.py`（但單元測試仍指向它） |
| `filter_wra_cogate_columns.py` | 一次性，產 `wra_cogate_obs_wide_gate_opening.csv` 後未再用 |
| `models/DLinearMix.py` | `DLinearMix2.py`（early fusion → late fusion） |
| `exp/exp_Main.py` | `exp_Main2.py` |
| `visualize.py` / `model_visualize.py` | 較舊的視覺化 |

## 5. 外來參考碼（非本專案撰寫，不要審）

`Source_Code/` — LTSF-Linear 原始碼（Apache-2.0）：`DLinear.py`、`Linear.py`、`NLinear.py`、
`exp_basic.py`、`exp_main.py`、`data_factory.py`、`data_loader.py`。
⚠️ **目前沒有 NOTICE 檔標註來源與授權**，這是 TODO。

## 6. End-to-end 重現指令鏈

```bash
# 0) （需 SQL 環境；資料已凍結，通常跳過）
python Data_From_SQL_all.py                     # → dataset/all_minute_wide.csv

# 1) 閘門欄就地重建（已執行過；會自動備份 .gatev1.bak.csv）
python rebuild_gate_columns.py --dry-run
python rebuild_gate_columns.py

# 2) drycut 切分 meta（L=3h, buffer=60min）
python build_drycut_segments_meta.py --l-hours 3 --buffer 60
#   → dataset/rain_segments_meta_drycut_L3h_buf60.csv

# 3) meta → 訓練 CSV
python build_training_csv_from_meta.py --meta dataset/rain_segments_meta_drycut_L3h_buf60.csv
#   → dataset/train_drycut_L3h_buf60.csv
# 舊法對照組（有 32 對重疊，需明示接受）
python build_training_csv_from_meta.py --meta dataset/rain_segments_meta.csv --allow-overlap
#   → dataset/train_old.csv

# 4) 切分 + rolling-origin folds
python build_splits.py --data-path dataset/train_drycut_L3h_buf60.csv
python build_splits.py --data-path dataset/train_old.csv
#   → dataset/splits_*.csv

# 5) 訓練（審查者請勿執行）
python run.py --model DLinearMix2 --data custom \
  --data_path train_drycut_L3h_buf60.csv \
  --segment_col segment_id --features S --target HL01 \
  --input_col 'HL02,HL03,HL04,HL05,HL06' \
  --exog_col 'min_since_rain,Past10Min,Past1Hr,Now,*gate_opening*' \
  --split_mode file --split_file dataset/splits_train_drycut_L3h_buf60.csv --fold 1 \
  --seq_len 96 --pred_len 15 --label_len 30 \
  --batch_size 64 --train_epochs 80 --patience 15 \
  --learning_rate 1e-3 --dropout 0.1 --early_stop_metric corr
```

> ⚠️ 上面第 5 步的 `--input_col` 刻意寫成明列的 `HL02..HL06`，**不是** `PROGRESS.md` §5
> 與 `run_dlinearmix2_sweep.sh` 用的 `'HL*'`。原因見 `05-traceability.md` T-01。
