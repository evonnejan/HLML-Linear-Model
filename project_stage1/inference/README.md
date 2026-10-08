# HL01 預測程式

這個資料夾是 HL01 水位預測的推論程式：輸入 60 分鐘的資料，輸出未來 15 分鐘的 HL01 預測。input 的準備方式見 **INPUT_MANUAL**。

本文件用 **t** 表示預測的起點，也就是 input 最後一列（最新一分鐘）的時間：input 是 t−59 到 t 共 60 列，預測的是 t+1 到 t+15 分鐘的 HL01。例如 t 是 11:00 時，input 是 10:01 到 11:00，預測 11:01 到 11:15。

## 檔案

| 檔案 | 用途 |
|---|---|
| `predictor.py` | 預測入口。`Predictor` 載入模型，`predict()` 檢查 input 後回傳 15 筆預測 |
| `validation.py` | 預測前檢查 input，`predict()` 會自動呼叫 |
| `input_schema.json` | 檢查規格：欄位、列數、時間格式、不可為負數的欄位 |
| `model.py` | 模型的網路結構 |
| `model_bundle/config.json` | 模型設定與特徵順序 |
| `model_bundle/scaler.npz` | 標準化參數 |
| `model_bundle/checkpoint.pth` | 模型權重 |
| `__init__.py` | 讓 `from project_stage1.inference import Predictor` 可以使用 |
| `run_forecast.py` | 基本用法範例：取 60 列、預測，並把結果附加到 `outputs/hl01_forecast.csv` |
| `forecast_from_table.py` | 從分鐘表選 t 的範例：找最新可預測的 t 或使用指定的 t，取 60 列預測，並把結果附加到 `outputs/hl01_forecast.csv` |
| `outputs/hl01_forecast.csv` | 範例程式保存的預測結果，每一輪一列；第一次執行時自動建立 |

`model_bundle/` 的三個檔案是同一次訓練的產物，必須一起使用，不要單獨替換其中一個。

## 要改與不用改的檔案

- 要改：`run_forecast.py` 或 `forecast_from_table.py`，依你們的用法選一支當基礎，把讀資料和保存結果改成自己系統的寫法（見「使用方式」）；也可以複製到自己的專案再改。
- 不用改：其他所有檔案（`predictor.py`、`validation.py`、`input_schema.json`、`model.py`、`model_bundle/`、`__init__.py`）。檢查規則、特徵順序、模型結構和權重都必須和訓練時一致，修改會讓預測出錯。模型需要更新時，會整份替換 `model_bundle/`。

## 環境

- Python 3.11 以上（pandas 3.0.1、NumPy 2.4.3 的要求），已驗證 Python 3.14。只需要 CPU，不需要 GPU。
- 安裝：`python -m pip install torch==2.10.0 pandas==3.0.1 numpy==2.4.3`。只需要這三個，它們用到的其他套件 pip 會自動一起安裝。
- Linux 上 pip 預設會安裝含 CUDA 的 PyTorch，下載量很大；沒有 GPU 時可以改用 CPU 版：`python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu`
- 用 `pd.read_sql` 從資料庫取資料時，另外安裝你們資料庫的連線套件。
- 在包含 `project_stage1/` 的資料夾執行程式。

## 使用方式

有兩支範例程式，選一支當基礎修改即可：

| 程式 | 適合的情況 |
|---|---|
| [`run_forecast.py`](run_forecast.py) | 基本用法：自己取好 t−59 到 t 的 60 列，直接預測 |
| [`forecast_from_table.py`](forecast_from_table.py) | 從分鐘表選 t：定期排程或有人臨時要預測時，找最新可預測的 t；也可以指定 t |

先在包含 `project_stage1/` 的資料夾執行一次，確認環境沒問題：

```bash
python -m project_stage1.inference.run_forecast
python -m project_stage1.inference.forecast_from_table
```

兩支都會印出 t 為 2024-10-31 11:00:00 的預測。結束碼：

| 結束碼 | 意思 |
|---|---|
| 0 | 預測成功 |
| 99 | input 沒有通過檢查（見「檢查沒通過時」），這一輪不預測 |
| 98 | 找不到可預測的 t（見「找不到可預測的 t 時」），這一輪不預測；只有 `forecast_from_table.py` 會出現 |

### run_forecast.py：基本用法

示範完整的一輪：取 60 列 → 預測 → 整理成結果表 → 保存。接到自己的系統時，改這兩處：

| 步驟 | 範例的寫法 | 改成 |
|---|---|---|
| 1. 取 60 列 | 讀分鐘表範例 `examples/example_minute.csv` 的最後 60 列 | 從 `hl01_minute` 取 t−59 到 t（手冊 4.3），例如 `pd.read_sql(...)` |
| 4. 保存結果 | 附加到 `outputs/hl01_forecast.csv` | 依你們的系統決定，例如定期把 CSV 匯入資料庫、直接 `forecast.to_sql(...)`，或直接使用回傳值（見「結果怎麼用」） |

### forecast_from_table.py：從分鐘表選 t

`forecast(predictor, t=None, now=None)` 先選出 t，再從分鐘表取 t−59 到 t 共 60 列（依時間由舊到新排好）預測，回傳和 `predict()` 相同的 dict。

| 呼叫方式 | t 怎麼決定 |
|---|---|
| `forecast(predictor)` | 從分鐘表最新一列往前找 HL01 是實測值的分鐘當 t，最早只找到「執行時間 − 14 分鐘」，這樣 t+15 才會在執行時間之後；找不到就不預測。執行時間是呼叫當下的台灣時間 |
| `forecast(predictor, t="2024-10-31 10:30:00")` | 直接用這個 t，不管多久以前；t 不在分鐘表裡，或 HL01 在 t 不是實測值，就不預測，也不往前找 |

例如 13:50 執行、分鐘表最新到 13:45：t 在 13:36 到 13:45 之間找；如果分鐘表停在 12:00，這一輪就不預測。「HL01 是實測值」指修改紀錄表 `hl01_change_log` 裡沒有這一分鐘 HL01 的紀錄（手冊 4.3）。

範例資料是 2024 年的，所以直接執行時用 `forecast(predictor, now="2024-10-31 11:05:00")`，假設在 11:05 執行。接到自己的系統時，改這幾處：

| 位置 | 範例的寫法 | 改成 |
|---|---|---|
| `read_minute_table()` | 讀 `examples/example_minute.csv` | 從 `hl01_minute` 讀指定時間範圍的列，例如 `pd.read_sql(...)` |
| `read_hl01_changes()` | 沒有修改紀錄 | 從 `hl01_change_log` 讀 HL01 被修改過的分鐘，例如 `pd.read_sql(...)` |
| `main()` 1. 選出 t 並預測 | `forecast(predictor, now="2024-10-31 11:05:00")` | `forecast(predictor)`；要指定時間時 `forecast(predictor, t=...)` |
| `main()` 3. 保存結果 | 附加到 `outputs/hl01_forecast.csv` | 同 `run_forecast.py` 的「4. 保存結果」 |

### 兩支共通

- `Predictor()` 只需要在程式啟動時建立一次，之後每一輪重複呼叫 `predict()`。找不到 `model_bundle/`，或三個檔案彼此不一致時，`Predictor()` 會直接報錯。
- `predict()` 的 input 是 pandas DataFrame：60 列、15 欄（手冊第 0 節）。`date` 可以是 `YYYY-MM-DD HH:MM:SS` 文字，也可以是不含時區的時間。

## 輸出

`predict()` 回傳一個 dict：

```python
{"time": "2024-10-31 11:00:00",
 "pred": [1450.0, 1437.681242280028, ..., 1279.3590226036636]}
```

- `time`：t，也就是這份 input 最後一列的時間，格式和 input 相同。
- `pred`：15 個 HL01 預測值，依序是 t+1 到 t+15 分鐘；單位和 input 的 HL01 相同，程式不做四捨五入。
- `predict()` 不會存檔，保存由呼叫端負責。

### 結果怎麼用

現在的設計是範例程式先把每一輪的預測寫到一個 CSV 檔（`outputs/hl01_forecast.csv`），之後可以再把這個檔案裡的資料寫入資料庫。這只是其中一種作法，可依你們的系統決定，例如：

- 定期把 CSV 裡的資料匯入資料庫。
- 在程式裡直接 `to_sql(...)`，每一輪寫進資料庫。
- 不存檔，直接使用 `predict()` 的回傳值串到自己的系統，例如 `result["pred"]` 就是 15 個預測值的 list。

兩支範例程式為了方便，都把結果附加到 `outputs/hl01_forecast.csv`，格式相同，每一輪一列。實際要存在哪裡、要不要分開存、要不要寫進資料庫，由你們決定。

寫入資料庫的時段可參考 INPUT_MANUAL 4.5。

### 結果表（建議格式）

要存成表時，建議每一輪一列：

| time | pred_1 | pred_2 | … | pred_15 |
|---|---|---|---|---|
| 2024-10-31 11:00:00 | 1450.0 | 1437.681242 | … | 1279.359023 |

- `time`：這一輪的 t，每一輪只有一列，可以用它唯一識別一筆。
- `pred_1` 到 `pred_15`：t+1 到 t+15 分鐘的 HL01 預測。
- 表名 `hl01_forecast` 只是建議，可依你們的命名習慣決定。

CSV 只會在最後面附加新的一列，所以：

- 同一個 t 重跑時會多一列相同的 `time`，匯入資料庫時以 `time` 去重即可。
- `time` 不一定由舊到新。例如平常排程依序附加了 t 為 2024-10-31 11:00、11:10、11:20 的三列，之後有人指定 t 為 2024-10-01 08:00 重算，這一列會接在最後面；需要依時間排列時，以 `time` 排序即可。

## 檢查沒通過時

`predict()` 會丟出 `InputValidationError`，這一輪不會預測；範例程式會印出原因，結束碼為 99。一次只回報第一個問題，`error.as_dict()` 的內容：

| 鍵 | 內容 |
|---|---|
| `code` | 錯誤碼，見下表 |
| `message` | 說明 |
| `field` | 出問題的欄位；沒有特定欄位時為 `None` |
| `row` | 出問題的列號，從 0 起算（第 1 列是 0）；沒有特定列時為 `None` |

例如少了一列，或第 31 列的 Past10Min 留下沒處理的 -99：

```python
{'code': 'ROW_COUNT', 'message': 'Expected exactly 60 rows, got 59', 'field': None, 'row': None}
{'code': 'NEGATIVE_RAINFALL', 'message': 'Value must be nonnegative; prepare negative source values upstream', 'field': 'Past10Min', 'row': 30}
```

| 錯誤碼 | 意思 |
|---|---|
| `MISSING_COLUMNS`、`EXTRA_COLUMNS`、`DUPLICATE_COLUMNS` | 欄位缺少、多出或重複 |
| `ROW_COUNT` | 不是剛好 60 列 |
| `TIMESTAMP_FORMAT`、`TIMESTAMP_MISSING`、`MINUTE_ALIGNMENT` | 時間格式不對、時間空白、時間不在整分鐘 |
| `DUPLICATE_TIMES`、`TIME_ORDER`、`TIME_GAP` | 時間重複、沒有由舊到新、相鄰兩列不是相隔 1 分鐘 |
| `NUMERIC_TYPE`、`NONFINITE_VALUE`、`NUMERIC_RANGE` | 不是數字、有空值或無限大、數值過大 |
| `NEGATIVE_WATER_LEVEL`、`NEGATIVE_RAINFALL`、`NEGATIVE_GATE` | 水位、雨量、閘門出現負數 |
| `INPUT_TYPE` | input 不是 pandas DataFrame |

## 找不到可預測的 t 時

只有 `forecast_from_table.py` 會發生。`forecast()` 會丟出 `NoForecastTimeError`，這一輪不會預測；範例程式會印出原因，結束碼為 98。原因有三種：

| 印出的訊息（例） | 意思 |
|---|---|
| `No minute with a measured HL01 between 2024-10-31 10:51:00 and 2024-10-31 11:05:00` | 沒給 t：往前 14 分鐘內找不到 HL01 是實測值的分鐘，或分鐘表停在更早的時間 |
| `2024-10-31 12:00:00 is not in the minute table` | 指定的 t 不在分鐘表裡 |
| `HL01 at 2024-10-31 10:30:00 is not a measured value` | 指定的 t 的 HL01 不是實測值 |
