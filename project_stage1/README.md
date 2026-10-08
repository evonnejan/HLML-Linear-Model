# HL01 水位預測

這一包是 HL01 水位預測的 input 準備手冊與推論程式：把原始紀錄整理成每分鐘一列的資料，再用最近 60 分鐘的資料預測未來 15 分鐘的 HL01。

## 資料夾結構

```text
project_stage1/
├── README.md
├── README.html
├── INPUT_MANUAL.md
├── INPUT_MANUAL.html
├── examples/
│   └── example_minute.csv
└── inference/
    ├── README.md
    ├── README.html
    ├── run_forecast.py
    ├── forecast_from_table.py
    ├── predictor.py
    ├── validation.py
    ├── input_schema.json
    ├── model.py
    ├── __init__.py
    └── model_bundle/
        ├── config.json
        ├── scaler.npz
        └── checkpoint.pth
```

| 檔案或資料夾 | 說明 |
|---|---|
| `README` | 本文件，說明這一包的內容 |
| `INPUT_MANUAL` | Input 準備手冊：原始紀錄怎麼取值、對齊時間、處理特殊碼與遺失值，存成分鐘表與修改紀錄表，再取出 60 列 input |
| `examples/example_minute.csv` | 分鐘表範例：2024-10-31 09:31 到 11:00，每分鐘一列，共 90 列 |
| `inference/` | 推論程式：檢查 input、載入模型、預測未來 15 分鐘的 HL01，並附兩支呼叫範例；各檔案的用途與使用方式見 `inference/README` |

`.md` 和 `.html` 的內容相同，`.html` 可以直接用瀏覽器開啟。

## 流程概覽

原始紀錄 → 依 `INPUT_MANUAL` 整理成分鐘表與修改紀錄表 → 從分鐘表取 60 列 → `inference/` 預測未來 15 分鐘的 HL01 → 保存結果
