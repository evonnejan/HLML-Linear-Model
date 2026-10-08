"""HL01 預測的呼叫範例：取 60 列 input、預測未來 15 分鐘的 HL01，整理成結果表並保存。

以這支程式為基礎，把「1. 取 60 列」和「4. 保存結果」改成自己系統的寫法即可。
現在的設計是先把每一輪的結果附加到 outputs/hl01_forecast.csv，之後可以再把檔案裡的資料寫入資料庫；
也可以直接 to_sql，或不存檔、直接使用 predict() 的回傳值，依你們的系統決定。
在包含 project_stage1/ 的資料夾執行：
    python -m project_stage1.inference.run_forecast

結束碼：0 表示預測成功；99 表示 input 沒有通過檢查，這一輪不預測。
"""
from pathlib import Path

import pandas as pd

from project_stage1.inference import InputValidationError, Predictor

EXAMPLE_MINUTE = Path(__file__).resolve().parents[1] / "examples" / "example_minute.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "outputs" / "hl01_forecast.csv"


def main() -> int:
    predictor = Predictor()  # 載入 model_bundle/；程式常駐執行時只需建立一次

    # 1. 取 t−59 到 t 共 60 列（見手冊 4.3）；範例取分鐘表範例的最後 60 列
    window = pd.read_csv(EXAMPLE_MINUTE).tail(60).reset_index(drop=True)
    # window = pd.read_sql("SELECT ... FROM hl01_minute ...", conn)  # 寫法依資料庫而定

    # 2. 預測
    try:
        result = predictor.predict(window)
    except InputValidationError as error:
        print(error.as_dict())  # 這一輪不預測，記錄原因
        return 99

    # 3. 整理成結果表：一列放這一輪的全部預測；time 是 t，pred_k 是 t+k 分鐘的預測
    row = {"time": result["time"], **{f"pred_{k}": value for k, value in enumerate(result["pred"], start=1)}}
    forecast = pd.DataFrame([row])
    print(forecast.T.to_string(header=False))  # 轉置後印出，比較好讀

    # 4. 保存結果：每一輪附加一列到 outputs/hl01_forecast.csv（第一次執行時建立資料夾與表頭）
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT_CSV, mode="a", header=not OUTPUT_CSV.exists(), index=False)
    print(f"已附加到 {OUTPUT_CSV}")
    # forecast.to_sql("hl01_forecast", conn, if_exists="append", index=False)  # 直接寫進資料庫；寫法依資料庫而定
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
