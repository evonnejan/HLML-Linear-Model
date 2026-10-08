"""HL01 預測的範例：從分鐘表選出 t，取 t−59 到 t 共 60 列，預測 t+1 到 t+15 分鐘的 HL01。

- 不給 t：從分鐘表最新一列往前找 HL01 是實測值的分鐘當 t，最早只找到「執行時間 − 14 分鐘」；
  找不到就不預測。定期排程和有人臨時要預測時都用這個。
- 給 t：直接用這個 t；t 不在分鐘表裡，或 HL01 在 t 不是實測值，就不預測，也不往前找。

以這支程式為基礎，把 read_minute_table() 和 read_hl01_changes() 改成自己系統的寫法即可。
範例資料是 2024 年的，所以範例把執行時間設成 2024-10-31 11:05:00。
在包含 project_stage1/ 的資料夾執行：
    python -m project_stage1.inference.forecast_from_table

結束碼：0 表示預測成功；99 表示 input 沒有通過檢查；98 表示找不到可預測的 t。後兩種這一輪都不預測。
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from project_stage1.inference import InputValidationError, Predictor

EXAMPLE_MINUTE = Path(__file__).resolve().parents[1] / "examples" / "example_minute.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "outputs" / "hl01_forecast.csv"
WINDOW = pd.Timedelta(minutes=59)  # t−59 到 t 共 60 列
MAX_AGE = pd.Timedelta(minutes=14)  # t 最早到執行時間 − 14 分鐘，t+15 才會在執行時間之後
TAIWAN = timezone(timedelta(hours=8))


class NoForecastTimeError(Exception):
    """找不到可預測的 t。"""


def read_minute_table(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """讀分鐘表裡 start 到 end（含）的列。"""
    table = pd.read_csv(EXAMPLE_MINUTE)
    # table = pd.read_sql("SELECT * FROM hl01_minute WHERE date BETWEEN ? AND ?", conn, params=[start, end])  # 寫法依資料庫而定
    dates = pd.to_datetime(table["date"], errors="coerce")
    return table[dates.between(start, end)]


def read_hl01_changes(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    """讀修改紀錄表，回傳 start 到 end（含）之間 HL01 不是實測值的分鐘。"""
    log = pd.DataFrame({"date": []})  # 範例沒有修改紀錄
    # log = pd.read_sql("SELECT date FROM hl01_change_log WHERE column_name = 'HL01' AND date BETWEEN ? AND ?", conn, params=[start, end])
    return set(pd.to_datetime(log["date"]))


def forecast(predictor: Predictor, t=None, now=None) -> dict:
    """選出 t，取 t−59 到 t 共 60 列交給 predictor.predict()，回傳和 predict() 相同的 dict。

    t：要預測的時間點（YYYY-MM-DD HH:MM:SS）；不給時從分鐘表找最新可預測的 t。
    now：執行時間，只在不給 t 時使用；不給時用現在的台灣時間。
    找不到可預測的 t 時丟出 NoForecastTimeError；60 列沒有通過檢查時丟出 InputValidationError。
    """
    if t is None:
        now = pd.Timestamp(now if now is not None else datetime.now(TAIWAN).replace(tzinfo=None)).floor("min")
        earliest = now - MAX_AGE
        table = read_minute_table(earliest - WINDOW, now)
        dates = pd.to_datetime(table["date"], errors="coerce")
        changed = read_hl01_changes(earliest, now)
        # 從最新一列往前找 HL01 是實測值的分鐘
        candidates = [d for d in dates.sort_values(ascending=False) if d >= earliest and d not in changed]
        if not candidates:
            raise NoForecastTimeError(f"No minute with a measured HL01 between {earliest} and {now}")
        t = candidates[0]
    else:
        t = pd.Timestamp(t)
        table = read_minute_table(t - WINDOW, t)
        dates = pd.to_datetime(table["date"], errors="coerce")
        if not (dates == t).any():
            raise NoForecastTimeError(f"{t} is not in the minute table")
        if t in read_hl01_changes(t, t):
            raise NoForecastTimeError(f"HL01 at {t} is not a measured value")

    # 取 t−59 到 t，依時間由舊到新排好
    in_window = dates.between(t - WINDOW, t)
    window = table.loc[dates[in_window].sort_values().index].reset_index(drop=True)
    return predictor.predict(window)


def main() -> int:
    predictor = Predictor()  # 載入 model_bundle/

    # 1. 選出 t 並預測
    try:
        result = forecast(predictor, now="2024-10-31 11:05:00")  # 範例：假設在 11:05 執行
        # result = forecast(predictor)  # 正式使用：找最新可預測的 t
        # result = forecast(predictor, t="2024-10-31 10:30:00")  # 指定 t
    except NoForecastTimeError as error:
        print(error)  # 這一輪不預測，記錄原因
        return 98
    except InputValidationError as error:
        print(error.as_dict())  # 這一輪不預測，記錄原因
        return 99

    # 2. 整理成結果表：一列放這一輪的全部預測；time 是 t，pred_k 是 t+k 分鐘的預測
    row = {"time": result["time"], **{f"pred_{k}": value for k, value in enumerate(result["pred"], start=1)}}
    result_table = pd.DataFrame([row])
    print(result_table.T.to_string(header=False))  # 轉置後印出，比較好讀

    # 3. 保存結果：每一輪附加一列到 outputs/hl01_forecast.csv（第一次執行時建立資料夾與表頭）
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(OUTPUT_CSV, mode="a", header=not OUTPUT_CSV.exists(), index=False)
    print(f"已附加到 {OUTPUT_CSV}")
    # result_table.to_sql("hl01_forecast", conn, if_exists="append", index=False)  # 直接寫進資料庫；寫法依資料庫而定
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
