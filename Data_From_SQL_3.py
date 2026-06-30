"""Build the original rain-window water-level dataset with macOS/Docker SQL access."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Data_From_SQL_4 import (
    MIN_RAIN_SEGMENT_MINUTES,
    RAIN_GAP_MINUTES,
    RAIN_THRESHOLD,
    SQLServerClient,
    build_event_windows,
    load_rain_data,
    load_water_data,
    segment_intervals,
)

OUTPUT_CSV = Path("dataset/water_level_all3.csv")


def main() -> None:
    client = SQLServerClient()
    water_df = load_water_data(client).sort_values(["measure_time", "device_id"]).reset_index(drop=True)
    rain_df, rain_value_col = load_rain_data(client)

    print(f"使用雨量欄位：{rain_value_col}")
    print("開始篩選資料...")

    rain_segments = segment_intervals(
        rain_df,
        time_col="ObsTime",
        value_col=rain_value_col,
        threshold=RAIN_THRESHOLD,
        gap_minutes=RAIN_GAP_MINUTES,
        min_duration_minutes=MIN_RAIN_SEGMENT_MINUTES,
    )
    rain_windows = build_event_windows(rain_segments)

    if rain_windows.empty:
        raise ValueError("沒有找到符合條件的降雨事件，請調整 threshold 或 gap_minutes。")

    segments_data_list = []
    for _, row in rain_windows.iterrows():
        mask = (water_df["measure_time"] >= row["WinStart"]) & (water_df["measure_time"] <= row["WinEnd"])
        sub_df = water_df.loc[mask].copy()
        if sub_df.empty:
            continue

        sub_df["segment_id"] = row["segment_id"]
        sub_df["WinStart"] = row["WinStart"]
        sub_df["WinEnd"] = row["WinEnd"]
        sub_df["SegmentStart"] = row["SegmentStart"]
        sub_df["SegmentEnd"] = row["SegmentEnd"]
        sub_df["isRain"] = (
            (sub_df["measure_time"] >= row["SegmentStart"]) & (sub_df["measure_time"] <= row["SegmentEnd"])
        )
        segments_data_list.append(sub_df)

    if not segments_data_list:
        raise ValueError("降雨事件範圍內沒有可用的水位資料。")

    merged = pd.concat(segments_data_list, ignore_index=True)
    wide = merged.pivot_table(
        index=["measure_time", "SegmentStart", "SegmentEnd", "segment_id", "WinStart", "WinEnd", "isRain"],
        columns="device_id",
        values="val",
    )
    wide = wide.reset_index().rename(columns={"measure_time": "date"})
    wide.to_csv(OUTPUT_CSV, index=False)

    print("CSV 檔案儲存成功！")
    print(f"輸出檔案：{OUTPUT_CSV}")
    print(f"資料列數：{len(wide):,}")


if __name__ == "__main__":
    main()
