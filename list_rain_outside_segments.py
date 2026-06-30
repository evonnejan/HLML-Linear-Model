"""列出 all_minute_wide.csv 中 Past1Hr>=0.1 且不在任何 segment 事件視窗內的分鐘，
切成一段段、標註持續時間與『是否真的有降雨(Past10Min)』，並匯出 CSV 供檢視。

輸出:
  dataset/rain_outside_segments_runs.csv     -- 每一段一列(摘要)
  dataset/rain_outside_segments_minutes.csv  -- 每一分鐘一列(原始值)
"""
import pandas as pd

THRESHOLD = 0.1
GAP_MINUTES = 30

df = pd.read_csv("dataset/all_minute_wide.csv", usecols=["date", "Past10Min", "Past1Hr"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

meta = pd.read_csv("dataset/rain_segments_meta.csv",
                   parse_dates=["WinStart", "WinEnd", "SegmentStart", "SegmentEnd"])

# 標註：在任一事件視窗(Win = segment +/-60min)內
in_window = pd.Series(False, index=df.index)
for _, r in meta.iterrows():
    in_window |= (df["date"] >= r["WinStart"]) & (df["date"] <= r["WinEnd"])

rain = df["Past1Hr"] >= THRESHOLD
mask = rain & ~in_window

minutes = df.loc[mask].copy().reset_index(drop=True)

# 用相同 gap>30min 規則切段
gap = minutes["date"].diff().dt.total_seconds().div(60)
minutes["run_id"] = (gap.isna() | (gap > GAP_MINUTES)).cumsum()

# 段摘要
runs = minutes.groupby("run_id").agg(
    start=("date", "min"),
    end=("date", "max"),
    rainy_min=("date", "count"),
    max_Past1Hr=("Past1Hr", "max"),
    max_Past10Min=("Past10Min", "max"),
).reset_index()
runs["span_min"] = (runs["end"] - runs["start"]).dt.total_seconds().div(60).astype(int) + 1
# 段內有任何一筆 Past10Min>=0.1 = 這段期間真的有雨落下(不是純Past1Hr累積尾巴)
runs["has_real_rain"] = runs["max_Past10Min"] >= THRESHOLD
runs = runs[["run_id", "start", "end", "span_min", "rainy_min",
             "max_Past1Hr", "max_Past10Min", "has_real_rain"]]

# 匯出
runs.to_csv("dataset/rain_outside_segments_runs.csv", index=False)
minutes.drop(columns="run_id").assign(run_id=minutes["run_id"]) \
       .to_csv("dataset/rain_outside_segments_minutes.csv", index=False)

# 列印全部 86 段
pd.set_option("display.max_rows", None, "display.width", 140)
print(f"視窗外下雨分鐘總數: {int(mask.sum())}   形成 {len(runs)} 段")
print(f"  其中 has_real_rain=True (真有降雨落下): {int(runs['has_real_rain'].sum())} 段")
print(f"        has_real_rain=False(純Past1Hr累積尾巴): {int((~runs['has_real_rain']).sum())} 段")
print("\n=== 全部 86 段 (依時間排序) ===")
print(runs.sort_values("start").to_string(index=False))
print("\n已匯出:")
print("  dataset/rain_outside_segments_runs.csv     (段摘要, 86 列)")
print("  dataset/rain_outside_segments_minutes.csv  (逐分鐘原始值, 4933 列)")
