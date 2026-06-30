"""Quantify Past1Hr>=0.1 minutes in all_minute_wide.csv that fall OUTSIDE any
rain-segment window, and inspect the short "20-min drizzle" bursts that the
segment filter (gap>30min split, Points>1, Duration>=60min) drops.

Read-only analysis. No files written.
"""
import pandas as pd

ALL_CSV = "dataset/all_minute_wide.csv"
META_CSV = "dataset/rain_segments_meta.csv"
THRESHOLD = 0.1
GAP_MINUTES = 30          # same as RAIN_GAP_MINUTES
MIN_SEG_MINUTES = 60      # same as MIN_RAIN_SEGMENT_MINUTES

df = pd.read_csv(ALL_CSV, usecols=["date", "Past1Hr"])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

meta = pd.read_csv(META_CSV, parse_dates=["WinStart", "WinEnd", "SegmentStart", "SegmentEnd"])

# --- 1) flag each minute: inside ANY segment WINDOW (Win = segment +/- 60min) ---
in_window = pd.Series(False, index=df.index)
for _, r in meta.iterrows():
    in_window |= (df["date"] >= r["WinStart"]) & (df["date"] <= r["WinEnd"])
df["in_window"] = in_window

# also flag inside the strict segment span (SegmentStart..SegmentEnd, no buffer)
in_seg = pd.Series(False, index=df.index)
for _, r in meta.iterrows():
    in_seg |= (df["date"] >= r["SegmentStart"]) & (df["date"] <= r["SegmentEnd"])
df["in_segment"] = in_seg

rain = df["Past1Hr"] >= THRESHOLD
total_rain = int(rain.sum())
total_rows = len(df)

print("=" * 70)
print(f"資料範圍: {df['date'].min()} ~ {df['date'].max()}  (共 {total_rows:,} 分鐘)")
print(f"segment 數: {len(meta)}")
print("=" * 70)
print(f"\nPast1Hr >= {THRESHOLD} 的分鐘總數: {total_rain:,}  "
      f"({total_rain/total_rows*100:.2f}% of all minutes)")

for label, col in [("事件視窗 Win(±60min)", "in_window"), ("嚴格 segment 內", "in_segment")]:
    inside = int((rain & df[col]).sum())
    outside = total_rain - inside
    print(f"\n  以「{label}」界定:")
    print(f"    在範圍內: {inside:,} 分鐘 ({inside/total_rain*100:.2f}% of rainy)")
    print(f"    在範圍外: {outside:,} 分鐘 ({outside/total_rain*100:.2f}% of rainy)  <-- 被忽略的雨")

# --- 2) reconstruct rain runs OUTSIDE windows, grouped with gap>30min ---
print("\n" + "=" * 70)
print("把『落在事件視窗外』的下雨分鐘，用 >30min 間隔切成一段段，看時長分布")
print("=" * 70)

out = df.loc[rain & ~df["in_window"], ["date"]].copy().reset_index(drop=True)
if out.empty:
    print("沒有任何視窗外的下雨分鐘。")
else:
    gap = out["date"].diff().dt.total_seconds().div(60)
    out["new"] = gap.isna() | (gap > GAP_MINUTES)
    out["grp"] = out["new"].cumsum()
    runs = out.groupby("grp").agg(start=("date", "min"),
                                  end=("date", "max"),
                                  minutes=("date", "count"))
    # duration in wall-clock minutes (end-start), and # of rainy minutes
    runs["span_min"] = (runs["end"] - runs["start"]).dt.total_seconds().div(60).astype(int) + 1

    print(f"\n視窗外總共形成 {len(runs)} 段(用相同 gap=30min 規則)。")
    print("\n時長分布 (span = 首末下雨分鐘之間的跨度):")
    bins = [0, 5, 10, 20, 30, 45, 60, 120, 10**9]
    labels = ["<5", "5-10", "10-20", "20-30", "30-45", "45-60", "60-120", ">=120"]
    cat = pd.cut(runs["span_min"], bins=bins, labels=labels, right=False)
    dist = cat.value_counts().reindex(labels)
    for lab, n in dist.items():
        print(f"    {lab:>8} min : {n:>4} 段")

    short = runs[runs["span_min"] < MIN_SEG_MINUTES]
    print(f"\n其中 < {MIN_SEG_MINUTES}min(撐不到 segment 門檻)的短雨: "
          f"{len(short)} 段，佔視窗外總雨分鐘的 "
          f"{short['minutes'].sum()/max(1,runs['minutes'].sum())*100:.1f}%")

    # --- 3) the "20-min drizzle" examples ---
    print("\n" + "-" * 70)
    print("『一陣 ~20 分鐘小雨』範例 (span 介於 10~30 分鐘，列最早 15 筆):")
    print("-" * 70)
    drizzle = runs[(runs["span_min"] >= 10) & (runs["span_min"] < 30)].sort_values("start")
    print(f"符合的有 {len(drizzle)} 段。")
    for _, r in drizzle.head(15).iterrows():
        print(f"    {r['start']} ~ {r['end']}  span={r['span_min']:>3}min  rainy_min={r['minutes']}")
