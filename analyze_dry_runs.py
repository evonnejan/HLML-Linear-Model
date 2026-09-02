"""Dry-run-first segmentation feasibility stats (7/2 meeting: 從沒下雨的部分出發).

Instead of detecting rain events, detect long confirmed-dry runs
(Past10Min == 0 sustained >= L hours), remove them, and keep everything else.
This script quantifies, for L = 3/4/5/6 hours:
  - how many long dry runs exist and how many minutes they remove
  - the kept-segment count / duration distribution / usability (>= seq+pred)
  - overlap vs the current Past1Hr-based rain_segments_meta windows

Read-only analysis. No files written.
"""
import pandas as pd

ALL_CSV = "dataset/all_minute_wide.csv"
META_CSV = "dataset/rain_segments_meta.csv"
L_HOURS = [3, 4, 5, 6]
SEQ_LEN = 60              # run.py default --seq_len
PRED_LEN = 15             # run.py default --pred_len
MIN_USABLE = SEQ_LEN + PRED_LEN

df = pd.read_csv(ALL_CSV, usecols=["date", "Past10Min"], parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
total_rows = len(df)

meta = pd.read_csv(META_CSV, parse_dates=["WinStart", "WinEnd", "SegmentStart", "SegmentEnd"])
in_window = pd.Series(False, index=df.index)
for _, r in meta.iterrows():
    in_window |= (df["date"] >= r["WinStart"]) & (df["date"] <= r["WinEnd"])

# NaN is "unknown", not "confirmed dry": it must break a dry run.
dry = df["Past10Min"].eq(0).fillna(False)
wet_total = int((~dry).sum())

print("=" * 70)
print(f"資料範圍: {df['date'].min()} ~ {df['date'].max()}  (共 {total_rows:,} 分鐘)")
print(f"Past10Min == 0 的分鐘: {int(dry.sum()):,} ({dry.mean()*100:.2f}%)   "
      f"有雨分鐘: {wet_total:,}")
print(f"現行 rain-segment 視窗 (Past1Hr 版, ±60min) 覆蓋: {int(in_window.sum()):,} 分鐘 "
      f"({in_window.mean()*100:.2f}%), segments={len(meta)}")
print("=" * 70)

# --- 1) maximal dry-run length distribution ---
run_id = (dry != dry.shift()).cumsum()
runs = df.groupby(run_id).agg(is_dry=("Past10Min", lambda s: bool(s.eq(0).all())),
                              start=("date", "min"), end=("date", "max"),
                              minutes=("date", "count"))
dry_runs = runs.loc[runs["is_dry"]].copy()

print("\n乾段（連續 Past10Min==0）長度分布（首尾不完整的乾段也計入）:")
bins = [0, 30, 60, 120, 180, 240, 300, 360, 720, 1440, float("inf")]
labels = ["<30m", "30-60m", "1-2h", "2-3h", "3-4h", "4-5h", "5-6h", "6-12h", "12-24h", ">24h"]
dist = pd.cut(dry_runs["minutes"], bins=bins, labels=labels, right=False).value_counts().sort_index()
summary = pd.DataFrame({"段數": dist})
summary["總分鐘"] = [int(dry_runs.loc[pd.cut(dry_runs['minutes'], bins=bins, labels=labels,
                                              right=False) == b, "minutes"].sum()) for b in labels]
print(summary.to_string())

# --- 2) for each L: remove dry runs >= L, characterize the kept segments ---
rows = []
for L in L_HOURS:
    L_min = L * 60
    long_dry = dry_runs.loc[dry_runs["minutes"] >= L_min]
    removed = pd.Series(False, index=df.index)
    removed_run_ids = set(long_dry.index)
    removed = run_id.isin(removed_run_ids)

    kept = ~removed
    kept_run_id = (kept != kept.shift()).cumsum()
    seg = (df.loc[kept].groupby(kept_run_id[kept])
             .agg(start=("date", "min"), end=("date", "max"), minutes=("date", "count")))
    usable = seg.loc[seg["minutes"] >= MIN_USABLE]

    kept_wet = int((~dry & kept).sum())
    rows.append({
        "L(h)": L,
        "長乾段數": len(long_dry),
        "剔除分鐘": int(removed.sum()),
        "剔除%": round(removed.mean() * 100, 1),
        "保留分鐘": int(kept.sum()),
        "保留%": round(kept.mean() * 100, 1),
        "seg數": len(seg),
        f"可用seg(>= {MIN_USABLE}m)": len(usable),
        "可用分鐘": int(usable["minutes"].sum()),
        "seg中位長(m)": int(seg["minutes"].median()),
        "seg最長(h)": round(seg["minutes"].max() / 60, 1),
        "雨分鐘保留率%": round(kept_wet / wet_total * 100, 1),
    })

print("\n各 L 門檻的剔除/保留統計（保留 = 全資料剔除長乾段後的殘餘）:")
print(pd.DataFrame(rows).to_string(index=False))

# --- 3) vs current pipeline: what does the new kept set add / drop? ---
print("\n與現行 rain-segment 視窗 (in_window) 的對照:")
for L in L_HOURS:
    L_min = L * 60
    removed = run_id.isin(set(dry_runs.loc[dry_runs["minutes"] >= L_min].index))
    kept = ~removed
    both = int((kept & in_window).sum())
    new_only = int((kept & ~in_window).sum())
    lost = int((~kept & in_window).sum())
    print(f"  L={L}h: 兩者皆保留 {both:,} 分鐘 | 新增(視窗外但保留) {new_only:,} 分鐘 | "
          f"現行視窗內但被新法剔除 {lost:,} 分鐘")

# --- 4) kept-segment duration distribution per L (for boundary-rule decision) ---
print("\n各 L 的保留 segment 長度分佈 (小時):")
for L in L_HOURS:
    L_min = L * 60
    removed = run_id.isin(set(dry_runs.loc[dry_runs["minutes"] >= L_min].index))
    kept = ~removed
    kept_run_id = (kept != kept.shift()).cumsum()
    seg_min = df.loc[kept].groupby(kept_run_id[kept])["date"].count()
    q = (seg_min / 60).describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(1)
    print(f"  L={L}h: n={int(q['count'])}  min={q['min']}  p25={q['25%']}  "
          f"median={q['50%']}  p75={q['75%']}  p90={q['90%']}  max={q['max']}")
