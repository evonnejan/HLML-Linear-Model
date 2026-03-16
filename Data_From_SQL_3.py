import pandas as pd
import math
from tqdm import tqdm
from sqlalchemy import create_engine, text

DB_NAME = "HLMLDataDb_2509"

conn_str = (
    f"mssql+pyodbc://localhost/{DB_NAME}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
)

print("正在查詢資料總筆數...")
count_sql = "SELECT COUNT(*) FROM WaterLevelObsData_Alt"
total_rows = pd.read_sql(count_sql, conn_str).iloc[0, 0]
print(f"資料庫中共有 {total_rows:,} 筆資料準備下載。")

chunk_size = 50000
total_chunks = math.ceil(total_rows / chunk_size)

sql = """
select *
from WaterLevelObsData_Alt
order by measure_time, device_id
"""

chunks = []
for chunk in tqdm(pd.read_sql(sql, conn_str, chunksize=chunk_size), 
                  total=total_chunks, 
                  desc="下載進度", 
                  unit="塊"):
    chunks.append(chunk)

print("\n下水道水位資料下載完畢")
df = pd.concat(chunks, ignore_index=True)

device_map = {
    '8307': 'HL01', '8308': 'HL02', '8309': 'HL03', 
    '8310': 'HL04', '8311': 'HL05', '8312': 'HL06'
}
df['device_id'] = df['device_id'].map(device_map)

df['measure_time'] = pd.to_datetime(df['measure_time'])


print("\n開始篩選資料")
sql = """
select *
from CWARainObsData
"""
df_rain = pd.read_sql(sql, conn_str)
# print(df_rain.shape)
# df_rain.head()

df_rain["ObsTime"] = pd.to_datetime(df_rain["ObsTime"])
df_rain["Past1Hr"] = pd.to_numeric(df_rain["Past1Hr"], errors='coerce')
df_rain = df_rain.dropna(subset=["ObsTime", "Past1Hr"])
# print("After dropna:", df_rain.shape)

threshold = 0.1
df_rain_thr = df_rain[df_rain["Past1Hr"] >= threshold].copy()
# print("After applying threshold:", df_rain_thr.shape)
# df_rain_thr.head()

def segment_intervals(df, time_col='ObsTime', value_col='Past1Hr', gap_minutes=20):
    """
    df: 已經過門檻篩選的資料
    回傳: (points 帶 segment_id, segments 摘要)
    """
    df = df.sort_values([time_col]).copy()

    # 計算與上一筆的分鐘差
    dt_1stamp = (df[time_col] - df[time_col].shift(1)).dt.total_seconds().div(60)
    # 是否開新段
    df['is_new'] = df[time_col].shift(1).isna() | (dt_1stamp > gap_minutes)
    # 累計成 segment_id（依分組各自編號）
    df['segment_id'] = df['is_new'].cumsum().astype('int64')

    # 彙總每個 segment
    agg = {
        time_col: ['min', 'max', 'count']
    }
    seg = (df.groupby(['segment_id']).agg(agg))
    # 攤平欄名
    seg.columns = ['SegmentStart','SegmentEnd','Points']
    seg = seg.reset_index()
    seg['DurationMinutes'] = (seg['SegmentEnd'] - seg['SegmentStart']).dt.total_seconds().div(60).astype('int64')
    # 去除只有一筆的段
    seg = seg[seg['Points'] > 1]
    # 重新編號 segment_id
    seg['segment_id'] = range(1, len(seg) + 1)
    # 排序
    seg = seg.sort_values(['SegmentStart'])
    return seg

gap_minutes = 30
segments = segment_intervals(df_rain_thr, gap_minutes=gap_minutes)
# print(segments.shape)
long_segments = segments[segments['DurationMinutes'] >= 60]
# print(long_segments.shape)
# long_segments.sort_values(by='DurationMinutes', ascending=False).head(8)

pre_window  = pd.Timedelta(minutes=60)
post_window = pd.Timedelta(minutes=60)

long_segments['WinStart'] = long_segments['SegmentStart'] - pre_window
long_segments['WinEnd']   = long_segments['SegmentEnd']   + post_window
long_segments = long_segments.sort_values('WinStart')

df_water_filt = df.copy().sort_values(['measure_time', 'device_id'])
segments_data_list = []

for idx, row in long_segments.iterrows():
    t_start = row['WinStart']
    t_end   = row['WinEnd']
    seg_start = row['SegmentStart']
    seg_end   = row['SegmentEnd']
    seg_id  = row['segment_id']
    
    mask = (df_water_filt['measure_time'] >= t_start) & \
           (df_water_filt['measure_time'] <= t_end)
    
    sub_df = df_water_filt[mask].copy()
    
    if not sub_df.empty:
        sub_df['segment_id'] = seg_id
        sub_df['WinStart'] = t_start
        sub_df['WinEnd'] = t_end
        sub_df['SegmentStart'] = seg_start
        sub_df['SegmentEnd']   = seg_end
        
        segments_data_list.append(sub_df)

if segments_data_list:
    df_water_filt2 = pd.concat(segments_data_list, ignore_index=True)
else:
    df_water_filt2 = pd.DataFrame()
    
df_water_filt2 = df_water_filt2.drop(columns=['upload_time', 'battery', 'rssi', 'notify'])
# print(df_water_filt.shape)

def rain_label(row):
    if (row['SegmentStart'] <= row['measure_time']) and (row['measure_time'] <= row['SegmentEnd']):
        return True
    else:
        return False
    
df_water_filt2['isRain'] = df_water_filt2.apply(rain_label, axis=1)

df_wide = df_water_filt2.pivot_table(index=['measure_time', 'SegmentStart', 'SegmentEnd', 'segment_id',
                                            'WinStart', 'WinEnd', 'isRain'], columns='device_id', values='val')
# df_wide = df_wide.resample('1min').ffill()
df_wide = df_wide.reset_index()
df_wide = df_wide.rename(columns={'measure_time': 'date'})
df_wide.to_csv('dataset/water_level_all3.csv', index=False)

print("CSV 檔案儲存成功！")


