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

print("\n資料下載完畢")
df = pd.concat(chunks, ignore_index=True)

device_map = {
    '8307': 'HL01', '8308': 'HL02', '8309': 'HL03', 
    '8310': 'HL04', '8311': 'HL05', '8312': 'HL06'
}
df['device_id'] = df['device_id'].map(device_map)

df['measure_time'] = pd.to_datetime(df['measure_time'])

df_wide = df.pivot_table(index='measure_time', columns='device_id', values='val')
df_wide = df_wide.resample('1min').ffill()
df_wide = df_wide.reset_index()
df_wide = df_wide.rename(columns={'measure_time': 'date'})
df_wide.to_csv('dataset/water_level_all.csv', index=False)

print("CSV 檔案儲存成功！")