"""Build minute-level wide gate data from WraCoGate tables."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from Data_From_SQL_4 import SQLServerClient

STATIONS_TABLE = "WraCoGateStations"
OBS_TABLE = "WraCoGateObsData"
CHUNK_SIZE = 100000

MAPPING_OUTPUT_CSV = Path("dataset/wra_cogate_pqid_fullname_map.csv")
LONG_OUTPUT_CSV = Path("dataset/wra_cogate_obs_long.csv")
WIDE_OUTPUT_CSV = Path("dataset/wra_cogate_obs_wide.csv")

MAPPING_COLUMNS = ["PqId", "FullName"]
OBS_COLUMNS = ["ObsTime", "PqId", "FullName", "ObsValue"]


def build_gate_base_cte() -> str:
    return f"""
    WITH mapping AS (
        SELECT DISTINCT
            CAST(PQ_id AS varchar(50)) AS PqId,
            CAST(FullName AS nvarchar(255)) AS FullName
        FROM {STATIONS_TABLE}
        WHERE PQ_id IS NOT NULL
          AND FullName IS NOT NULL
    ),
    dedup_obs AS (
        SELECT DISTINCT
            CAST(PqId AS varchar(50)) AS PqId,
            CAST(ObsTime AS datetime2(3)) AS ObsTime,
            TRY_CAST(ObsValue AS float) AS ObsValue
        FROM {OBS_TABLE}
        WHERE PqId IS NOT NULL
          AND ObsTime IS NOT NULL
          AND TRY_CAST(ObsValue AS float) IS NOT NULL
    )
    """


def build_mapping_query() -> str:
    return f"""
    SELECT DISTINCT
        CAST(PQ_id AS varchar(50)) AS PqId,
        CAST(FullName AS nvarchar(255)) AS FullName
    FROM {STATIONS_TABLE}
    WHERE PQ_id IS NOT NULL
      AND FullName IS NOT NULL
    ORDER BY FullName, PqId
    """


def build_gate_obs_count_query() -> str:
    return (
        build_gate_base_cte()
        + """
    SELECT COUNT(*) AS total_rows
    FROM dedup_obs
    """
    )


def build_gate_obs_query(*, offset: int | None = None, fetch: int | None = None) -> str:
    order_and_page = "ORDER BY ObsTime, PqId"
    if offset is not None and fetch is not None:
        order_and_page += f" OFFSET {offset} ROWS FETCH NEXT {fetch} ROWS ONLY"

    return (
        build_gate_base_cte()
        + f"""
    SELECT
        CONVERT(varchar(23), ObsTime, 121) AS ObsTime,
        CAST(dedup_obs.PqId AS varchar(50)) AS PqId,
        CAST(mapping.FullName AS nvarchar(255)) AS FullName,
        CAST(ObsValue AS varchar(50)) AS ObsValue
    FROM dedup_obs
    LEFT JOIN mapping
        ON dedup_obs.PqId = mapping.PqId
    {order_and_page}
    """
    )


def load_mapping(client: SQLServerClient) -> pd.DataFrame:
    mapping_df = client.read_query(build_mapping_query(), columns=MAPPING_COLUMNS)
    if mapping_df.empty:
        raise ValueError(f"{STATIONS_TABLE} 沒有可用的 PqId / FullName 對照資料。")

    mapping_df = mapping_df.dropna(subset=["PqId", "FullName"]).drop_duplicates().reset_index(drop=True)
    return mapping_df


def validate_mapping(mapping_df: pd.DataFrame) -> None:
    pq_dup_count = int(mapping_df.duplicated(subset=["PqId"]).sum())
    name_dup_count = int(mapping_df.duplicated(subset=["FullName"]).sum())

    if pq_dup_count or name_dup_count:
        raise ValueError(
            f"PqId / FullName 對照不是一對一，PqId 重複 {pq_dup_count} 筆，FullName 重複 {name_dup_count} 筆。"
        )


def load_gate_obs(client: SQLServerClient) -> pd.DataFrame:
    print("正在查詢閘門觀測資料筆數...")
    total_rows = client.read_query(build_gate_obs_count_query(), columns=["total_rows"]).iloc[0, 0]
    total_rows = int(total_rows)
    print(f"分鐘級觀測資料共有 {total_rows:,} 筆，準備下載。")

    if client.mode == "odbc":
        chunks = []
        query = build_gate_obs_query()
        chunk_iter = pd.read_sql(query, client.engine, chunksize=CHUNK_SIZE)
        total_chunks = math.ceil(total_rows / CHUNK_SIZE)
        for chunk in tqdm(chunk_iter, total=total_chunks, desc="下載閘門分鐘資料", unit="塊"):
            chunks.append(chunk)
    else:
        chunks = []
        total_chunks = math.ceil(total_rows / CHUNK_SIZE)
        for offset in tqdm(range(0, total_rows, CHUNK_SIZE), total=total_chunks, desc="下載閘門分鐘資料", unit="塊"):
            chunk = client.read_query(
                build_gate_obs_query(offset=offset, fetch=CHUNK_SIZE),
                columns=OBS_COLUMNS,
            )
            if not chunk.empty:
                chunks.append(chunk)

    if not chunks:
        raise ValueError(f"{OBS_TABLE} 沒有可用的分鐘級觀測資料。")

    obs_df = pd.concat(chunks, ignore_index=True)
    obs_df["ObsTime"] = pd.to_datetime(obs_df["ObsTime"], errors="coerce")
    obs_df["ObsValue"] = pd.to_numeric(obs_df["ObsValue"], errors="coerce")
    obs_df = obs_df.dropna(subset=["ObsTime", "PqId", "FullName", "ObsValue"]).copy()
    obs_df = obs_df.sort_values(["ObsTime", "PqId"]).reset_index(drop=True)
    return obs_df


def build_wide_table(obs_df: pd.DataFrame) -> pd.DataFrame:
    minute_counts = obs_df.groupby(["ObsTime", "FullName"]).size()
    duplicate_count = int((minute_counts > 1).sum())
    if duplicate_count:
        raise ValueError(
            "目前資料在相同 ObsTime 與 FullName 仍有重複，無法直接 pivot 成 wide table。"
            f" 請先決定篩選或聚合規則，目前共有 {duplicate_count:,} 組重複分鐘資料。"
        )

    merged = obs_df[["ObsTime", "FullName", "ObsValue"]].copy()
    wide_df = (
        merged.pivot(index="ObsTime", columns="FullName", values="ObsValue")
        .sort_index()
    )
    wide_df.index.name = "date"
    wide_df = wide_df.reset_index()
    return wide_df


def main() -> None:
    client = SQLServerClient()

    print("建立 PqId -> FullName 對照表...")
    mapping_df = load_mapping(client)
    validate_mapping(mapping_df)
    MAPPING_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(MAPPING_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    obs_df = load_gate_obs(client)
    merged_obs_df = obs_df.merge(mapping_df, on=["PqId", "FullName"], how="inner")
    merged_obs_df.to_csv(LONG_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    try:
        wide_df = build_wide_table(merged_obs_df)
        wide_df.to_csv(WIDE_OUTPUT_CSV, index=False, encoding="utf-8-sig")
        wide_rows = len(wide_df)
        wide_cols = len(wide_df.columns)
        wide_range = f"{wide_df['date'].min()} ~ {wide_df['date'].max()}"
    except ValueError as exc:
        wide_df = None
        wide_rows = None
        wide_cols = None
        wide_range = str(exc)

    print("資料產生完成。")
    print(f"對照表輸出：{MAPPING_OUTPUT_CSV}")
    print(f"長表輸出：{LONG_OUTPUT_CSV}")
    print(f"分鐘 wide table 輸出：{WIDE_OUTPUT_CSV}")
    print(f"對照表筆數：{len(mapping_df):,}")
    print(f"長表筆數：{len(merged_obs_df):,}")
    if wide_df is not None:
        print(f"wide table 列數：{wide_rows:,}")
        print(f"wide table 欄數：{wide_cols:,}")
        print(f"時間範圍：{wide_range}")
    else:
        print(f"wide table 尚未輸出：{wide_range}")


if __name__ == "__main__":
    main()
