"""Build a merged water-level and rainfall dataset from SQL Server.

macOS notes:
- Direct Windows-integrated authentication is no longer used.
- The script first tries a normal ODBC connection to the SQL Server port
  exposed by Docker (`127.0.0.1:1433` by default).
- If direct ODBC is unavailable, it falls back to running `sqlcmd`
  inside the Docker container.

Environment variables you can override:
- DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
- SQL_SERVER_ODBC_DRIVER
- SQL_QUERY_MODE: auto | odbc | docker
- SQLCMD_CONTAINER
"""

from __future__ import annotations

import io
import math
import os
import subprocess
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote_plus

import pandas as pd
from tqdm import tqdm

try:
    from sqlalchemy import create_engine
except ImportError:  # pragma: no cover - only happens in incomplete environments
    create_engine = None

DB_NAME = os.getenv("DB_NAME", "HLMLDataDb_2509")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "1433"))
DB_USER = os.getenv("DB_USER", "sa")
DB_PASSWORD = os.getenv("DB_PASSWORD")
ODBC_DRIVER = os.getenv("SQL_SERVER_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")
SQL_QUERY_MODE = os.getenv("SQL_QUERY_MODE", "auto").strip().lower()
SQLCMD_CONTAINER = os.getenv("SQLCMD_CONTAINER", "sql1")

WATER_TABLE = "WaterLevelObsData_Alt"
RAIN_TABLE = "CWARainObsData"
OUTPUT_CSV = Path("dataset/water_level_rain_all4.csv")

WATER_DEVICE_MAP = {
    "8307": "HL01",
    "8308": "HL02",
    "8309": "HL03",
    "8310": "HL04",
    "8311": "HL05",
    "8312": "HL06",
}

RAIN_VALUE_CANDIDATES = ["Past1Hr", "Past10Min", "Past3Hr", "Past6Hr"]
RAIN_THRESHOLD = 0.1
RAIN_GAP_MINUTES = 30
MIN_RAIN_SEGMENT_MINUTES = 60
PRE_WINDOW_MINUTES = 60
POST_WINDOW_MINUTES = 60
CHUNK_SIZE = 50000

WATER_EXPORT_COLUMNS = ["measure_time", "device_id", "val"]
RAIN_EXPORT_COLUMNS = [
    "StationId",
    "ObsTime",
    "Past10Min",
    "Past1Hr",
    "Past3Hr",
    "Past6Hr",
    "Past12Hr",
    "Past24Hr",
    "Past2Day",
    "Past3Day",
    "Now",
]


def run_command(args: Sequence[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout


def get_db_password() -> str:
    if DB_PASSWORD:
        return DB_PASSWORD

    if SQL_QUERY_MODE == "odbc":
        raise ValueError("找不到 DB_PASSWORD，請先設定資料庫密碼。")

    try:
        output = run_command(
            [
                "docker",
                "inspect",
                SQLCMD_CONTAINER,
                "--format",
                "{{range .Config.Env}}{{println .}}{{end}}",
            ]
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ValueError("找不到 DB_PASSWORD，且無法從 Docker 容器讀取 MSSQL_SA_PASSWORD。") from exc

    for line in output.splitlines():
        if line.startswith("MSSQL_SA_PASSWORD="):
            return line.split("=", 1)[1].strip()

    raise ValueError("找不到 DB_PASSWORD，Docker 容器內也沒有 MSSQL_SA_PASSWORD。")


def build_odbc_connection_url(password: str) -> str:
    return (
        f"mssql+pyodbc://{quote_plus(DB_USER)}:{quote_plus(password)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        f"?driver={quote_plus(ODBC_DRIVER)}&Encrypt=yes&TrustServerCertificate=yes"
    )


class SQLServerClient:
    def __init__(self, *, mode: str | None = None) -> None:
        self.mode = (mode or SQL_QUERY_MODE).lower()
        self.password = get_db_password()
        self.engine = None

        if self.mode not in {"auto", "odbc", "docker"}:
            raise ValueError("SQL_QUERY_MODE 只能是 auto、odbc 或 docker。")

        if self.mode in {"auto", "odbc"}:
            self.engine = self._try_create_engine()
            if self.engine is not None:
                self.mode = "odbc"
            elif self.mode == "odbc":
                raise ValueError("ODBC 連線失敗，請確認 Docker port、帳號密碼與 ODBC Driver 設定。")

        if self.engine is None:
            self.mode = "docker"
            self._check_docker_sqlcmd()

        print(f"目前使用的 SQL 連線模式：{self.mode}")

    def _try_create_engine(self):
        if create_engine is None:
            return None

        try:
            engine = create_engine(build_odbc_connection_url(self.password))
            pd.read_sql("SELECT 1 AS ok", engine)
            return engine
        except Exception:
            return None

    def _check_docker_sqlcmd(self) -> None:
        try:
            run_command(
                [
                    "docker",
                    "exec",
                    SQLCMD_CONTAINER,
                    "/opt/mssql-tools18/bin/sqlcmd",
                    "-S",
                    "localhost",
                    "-U",
                    DB_USER,
                    "-P",
                    self.password,
                    "-C",
                    "-Q",
                    "SELECT 1 AS ok;",
                ]
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise ValueError("無法透過 Docker 容器中的 sqlcmd 連到 SQL Server。") from exc

    def count_rows(self, table_name: str) -> int:
        query = f"SELECT COUNT(*) AS total_rows FROM {table_name}"
        result = self.read_query(query, columns=["total_rows"])
        return int(result.iloc[0, 0])

    def read_query(self, query: str, *, columns: Sequence[str]) -> pd.DataFrame:
        if self.mode == "odbc":
            return pd.read_sql(query, self.engine)

        return self._read_query_via_docker(query, columns=columns)

    def _read_query_via_docker(self, query: str, *, columns: Sequence[str]) -> pd.DataFrame:
        sql = f"SET NOCOUNT ON; {query}"
        output = run_command(
            [
                "docker",
                "exec",
                SQLCMD_CONTAINER,
                "/opt/mssql-tools18/bin/sqlcmd",
                "-S",
                "localhost",
                "-U",
                DB_USER,
                "-P",
                self.password,
                "-C",
                "-d",
                DB_NAME,
                "-W",
                "-w",
                "65535",
                "-s",
                "\t",
                "-h",
                "-1",
                "-Q",
                sql,
            ]
        )

        lines = [line for line in output.splitlines() if line.strip()]
        if not lines:
            return pd.DataFrame(columns=columns)

        return pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", names=list(columns), dtype=str)


def pick_rain_value_col(df: pd.DataFrame) -> str:
    for column in RAIN_VALUE_CANDIDATES:
        if column in df.columns:
            return column

    numeric_candidates = [column for column in df.columns if column != "ObsTime"]
    if not numeric_candidates:
        raise ValueError("找不到可用的雨量欄位，請確認 CWARainObsData 的欄位結構。")
    return numeric_candidates[0]


def segment_intervals(
    df: pd.DataFrame,
    *,
    time_col: str,
    value_col: str,
    threshold: float,
    gap_minutes: int,
    min_duration_minutes: int,
) -> pd.DataFrame:
    filtered = df.loc[df[value_col] >= threshold, [time_col, value_col]].copy()
    filtered = filtered.dropna(subset=[time_col, value_col]).sort_values(time_col)

    if filtered.empty:
        return pd.DataFrame(columns=["segment_id", "SegmentStart", "SegmentEnd", "Points", "DurationMinutes"])

    dt_minutes = (filtered[time_col] - filtered[time_col].shift(1)).dt.total_seconds().div(60)
    filtered["is_new"] = filtered[time_col].shift(1).isna() | (dt_minutes > gap_minutes)
    filtered["segment_id"] = filtered["is_new"].cumsum().astype("int64")

    segments = (
        filtered.groupby("segment_id", as_index=False)
        .agg(SegmentStart=(time_col, "min"), SegmentEnd=(time_col, "max"), Points=(time_col, "count"))
    )
    segments["DurationMinutes"] = (
        (segments["SegmentEnd"] - segments["SegmentStart"]).dt.total_seconds().div(60).astype("int64")
    )
    segments = segments.loc[segments["Points"] > 1].copy()
    segments = segments.loc[segments["DurationMinutes"] >= min_duration_minutes].copy()
    segments = segments.sort_values("SegmentStart").reset_index(drop=True)
    segments["segment_id"] = range(1, len(segments) + 1)
    return segments


def build_event_windows(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return segments.copy()

    windows = segments.copy()
    windows["WinStart"] = windows["SegmentStart"] - pd.Timedelta(minutes=PRE_WINDOW_MINUTES)
    windows["WinEnd"] = windows["SegmentEnd"] + pd.Timedelta(minutes=POST_WINDOW_MINUTES)
    return windows.sort_values("WinStart").reset_index(drop=True)


def to_minute_wide(df: pd.DataFrame, *, time_col: str, value_cols: Iterable[str]) -> pd.DataFrame:
    minute_df = (
        df.set_index(time_col)
        .sort_index()
        .resample("1min")
        .ffill()
        .reset_index()
        .rename(columns={time_col: "date"})
    )
    return minute_df.loc[:, ["date", *value_cols]].copy()


def build_water_wide(water_df: pd.DataFrame) -> pd.DataFrame:
    water_wide = (
        water_df.pivot_table(index="measure_time", columns="device_id", values="val")
        .sort_index()
        .resample("1min")
        .ffill()
        .reset_index()
        .rename(columns={"measure_time": "date"})
    )
    return water_wide


def build_rain_wide(rain_df: pd.DataFrame) -> pd.DataFrame:
    rain_cols = [column for column in rain_df.columns if column != "ObsTime"]
    return to_minute_wide(rain_df, time_col="ObsTime", value_cols=rain_cols)


def label_rain_minutes(df: pd.DataFrame, *, segment_start: pd.Timestamp, segment_end: pd.Timestamp) -> pd.Series:
    return (df["date"] >= segment_start) & (df["date"] <= segment_end)


def load_water_data(client: SQLServerClient) -> pd.DataFrame:
    print("正在查詢下水道水位資料總筆數...")
    total_rows = client.count_rows(WATER_TABLE)
    print(f"水位資料共有 {total_rows:,} 筆，準備下載。")

    total_chunks = math.ceil(total_rows / CHUNK_SIZE)
    chunks: list[pd.DataFrame] = []

    for offset in tqdm(range(0, total_rows, CHUNK_SIZE), total=total_chunks, desc="下載下水道水位資料", unit="塊"):
        query = f"""
        SELECT
            CONVERT(varchar(19), measure_time, 120) AS measure_time,
            CAST(device_id AS varchar(20)) AS device_id,
            CAST(val AS varchar(50)) AS val
        FROM {WATER_TABLE}
        ORDER BY measure_time, device_id
        OFFSET {offset} ROWS FETCH NEXT {CHUNK_SIZE} ROWS ONLY
        """
        chunk = client.read_query(query, columns=WATER_EXPORT_COLUMNS)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"{WATER_TABLE} 沒有資料。")

    water_df = pd.concat(chunks, ignore_index=True)
    water_df["measure_time"] = pd.to_datetime(water_df["measure_time"], errors="coerce")
    water_df["device_id"] = water_df["device_id"].astype(str).map(WATER_DEVICE_MAP).fillna(water_df["device_id"])
    water_df["val"] = pd.to_numeric(water_df["val"], errors="coerce")
    water_df = water_df.dropna(subset=["measure_time", "device_id", "val"]).copy()
    return water_df


def load_rain_data(client: SQLServerClient) -> tuple[pd.DataFrame, str]:
    print("正在查詢雨量資料總筆數...")
    total_rows = client.count_rows(RAIN_TABLE)
    print(f"雨量資料共有 {total_rows:,} 筆，準備下載。")

    query = f"""
    SELECT
        CAST(StationId AS varchar(50)) AS StationId,
        CONVERT(varchar(19), ObsTime, 120) AS ObsTime,
        CAST(Past10Min AS varchar(50)) AS Past10Min,
        CAST(Past1Hr AS varchar(50)) AS Past1Hr,
        CAST(Past3Hr AS varchar(50)) AS Past3Hr,
        CAST(Past6Hr AS varchar(50)) AS Past6Hr,
        CAST(Past12Hr AS varchar(50)) AS Past12Hr,
        CAST(Past24Hr AS varchar(50)) AS Past24Hr,
        CAST(Past2Day AS varchar(50)) AS Past2Day,
        CAST(Past3Day AS varchar(50)) AS Past3Day,
        CAST([Now] AS varchar(50)) AS [Now]
    FROM {RAIN_TABLE}
    ORDER BY ObsTime
    """
    rain_df = client.read_query(query, columns=RAIN_EXPORT_COLUMNS)
    if rain_df.empty:
        raise ValueError(f"{RAIN_TABLE} 沒有資料。")

    rain_df["ObsTime"] = pd.to_datetime(rain_df["ObsTime"], errors="coerce")
    rain_df = rain_df.dropna(subset=["ObsTime"]).copy()

    for column in rain_df.columns:
        if column != "ObsTime" and column != "StationId":
            rain_df[column] = pd.to_numeric(rain_df[column], errors="coerce")

    rain_value_col = pick_rain_value_col(rain_df)
    rain_df = rain_df.dropna(subset=[rain_value_col]).copy()
    return rain_df, rain_value_col


def build_final_dataset(water_df: pd.DataFrame, rain_df: pd.DataFrame, rain_value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"使用雨量欄位：{rain_value_col}")
    print("開始偵測降雨事件區間...")

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

    print(f"找到 {len(rain_windows):,} 段降雨事件。")
    print("建立 1 分鐘水位與雨量資料表...")

    water_wide = build_water_wide(water_df)
    rain_wide = build_rain_wide(rain_df)
    merged_wide = water_wide.merge(rain_wide, on="date", how="left").sort_values("date").reset_index(drop=True)

    segments_data_list = []
    for _, row in rain_windows.iterrows():
        mask = (merged_wide["date"] >= row["WinStart"]) & (merged_wide["date"] <= row["WinEnd"])
        sub_df = merged_wide.loc[mask].copy()
        if sub_df.empty:
            continue

        sub_df["segment_id"] = row["segment_id"]
        sub_df["SegmentStart"] = row["SegmentStart"]
        sub_df["SegmentEnd"] = row["SegmentEnd"]
        sub_df["WinStart"] = row["WinStart"]
        sub_df["WinEnd"] = row["WinEnd"]
        sub_df["isRain"] = label_rain_minutes(sub_df, segment_start=row["SegmentStart"], segment_end=row["SegmentEnd"])
        segments_data_list.append(sub_df)

    if not segments_data_list:
        raise ValueError("降雨事件範圍內沒有可用的水位資料。")

    final_df = pd.concat(segments_data_list, ignore_index=True)

    segment_cols = ["date", "SegmentStart", "SegmentEnd", "segment_id", "WinStart", "WinEnd", "isRain"]
    other_cols = [column for column in final_df.columns if column not in segment_cols]
    final_df = final_df.loc[:, [*segment_cols, *other_cols]].sort_values(["segment_id", "date"]).reset_index(drop=True)
    return final_df, rain_windows


def main() -> None:
    client = SQLServerClient()
    water_df = load_water_data(client)
    rain_df, rain_value_col = load_rain_data(client)
    final_df, rain_windows = build_final_dataset(water_df, rain_df, rain_value_col)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("資料產生完成。")
    print(f"輸出檔案：{OUTPUT_CSV}")
    print(f"最終資料列數：{len(final_df):,}")
    print(f"欄位數：{len(final_df.columns):,}")
    print(f"事件段數：{len(rain_windows):,}")
    print(f"資料時間範圍：{final_df['date'].min()} ~ {final_df['date'].max()}")


if __name__ == "__main__":
    main()
