from pathlib import Path

import pandas as pd

INPUT_CSV = Path("dataset/wra_cogate_obs_wide.csv")
OUTPUT_CSV = Path("dataset/wra_cogate_obs_wide_gate_opening.csv")

TARGET_COLUMNS = [
    "開度_北側閘門(1)",
    "開度_北側閘門(2)",
    "開度_北側閘門(3)",
    "開度_北側閘門(4)",
    "開度_南側閘門(1)",
    "開度_南側閘門(2)",
    "開度_南側閘門(3)",
]

RENAME_MAP = {
    "date": "date",
    "開度_北側閘門(1)": "north_gate_opening_1",
    "開度_北側閘門(2)": "north_gate_opening_2",
    "開度_北側閘門(3)": "north_gate_opening_3",
    "開度_北側閘門(4)": "north_gate_opening_4",
    "開度_南側閘門(1)": "south_gate_opening_1",
    "開度_南側閘門(2)": "south_gate_opening_2",
    "開度_南側閘門(3)": "south_gate_opening_3",
}


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"找不到輸入檔案：{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_columns = ["date", *TARGET_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"輸入檔缺少欄位：{missing_columns}")

    filtered_df = df.loc[:, required_columns].copy()
    filtered_df = filtered_df.dropna(subset=TARGET_COLUMNS, how="all").copy()
    filtered_df["date"] = pd.to_datetime(filtered_df["date"], errors="coerce")
    filtered_df = filtered_df.dropna(subset=["date"]).copy()
    filtered_df["date"] = filtered_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    filtered_df = filtered_df.rename(columns=RENAME_MAP)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"輸出檔案：{OUTPUT_CSV}")
    print(f"資料列數：{len(filtered_df):,}")
    print(f"欄位數：{len(filtered_df.columns):,}")


if __name__ == "__main__":
    main()
