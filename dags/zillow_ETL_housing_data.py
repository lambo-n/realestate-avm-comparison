from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import requests
import boto3
import os
from io import BytesIO
from typing import List
import pandas as pd
import sqlalchemy

# ---- CONFIGURATION ----
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TMP_DIR = "/opt/airflow/tmp"

DB_HOSTNAME = "zillow-silver-postgresql-rdb.c4fkgcky4zra.us-east-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "lambo"
DB_TABLE_NAME = "zillow-merged-data"

S3_BUCKET = "zillow-housing-data-storage"
S3_PREFIX = "raw-zillow-data/"

# ---- SOURCE FILES ----
sourceList = [
    "https://files.zillowstatic.com/research/public_csvs/invt_fs/Metro_invt_fs_uc_sfrcondo_sm_month.csv?t=1753988584",
    "https://files.zillowstatic.com/research/public_csvs/new_listings/Metro_new_listings_uc_sfrcondo_sm_month.csv?t=1753988584",
    "https://files.zillowstatic.com/research/public_csvs/mlp/Metro_mlp_uc_sfrcondo_sm_month.csv?t=1753988584",
    "https://files.zillowstatic.com/research/public_csvs/sales_count_now/Metro_sales_count_now_uc_sfrcondo_month.csv?t=1753988584",
    "https://files.zillowstatic.com/research/public_csvs/median_sale_price_now/Metro_median_sale_price_now_uc_sfrcondo_month.csv?t=1753988584",
    "https://files.zillowstatic.com/research/public_csvs/mean_sale_price_now/Metro_mean_sale_price_now_uc_sfrcondo_month.csv?t=1753988584",
    "https://files.zillowstatic.com/research/public_csvs/total_transaction_value_now/Metro_total_transaction_value_now_uc_sfrcondo_month.csv?t=1753988584",
    "https://files.zillowstatic.com/research/public_csvs/market_temp_index/Metro_market_temp_index_uc_sfrcondo_month.csv?t=1753988584",
    "https://files.zillowstatic.com/research/public_csvs/new_con_sales_count_raw/Metro_new_con_sales_count_raw_uc_sfrcondo_month.csv?t=1754063616",
    "https://files.zillowstatic.com/research/public_csvs/new_con_median_sale_price/Metro_new_con_median_sale_price_uc_sfrcondo_month.csv?t=1754063616",
    "https://files.zillowstatic.com/research/public_csvs/new_homeowner_income_needed/Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1754063616",
    "https://files.zillowstatic.com/research/public_csvs/affordable_price/Metro_affordable_price_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1754063616",
    "https://files.zillowstatic.com/research/public_csvs/years_to_save/Metro_years_to_save_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1754063616",
    "https://files.zillowstatic.com/research/public_csvs/new_homeowner_affordability/Metro_new_homeowner_affordability_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1754063616",
    "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv?t=1753988584",
]

# COLUMN NAMES
column_name_list = [
    "For_Sale_Inventory",
    "New_Listings",
    "Median_List_Price",
    "Sales_Count",
    "Median_Sale_Price",
    "Mean_Sale_Price",
    "Total_Transaction_Value",
    "Market_Heat_Index",
    "New_Construction_Sales_Count",
    "New_Construction_Median_Price",
    "New_Homeowner_Income_Needed",
    "Affordable_Home_Price",
    "Years_To_Save",
    "New_Homeowner_Affordability",
    "Zillow_Home_Value_Index",
]

JOIN_KEYS = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 'Date']


# ---- TASKS ----

@task
def fetch_and_save(source: tuple):
    url, column_name = source
    response = requests.get(url)
    response.raise_for_status()

    filename = url.split('/')[-1].split('?')[0]
    os.makedirs(TMP_DIR, exist_ok=True)
    file_path = f"{TMP_DIR}/{filename}"

    with open(file_path, "wb") as f:
        f.write(response.content)

    return (file_path, column_name)


@task
def upload_to_s3(file_info: tuple):
    local_path, column_name = file_info
    s3_key = f"{S3_PREFIX}{os.path.basename(local_path)}"

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(Filename=local_path, Bucket=S3_BUCKET, Key=s3_key)

    return (s3_key, column_name)


@task
def transform_s3_csv(s3_info: tuple):
    s3_key, column_name = s3_info
    s3 = boto3.client("s3", region_name=AWS_REGION)
    response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    df = pd.read_csv(BytesIO(response['Body'].read()))

    common_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    date_cols = [col for col in df.columns if col not in common_cols]

    melted = df.melt(
        id_vars=common_cols,
        value_vars=date_cols,
        var_name="Date",
        value_name=column_name
    )

    # group to de-duplicate by join keys
    grouped = (
        melted
        .groupby(common_cols + ["Date"], dropna=False)
        .agg({column_name: "first"})
        .reset_index()
    )

    os.makedirs(TMP_DIR, exist_ok=True)
    output_path = f"{TMP_DIR}/transformed_{column_name}.csv"
    grouped.to_csv(output_path, index=False)
    print(f"Transformed file saved to {output_path}")
    return output_path


@task
def merge_transformed_data(local_paths: List[str]):
    # Read as-is (Date remains string)
    dfs = [pd.read_csv(p) for p in local_paths]

    base = dfs[0]
    for df in dfs[1:]:
        base = pd.merge(base, df, on=JOIN_KEYS, how="outer", sort=False)

    # Normalize dtypes where appropriate (leave Date as string)
    base["RegionID"] = base["RegionID"].astype("Int64")

    # Sort (note: Date is string, so sort is lexicographic)
    base = base.sort_values(by=["SizeRank", "Date"], ascending=[True, True], na_position="first").reset_index(drop=True)

    os.makedirs(TMP_DIR, exist_ok=True)
    output_path = f"{TMP_DIR}/zillow_merged.csv"
    base.to_csv(output_path, index=False)
    print(f"Final merged CSV saved to {output_path} with shape {base.shape}")
    return output_path


@task
def upload_to_relational_db(csv_path: str):
    # Keep Date as string when uploading to Postgres
    df = pd.read_csv(csv_path)

    # Ensure pandas treats Date as object/string
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str)

    DB_PASSWORD = os.getenv("DB_PASSWORD", "El3ph4nt6757:)1AS")

    engine = sqlalchemy.create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOSTNAME}:{DB_PORT}/{DB_NAME}",
        connect_args={"sslmode": "require"}
    )

    # No dtype override -> Date stored as TEXT/VARCHAR in Postgres
    df.to_sql(
        DB_TABLE_NAME,
        engine,
        if_exists="replace",
        index=False
    )
    print(f"✅ Uploaded {df.shape[0]} rows to RDS table {DB_TABLE_NAME}")


# ---- DAG ----
with DAG(
    dag_id="zillow_ETL_housing_data",
    start_date=datetime(2025, 1, 30),
    schedule_interval="@monthly",
    catchup=False,
    tags=["zillow", "etl"],
) as dag:

    sources = list(zip(sourceList, column_name_list))

    files = fetch_and_save.expand(source=sources)
    s3_keys = upload_to_s3.expand(file_info=files)
    transformed_paths = transform_s3_csv.expand(s3_info=s3_keys)
    merged_path = merge_transformed_data(transformed_paths)
    upload_to_relational_db(merged_path)
