from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import get_current_context
from datetime import datetime
import os
import io
import gzip
import boto3
from pathlib import Path  # <-- NEW

from pipelines.data_pipeline import DataPipeline
from pipelines.analysis_pipeline import AnalysisPipeline
from pipelines.ml_pipeline import ML_Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import pandas as pd
import sqlalchemy

# --- Config ---
S3_BUCKET = os.getenv("S3_BUCKET")          # silver/ML artifacts bucket
S3_PREFIX = "zillow-housing-ml"
S3_REGION = "us-east-1"

# Gold layer destination
GOLD_BUCKET = os.getenv("GOLD_BUCKET", os.getenv("S3_BUCKET"))
GOLD_PREFIX = "zillow-housing-ml"

DB_HOST = os.getenv("DB_HOSTNAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "postgres")
TABLE_NAME = os.getenv("DB_TABLE_NAME", "zillow-merged-data")

PLOTS_DIR = "/opt/airflow/dags/plots"
OUTPUTS_DIR = "/opt/airflow/dags/outputs"

TEST_SIZE = 0.3
RANDOM_STATE = 42
COLUMNS_NEED_LOOKUP = ["RegionName", "StateName"]
COLUMNS_NEED_DROP = ["RegionType"]

ZHVF_COLS = {
    1:  os.getenv("ZHVF_COL_H1",  "2025-07-31"),
    3:  os.getenv("ZHVF_COL_H3",  "2025-09-30"),
    12: os.getenv("ZHVF_COL_H12", "2026-06-30"),
}

# --- Local ZHVF config (file lives next to this DAG) ---
ZHVF_FILE_NAME = os.getenv("ZHVF_FILE_NAME", "zhvf_latest.csv")
DAGS_DIR = Path(__file__).resolve().parent
ZHVF_LOCAL_PATH = os.getenv("ZHVF_LOCAL_PATH", str(DAGS_DIR / ZHVF_FILE_NAME))

def get_engine():
    return sqlalchemy.create_engine(
        f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}",
        connect_args={"sslmode": "require"}
    )

def get_s3():
    region = os.getenv("AWS_REGION", S3_REGION)
    return boto3.client("s3", region_name=region)

def _try_parquet(df) -> (bytes, str):
    try:
        import pyarrow  # noqa: F401
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue(), "parquet"
    except Exception:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="w") as gz:
            with io.TextIOWrapper(gz, encoding="utf-8") as wrapper:
                df.to_csv(wrapper, index=False)
        return buf.getvalue(), "csv.gz"

def put_df_s3(df: pd.DataFrame, bucket: str, key: str) -> str:
    s3 = get_s3()
    body, ext = _try_parquet(df)
    if not key.endswith(ext):
        key = f"{key}.{ext}"
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    return key

def get_df_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = get_s3()
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    if key.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(body))
    if key.endswith(".csv.gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(body), mode="rb") as gz:
            return pd.read_csv(gz)
    if key.endswith(".csv"):
        return pd.read_csv(io.BytesIO(body))
    try:
        return pd.read_parquet(io.BytesIO(body))
    except Exception:
        return pd.read_csv(io.BytesIO(body))

@task
def remove_old_outputs():
    def clean_dir(path):
        if not os.path.exists(path):
            print(f"Skipping missing directory: {path}")
            return
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    clean_dir(PLOTS_DIR)
    clean_dir(OUTPUTS_DIR)

@task
def load_data_table():
    return TABLE_NAME

@task
def run_data_pipeline(table_name: str):
    ctx = get_current_context()
    ts_nodash = ctx["ts_nodash"]
    base_prefix = S3_PREFIX.strip("/")
    run_prefix = f"{base_prefix}/{ctx['dag'].dag_id}/{ts_nodash}"

    engine = get_engine()
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)

    data_pipeline = DataPipeline(
        df, TEST_SIZE, RANDOM_STATE,
        lookup_columns_list=COLUMNS_NEED_LOOKUP,
        drop_columns_list=COLUMNS_NEED_DROP
    )
    X_train, X_test, y_train, y_test, cleanedDF, targetY = data_pipeline.run()

    keys = {}
    keys["X_train"]   = put_df_s3(X_train,   S3_BUCKET, f"{run_prefix}/X_train")
    keys["X_test"]    = put_df_s3(X_test,    S3_BUCKET, f"{run_prefix}/X_test")
    keys["y_train"]   = put_df_s3(pd.DataFrame({targetY: y_train}), S3_BUCKET, f"{run_prefix}/y_train")
    keys["y_test"]    = put_df_s3(pd.DataFrame({targetY: y_test}),   S3_BUCKET, f"{run_prefix}/y_test")
    keys["cleanedDF"] = put_df_s3(cleanedDF, S3_BUCKET, f"{run_prefix}/cleanedDF")

    return {
        "bucket": S3_BUCKET,
        "prefix": run_prefix,
        "keys": keys,
        "targetY": targetY,
        "n_rows": int(len(cleanedDF)),
        "n_features": int(X_train.shape[1]),
    }

@task
def run_analysis_pipeline(meta: dict):
    bucket = meta["bucket"]
    keys = meta["keys"]
    targetY = meta["targetY"]

    X_train = get_df_s3(bucket, keys["X_train"])
    X_test  = get_df_s3(bucket, keys["X_test"])
    y_train = get_df_s3(bucket, keys["y_train"])[targetY]
    y_test  = get_df_s3(bucket, keys["y_test"])[targetY]
    cleaned = get_df_s3(bucket, keys["cleanedDF"])

    analysis_pipeline = AnalysisPipeline(X_train, X_test, y_train, y_test, cleaned, targetY)
    analysis_pipeline.run()
    print("ANALYSIS PIPELINE DONE")

@task
def run_ml_pipeline(meta: dict):
    # For gold output naming
    ctx = get_current_context()
    ts_nodash = ctx["ts_nodash"]
    dag_id = ctx['dag'].dag_id

    bucket = meta["bucket"]
    keys = meta["keys"]
    targetY = meta["targetY"]

    X_train = get_df_s3(bucket, keys["X_train"])
    X_test  = get_df_s3(bucket, keys["X_test"])
    y_train = get_df_s3(bucket, keys["y_train"])[targetY]
    y_test  = get_df_s3(bucket, keys["y_test"])[targetY]
    cleaned = get_df_s3(bucket, keys["cleanedDF"])

    pipeline = ML_Pipeline(X_train, X_test, y_train, y_test, cleaned, targetY)
    pipeline.add_model("Decision Tree", DecisionTreeRegressor(max_depth=4))
    pipeline.add_model("Random Forest", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))
    pipeline.add_model("KNN", KNeighborsRegressor(n_neighbors=5))
    pipeline.add_model("Neural Network", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE))
    pipeline.add_model("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE))
    pipeline.add_model("SVR (RBF)", SVR())
    pipeline.add_model("PCA Projection", None)
    pipeline.run()

    # --- create and push forecast to Gold layer ---
    forecast_df = pipeline.perform_model_prediction(horizons=(1, 3, 12), region_id_col="RegionID")
    if forecast_df is None or forecast_df.empty:
        print("[GOLD] No forecast produced; skipping S3 write.")
        return {"gold_bucket": GOLD_BUCKET, "gold_key": None, "public_key": None}

    # 1) Save FULL (with RegionID) for internal downstream steps
    gold_key_full = f"{GOLD_PREFIX}/{dag_id}/{ts_nodash}/rf_forecast_h6_full"
    saved_key_full = put_df_s3(forecast_df, GOLD_BUCKET, gold_key_full)
    print(f"[GOLD] Forecast (full) saved to s3://{GOLD_BUCKET}/{saved_key_full}")

    # 2) Save SANITIZED (drop RegionID) for external/consumer use
    drop_candidates = ["RegionID", "region_id", "regionid"]
    sanitized = forecast_df.drop(columns=[c for c in drop_candidates if c in forecast_df.columns], errors="ignore")
    gold_key_public = f"{GOLD_PREFIX}/{dag_id}/{ts_nodash}/rf_forecast_h6"
    saved_key_public = put_df_s3(sanitized, GOLD_BUCKET, gold_key_public)
    print(f"[GOLD] Forecast (sanitized) saved to s3://{GOLD_BUCKET}/{saved_key_public}")

    # Return the FULL key so `calculate_conclusions` can still join on RegionID
    return {"gold_bucket": GOLD_BUCKET, "gold_key": saved_key_full, "public_key": saved_key_public}

@task
def calculate_conclusions(meta: dict, model_out: dict, region_id_col: str = "RegionID", zhfv_path: str = ZHVF_LOCAL_PATH):
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # --- inputs/keys ---
    cleaned_key = meta["keys"]["cleanedDF"]
    silver_bucket = meta["bucket"]
    targetY = meta["targetY"]

    gold_bucket = model_out.get("gold_bucket") or silver_bucket
    forecast_key = model_out.get("gold_key")
    if not forecast_key:
        print("[WARN] No forecast key from model stage; cannot compute conclusions.")
        return None

    # --- load model forecast (levels) ---
    model_df = get_df_s3(gold_bucket, forecast_key)

    # --- load local ZHVF (% change columns named by dates) ---
    zh_path = Path(zhfv_path)
    if not zh_path.exists():
        raise FileNotFoundError(f"ZHVF file not found at {zh_path}")
    zh_df = pd.read_parquet(zh_path) if zh_path.suffix.lower() == ".parquet" else pd.read_csv(zh_path)

    # --- normalize RegionID column name if needed ---
    if region_id_col not in model_df.columns:
        for cand in ["regionid", "region_id", "RegionID"]:
            if cand in model_df.columns:
                region_id_col = cand
                break
    if region_id_col not in model_df.columns or region_id_col not in zh_df.columns:
        raise ValueError(f"Region ID column not found in both dataframes. "
                         f"Model cols: {model_df.columns.tolist()} | ZHVF cols: {zh_df.columns.tolist()}")

    # Ensure same dtype for join
    model_df[region_id_col] = model_df[region_id_col].astype(str)
    zh_df[region_id_col] = zh_df[region_id_col].astype(str)

    # --- required model columns ---
    base_col = f"{targetY}_last"  # e.g., Zillow_Home_Value_Index_last
    need_model_cols = [base_col] + [f"{targetY}_pred_t_plus_{h}" for h in (1, 3, 12)]
    missing = [c for c in need_model_cols if c not in model_df.columns]
    if missing:
        raise ValueError(f"Model forecast table missing required columns: {missing}")

    # --- required ZHVF columns (hardcoded) ---
    horizons = [1, 3, 12]
    zh_missing = [ZHVF_COLS[h] for h in horizons if ZHVF_COLS[h] not in zh_df.columns]
    if zh_missing:
        raise ValueError(f"ZHVF file missing required date columns: {zh_missing}")

    # --- MASE scaling from historical cleanedDF (seasonal naive m=12; fallback m=1) ---
    cleaned = get_df_s3(silver_bucket, cleaned_key)
    if targetY not in cleaned.columns or "Date_months_since_2000" not in cleaned.columns:
        raise ValueError("cleanedDF must contain target and Date_months_since_2000.")

    def seasonal_naive_mae(df_hist: pd.DataFrame, m: int = 12) -> float:
        errs = []
        use_cols = [region_id_col, "Date_months_since_2000", targetY]
        if any(c not in df_hist.columns for c in use_cols):
            return np.nan
        df_hist = df_hist.dropna(subset=use_cols)
        for _, g in df_hist.sort_values("Date_months_since_2000").groupby(region_id_col):
            y = g[targetY].to_numpy()
            if len(y) <= m:
                continue
            e = np.abs(y[m:] - y[:-m])
            e = e[np.isfinite(e)]
            if e.size:
                errs.append(np.mean(e))
        return float(np.mean(errs)) if errs else np.nan

    scale = seasonal_naive_mae(cleaned, m=12)
    if not np.isfinite(scale) or scale == 0:
        alt = seasonal_naive_mae(cleaned, m=1)
        scale = alt if (np.isfinite(alt) and alt > 0) else 1.0
        print(f"[INFO] Using fallback MASE scale={scale}")

    # --- align regions first ---
    common = set(model_df[region_id_col]).intersection(set(zh_df[region_id_col]))
    model_df = model_df[model_df[region_id_col].isin(common)].copy()
    zh_df = zh_df[zh_df[region_id_col].isin(common)].copy()

    rows = []
    eps = 1e-9

    for h in horizons:
        mcol = f"{targetY}_pred_t_plus_{h}"
        zcol = ZHVF_COLS[h]  # hardcoded ZHVF % change column for this horizon

        merged = pd.merge(
            model_df[[region_id_col, base_col, mcol]],
            zh_df[[region_id_col, zcol]],
            on=region_id_col,
            how="inner",
        ).dropna()

        if merged.empty:
            print(f"[WARN] No overlapping rows for horizon {h}; skipping.")
            continue

        # Convert % change -> level forecast using your base level
        zh_pct = merged[zcol].astype(float) / 100.0
        zh_level = merged[base_col].astype(float) * (1.0 + zh_pct)

        a = merged[mcol].astype(float).to_numpy()  # your model level forecast
        b = zh_level.to_numpy()                    # Zillow level forecast
        diff = a - b

        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        smape = float(np.mean(2.0 * np.abs(diff) / (np.abs(a) + np.abs(b) + eps)) * 100.0)
        mase = float(mae / (scale + eps))

        rows.append({
            "Horizon_Months": h,
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE_pct": smape,
            "MASE": mase,
            "N_Regions": int(len(merged))
        })

    if not rows:
        raise RuntimeError("No horizons could be evaluated; confirm ZHVF hardcoded columns and model forecast columns.")

    out_df = pd.DataFrame(rows).sort_values("Horizon_Months").reset_index(drop=True)

    # --- save locally ---
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    local_csv = os.path.join(OUTPUTS_DIR, "forecast_error_summary.csv")
    local_html = os.path.join(OUTPUTS_DIR, "forecast_error_summary.html")
    out_df.to_csv(local_csv, index=False)

    # Round ONLY the metric columns to 4 decimals for the HTML table
    metrics_cols = [c for c in ["MAE", "RMSE", "sMAPE_pct", "MASE"] if c in out_df.columns]
    df_display = out_df.copy()
    for c in metrics_cols:
        df_display[c] = df_display[c].astype(float).map(lambda x: f"{x:.4f}")  # one-ten-thousandth
    # keep integer-ish columns as-is (no .0000)
    intish = [c for c in ["Horizon_Months", "N_Regions"] if c in df_display.columns]
    for c in intish:
        df_display[c] = df_display[c].astype(int)

    try:
        import plotly.graph_objects as go
        fig = go.Figure(
            data=[go.Table(
                header=dict(values=list(df_display.columns), fill_color="paleturquoise", align="left"),
                cells=dict(values=[df_display[c] for c in df_display.columns], fill_color="lavender", align="left"),
            )]
        )
        fig.update_layout(title="Model vs ZHVF – Error Summary")
        fig.write_html(local_html)
    except Exception as e:
        print(f"[WARN] Could not write HTML summary: {e}")

    print(f"[INFO] Wrote local summary: {local_csv}")
    # --- save to S3 alongside your forecast ---
    s3_key_base = forecast_key.rsplit("/", 1)[0]
    saved_key = put_df_s3(out_df, gold_bucket, f"{s3_key_base}/forecast_error_summary")
    print(f"[GOLD] Error summary saved to s3://{gold_bucket}/{saved_key}")

    return {"summary_bucket": gold_bucket, "summary_key": saved_key, "local_csv": local_csv}


with DAG(
    dag_id="zillow_ML_housing_data",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "zillow"]
) as dag:
    cleanup = remove_old_outputs()
    table_ref = load_data_table()
    meta = run_data_pipeline(table_ref)
    analysis = run_analysis_pipeline(meta)
    modeling = run_ml_pipeline(meta)
    conclusions = calculate_conclusions(meta, modeling)

    cleanup >> table_ref >> meta >> [analysis, modeling] >> conclusions
