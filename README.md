# Real Estate AVM Comparison

A reproducible, end-to-end machine learning pipeline that ingests public Zillow housing datasets, engineers features, trains multiple regression models, and benchmarks their performance against Zillow's official Home Value Forecasts (ZHVF). The pipeline is orchestrated with Apache Airflow and designed to run locally via Docker on Windows.

---

## What This Does

This project answers the question: *how well can off-the-shelf ML models predict housing values compared to Zillow's own forecasts?*

It works in two phases:

**Phase 1 — ETL** (`zillow_ETL_housing_data` DAG): Pulls 15 Zillow Research CSV datasets (inventory, listings, prices, sales, affordability, ZHVI, etc.), transforms them from wide format to long format, merges them into a single relational table, and stores everything in AWS S3 (raw) and AWS RDS PostgreSQL (processed).

**Phase 2 — ML** (`zillow_ML_housing_data` DAG): Loads the merged dataset, cleans and encodes features, trains six regression models, generates EDA visualizations, and benchmarks Random Forest forecasts at 1-, 3-, and 12-month horizons against Zillow's published ZHVF predictions using MAE, RMSE, sMAPE, and MASE.

### Models Trained

| Model | Key Parameters |
|---|---|
| Decision Tree | max_depth=4 |
| Random Forest | n_estimators=200 |
| K-Nearest Neighbors | n_neighbors=5 |
| Neural Network (MLP) | hidden layers: 64→32, max_iter=300 |
| Gradient Boosting | n_estimators=200 |
| Support Vector Regressor | RBF kernel |

### Data Sources (15 Zillow Research Datasets)

- For-Sale Inventory
- New Listings
- Median & Mean List Price
- Sales Count & Median/Mean Sale Price
- Total Transaction Value
- Market Heat Index
- New Construction Sales Count & Median Price
- New Homeowner Income Needed
- Affordable Home Price
- Years To Save
- New Homeowner Affordability
- Zillow Home Value Index (ZHVI) — prediction target

---

## Repository Layout

```
.
├── dags/
│   ├── zillow_ETL_housing_data.py          # ETL DAG: fetch → S3 → transform → merge → RDS
│   ├── zillow_ML_housing pipeline_data.py  # ML DAG: load → clean → EDA → train → evaluate
│   ├── zhvf_latest.csv                     # Zillow Home Value Forecast reference file
│   ├── plots/                              # Generated EDA visualizations (HTML)
│   └── outputs/                            # Model results, forecasts, error summaries
├── pipelines/
│   ├── data_pipeline.py                    # DataPipeline: cleaning, encoding, train/test split
│   ├── analysis_pipeline.py                # AnalysisPipeline: EDA plots
│   └── ml_pipeline.py                      # ML_Pipeline: training, evaluation, forecasting
├── docker-compose.yaml                     # Airflow stack (webserver, scheduler, worker, Redis, Postgres)
└── .gitignore
```

---

## Prerequisites

- **Docker Desktop** with WSL2 backend enabled (tested on Windows)
- **Apache Airflow 2.5.1** — provided via Docker image in `docker-compose.yaml`; no separate local install required
- **AWS account** with:
  - An S3 bucket named `zillow-housing-data-storage`
  - An RDS PostgreSQL instance accessible from your machine (connection details are hardcoded in the ETL DAG — update them to match your setup)
  - IAM credentials with read/write access to S3 and RDS

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/lambo-n/ZestimateModelComparison.git
cd ZestimateModelComparison
```

### 2. Create a `.env` file

Create a `.env` file in the project root with your AWS credentials:

```env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
```

These are injected into all Airflow containers via `docker-compose.yaml`. Do not commit this file — it is already in `.gitignore`.

### 3. Configure your AWS RDS connection

Open `dags/zillow_ETL_housing_data.py` and update the database connection block near the top of the file to match your RDS instance:

```python
DB_HOSTNAME = "your-rds-endpoint.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "your_db_user"
DB_TABLE_NAME = "zillow-merged-data"
```

Set the `DB_PASSWORD` environment variable in your `.env` file:

```env
DB_PASSWORD=your_db_password
```

### 4. Create required local directories

Airflow expects these directories to exist before startup:

```bash
mkdir -p logs plugins
```

On Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force -Path logs, plugins
```

### 5. Start the Airflow stack

```bash
docker-compose up
```

This starts the following services:
- **airflow-webserver** — UI at `http://localhost:8080`
- **airflow-scheduler** — watches for DAG runs
- **airflow-worker** — Celery worker that executes tasks
- **airflow-triggerer** — handles deferred tasks
- **postgres** — Airflow metadata database (separate from your RDS instance)
- **redis** — Celery task broker

Default Airflow login: `airflow` / `airflow`

First startup takes a few minutes while `airflow-init` runs database migrations and creates the admin user.

### 6. Install Python dependencies inside the containers

The DAGs require additional Python packages (boto3, scikit-learn, plotly, sqlalchemy, etc.). Add them to the `_PIP_ADDITIONAL_REQUIREMENTS` variable in your `.env`:

```env
_PIP_ADDITIONAL_REQUIREMENTS=boto3 scikit-learn plotly sqlalchemy pandas numpy scipy psycopg2-binary
```

Then restart:

```bash
docker-compose down && docker-compose up
```

---

## Running the Pipelines

### ETL Pipeline

1. Open the Airflow UI at `http://localhost:8080`
2. Enable the `zillow_ETL_housing_data` DAG (toggle on)
3. Trigger it manually or wait for its monthly schedule
4. Tasks run in parallel where possible: fetch → upload to S3 → transform → merge → load to RDS

### ML Pipeline

Run this after the ETL pipeline has successfully populated the RDS table.

1. Enable the `zillow_ML_housing_data` DAG
2. Trigger it manually (it has no automatic schedule)
3. Results are written to:
   - `dags/plots/` — 44 interactive HTML EDA plots
   - `dags/outputs/model_results_table.html` — R², MAE, RMSE for all 6 models
   - `dags/outputs/rf_forecast_h1_3_12_wide.csv` — Random Forest forecasts at 1, 3, 12 months
   - `dags/outputs/forecast_error_summary.csv` — Benchmark vs. Zillow ZHVF

---

## Outputs

| File | Description |
|---|---|
| `dags/outputs/model_results_table.html` | Model comparison: R², MAE, RMSE for all 6 models |
| `dags/outputs/rf_forecast_h1_3_12_wide.csv` | Random Forest predictions per region at 1, 3, 12-month horizons |
| `dags/outputs/forecast_error_summary.csv` | MAE, RMSE, sMAPE, MASE comparing RF forecasts vs. Zillow |
| `dags/plots/*.html` | Histograms, violin plots, scatter matrix, correlation heatmap, PCA projection |

---

## Notes

- **ZHVF reference columns are hardcoded** in the ML DAG. If you update `dags/zhvf_latest.csv` to a newer Zillow forecast file, update the `ZHVF_COL_H1`, `ZHVF_COL_H3`, and `ZHVF_COL_H12` variables in the DAG to match the new date column names.
- The ETL DAG **replaces** the RDS table on each run (idempotent). Re-running it will overwrite existing data.
- The `pipelines/` package is mounted into the container at `/opt/airflow/pipelines` and added to `PYTHONPATH`, so DAGs can import from it directly without packaging.
- Tested on **Docker Desktop with WSL2 on Windows**.
