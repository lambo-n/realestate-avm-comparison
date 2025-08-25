# Real Estate AVM Comparison (ETL + Airflow + ML)

This project sets up a reproducible pipeline to ingest public housing datasets (e.g., Zillow Research CSVs), engineer features, train & compare automated valuation models (AVMs), and benchmark their performance over time. It’s organized as Airflow DAGs plus a small `pipelines/` package so you can run the whole workflow locally via Docker.

---

## Highlights

- **End-to-end pipeline**: ingest → clean/transform → train models → evaluate → export artifacts.
- **Airflow-orchestrated**: DAGs handle parameterized runs (e.g., by month/region) and dependency order.
- **Pluggable ML**: bring your own scikit-learn models and metrics (MAE/RMSE/SMAPE, etc.).
- **Local-first**: `docker-compose` creates an Airflow stack (webserver, scheduler, etc.) on your machine.

---

## Repository layout

.
├─ dags/ # Airflow DAG definitions (scheduling/orchestration)
├─ pipelines/ # Python modules for ETL, feature engineering, training, evaluation
├─ airflow/
│ └─ tmp/ # Local volume for logs/tmp state when running in Docker
├─ docker-compose.yaml # Local Airflow stack
└─ .gitignore


---

## Prerequisites

- **Docker Desktop** (WSL2 recommended on Windows)
- (Optional) **Python 3.10+** if you want to run modules or notebooks outside of Docker

---

## Quick start (Docker + Airflow)

1. **Clone**
   ```bash
   git clone https://github.com/lambo-n/ZestimateModelComparison.git
   cd ZestimateModelComparison
