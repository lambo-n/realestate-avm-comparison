import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils.multiclass import type_of_target

import plotly.express as px
import plotly.graph_objects as go


class ML_Pipeline:
    def __init__(self, X_train, X_test, y_train, y_test, cleanedDF, targetY, outputs_dir=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cleanedDF = cleanedDF
        self.targetY = targetY

        self.bestModel = None
        self.bestModelName = None

        # Resolve output directory
        default_outputs = Path(__file__).resolve().parents[1] / "dags" / "outputs"
        resolved = Path(os.getenv("OUTPUTS_DIR", outputs_dir or default_outputs))
        resolved.mkdir(parents=True, exist_ok=True)
        self.output_dir = resolved

        print(f"[ML_Pipeline] CWD: {os.getcwd()}")
        print(f"[ML_Pipeline] Output directory: {self.output_dir}")

        self.models = []

    def add_model(self, name, model):
        self.models.append((name, model))

    def train_and_evaluate(self):
        results = []

        y_type = type_of_target(self.y_train)
        is_regression = "continuous" in y_type

        for name, model in self.models:
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"Evaluating {name}...")

            y_pred = model.predict(self.X_test)

            if is_regression:
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                rmse = mean_squared_error(self.y_test, y_pred, squared=False)
                results.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
            else:
                # (classification branch kept for completeness)
                from sklearn.metrics import accuracy_score, roc_auc_score
                from sklearn.preprocessing import label_binarize
                acc = accuracy_score(self.y_test, y_pred)

                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(self.X_test)
                elif hasattr(model, "decision_function"):
                    df = model.decision_function(self.X_test)
                    df_min = df.min(axis=1, keepdims=True)
                    df_max = df.max(axis=1, keepdims=True)
                    denom = np.where((df_max - df_min) == 0, 1, (df_max - df_min))
                    y_proba = (df - df_min) / denom
                    if y_proba.ndim == 1:
                        y_proba = np.column_stack([1 - y_proba, y_proba])
                else:
                    y_proba = None

                auc = None
                if y_proba is not None:
                    classes_sorted = np.unique(self.y_test)
                    class_count = len(classes_sorted)
                    if class_count == 2:
                        if y_proba.ndim == 1:
                            y_proba = np.column_stack([1 - y_proba, y_proba])
                        auc = roc_auc_score(self.y_test, y_proba[:, 1])
                    else:
                        y_true_bin = label_binarize(self.y_test, classes=classes_sorted)
                        if y_proba.ndim == 2 and y_proba.shape[1] == class_count:
                            auc = roc_auc_score(y_true_bin, y_proba, multi_class="ovr")

                row = {"Model": name, "Accuracy": acc}
                if auc is not None:
                    row["AUC"] = auc
                results.append(row)

            print(f"Metrics saved for {name}\n")

            # Choose Random Forest as the "best" model for forecasting
            if name == "Random Forest":
                self.bestModel = model
                self.bestModelName = name

        self.results_df = pd.DataFrame(results)
        print("\nModel Evaluation Summary:\n")
        print(self.results_df)

    def _numeric_features_only(self):
        features = self.cleanedDF.drop(columns=self.targetY).copy()
        for c in features.columns:
            if is_numeric_dtype(features[c]):
                features[c] = pd.to_numeric(features[c], errors='coerce')
        non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"[ML_Pipeline] Dropping non-numeric columns for PCA/t-SNE: {non_numeric}")
            features = features.drop(columns=non_numeric)
        features = features.astype("float64")
        return features

    def display_results_table(self):
        if not hasattr(self, "results_df"):
            print("No results to display. Run training first.")
            return

        df = self.results_df.copy()
        if "R2" in df.columns:
            df = df.sort_values(by="R2", ascending=False, na_position="last").reset_index(drop=True)
        else:
            print("[WARN] 'R2' column not found; leaving original order.")
        df_display = df.round(4)

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=list(df_display.columns), fill_color="paleturquoise", align="left"),
                    cells=dict(values=[df_display[col] for col in df_display.columns], fill_color="lavender", align="left"),
                )
            ]
        )
        fig.update_layout(title="Model Evaluation Summary (sorted by R2)")

        table_path = self.output_dir / "model_results_table.html"
        fig.write_html(str(table_path))
        print(f"[INFO] Results table saved: {table_path}")

    def run_pca_projection(self):
        features = self._numeric_features_only()
        if features.empty:
            print("All features must be numeric for PCA/t-SNE.")
            return
        labels = self.cleanedDF[self.targetY]
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        pca_df = features.copy()
        pca_df["PC1"] = pca_result[:, 0]
        pca_df["PC2"] = pca_result[:, 1]
        pca_df[self.targetY] = labels

        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color=self.targetY,
            title="2D PCA Projection", labels={"color": self.targetY}, opacity=0.7,
        )
        fig_pca.update_layout(
            coloraxis_colorbar=dict(
                tickmode="array",
                ticktext=sorted(self.cleanedDF[self.targetY].unique()),
                tickvals=sorted(self.cleanedDF[self.targetY].unique()),
            )
        )
        pca_path = self.output_dir / "pca_2d_projection.html"
        fig_pca.write_html(str(pca_path))
        print(f"Saved PCA plot: {pca_path}")

    def run_tsne_projection(self):
        features = self._numeric_features_only()
        if features.empty:
            print("All features must be numeric for PCA/t-SNE.")
            return
        labels = self.cleanedDF[self.targetY]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
        tsne_result = tsne.fit_transform(features)
        tsne_df = features.copy()
        tsne_df["TSNE1"] = tsne_result[:, 0]
        tsne_df["TSNE2"] = tsne_result[:, 1]
        tsne_df[self.targetY] = labels

        fig_tsne = px.scatter(
            tsne_df, x="TSNE1", y="TSNE2", color=self.targetY,
            title="2D t-SNE Projection", labels={"color": self.targetY}, opacity=0.7,
        )
        fig_tsne.update_layout(
            coloraxis_colorbar=dict(
                tickmode="array",
                ticktext=sorted(self.cleanedDF[self.targetY].unique()),
                tickvals=sorted(self.cleanedDF[self.targetY].unique()),
            )
        )
        tsne_path = self.output_dir / "tsne_2d_projection.html"
        fig_tsne.write_html(str(tsne_path))
        print(f"Saved t-SNE plot: {tsne_path}\n")

    def perform_model_prediction(
        self,
        horizons=(1, 3, 12),
        region_id_col: str = "RegionID",
        return_long: bool = False,
    ):
        # Ensure best model is RF
        if self.bestModel is None or self.bestModelName != "Random Forest":
            print("[WARN] Random Forest is not set as bestModel. Make sure it's added and trained.")
            for name, model in self.models:
                if name == "Random Forest":
                    self.bestModel = model
                    self.bestModelName = name
                    break
            if self.bestModel is None:
                print("[ERROR] Random Forest model not found. Aborting forecast.")
                return None

        # Validate horizons
        try:
            horizons = [int(h) for h in horizons if int(h) > 0]
        except Exception:
            print("[ERROR] 'horizons' must be an iterable of positive integers.")
            return None
        if not horizons:
            print("[ERROR] No valid horizons provided.")
            return None

        # Expected feature columns
        x_cols = list(self.X_train.columns)
        if "Date_months_since_2000" not in x_cols:
            print("[ERROR] 'Date_months_since_2000' is required in features for horizon shifting.")
            return None
        if region_id_col not in self.cleanedDF.columns:
            print(f"[ERROR] '{region_id_col}' column not found in cleanedDF.")
            return None
        if "Date_months_since_2000" not in self.cleanedDF.columns:
            print("[ERROR] cleanedDF is missing 'Date_months_since_2000'.")
            return None

        # Latest row per region
        df = self.cleanedDF.copy().sort_values([region_id_col, "Date_months_since_2000"])
        idx = df.groupby(region_id_col)["Date_months_since_2000"].idxmax()
        last_rows = df.loc[idx].copy()

        # Prepare predictions for each horizon
        preds = {}
        fcast_months = {}
        for h in horizons:
            X_future = last_rows[x_cols].copy()
            X_future["Date_months_since_2000"] = X_future["Date_months_since_2000"] + h
            X_future = X_future.reindex(columns=x_cols, fill_value=0)
            preds[h] = self.bestModel.predict(X_future)
            fcast_months[h] = X_future["Date_months_since_2000"].values

        # Build output (wide by default)
        base = pd.DataFrame({
            region_id_col: last_rows[region_id_col].values,
            "last_months_since_2000": last_rows["Date_months_since_2000"].values,
            f"{self.targetY}_last": last_rows[self.targetY].values if self.targetY in last_rows.columns else np.nan,
        })

        if return_long:
            # one row per horizon
            records = []
            for h in horizons:
                records.append(pd.DataFrame({
                    region_id_col: base[region_id_col],
                    "last_months_since_2000": base["last_months_since_2000"],
                    "forecast_months_since_2000": fcast_months[h],
                    "horizon_months": h,
                    f"{self.targetY}_last": base[f"{self.targetY}_last"],
                    f"{self.targetY}_pred": preds[h],
                }))
            out = pd.concat(records, ignore_index=True)
            suffix = f"h{'_'.join(map(str, horizons))}_long"
        else:
            out = base.copy()
            for h in horizons:
                out[f"forecast_months_since_2000_h{h}"] = fcast_months[h]
                out[f"{self.targetY}_pred_t_plus_{h}"] = preds[h]
            suffix = f"h{'_'.join(map(str, horizons))}_wide"

        # Save locally
        csv_path = self.output_dir / f"rf_forecast_{suffix}.csv"
        out.to_csv(csv_path, index=False)
        print(f"[INFO] Saved forecast CSV: {csv_path}")

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=list(out.columns), fill_color="paleturquoise", align="left"),
                    cells=dict(values=[out[c] for c in out.columns], fill_color="lavender", align="left"),
                )
            ]
        )
        fig.update_layout(title=f"Random Forest Forecast ({suffix})")
        html_path = self.output_dir / f"rf_forecast_{suffix}.html"
        fig.write_html(str(html_path))
        print(f"[INFO] Saved forecast HTML: {html_path}")

        return out

    def run(self):
        pca_model = ("PCA Projection", None)
        if pca_model in self.models:
            print("PCA found")
            self.models.remove(pca_model)
            self.run_pca_projection()

        tsne_model = ("t-SNE Clustering", None)
        if tsne_model in self.models:
            print("t-SNE found")
            self.models.remove(tsne_model)
            self.run_tsne_projection()

        self.train_and_evaluate()
        self.display_results_table()
