from pathlib import Path
import os
import numpy as np
import pandas as pd
import plotly.express as px


class AnalysisPipeline:
    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        cleanedDF,
        targetY,
        plots_dir=None,
        # --- EDA sampling knobs ---
        sample_rows=10_000,
        scatter_sample_rows=5_000,
        corr_sample_rows=None,
        max_scatter_dims=10,
        # --- Histogram knobs ---
        hist_top_categories=20,        # for categorical histograms (bars)
        use_color_if_levels_le=12,     # only color when target has ≤ this many distinct levels
        max_hist_cols=None,            # limit how many columns to draw histograms for (None = all)
        random_state=42,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cleanedDF = cleanedDF
        self.targetY = targetY

        # Resolve plots directory (../dags/plots by default)
        default_plots = Path(__file__).resolve().parents[1] / "dags" / "plots"
        resolved = Path(os.getenv("PLOTS_DIR", plots_dir or default_plots))
        resolved.mkdir(parents=True, exist_ok=True)
        self.plots_dir = resolved

        # Ensure Date is datetime but DO NOT drop it
        self._ensure_datetime_date()

        # Sampling + controls
        self.sample_rows = sample_rows
        self.scatter_sample_rows = scatter_sample_rows
        self.corr_sample_rows = corr_sample_rows
        self.max_scatter_dims = max_scatter_dims
        self.hist_top_categories = hist_top_categories
        self.use_color_if_levels_le = use_color_if_levels_le
        self.max_hist_cols = max_hist_cols
        self.random_state = random_state

        # Create sampled frames
        self.df_plot = self._make_sample(self.cleanedDF, self.sample_rows, tag="plots")
        self.df_scatter = self._make_sample(self.cleanedDF, self.scatter_sample_rows, tag="scatter")
        self.df_corr = (
            self._make_sample(self.cleanedDF, self.corr_sample_rows, tag="corr")
            if self.corr_sample_rows
            else self.cleanedDF
        )

        # Optional sanity logging
        print(f"[AnalysisPipeline] CWD: {os.getcwd()}")
        print(f"[AnalysisPipeline] Plots directory: {self.plots_dir}")
        print(
            f"[AnalysisPipeline] Shapes -> plot:{self.df_plot.shape}, "
            f"scatter:{self.df_scatter.shape}, corr:{self.df_corr.shape}"
        )

    # ------------------------ helpers ------------------------

    def _make_sample(self, df: pd.DataFrame, n: int, tag: str):
        if n is None or n >= len(df):
            print(f"[AnalysisPipeline] Using full dataset for {tag}: {len(df)} rows")
            return df
        out = df.sample(n=n, random_state=self.random_state)
        print(f"[AnalysisPipeline] Sampled {n} rows for {tag}")
        return out

    def _ensure_datetime_date(self):
        if "Date" in self.cleanedDF.columns:
            before = self.cleanedDF["Date"].dtype
            self.cleanedDF["Date"] = pd.to_datetime(
                self.cleanedDF["Date"], format="%m/%d/%Y", errors="coerce"
            )
            nat_count = int(self.cleanedDF["Date"].isna().sum())
            print(
                f"[AnalysisPipeline] Converted 'Date' from {before} -> {self.cleanedDF['Date'].dtype}. "
                f"NaT rows: {nat_count}"
            )

    def _fd_nbins(self, s: pd.Series, min_bins=10, max_bins=120) -> int:
        x = pd.to_numeric(s.dropna(), errors="coerce")
        if x.empty:
            return min_bins
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = float(q3 - q1) or 1.0
        n = len(x)
        width = 2 * iqr / (n ** (1 / 3))
        if width <= 0:
            return min_bins
        nbins = int(np.ceil((x.max() - x.min()) / width))
        return int(max(min_bins, min(max_bins, nbins)))

    def _small_target_palette_ok(self) -> bool:
        if self.targetY not in self.df_plot.columns:
            return False
        tgt = self.df_plot[self.targetY]
        # Consider categorical or small-cardinality integers as OK to color
        is_cat_like = (
            tgt.dtype == "object"
            or pd.api.types.is_categorical_dtype(tgt)
            or (pd.api.types.is_integer_dtype(tgt) and tgt.nunique(dropna=True) <= self.use_color_if_levels_le)
        )
        return is_cat_like and (tgt.nunique(dropna=True) <= self.use_color_if_levels_le)

    # ------------------------ plots ------------------------

    def violin_plot(self):
        if self.targetY not in self.df_plot.columns:
            print(f"Target column '{self.targetY}' not found.")
            return

        for col in self.df_plot.columns:
            if col == self.targetY:
                continue

            fig = px.violin(
                self.df_plot,
                y=col,
                title=f"Violin Plot of {col} grouped by {self.targetY}",
            )

            file_path = self.plots_dir / f"{col}_violin_plot.html"
            fig.write_html(str(file_path))
            print(f"Saved: {file_path}")

    def histogram(self):
        if self.targetY not in self.df_plot.columns:
            print(f"Target column '{self.targetY}' not found.")
            return

        cols = [c for c in self.df_plot.columns if c != self.targetY]
        if self.max_hist_cols:
            cols = cols[: self.max_hist_cols]
            print(f"[AnalysisPipeline] Limiting histograms to first {len(cols)} columns")

        use_color = self._small_target_palette_ok()
        color_kw = {"color": self.targetY} if use_color else {}
        if not use_color:
            print("[AnalysisPipeline] Skipping color by target for histograms (too many levels or not categorical)")

        for col in cols:
            try:
                print(f"[AnalysisPipeline] Histogram for '{col}' ...")
                s = self.df_plot[col]

                # Datetime -> bin monthly
                if pd.api.types.is_datetime64_any_dtype(s):
                    df_h = self.df_plot.copy()
                    df_h["__month__"] = s.dt.to_period("M").dt.to_timestamp()
                    nbins = min(120, max(12, df_h["__month__"].nunique()))
                    fig = px.histogram(
                        df_h,
                        x="__month__",
                        nbins=nbins,
                        title=f"Histogram of {col} (monthly bins)",
                        **color_kw,
                    )

                # Numeric -> robust nbins (Freedman–Diaconis)
                elif pd.api.types.is_numeric_dtype(s):
                    nb = self._fd_nbins(s)
                    fig = px.histogram(
                        self.df_plot,
                        x=col,
                        nbins=nb,
                        title=f"Histogram of {col} (nbins={nb})",
                        **color_kw,
                    )

                # Categorical/text -> top-K categories + Other
                else:
                    vc = s.astype("string").fillna("NA").value_counts()
                    top = vc.head(self.hist_top_categories)
                    other = vc.iloc[self.hist_top_categories:].sum()
                    plot_df = top.rename_axis(col).reset_index(name="count")
                    if other > 0:
                        plot_df = pd.concat(
                            [plot_df, pd.DataFrame({col: ["Other"], "count": [other]})],
                            ignore_index=True,
                        )
                    fig = px.bar(
                        plot_df,
                        x=col,
                        y="count",
                        title=f"Top {self.hist_top_categories} {col} (others grouped)",
                    )

                fig.update_traces(marker_line_width=0)
                file_path = self.plots_dir / f"{col}_histogram.html"
                # Use CDN to keep HTML light and speed up writes
                fig.write_html(str(file_path), include_plotlyjs="cdn")
                print(f"Saved: {file_path}")

            except Exception as e:
                print(f"[AnalysisPipeline] Skipped histogram for '{col}' due to: {e}")

    def scatter_pairs_matrix(self):
        if self.targetY not in self.df_scatter.columns:
            print(f"Target column '{self.targetY}' not found.")
            return

        numeric_df = self.df_scatter.select_dtypes(include="number").copy()
        if self.targetY in numeric_df.columns:
            numeric_df.drop(columns=self.targetY, inplace=True)

        if numeric_df.empty:
            print("No numeric predictors available for scatter matrix.")
            return

        # Cap dimensions by variance
        var_ranked = numeric_df.var(skipna=True).sort_values(ascending=False)
        dims = var_ranked.head(self.max_scatter_dims).index.tolist()

        fig = px.scatter_matrix(
            self.df_scatter,
            dimensions=dims,
            color=self.targetY if self.targetY in self.df_scatter.columns else None,
            title=f"Scatter Matrix (top {len(dims)} numeric features by variance)",
            height=800,
            width=800,
        )

        uniques = self.df_scatter[self.targetY].unique() if self.targetY in self.df_scatter.columns else []
        if len(uniques) and len(uniques) <= 20:
            fig.update_layout(
                coloraxis_colorbar=dict(
                    tickmode="array",
                    ticktext=sorted(uniques),
                    tickvals=sorted(uniques),
                )
            )

        file_path = self.plots_dir / "scatter_matrix.html"
        fig.write_html(str(file_path))
        print(f"Saved scatter matrix: {file_path}")

    def correlation_heatmap(self):
        numeric_df = self.df_corr.select_dtypes(include="number").copy()
        if self.targetY in numeric_df.columns:
            numeric_df.drop(columns=self.targetY, inplace=True)

        if numeric_df.empty or numeric_df.shape[1] < 2:
            print("Not enough numeric features for a correlation heatmap.")
            return

        corr_matrix = numeric_df.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title=f"Correlation Heatmap of Numeric Features (rows={len(numeric_df)})",
            zmin=-1,
            zmax=1,
        )

        file_path = self.plots_dir / "correlation_heatmap.html"
        fig.write_html(str(file_path))
        print(f"Saved: {file_path}")

    # ------------------------ driver ------------------------

    def run(self):
        self.violin_plot()
        self.histogram()
        self.scatter_pairs_matrix()
        self.correlation_heatmap()
