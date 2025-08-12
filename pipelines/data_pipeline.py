import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class DataPipeline:
    def __init__(self, unprocessedDF, test_size, random_state, lookup_columns_list=None, drop_columns_list=None):
        self.unprocessedDF = unprocessedDF
        self.test_size = test_size
        self.random_state = random_state
        self.lookup_columns_list = lookup_columns_list or []
        self.drop_columns_list = drop_columns_list or []

    def load_data(self):
        # target is last column
        self.targetY = self.unprocessedDF.columns[-1]
        # cleanedDF starts as features-only copy
        self.cleanedDF = self.unprocessedDF.copy().drop(columns=self.targetY)

    # drop rows with missing target + rows with too many missing features
    def drop_unnecessary_rows(self, threshold: float = 0.8):
        # 1) Drop rows where target is missing and keep alignment between frames
        target_mask = ~self.unprocessedDF[self.targetY].isna()
        dropped_target = int((~target_mask).sum())
        if dropped_target:
            print(f"Dropped {dropped_target} rows with missing target '{self.targetY}'.")
        self.unprocessedDF = self.unprocessedDF.loc[target_mask].copy()
        self.cleanedDF = self.cleanedDF.loc[target_mask].copy()

        # 2) Drop rows with >= threshold proportion of missing feature values
        null_fraction_per_row = self.cleanedDF.isnull().mean(axis=1)
        rows_to_drop = null_fraction_per_row >= threshold
        num_dropped = int(rows_to_drop.sum())
        self.cleanedDF = self.cleanedDF.loc[~rows_to_drop].copy()
        # keep source aligned to cleanedDF index
        self.unprocessedDF = self.unprocessedDF.loc[self.cleanedDF.index].copy()
        print(f"Dropped {num_dropped} rows with >= {threshold*100:.0f}% missing feature values.")

    # optional: drop columns with high missing or low variance
    def drop_columns(self, columns_to_drop: list):
        if not columns_to_drop:
            print("No columns requested to drop.")
            return
        existing = [c for c in columns_to_drop if c in self.cleanedDF.columns]
        missing  = [c for c in columns_to_drop if c not in self.cleanedDF.columns]
        if existing:
            self.cleanedDF.drop(columns=existing, inplace=True, errors='ignore')
        if missing:
            print(f"Skipped (not present): {missing}")
        print(f"\nColumns after drop: {self.cleanedDF.columns.tolist()}")

    def encode_lookup_columns(self, columns_to_encode: list):
        for col in columns_to_encode:
            if col not in self.cleanedDF.columns:
                print(f"Column '{col}' not found in cleanedDF, skipping.")
                continue
            # Build lookup (stable order)
            unique_values = self.cleanedDF[col].dropna().astype(str).unique()
            lookup = {val: idx for idx, val in enumerate(sorted(unique_values))}
            # Save for downstream use
            setattr(self, f"{col}_lookup_table", lookup)
            # Apply encoding (cast to str to match keys)
            self.cleanedDF[col] = self.cleanedDF[col].astype(str).map(lookup)
            print(f"Encoded '{col}' with {len(lookup)} unique values.")

    def _replace_date_with_months_since_2000(self):
        if 'Date' not in self.cleanedDF.columns:
            return

        # Remember position
        idx = list(self.cleanedDF.columns).index('Date')

        # Parse safely
        parsed = pd.to_datetime(self.cleanedDF['Date'], format='%m/%d/%Y', errors='coerce')
        if parsed.isna().all():
            parsed = pd.to_datetime(self.cleanedDF['Date'], errors='coerce', infer_datetime_format=True)

        years = parsed.dt.year.astype('Int64')
        months = parsed.dt.month.astype('Int64')
        months_since_2000 = (years - 2000) * 12 + (months - 1)

        # Drop original Date and insert numeric feature at the same position
        self.cleanedDF = self.cleanedDF.drop(columns=['Date'])
        self.cleanedDF.insert(idx, 'Date_months_since_2000', months_since_2000)

        na_cnt = int(self.cleanedDF['Date_months_since_2000'].isna().sum())
        print(f"Replaced 'Date' with 'Date_months_since_2000' at position {idx}. NaNs: {na_cnt}")

    # one-hot encode categorical variables
    def encode_categorical_variables(self):
        # replace infs to avoid issues in later imputation
        self.cleanedDF.replace([np.inf, -np.inf], np.nan, inplace=True)

        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

        categorical_columns = self.cleanedDF.select_dtypes(
            include=['object', 'category', 'string']
        ).columns.tolist()

        if categorical_columns:
            print(f"\nCategorical columns found: {categorical_columns}")
            num_encoded = encoder.fit_transform(self.cleanedDF[categorical_columns])
            # feature names across sklearn versions
            try:
                feature_names = encoder.get_feature_names_out(categorical_columns)
            except AttributeError:
                feature_names = encoder.get_feature_names(categorical_columns)

            temp_encoded_df = pd.DataFrame(
                num_encoded, columns=feature_names, index=self.cleanedDF.index
            )
            numeric_df = self.cleanedDF.drop(columns=categorical_columns)
            self.cleanedDF = pd.concat([numeric_df, temp_encoded_df], axis=1)

        print(f"\nColumns after encoding: {self.cleanedDF.columns.tolist()}")

    # handle nulls/infs in features only (target handled earlier)
    def handle_missing_values(self):
        # replace infs just in case any slipped in
        self.cleanedDF.replace([np.inf, -np.inf], np.nan, inplace=True)

        cols_with_na = self.cleanedDF.columns[self.cleanedDF.isnull().any()].tolist()
        if cols_with_na:
            print(f"\nColumns with missing values: {cols_with_na}")
            for col in cols_with_na:
                if pd.api.types.is_numeric_dtype(self.cleanedDF[col]):
                    # Numeric -> mean imputation
                    mean_val = self.cleanedDF[col].mean()
                    self.cleanedDF[col] = self.cleanedDF[col].fillna(mean_val)
                    remaining = int(self.cleanedDF[col].isnull().sum())
                    print(f"- {col}: filled NAs with mean={mean_val:.4f}; remaining={remaining}")
                elif np.issubdtype(self.cleanedDF[col].dtype, np.datetime64):
                    # Datetime (if any remain) -> mode or ffill/bfill
                    mode_val = self.cleanedDF[col].mode(dropna=True)
                    if not mode_val.empty:
                        fill_val = mode_val.iloc[0]
                        self.cleanedDF[col] = self.cleanedDF[col].fillna(fill_val)
                        print(f"- {col}: filled NaT with mode={fill_val}")
                    else:
                        self.cleanedDF[col] = self.cleanedDF[col].ffill().bfill()
                        print(f"- {col}: filled NaT with ffill/bfill fallback")
                else:
                    # Any other dtype -> mode imputation
                    mode_val = self.cleanedDF[col].mode(dropna=True)
                    mode_val = mode_val.iloc[0] if not mode_val.empty else None
                    self.cleanedDF[col] = self.cleanedDF[col].fillna(mode_val)
                    print(f"- {col}: filled NAs with mode={mode_val}")
        else:
            print("\nNo missing values found in the dataset")

        if self.cleanedDF.isnull().values.any():
            raise ValueError("Missing values still exist after imputation!")

    # test-train split
    def split_data(self):
        # reattach aligned target
        self.cleanedDF = pd.concat([self.cleanedDF, self.unprocessedDF[self.targetY]], axis=1)

        # Split features/target
        X = self.cleanedDF.drop(columns=self.targetY)
        y = self.cleanedDF[self.targetY]

        # final guard: ensure finiteness for numeric columns, non-null for others
        num_cols = X.select_dtypes(include=[np.number]).columns
        other_cols = X.columns.difference(num_cols)

        # 1) Make sure numeric block is true numeric (no pandas nullable/object)
        numeric_ok = pd.Series(True, index=X.index)
        if len(num_cols) > 0:
            # coerce to float64 to avoid object arrays/nullable Int64 issues
            X_num = X[num_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
            numeric_ok = np.isfinite(X_num.to_numpy(dtype=np.float64)).all(axis=1)
        else:
            X_num = pd.DataFrame(index=X.index)  # empty

        # 2) Non-numeric columns just need to be non-null
        other_ok = pd.Series(True, index=X.index)
        if len(other_cols) > 0:
            other_ok = X[other_cols].notna().all(axis=1)

        # 3) Target
        y_ok = y.notna()
        if pd.api.types.is_numeric_dtype(y):
            y_num = pd.to_numeric(y, errors="coerce").astype("float64")
            y_ok = y_num.notna() & np.isfinite(y_num.to_numpy())

        finite_mask = numeric_ok & other_ok & y_ok

        if not finite_mask.all():
            removed = int((~finite_mask).sum())
            print(f"Removed {removed} rows with invalid values before train/test split.")
            X = X.loc[finite_mask]
            y = y.loc[finite_mask]

            # Re-apply coercion to the filtered data (keeps types clean for downstream)
            if len(num_cols) > 0:
                X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print("\nSplit:")
        print(f"X_train shape: {self.X_train.shape} (rows, columns)")
        print(f"X_test shape: {self.X_test.shape} (rows, columns)")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        print(f"\nFeature columns: {X.columns.tolist()}")
        print(f"Target column: {self.targetY}\n")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def run(self):
        self.load_data()
        self.drop_unnecessary_rows()
        self.drop_columns(self.drop_columns_list)

        # Replace Date with numeric months-since-2000 at the same position
        self._replace_date_with_months_since_2000()

        if self.lookup_columns_list:
            self.encode_lookup_columns(self.lookup_columns_list)

        self.encode_categorical_variables()
        self.handle_missing_values()

        print(f"Total columns: {len(self.cleanedDF.columns)}")
        print("Column names:")
        print(list(self.cleanedDF.columns))
        print("\nFirst 2 rows:")
        try:
            print(self.cleanedDF.head(2).to_string(index=False))
        except Exception:
            print(self.cleanedDF.head(2))

        X_train, X_test, y_train, y_test = self.split_data()
        return X_train, X_test, y_train, y_test, self.cleanedDF, self.targetY
