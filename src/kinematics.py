import pandas as pd
import numpy as np
import os    
import json  
import re
from fitter import Fitter
import matplotlib.pyplot as plt

def get_initial_observation(df):
    """
    Performs structural analysis (Parte A).
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "types": {"numerical": numerical_cols, "categorical": categorical_cols},
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "low_cardinality": [col for col in df.columns if df[col].nunique() < 10]
    }

def get_statistical_description(df):
    """
    Calculates moments of distribution and correlations (Parte B).
    """
    # 1. Numerical Summary
    num_df = df.select_dtypes(include=[np.number])
    summary = num_df.describe().to_dict()
    for col in num_df.columns:
        summary[col]["iqr"] = float(num_df[col].quantile(0.75) - num_df[col].quantile(0.25))

    # 2. Categorical Summary
    cat_df = df.select_dtypes(exclude=[np.number])
    cat_summary = {col: {"counts": cat_df[col].value_counts().to_dict(), 
                         "percentages": (cat_df[col].value_counts(normalize=True)*100).to_dict()} 
                   for col in cat_df.columns}

    # 3. Correlations
    pearson = num_df.corr(method='pearson').to_dict()
    spearman = num_df.corr(method='spearman').to_dict()

    return {
        "numerical": summary,
        "categorical": cat_summary,
        "correlations": {"pearson": pearson, "spearman": spearman}
    }

def log_hypotheses(hypotheses_list, filename="06_hypotheses_log.json"):
    """
    Persists a list of falsifiable hypotheses as a JSON artifact.
    """
    # Ensure the path is relative to the project root
    path = os.path.join("..", "artifacts", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w") as f:
        json.dump({"hypotheses": hypotheses_list}, f, indent=4)
    return path

def save_final_report(report_data, filename="07_conclusions.json"):
    """
    Persists the final conclusions and researcher questions as a JSON artifact.
    """
    path = os.path.join("..", "artifacts", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    return path


class TimeSeriesEngine:
    """Engine for Lab 2 time-series operations on EUR/USD data."""

    def __init__(self, data_frame):
        self.DataFrame = data_frame.copy()

    @staticmethod
    def _to_pascal_case(name):
        parts = re.split(r"[^A-Za-z0-9]+", str(name))
        parts = [part for part in parts if part]
        return "".join(part[:1].upper() + part[1:] for part in parts)

    @classmethod
    def from_csv(cls, csv_path_or_url):
        data_frame = pd.read_csv(csv_path_or_url)
        if "Unnamed: 0" in data_frame.columns:
            data_frame = data_frame.drop(columns=["Unnamed: 0"])
        return cls(data_frame)

    def normalize_columns_pascal_case(self):
        rename_map = {col: self._to_pascal_case(col) for col in self.DataFrame.columns}
        self.DataFrame = self.DataFrame.rename(columns=rename_map)
        return self.DataFrame

    def set_time_index(self, time_column="Time"):
        self.DataFrame[time_column] = pd.to_datetime(self.DataFrame[time_column], errors="coerce")
        self.DataFrame = self.DataFrame.set_index(time_column).sort_index()
        return self.DataFrame

    def get_data_frame_info(self):
        return {
            "Shape": {
                "Rows": int(self.DataFrame.shape[0]),
                "Columns": int(self.DataFrame.shape[1])
            },
            "Dtypes": {col: str(dtype) for col, dtype in self.DataFrame.dtypes.items()}
        }

    def get_null_nan_report(self):
        null_counts = self.DataFrame.isna().sum().to_dict()
        return {
            "NullOrNaNByColumn": {k: int(v) for k, v in null_counts.items()},
            "TotalNullOrNaN": int(self.DataFrame.isna().sum().sum())
        }

    def keep_only_close_price(self, close_column="Close"):
        self.DataFrame = self.DataFrame[[close_column]].copy()
        return self.DataFrame

    def add_diff_price(self, close_column="Close", diff_column="DiffPrice"):
        self.DataFrame[diff_column] = self.DataFrame[close_column].diff()
        return self.DataFrame

    def get_best_distribution(self, column="DiffPrice", distributions=None):
        if distributions is None:
            distributions = ["gamma", "lognorm", "beta", "burr", "norm"]

        sample = self.DataFrame[column].dropna().values
        fitter = Fitter(sample, distributions=distributions)
        fitter.fit()

        best = fitter.get_best(method="sumsquare_error")
        best_name = next(iter(best))
        return {
            "BestDistribution": best_name,
            "Parameters": {k: float(v) for k, v in best[best_name].items()},
            "AllSSE": {k: float(v) for k, v in fitter.summary()["sumsquare_error"].to_dict().items()}
        }

    def filter_year(self, year):
        self.DataFrame = self.DataFrame[self.DataFrame.index.year == year].copy()
        return self.DataFrame

    def grouped_mean_close(self, close_column="Close"):
        return {
            "15D": self.DataFrame.groupby(pd.Grouper(freq="15D"))[close_column].mean(),
            "1W": self.DataFrame.groupby(pd.Grouper(freq="1W"))[close_column].mean(),
            "1M": self.DataFrame.groupby(pd.Grouper(freq="ME"))[close_column].mean()
        }

    def monthly_histograms(self, year, close_column="Close", output_dir="artifacts/lab02_monthly_hist"):
        os.makedirs(output_dir, exist_ok=True)
        data_year = self.DataFrame[self.DataFrame.index.year == year]
        saved_paths = []

        for month in range(1, 13):
            month_data = data_year[data_year.index.month == month][close_column].dropna()
            if month_data.empty:
                continue

            plt.figure(figsize=(8, 4))
            plt.hist(month_data, bins=30, color="#2a9d8f", edgecolor="white", alpha=0.9)
            plt.title(f"Histogram of {close_column} - {year}-{month:02d}")
            plt.xlabel(close_column)
            plt.ylabel("Frequency")
            file_path = os.path.join(output_dir, f"hist_{year}_{month:02d}.png")
            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()
            saved_paths.append(file_path)

        return saved_paths


class FeatureEngineeringEngine:
    """Engine for Lab 2 preprocessing operations on the breast cancer dataset."""

    def __init__(self, features, targets):
        self.Features = features.copy()
        self.Targets = targets.copy()
        self.SystemDataFrame = None

    @staticmethod
    def _to_pascal_case(name):
        parts = re.split(r"[^A-Za-z0-9]+", str(name))
        parts = [part for part in parts if part]
        return "".join(part[:1].upper() + part[1:] for part in parts)

    def build_system_dataframe(self, target_column_name="Diagnosis"):
        target_series = self.Targets.squeeze().rename(target_column_name)
        self.SystemDataFrame = pd.concat([self.Features, target_series], axis=1)
        return self.SystemDataFrame

    def normalize_columns_pascal_case(self):
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")
        rename_map = {col: self._to_pascal_case(col) for col in self.SystemDataFrame.columns}
        self.SystemDataFrame = self.SystemDataFrame.rename(columns=rename_map)
        return self.SystemDataFrame

    def map_diagnosis_numeric(self, source_column="Diagnosis", target_column="DiagnosisNumeric"):
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")
        mapping = {"B": 0, "M": 1}
        self.SystemDataFrame[target_column] = self.SystemDataFrame[source_column].map(mapping)
        return self.SystemDataFrame

    def get_null_profile(self):
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")
        null_counts = self.SystemDataFrame.isna().sum().to_dict()
        return {
            "NullCountByColumn": {k: int(v) for k, v in null_counts.items()},
            "TotalNullCount": int(self.SystemDataFrame.isna().sum().sum())
        }

    def get_target_distribution(self, target_column="Diagnosis"):
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")
        distribution = self.SystemDataFrame[target_column].value_counts(dropna=False).to_dict()
        return {str(k): int(v) for k, v in distribution.items()}

    def zscore_normalize_features(self, exclude_columns=None):
        """Applies vectorized z-score normalization to numeric columns."""
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")

        if exclude_columns is None:
            exclude_columns = []

        numeric_cols = self.SystemDataFrame.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in set(exclude_columns)]

        means = self.SystemDataFrame[cols_to_normalize].mean(axis=0)
        stds = self.SystemDataFrame[cols_to_normalize].std(axis=0, ddof=0)
        stds = stds.replace(0, np.nan)

        self.SystemDataFrame.loc[:, cols_to_normalize] = (
            (self.SystemDataFrame[cols_to_normalize] - means) / stds
        )
        return self.SystemDataFrame

    def add_grouped_feature_averages_regex(self):
        """Groups feature triplets (1,2,3) by regex pattern and stores their row-wise means."""
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")

        base_to_group = {
            "Radius": "RadiusMean",
            "Texture": "TextureMean",
            "Perimeter": "PerimeterMean",
            "Area": "AreaMean",
            "Smoothness": "SmoothnessMean",
            "Compactness": "CompactnessMean",
            "Concavity": "ConcavityMean",
            "ConcavePoints": "ConcavePointsMean",
            "Symmetry": "SymmetryMean",
            "FractalDimension": "FractalDimensionMean"
        }

        grouped_sources = {}
        for base_name, grouped_name in base_to_group.items():
            pattern = re.compile(rf"^{base_name}(1|2|3)$")
            matched_cols = [col for col in self.SystemDataFrame.columns if pattern.match(col)]
            if not matched_cols:
                continue

            self.SystemDataFrame[grouped_name] = self.SystemDataFrame[matched_cols].mean(axis=1)
            grouped_sources[grouped_name] = matched_cols

        return self.SystemDataFrame, grouped_sources

    def detect_outliers_iqr(self, column):
        """Returns row indices outside the IQR bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")

        series = self.SystemDataFrame[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (self.SystemDataFrame[column] < lower) | (self.SystemDataFrame[column] > upper)
        return self.SystemDataFrame.index[mask].tolist()

    def detect_outliers_zscore(self, column, threshold=3.0):
        """Returns row indices where abs(z-score) > threshold."""
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")

        series = self.SystemDataFrame[column]
        mu = series.mean()
        sigma = series.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return []

        z_scores = (series - mu) / sigma
        mask = z_scores.abs() > threshold
        return self.SystemDataFrame.index[mask].tolist()

    def remove_outliers(self, outlier_indices):
        """Removes outlier rows from SystemDataFrame and returns the cleaned DataFrame."""
        if self.SystemDataFrame is None:
            raise ValueError("SystemDataFrame is not initialized. Run build_system_dataframe first.")

        self.SystemDataFrame = self.SystemDataFrame.drop(index=outlier_indices, errors="ignore")
        return self.SystemDataFrame