from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class GroupwiseMedianNoiseImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, noise_scale=0.1):
        self.group_col = group_col
        self.noise_scale = noise_scale
        self.group_medians_ = {}

    def fit(self, X, y=None):
        self.group_stats_ = {}
        df = X.copy()

        for group in df[self.group_col].dropna().unique():
            sub = df[df[self.group_col] == group]
            self.group_stats_[group] = {}
            for col in df.columns:
                if col == self.group_col:
                    continue
                median = sub[col].median()
                iqr = sub[col].quantile(0.75) - sub[col].quantile(0.25)
                nonzero_min = sub[col][sub[col] > 0].min()
                self.group_stats_[group][col] = {
                    'median': median,
                    'iqr': iqr,
                    'nonzero_min': nonzero_min
                }
        return self

    def transform(self, X):
        df = X.copy()
        for idx, row in df.iterrows():
            group = row[self.group_col]
            for col in df.columns:
                if col == self.group_col:
                    continue
                if pd.isna(row[col]):
                    if group in self.group_stats_ and col in self.group_stats_[group]:
                        stats = self.group_stats_[group][col]
                        median = stats['median']
                        iqr = stats['iqr']
                        min_val = stats['nonzero_min']
                        noise = np.random.normal(loc=0, scale=self.noise_scale * iqr)
                        value = median + noise
                        df.at[idx, col] = max(value, min_val)  # 限制下界
        return df.drop(columns=[self.group_col])