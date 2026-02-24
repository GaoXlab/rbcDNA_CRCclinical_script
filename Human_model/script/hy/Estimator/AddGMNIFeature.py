import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .GroupwiseMedianNoiseImputer import GroupwiseMedianNoiseImputer

class AddGMNIFeature(BaseEstimator, TransformerMixin):
    """
    为 DataFrame 添加指定列的转换器

    参数:
    ----------
    column_name : str
        要添加的列名
    column_data : array-like, Series or callable
        列数据，可以是:
        - 固定值 (如 0 或 'default')
        - 与 DataFrame 长度相同的数组/Series
        - 一个函数，接受 DataFrame 返回列数据
    """

    def __init__(self, column_name, sampleinfo):
        self.column_name = column_name
        self.sampleinfo = sampleinfo
        self.imputer = GroupwiseMedianNoiseImputer(group_col='target', noise_scale=0.1)

    def fit(self, X, y=None):
        """拟合方法（此处不需要实际拟合）"""

        self.imputer.fit(self.sampleinfo.loc[X.index][[self.column_name, 'target']])
        return self

    def transform(self, X):
        """
        添加指定列到 DataFrame

        参数:
        ----------
        X : pd.DataFrame
            输入数据

        返回:
        ----------
        X_transformed : pd.DataFrame
            添加了新列的数据
        """
        column_to_merge = self.sampleinfo[[self.column_name, 'target']]
        column_to_merge = self.imputer.transform(column_to_merge)

        # 根据索引合并
        merged = X.join(column_to_merge, how='left')
        print(merged)
        return merged