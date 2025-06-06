"""
This module contains the SimpleFilter class, which is used to filter features
from a dataset based on various criteria. The class includes methods for
removing low variance features, duplicate features, and correlated features.
It also includes a method for advanced feature selection using a linear regression
model.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from feature_engine.selection import DropCorrelatedFeatures, DropDuplicateFeatures
from feature_engine.selection import ProbeFeatureSelection
from sklearn.linear_model import LinearRegression
from scipy import stats


class SimpleFilter:

    # Filter for low variance features
    def __init__(self, variance_threshold=0.0, corr_threshold=0.9):
        self.lowVarianceFilter = VarianceThreshold()
        self.filter_duplicates = DropDuplicateFeatures()
        self.correlated_filter = DropCorrelatedFeatures(threshold=corr_threshold)
        self.advanced_filtered = ProbeFeatureSelection(
            estimator=LinearRegression(),
            scoring="neg_mean_absolute_percentage_error",
            n_probes=3,
            distribution="normal",
        )

    
    def fit(self, X_data, y_data):
        print(X_data.shape)
        self.lowVarianceFilter.fit(X_data)
        lv = self.lowVarianceFilter.transform(X_data)

        lv_df = pd.DataFrame(
            data=lv,
            columns=self.lowVarianceFilter.get_feature_names_out(),
            index=X_data.index,
        )
        lv_df = lv_df.loc[:,~lv_df.columns.duplicated()].copy()
        print(lv_df.shape)
        self.filter_duplicates.fit(lv_df)
        no_dup = self.filter_duplicates.transform(lv_df)
        print(no_dup.shape)
        self.correlated_filter.fit(no_dup)
        not_corr = self.correlated_filter.transform(no_dup)
        print(not_corr.shape)
        self.advanced_filtered.fit(not_corr, y_data)

    def transform(self, X_data, y_data):
        X_data = X_data.copy()
        X_data_low = self.lowVarianceFilter.transform(X_data)
        X_data_low_df = pd.DataFrame(
            data=X_data_low,
            columns=self.lowVarianceFilter.get_feature_names_out(),
            index=X_data.index,
        )   
        X_data_low_df = X_data_low_df.loc[:,~X_data_low_df.columns.duplicated()].copy()
        print(X_data_low_df.shape)
        no_dup = self.filter_duplicates.transform(X_data_low_df)
        print(no_dup.shape)
        not_corr = self.correlated_filter.transform(no_dup)
        print(not_corr.shape)
        X_transformed = self.advanced_filtered.transform(not_corr)
        
        return X_transformed, y_data

class PriceOutlierFilter:
    """
    Filtro de outliers para el target (Price), basado en percentiles.
    
    Por defecto, elimina el 1% más bajo y el 1% más alto.
    """

    def __init__(self, lower_pct=0.1, upper_pct=0.85):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X_data, y_data):
        self.lower_bound = y_data.quantile(self.lower_pct)
        self.upper_bound = y_data.quantile(self.upper_pct)
        return self

    def transform(self, X_data, y_data):
        mask = (y_data >= self.lower_bound) & (y_data <= self.upper_bound)
        return X_data[mask], y_data[mask]
