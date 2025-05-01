from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import streamlit as st


class RobustStandardScaler(BaseEstimator, TransformerMixin):

    def fit(self, x):
        """
        Compute the mean and std, ignoring NaN values, for scaling.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the feature's axis.
        """
        # x = x.astype(float)
        # Compute mean ignoring NaN values
        self.mean_ = np.nanmean(x, axis=0)
        # Compute variance ignoring NaN values
        self.var_ = np.array(list(np.nanvar(x, axis=0)))
        # Compute standard deviation
        self.scale_ = np.sqrt(self.var_)
        return self

    def transform(self, x):
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data that needs to be transformed.
        """
        # Apply transformation ignoring NaN values
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x):
        """
        Fit to data, then transform it.
        """
        return self.fit(x).transform(x)

    def inverse_transform(self, x_scaled):
        """
        Undo the standardization to get back original values.

        Parameters
        ----------
        x_scaled : array-like, shape [n_samples, n_features]
            The standardized data to be inverse transformed.
        """
        # Revert transformation ignoring NaN values
        return (x_scaled * self.scale_) + self.mean_