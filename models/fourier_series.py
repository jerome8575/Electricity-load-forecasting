
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from data.get_data import HQ_data
from statsmodels.tsa.arima.model import ARIMA

class Fourier_series:

    def get_predictions(self, data, train_start, train_end, test):

        pf = PolynomialFeatures(degree=2)

        data = data.loc[train_start:test, :].copy()

        # get fourier series features
        ff_week = self.get_fourier_features(5, 7, data.loc[:, "day"])
        ff_24h = self.get_fourier_features(5, 24, data.loc[:, "hour"])
        fourier_features = pd.concat([ff_week, ff_24h], ignore_index=True, axis=1)

        # add extra predictors
        features = ["scaled_temp", "temp_lag_15", "temp_index_15", 
                    "demand_lag_24", "is_clear", "rel_hum", "scaled_temp_diff_24", "scaled_temp_diff_48"]

        X = fourier_features
        X[features] = data.loc[:, features]

        # train model

        X_train = pf.fit_transform(X.loc[train_start:train_end, :])
        y_train = data.loc[train_start:train_end, "log_demand"]

        model = sm.OLS(y_train, X_train).fit()

        # get forecast
        X_test = pf.fit_transform(np.array(X.loc[test, :]).reshape(1, -1))
        forecast  = np.exp(model.predict(X_test))

        return forecast
    

    def get_fourier_features(self, n_order, period, values):
        fourier_features = pd.DataFrame(
            {
                f"fourier_{func}_order_{order}_{period}": getattr(np, func)(
                    2 * order * np.pi * values / period
                )
                for order in range(1, n_order + 1)
                for func in ("sin", "cos")
            }
        )
        return fourier_features