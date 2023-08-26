import datetime
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from data.get_data import HQ_data
from statsmodels.tsa.arima.model import ARIMA

class ARMA_model:

    def get_predictions(self, data, train_start, train_end, test):

        # get training data
        data = data.loc[train_start:test, :].copy()

        # with this model, we treat the electricity demand for every hour individually
        # get hourly demand for hour we want to predict

        hour = test.hour
        data_hour = data.loc[data.loc[:, "hour"] == hour, :].copy()

        features = ["demand_lag_24", "is_clear", "is_weekend", "temp_lag_1", "temp_lag_15"]

        # train model
        X = data_hour.loc[:, features]
        y = data_hour.loc[:, "log_demand"]

        X_train = X.loc[train_start:train_end, :]
        y_train = y.loc[train_start:train_end]

        model = ARIMA(y_train, order=(1, 1, 1), exog=X_train)
        model_fit = model.fit()

        # get prediction

        X_test = X.loc[test, :]
        forecast = model_fit.forecast(1, exog=X_test)

        return np.exp(forecast)