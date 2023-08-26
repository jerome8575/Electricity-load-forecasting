import datetime
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.api as sm
from data.get_data import HQ_data
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor

class MLP_model:

    def get_predictions(self, data, train_start, train_end, test):

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        # get training data
        data = data.loc[train_start:test, :].copy()

        # with this model, we treat the electricity demand for every hour individually
        # get hourly demand for hour we want to predict

        hour = test.hour
        data_hour = data.loc[data.loc[:, "hour"] == hour, :].copy()

        # train mlp model

        features = ["demand_lag_24", "is_clear", "is_weekend", "temp_lag_1", "temp_lag_15"]

        X = data_hour.loc[:, features]
        y = data_hour.loc[:, "log_demand"]

        X_train = feature_scaler.fit_transform(X.loc[train_start:train_end, :])
        y_train = target_scaler.fit_transform(np.array(y.loc[train_start:train_end]).reshape(-1, 1))

        model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=1).fit(X_train, y_train)

        # get prediction

        X_test = feature_scaler.transform(np.array(X.loc[test, :]).reshape(1, -1))
        forecast = target_scaler.inverse_transform(model.predict(X_test).reshape(-1, 1))

        return np.exp(forecast)[0]