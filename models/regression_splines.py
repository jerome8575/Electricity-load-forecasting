import datetime
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from data.get_data import HQ_data
from statsmodels.tsa.arima.model import ARIMA

class SplineRegression:

    def get_predictions(self, data, train_start, train_end, test):

        # get training data
        data = data.loc[train_start:test, :].copy()

        # with this model, we treat the electricity demand for every hour individually
        # get hourly demand for hour we want to predict

        hour = test.hour
        data_hour = data.loc[data.loc[:, "hour"] == hour, :].copy()

        # train regression splines model
        # knots for regression is based off scaled temperature

        scaled_temp = data_hour["scaled_temp"].values

        # create basis matrix

        basis_matrix = dmatrix("bs(scaled_temp, knots=(-1, -0.5, 0, 0.5, 1), degree=3, include_intercept=True)", {"scaled_temp": scaled_temp}, return_type='dataframe')
        basis_matrix["date_time"] = data_hour.index.values
        basis_matrix.set_index("date_time", inplace=True)

        # add extra predictors

        features = ["demand_lag_24", "is_clear", "is_weekend", "temp_lag_1", "temp_lag_15"]

        # train model
        X = pd.concat([basis_matrix, data_hour.loc[:, features]], axis=1, ignore_index=True)
        y = data_hour.loc[:, "log_demand"]

        X_train = X.loc[train_start:train_end, :]
        y_train = y.loc[train_start:train_end]

        model = sm.OLS(y_train, X_train).fit()

        # fit arma errors model

        arma_model_start = train_end - datetime.timedelta(days=15)
        residuals = model.predict(X_train.loc[arma_model_start:train_end, :]) - y_train.loc[arma_model_start:train_end]
        residual_predictors = ["is_weekend"]
        res_model = ARIMA(residuals, order=(1, 0, 1)).fit()

        # get prediction

        X_test = X.loc[test, :]
        forecast = model.predict(X_test) - res_model.forecast(steps=1)[0]

        return np.exp(forecast)