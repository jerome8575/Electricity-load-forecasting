import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse


class Ridge_ensemble:

    def __init__(self, train_start, training_time_window):

        # read forecasts from base models

        mlp = pd.read_csv("Ensemble_methods\\base_model_forecasts\\results_mlp_2021.csv")
        """spline = pd.read_csv("base_model_forecasts/results_spline_2021.csv", index_col=0)
        fourier = pd.read_csv("base_model_forecasts/results_fourier_2021.csv", index_col=0)"""
        arma = pd.read_csv("Ensemble_methods\\base_model_forecasts/results_arma_2021.csv")

        self.forecasts = pd.concat([mlp.loc[:, ["date_time", "demand", "forecast"]],
                                    arma.loc[:, "forecast"]], axis=1)

        self.forecasts.columns = ["date_time", "demand", "mlp", "arma"]
        self.forecasts["date_time"] = pd.to_datetime(self.forecasts["date_time"])
        self.forecasts.set_index("date_time", inplace=True)

        self.time_window = training_time_window
        self.train_start = train_start

    def get_predictions(self):

        train_start = self.train_start
        train_end = train_start + datetime.timedelta(days=self.time_window)

        # simulation

        forecasts = []
        for _ in range(len(self.forecasts) - (self.time_window * 24) - 24):

            X_train = self.forecasts.loc[train_start:train_end, :].drop(columns=["demand"])
            y_train = self.forecasts.loc[train_start:train_end, "demand"]

            X_test = self.forecasts.loc[train_end + datetime.timedelta(days=1), ["mlp", "arma"]]
            print(X_test)

            # train model

            model = sm.OLS(y_train, X_train).fit_regularized(alpha=0.05, L1_wt=0.0)

            # get forecast

            forecast = model.predict(X_test)
            forecasts.append(forecast)

            # update train_start and train_end

            train_start = train_start + datetime.timedelta(hours=1)
            train_end = train_end + datetime.timedelta(hours=1)

        return np.array(forecasts).flatten()
        


train_start = datetime.datetime(2021, 1, 1, 0, 0, 0)
time_window = 30

print(train_start + datetime.timedelta(days=time_window + 1))

estimator = Ridge_ensemble(train_start, time_window)
forecasts = estimator.get_predictions()

results = estimator.forecasts.loc[train_start + datetime.timedelta(days=time_window + 1):datetime.datetime(2021, 12, 31, 23, 0, 0), :]
print(results)
results["forecast"] = forecasts

print("RMSE")
print(np.sqrt(mse(results.loc[:, "demand"], results.loc[:, "forecast"])))

print("MAPE")
print(mape(results.loc[:, "demand"], results.loc[:, "forecast"]))









