import sys
sys.path.insert(0, "C:\\Users\\jerom\\electricity_load_forecasting\\Electricity-load-forecasting\\")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from data.get_data import HQ_data
from models.regression_splines import SplineRegression
from models.fourier_series import Fourier_series
from models.arma import ARMA_model
from models.mlp import MLP_model

class Simulation:

    def __init__(self, num_iters, train_start, train_end):

        self.num_iters = num_iters
        self.train_start = train_start
        self.train_end = train_end
        self.test = self.train_end + datetime.timedelta(days=1)
        self.data = HQ_data()
        self.data = self.data.get_history()

    def get_prediction(self, train_start, train_end, test):

        """ 
        Implement algorithm or call model here. 
        Model should have get_predictions method that takes in data, train_start, train_end, test 
        and returns a single forecast for the time step test: 24 hour after train_end

        """

        """mlp = MLP_model()
        forecasts = mlp.get_predictions(self.data, train_start, train_end, test)"""

        """spline_reg = SplineRegression()
        forecasts = spline_reg.get_predictions(self.data, train_start, train_end, test)"""

        fourier = Fourier_series()
        forecasts = fourier.get_predictions(self.data, train_start, train_end, test)

        """arma = ARMA_model()
        forecasts = arma.get_predictions(self.data, train_start, train_end, test)"""

        return forecasts

    def run_simulation(self):
        
        train_start = self.train_start
        train_end = self.train_end
        test = self.test

        forecasts = []

        for i in range(self.num_iters):

            forecast = self.get_prediction(train_start, train_end, test)
            forecasts.append(forecast)

            print("********************************************")
            print("At iteration"+ str(i))
            print("forecast: ", forecast)
            print("********************************************")

            train_start = train_start + datetime.timedelta(hours=1)
            train_end = train_end + datetime.timedelta(hours=1)
            test = test + datetime.timedelta(hours=1)

        return np.array(forecasts).flatten()

    def plot_sim_results(self, forecasts):

        sim_start = self.train_end + datetime.timedelta(days=1)
        sim_end = sim_start + datetime.timedelta(hours = self.num_iters) - datetime.timedelta(hours=1)

        results = self.data.loc[sim_start:sim_end, ["demand", "scaled_temp"]]
        results["forecast"] = forecasts

        # save forecasts to csv
        results.to_csv("results\\results_fourier_2021.csv")

        residuals = results.loc[:, "demand"] - results.loc[:, "forecast"]

        print("RMSE")
        print(np.sqrt(np.mean(residuals**2)))

        print("MAPE")
        print(np.mean(abs(residuals)/results.loc[:, "demand"]))

        print("Percentage within 1000 mwh")
        print(sum(list(map(lambda x: int(x <= 1000), abs(residuals))))/len(residuals))

        print("Percentage within 500 mwh")
        print(sum(list(map(lambda x: int(x <= 500), abs(residuals))))/len(residuals))

        plt.plot(results.loc[:, "demand"], label="Demand")
        plt.plot(results.loc[:, "forecast"], label="Forecast")
        plt.legend()
        plt.title("24 hour ahead energy demand forecast for year 2022")
        plt.show()