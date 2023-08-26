
import pandas as pd
import numpy as np
import datetime 
from sklearn.preprocessing import StandardScaler


class HQ_data():

    def __init__(self):

        self.start = datetime.datetime(2019, 1, 1, 1, 0, 0)
        self.end = datetime.datetime(2022, 12, 31, 23, 0, 0)

        # read electricity demand data
        data_2019 = pd.read_excel("data\\data_files\\2019-demande-electricite-quebec.xlsx")
        data_2020 = pd.read_excel("data\\data_files\\2020-demande-electricite-quebec.xlsx")
        data_2021 = pd.read_excel("data\\data_files\\2021-demande-electricite-quebec.xlsx")
        data_2022 = pd.read_excel("data\\data_files\\2022-demande-electricite-quebec.xlsx")

        demand_data = pd.concat([data_2019, data_2020, data_2021, data_2022])
        
        weather_data = pd.read_csv("data\\data_files\\montreal_weather.csv")

        self.data = self.create_feature_dataframe(demand_data, weather_data)

    def create_feature_dataframe(self, demand_data, weather_data):

        # Read data and create various features for forecasting models

        scaler = StandardScaler()

        demand_data.set_index("Date", inplace=True)
        datetime_index = list(map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d %H:%M'), weather_data.loc[:, "datetime"]))

        weather_data["datetime_index"] = datetime_index
        weather_data.set_index("datetime_index", inplace=True)
 
        day = list(map(lambda t: t.weekday(), demand_data.loc[self.start:self.end, :].index))
        weekend = [5, 6]
        is_weekend = list(map(lambda x: int(x in weekend), day))
        summer = [6, 6, 7, 8, 9]
        is_summer = list(map(lambda x: int(x in summer), demand_data.loc[self.start:self.end, :].index.month))

        data = pd.DataFrame({
            "demand": demand_data.loc[self.start:self.end, "Moyenne (MW)"].values,
            "temp": weather_data.loc[self.start:self.end, "temp"].values,
            "is_cloudy": weather_data.loc[self.start:self.end, "is_cloudy"].values,
            "is_clear": weather_data.loc[self.start:self.end, "is_clear"].values,
            "is_snowing": weather_data.loc[self.start:self.end, "is_snowing"].values,
            "date_time": demand_data.index[0:-1].values,
            "day": day,
            "hour": list(map(lambda t: t.hour, demand_data.loc[self.start:self.end, :].index)),
            "is_weekend": is_weekend,
            "wind_speed": weather_data.loc[self.start:self.end, "wind_speed"].values,
            "wind_chill": weather_data.loc[self.start:self.end, "wind_chill"].values,
            "rel_hum": weather_data.loc[self.start:self.end, "rel_hum"].values
        })
        data["scaled_temp"] = scaler.fit_transform(np.array(data.loc[:, "temp"]).reshape(-1, 1))
        data["log_demand"] = np.log(data.loc[:, "demand"])
        data["demand_lag_24"] = data.loc[:, "demand"].shift(24)
        data["demand_lag_48"] = data.loc[:, "demand"].shift(48)

        for i in range(1, 24):
            data["temp_lag_" + str(i)] = data.loc[:, "temp"].shift(i)
        for i in range(1,24):
            data["temp_index_"+ str(i)] = data["temp_lag_" + str(i)] * data.loc[:, "scaled_temp"]

        l = list(data.loc[:, "scaled_temp"])
        diff = [l[i] - l[i-24] for i in range(24, len(l))]
        data["scaled_temp_diff_24"] = [0] * 24 + diff

        l = list(data.loc[:, "scaled_temp"])
        diff = [l[i] - l[i-48] for i in range(48, len(l))]
        data["scaled_temp_diff_48"] = [0] * 48 + diff
        
        data.set_index("date_time", inplace=True)
        data.fillna(method="backfill", inplace=True)

        return data
    
    def get_history(self, start = datetime.datetime(2019, 1, 1, 1, 0, 0), end = datetime.datetime(2022, 12, 31, 23, 0, 0)):

        # Returns slice of data from start to end as pandas dataframe
        return self.data.loc[start:end, :]
    