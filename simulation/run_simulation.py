import sys
sys.path.insert(0, "C:\\Users\\jerom\\electricity_load_forecasting\\Electricity-load-forecasting\\")


from simulation import Simulation
import datetime

train_start = datetime.datetime(2019, 1, 1, 0, 0, 0)
train_end = datetime.datetime(2020, 12, 31, 23, 0, 0)

sim = Simulation(365 * 24, train_start, train_end)
forecasts = sim.run_simulation()
sim.plot_sim_results(forecasts)


