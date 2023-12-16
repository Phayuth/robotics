import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import pickle
from icecream import ic

matplotlib.rcParams.update({'font.size': 10})


def get_data_plot(loadedList):
    # Determine a common set of x-values based on the range of x-values in the datasets
    x_min = min(min(data_set, key=lambda x: x[0])[0] for data_set in loadedList)
    # x_max = max(max(data_set, key=lambda x: x[0])[0] for data_set in loadedList)
    common_x_values = np.arange(x_min, 3000, 1)

    # Interpolate y-values for the common_x_values in each dataset
    interpolated_data_sets = []
    for data_set in loadedList:
        x_values, y_values = zip(*data_set)
        interpolated_y_values = np.interp(common_x_values, x_values, y_values)
        interpolated_data_sets.append(interpolated_y_values)

    # Calculate the mean curve
    mean_curve = np.mean(interpolated_data_sets, axis=0)
    std_curve = np.std(interpolated_data_sets, axis=0)
    mean_p_std = mean_curve + std_curve
    mean_m_std = mean_curve - std_curve
    return common_x_values, mean_curve, mean_p_std, mean_m_std


############################################################# 6D Single ###################################################################################

nameList = ["inform", "connect", "connect_locgap", "starconnect", "starconnect_locgap"]

costGraphList = []
for name in nameList:
    with open(f"./datasave/planner_performance/single6dof/{name}_costgraph.pkl", "rb") as file:
        loadedList = pickle.load(file)
        costGraphList.append(loadedList)

for costGraphPerPlanner in costGraphList:
    countfail = np.sum([1 for innerlist in costGraphPerPlanner if len(innerlist) == 0])
    ic(countfail)

costGraphListFilter = []
for costGraphPerPlanner in costGraphList:
    costGraphPerPlanner = [innerlist for innerlist in costGraphPerPlanner if len(innerlist) != 0]
    costGraphListFilter.append(costGraphPerPlanner)

common_x_values_inform, mean_curve_inform, _, _ = get_data_plot(costGraphListFilter[0])
common_x_values_connect, mean_curve_connect, _, _ = get_data_plot(costGraphListFilter[1])
common_x_values_connect_loc, mean_curve_connect_loc, _, _ = get_data_plot(costGraphListFilter[2])
common_x_values_connectstar, mean_curve_connectstar, _, _ = get_data_plot(costGraphListFilter[3])
common_x_values_connectstar_loc, mean_curve_connectstar_loc, _, _ = get_data_plot(costGraphListFilter[4])

scale = 0.9
fig = plt.figure(figsize=(scale * 3.40067 * 2, scale * 3.40067), frameon=True, layout='tight')
ax = plt.subplot(1, 1, 1)

ax.plot(common_x_values_inform, mean_curve_inform, label="Inform-RRT*", linewidth=3, linestyle='solid', color="#558b2f")
ax.plot(common_x_values_connect, mean_curve_connect, label="RRT-Connect", linewidth=3, linestyle='solid', color="#ff8f00")
ax.plot(common_x_values_connect_loc, mean_curve_connect_loc, label="RRT-Connect + LocGap", linewidth=3, linestyle='--', color="#1565c0")
ax.plot(common_x_values_connectstar, mean_curve_connectstar, label="RRT*-Connect", linewidth=3, linestyle='--', color="#9e9d24")
ax.plot(common_x_values_connectstar_loc, mean_curve_connectstar_loc, label="RRT*-Connect + LocGap", linewidth=3, linestyle='--', color="#6a1b9a")

ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.grid(axis='y')
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis="both", which="major", direction='in', length=3, width=1, colors='black', grid_color='gray', grid_alpha=0.2)
ax.tick_params(axis="both", which="minor", direction='in', length=1, width=1, colors='black', grid_color='gray', grid_alpha=0.1)
ax.set_xlim((0, 3025))
ax.legend(loc=1)

############################################################# 2D Multi ###################################################################################

nameList = ["connect", "connect_locgap", "starconnect", "starconnect_locgap"]

costGraphList = []
for name in nameList:
    with open(f"./datasave/planner_performance/multi6dof/{name}_costgraph.pkl", "rb") as file:
        loadedList = pickle.load(file)
        costGraphList.append(loadedList)

for costGraphPerPlanner in costGraphList:
    countfail = np.sum([1 for innerlist in costGraphPerPlanner if len(innerlist) == 0])
    ic(countfail)

costGraphListFilter = []
for costGraphPerPlanner in costGraphList:
    costGraphPerPlanner = [innerlist for innerlist in costGraphPerPlanner if len(innerlist) != 0]
    costGraphListFilter.append(costGraphPerPlanner)

# common_x_values_inform, mean_curve_inform, _, _ = get_data_plot(costGraphListFilter[0])
common_x_values_connect, mean_curve_connect, _, _ = get_data_plot(costGraphListFilter[0])
common_x_values_connect_loc, mean_curve_connect_loc, _, _ = get_data_plot(costGraphListFilter[1])
common_x_values_connectstar, mean_curve_connectstar, _, _ = get_data_plot(costGraphListFilter[2])
common_x_values_connectstar_loc, mean_curve_connectstar_loc, _, _ = get_data_plot(costGraphListFilter[3])

fig1 = plt.figure(figsize=(scale * 3.40067 * 2, scale * 3.40067), frameon=True, layout='tight')
ax1 = plt.subplot(1, 1, 1)

ax1.plot(common_x_values_connect, mean_curve_connect, label="RRT-Connect", linewidth=3, linestyle='solid', color="#ff8f00")
ax1.plot(common_x_values_connect_loc, mean_curve_connect_loc, label="RRT-Connect + LocGap", linewidth=3, linestyle='--', color="#1565c0")
ax1.plot(common_x_values_connectstar, mean_curve_connectstar, label="RRT*-Connect", linewidth=3, linestyle='--', color="#9e9d24")
ax1.plot(common_x_values_connectstar_loc, mean_curve_connectstar_loc, label="RRT*-Connect + LocGap", linewidth=3, linestyle='--', color="#6a1b9a")

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Cost")
ax1.grid(axis='y')
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.tick_params(axis="both", which="major", direction='in', length=3, width=1, colors='black', grid_color='gray', grid_alpha=0.2)
ax1.tick_params(axis="both", which="minor", direction='in', length=1, width=1, colors='black', grid_color='gray', grid_alpha=0.1)
ax1.set_xlim((0, 3025))
ax1.legend(loc=1)

plt.show()