from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

FILE_DIR = '6.385E-6.xlsx'
INTERVAL_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64, 70, 75] # insert intervals in number of frames
#INTERVAL_LIST = [5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35] # smaller interval list for checks
FRAMERATE = 1/4.4 # insert framerate in secs (eg 1/15)
#PARTICLE_LIST = ['6.385E-6.xlsx',
#                  '7.071E-6.xlsx',
#                  '3.869E-6.xlsx',
#                  '3.436E-6.xlsx',
#                  '2.993E-6.xlsx',
#                  '2.428E-6.xlsx'] # week 1 particle list
PARTICLE_LIST = ['week2/40% - 2.753E-6.xlsx',
                 'week2/35% - 2.794E-6.xlsx',
                 'week2/30% - 2.893E-6.xlsx',
                 'week2/25% - 2.386E-6.xlsx',
                 'week2/10% - 2.910E-6.xlsx',
                 'week2/7% - 2.278E-6.xlsx',
                 'week2/7% - 2.238E-6.xlsx']

def get_data(filename: str) -> pd.DataFrame:
    ''' a function that takes the file name (csv) and returns data frame (also name for plotting)'''
    data = pd.read_excel(filename)
    data.columns = ['t', 'x', 'y']
    return data ,filename

def get_mean_r2_single_with_drift(data: pd.DataFrame, interval: int, drift: Tuple):
    ''' a function that receives the data then returns <r^2> for a single interval. interval is a number of frames'''
    r2_list = []
    idx = 0
    while (idx + interval) < len(data):
        curr_x0 = data.iloc[idx, 1]
        curr_y0 = data.iloc[idx, 2]
        dx = (data.iloc[idx + interval, 1]-drift[0]) - curr_x0
        dy = (data.iloc[idx + interval, 2]-drift[1]) - curr_y0
        curr_r2 = dx ** 2 + dy ** 2
        r2_list.append(curr_r2)
        idx += 1
    mean_r2 = np.mean(r2_list)
    return mean_r2

def get_mean_r2_single_no_drift(data: pd.DataFrame, interval: int):
    ''' a function that receives the data then returns <r^2> for a single interval. interval is a number of frames'''
    r2_list = []
    idx = 0
    while (idx + interval) < len(data):
        curr_x0 = data.iloc[idx, 1]
        curr_y0 = data.iloc[idx, 2]
        dx = (data.iloc[idx + interval, 1]) - curr_x0
        dy = (data.iloc[idx + interval, 2]) - curr_y0
        curr_r2 = dx ** 2 + dy ** 2
        r2_list.append(curr_r2)
        idx += 1
    mean_r2 = np.mean(r2_list)
    return mean_r2

def get_mean_r2_with_drift(data: pd.DataFrame, interval_list: list, framerate: float):
    ''' a function that receives the data, uses mean_r2_single for the entire interval list, and returns a final df of time/<r2>'''
    dt = framerate
    times = [i * dt for i in interval_list]
    mean_r2_list =[]
    for interval in interval_list:
        curr_mean_r2 = get_mean_r2_single_with_drift(data, interval, find_drift(data, 2))
        mean_r2_list.append(curr_mean_r2)
    new_df = pd.DataFrame({'time': times, 'mean_r2': mean_r2_list})
    return new_df

def get_mean_r2_no_drift(data: pd.DataFrame, interval_list: list, framerate: float):
    ''' a function that receives the data, uses mean_r2_single for the entire interval list, and returns a final df of time/<r2>'''
    dt = framerate
    times = [i * dt for i in interval_list]
    mean_r2_list =[]
    for interval in interval_list:
        curr_mean_r2 = get_mean_r2_single_no_drift(data, interval)
        mean_r2_list.append(curr_mean_r2)
    new_df = pd.DataFrame({'time': times, 'mean_r2': mean_r2_list})
    return new_df


def find_drift(df: pd.DataFrame, interval):
    drifts = df.iloc[:, [1, 2]].diff(periods=interval)
    means = drifts.mean()
    return means.iloc[0], means.iloc[1]

def find_drift2():
    ...
    # try to find drift using (<r^2) - (<r)^2

def linear_fit(x, a, b):
    return a * x + b

def plot_single_mean_r2_time(df: pd.DataFrame, size, filename):
    ''' plot function for a single particle of <r^2>/time'''
    plt.figure(figsize = (12, 7))
    size_microns = size * 1e6

    #fit:
    popt, pcov = curve_fit(linear_fit, df.time, df.mean_r2)
    slope, intercept = popt
    slope_err = np.sqrt(np.diag(pcov))[0]

    # f"{value:.2f}" rounds to 2 decimal places
    label_text = f"{size_microns:.2f} $\mu$m"

    scatter = plt.scatter(df.time, df.mean_r2, label=label_text)
    x_fit = np.linspace(df.time.min(), df.time.max(), 100)
    y_fit = linear_fit(x_fit, *popt)

    fit_label = f"Fit: slope={slope:.2e} $\pm$ {slope_err:.1e}"
    print(f"{filename}: {fit_label}")
    plt.plot(x_fit, y_fit, color='black', linestyle='-', label=fit_label)

    plt.title(filename)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Mean $\langle r^2 \rangle$ ($m^2$)')
    plt.grid(True, alpha=0.3)
    plt.legend(title="Particle Diameter")  # Adds a title to the box
    plt.show()


def plot_final_mean_r2(df_list: list):
    plt.figure(figsize=(12, 7))
    for df, size in df_list:
        # Convert Meters to Microns (multiply by 1 million)
        size_microns = size * 1e6

        # f"{value:.2f}" rounds to 2 decimal places
        label_text = f"{size_microns:.2f} $\mu$m"

        # plt.scatter(df.time, df.mean_r2, label=label_text)
        plt.plot(df.time, df.mean_r2,'o-', label=label_text)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Mean $\langle r^2 \rangle$ ($m^2$)')
    plt.grid(True, alpha=0.3)
    plt.legend(title="Particle Diameter")  # Adds a title to the box
    plt.show()

# def plot_final_mean_r2(df_list: list):
#     ''' final plot function of several particles <r^2>/time. it receives a LIST OF DF!!!
#         it gets a list of tuples (df, size)'''
#     plt.figure(figsize=(12, 7))
#     for df, size in df_list:
#         plt.plot(df.time, df.mean_r2)
#         plt.legend(size)
#     plt.xlabel('Time (s)')
#     plt.ylabel(r'Mean $r^2$')  # need to insert units
#     plt.grid(True, alpha=0.3)
#     plt.show()

def single():
    for FILE_DIR in PARTICLE_LIST:
        data, filename = get_data(FILE_DIR)
        size = float(filename[12:-5])
        plot_single_mean_r2_time(get_mean_r2_with_drift(data, INTERVAL_LIST, FRAMERATE), size, filename) #change to with drift t find drift

def final():
    res_list = []
    for particle in PARTICLE_LIST: #particle is a string of the filename
        particle_data, filename = get_data(particle)
        size = float(filename[12:-5])
        curr_mean_r2 = get_mean_r2_with_drift(particle_data, INTERVAL_LIST, FRAMERATE)
        res_list.append((curr_mean_r2, size))
    plot_final_mean_r2(res_list)

if __name__ == '__main__':
    single()
    final()


#test
