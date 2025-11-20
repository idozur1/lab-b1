import Signal_Processing
from scipy import signal as signal

from matplotlib import pyplot as plt
import numpy as np
import pandas as pn
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.interpolate as interp

# def sin(t, w, phi):
#     return np.sin(t*w + phi)
#
# def get_freq(x_data, y_data):
#     params = Signal_Processing.get_fit_params(sin, x_data, y_data)
#     fit_params = params[0]
#     return fit_params[0] #w

def get_freq(x_data, y_data, height, threshold):
    x_peaks = signal.find_peaks(y_data)
    return x_peaks

measurement = Signal_Processing.get_data2('Doppler_1311.xlsx')
for tab, data in measurement.items():
    x_data = [x for x in data.keys()]
    y_data = [y for y in data.keys()]
    peak_freqs = get_freq(x_data, y_data, height, threshold)
    print(freq)
# ###

def get_data1(filename: str) -> dict:
    ''' this function returns the data as a dict with keys the TAB NAMES and values the data as DF type'''
    all_sheet_dict = pn.read_csv(filename, sheet_name = None)
    for dist, data in all_sheet_dict.items():
        # sheet is a df
        all_sheet_dict[dist] = data.iloc[7:, :2]
    return all_sheet_dict

def find_freq(filename):
    df = get_data1(filename)
    t = np.array(df.iloc[:, 0])
    amps = np.array(df.iloc[:,1])
    peaks = scipy.signal.find_peaks(amps, distance = 100 # insert real interval)
    peak_num = 0
    for peak in peaks:
        peak_num += 1
    freq = peak_num / (t[-1]-t[0])

    return freq







