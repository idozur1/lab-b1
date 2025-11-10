from matplotlib import pyplot as plt
import numpy as np
import pandas as pn
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# 'frequency2hz.xlsx'
# 'frequency75hz.xlsx'
# def get_data(sheet_alias: str) -> dict:
#     # make a dictionary from all the tabs
#     # each tab is one wave measurement with varying distance from the LED
#     all_sheets_dict = pn.read_excel(sheet_alias, sheet_name=None)
#     for sheet in all_sheets_dict.values():
#         # cut unnecessary data
#         sheet = sheet.iloc[7:, :2]
#     return all_sheets_dict

def get_data(filename: str) -> dict:
    ''' this function returns the data as a dict with keys the TAB NAMES and values the data as DF type'''
    all_sheet_dict = pn.read_excel(filename, sheet_name = None)
    for dist, data in all_sheet_dict.items():
        # sheet is a df
        all_sheet_dict[dist] = data.iloc[7:, :2]
    return all_sheet_dict

#define fit

#THIS IS A TEST

def fit_func(r, k, b):
        return k/r**2 + b

def get_params(x_data, y_data):

    params, covariance = curve_fit(fit_func, x_data, y_data)
    output = params, covariance

    return output

def fit(x_data, y_data):
    params = get_params(x_data, y_data)[0]
    y_fit = [fit_func(i,params[0],params[1]) for i in x_data]

    return y_fit

#Analysis

def analyse_fourier(dataset: dict) -> dict:

    results = {}

    #calculate fft and save to dict

    for dist, data in dataset.items():
        volt_data = list(data.iloc[:,1])
        length = len(volt_data)
        fft = np.abs(np.fft.fft(volt_data, norm = 'forward')) #normalization problem - amp too high in close ditsances
        output_freqs = np.fft.fftfreq(length, 0.001) #since measurement is in 1/ms
        output = {}
        for i in range(len(output_freqs)):
            output[output_freqs[i]] = fft[i]

        #save result

        amp = max(fft)
        results[dist] = amp

        #trim dict to interest frequencies

        interest_output = {}
        for freq,value in output.items():
            if freq < 100 and freq > 0:
                interest_output[freq] = value

        max_value = max(interest_output.values())
        max_amp_freq, max_amp = [(key, value) for key, value in interest_output.items() if value == max_value][0]

        #plot fourier trasnform
        #
        # plt.title(f'Fourier Transform of measurement from a distance of {dist}')
        # plt.plot(interest_output.keys(), interest_output.values(), label='fft')
        # plt.annotate(f'Freq: {max_amp_freq:.2f}, Amplitude: {max_amp:.2f}',
        #              xy = (max_amp_freq, max_amp),
        #              xytext = (max_amp_freq, max_amp),
        #              arrowprops=dict(facecolor='black', shrink=0.05),  # Arrow properties
        #              fontsize=10,
        #              color='red')
        # plt.xlabel('Frequency(1/s)')
        # plt.ylabel('Amplitude (V)')
        # #plt.legend()
        # plt.show()

    return(results)

def analyze_min_max(data :dict, freq: int):
    result_dict ={}
    for dist in data:
        measurement = data[dist].iloc[:, 1].values
        # 10% error range for peak find:
        min_dist = int((len(measurement)/freq)*0.9)
        peaks = list(measurement[i] for i in find_peaks(measurement, distance=min_dist)[0])
        valleys = list(measurement[j] for j in find_peaks(-measurement, distance=min_dist)[0])
        amp = (np.mean(peaks) - np.mean(valleys)) / 2
        result_dict[int(dist[:2])] = amp
    return result_dict

# This is a test 2

def min_max_fit_model(r, k, c):
    # model of decaying light when light is transmitted as a point source is proportional to 1/r^2.
    # constant is added for noise in the system'
    return k/(r**2) + c

def min_max_fit_func(data_dict: dict, fit_model):
    x_values = np.array([key for key in data_dict.keys()])
    y_values = np.array([val for val in data_dict.values()])
    popt, pcov = curve_fit(fit_model, x_values, y_values)
    x_fit = np.linspace(min(x_values), max(x_values), 100)
    y_fit = fit_model(x_fit, *popt)
    return (x_fit, y_fit, popt)

#results plot

def get_fourier_plot(files,x_mistake):
    plt.title("Amplitude VS Distance - Fourier method")
    atten_results = {}
    for freq, filepath in files.items():
        dataset = get_data(filepath)
        results = analyse_fourier(get_data(filepath))
        for key, value in results.items():
            atten_results[int(key[0:len(files.keys())])] = value-x_mistake

        x_data = list(atten_results.keys())
        y_data = list(atten_results.values())

        x_for_fit = [i for i in range(10,100)]
        params = get_params(x_data,y_data)[0]
        y_for_fit = [fit_func(i, params[0],params[1]) for i in x_for_fit]

        plt.scatter(x_data, y_data, label = f"{freq} Hz Data")
        #plt.errorbar(x_data, y_data, xerr = 0.001, yerr = 0.001, fmt='o', capsize=3) #Errorbars are too small to see

        plt.plot(x_for_fit, y_for_fit, label = f"fit - {freq} Hz Data"),
        plt.legend()
    plt.show()

def get_min_max_plot(amp_dist_dict: dict, name: str):
    # scatter the measurement's data
    x_values = [i for i in amp_dist_dict.keys()]
    y_values = [j for j in amp_dist_dict.values()]
    plt.scatter(x_values, y_values, label = f'{name} Data', s=25)
    #plot a fit line:
    fit = min_max_fit_func(amp_dist_dict, min_max_fit_model)
    plt.plot(fit[0], fit[1], linestyle='-', label=f'{name} Fit', lw = 1)
    # graph visualisations:
    plt.legend()
    plt.xlabel("Distance from LED (cm)")
    plt.ylabel("Amplitude (V)")
    plt.title("Amplitude VS Distance - min-max method")

files = {"2": "frequency2hz.xlsx",
            "75": "frequency75hz.xlsx"}
x_mistake = 0.0097

get_fourier_plot(files,x_mistake)
get_min_max_plot(analyze_min_max(get_data(files["2"]), 2), "2Hz")
get_min_max_plot(analyze_min_max(get_data(files["75"]), 75), "75Hz")
plt.show()

#fit test
#
# x_data = [1,2,3,4]
# y_data = [1,1/4,1/9,1/16]
#
# params = get_params(x_data,y_data)[0]
# y_fit = fit(x_data,y_data)
#
# plt.scatter(x_data,y_fit, label = 'fit')
# plt.scatter(x_data, y_data, label = 'org')
# plt.legend()
# plt.show()
#title = 'Ampltiude (V) as a function of distance (m)'

# week 2:

def fourier_trans(data: list, period) -> tuple:
    #example usage: fourier_trans([1,8,1,0,0,5,4,2 5], 0.5) for a 4 sec measurement in 2 Hz (4*2=8 items in list)

    length = len(data)
    fft = np.abs(np.fft.fft(data,
                            norm='forward'))  # norm = 'forward' divides rsults by n number of counts keeping normalization (inverse fourier requires no further division)
    output_freqs = np.fft.fftfreq(length, period)
    output_data = {}
    for i in range(length):
        output_data[output_freqs[i]] = fft[i]

    return output_data, period

BG_NOISE = pn.read_excel("noise.xlsx")
NOISE_FFT = fourier_trans(BG_NOISE.iloc[:,1].tolist(), 0.0002)

def noise_reduction(data: dict) -> dict:
     # recieves a dict of freq:volt values in frequency doamin
     new_data = {}
     for freq in data.keys():
         new_data[freq] = data[freq] - NOISE_FFT[freq]
     return new_data

# Aliasing:

def alias_plot(filename):
    alias_data = get_data(filename)
    for name, df in alias_data.items():
        time_vals = df.iloc[:,0].tolist()
        interval = time_vals[1] - time_vals[0]
        volt_vals = df.iloc[:,1].tolist()
        meas_spectrum = fourier_trans(volt_vals, interval)[0]

        #plot:
        x_data = []
        y_data = []
        for freq, amp in meas_spectrum.items():
            if freq >= 0:
                x_data.append(freq)
                y_data.append(amp)
        sorted_pairs = sorted(zip(x_data, y_data))
        freqs = [pair[0] for pair in sorted_pairs]
        amps = [pair[1] for pair in sorted_pairs]
        plt.figure(figsize = (10, 5))
        plt.plot(freqs, amps)
        plt.title(str(name))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (V)")
        plt.grid(True)
        plt.show()

alias_plot("alias.xlsx")






# week 3:

def demodulate_AM_wave(filename, carrier_freq) -> dict:
    # works only for cosine carrier waves. doesnt work for sine or a combination.
    # take the df of the wave in time space
    am_wave = get_data(filename).values()
    t = np.array(am_wave[:, 0])
    # make local carrier cos(w0t)
    loc_carrier = np.cos(2*np.pi*carrier_freq*t)
    # multiply with original signal
    mod_wave = am_wave[:, 1] * loc_carrier
    # analyze fourier:
    interval = am_wave.iloc[2,0]-am_wave.iloc[1,0]
    mod_wave_spectrum = fourier_trans(mod_wave,interval)
    # READ THIS!!
    # need to finish -
        # 1. take mod wave (now at freq domain) and filter it
        # understand how to filter -> we need to filter out a(2-2w0) and a(w+20)
        # ifft the filtered wave and get a(t)



    # original_data_signal = ifft(fft_am_wave)
     #return original_data_signal

#old demodulating AM thoruhg moving freq spectrum. too hard.

#def demodulate_AM_wave(filename, carrier_freq) -> dict:
    # works only for cosine carrier waves. doesnt work for sine or a combination.
    # take the df of the wave in time space
 #   am_wave = get_data(filename).values()
    # analyze fourier:
  #  interval = am_wave.iloc[2,0]-am_wave.iloc[1,0]
   # am_wave_spectrum = fourier_trans(am_wave.iloc[:,1],interval)
    # shift by w0 to the right
    #move_right_wave = deepcopy(am_wave_spectrum)
  #  for freq in move_right_wave[0].keys():
     #   move_right_wave[freq + carrier_freq] = move_right_wave[freq]
      #  move_right_wave[freq] = 0
    # shift by w0 to the right:
    #move_left_wave = deepcopy(am_wave_spectrum)
    #for freq in move_left_wave[0].keys():
     #   move_left_wave[freq] -= carrier_freq
    # combine them. we get a^(w) + 1/2 a^(w-2w0) + 1/2 a^(w+2w0):
    ##   am_wave_spectrum[freq] = move_right_wave[freq] + move_left_wave[freq]
    # filter w-2w0, w+2w0: