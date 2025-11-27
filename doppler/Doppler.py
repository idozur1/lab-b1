from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.interpolate as interp


def get_data(filename):
    data = pd.read_csv(filename, header=None)
    df = data.iloc[:, 3:5]
    time = df.iloc[:, 0].values
    amps = df.iloc[:, 1].values
    dt = float(data.iloc[1,1])
    return (data, df, time, amps, dt)

def fft(data: tuple):
    # data = (data, df, time, amps, dt)
    dt = data[4]
    N = len(data[3])
    # calculate fft:
    fft_output = np.fft.rfft(data[3])
    fft_mag = np.abs(fft_output)
    #normlize magnitudes:
    fft_mag_volts = fft_mag * (2/N)
    freqs = np.fft.rfftfreq(N, dt)
    print(len(freqs))
    # find max value (peak) - main freq:
    search_spectrum = fft_mag[1:]
    peak_height_threshold = np.max(search_spectrum) * 0.1
    peak_indices, _ = find_peaks(search_spectrum, height=peak_height_threshold)
    main_freq, main_mag = 0, 0
    if len(peak_indices) > 0:
        real_peak_indices = peak_indices + 1
        main_peak_index = real_peak_indices[np.argmax(fft_mag_volts[real_peak_indices])]
        main_freq = freqs[main_peak_index]
        main_mag = fft_mag_volts[main_peak_index]

    freqs_khz = freqs / 1000
    main_freq_khz = main_freq / 1000

    # plot
    plt.figure(figsize=(10,5))
    plt.plot(freqs_khz, fft_mag_volts)
    if main_mag > 0:
        plt.scatter(main_freq_khz, main_mag, color='red', marker='o', s=100, zorder=5,
                    label=f'Main Peak: {main_freq:.2f} Hz')
    plt.xlabel('Frequency (KHz)')
    plt.ylabel('Amplitude(V)')
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


fft(get_data('foreward_with_trigger.csv'))