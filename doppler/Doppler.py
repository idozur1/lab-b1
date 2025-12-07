from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def get_data(filename):
    data = pd.read_csv(filename, header=None)
    df = data.iloc[:, 3:5]
    time = df.iloc[:, 0].values
    amps = df.iloc[:, 1].values
    dt = float(data.iloc[1, 1])
    return (data, df, time, amps, dt)


def fft(data: tuple):
    data, df, time, amps, dt = data
    dt_real = dt
    N = len(amps)

    # Zero-padding to get better frequency resolution
    # Pad to make duration = 0.1 seconds for ~10 Hz resolution
    desired_duration = 0.5  # seconds
    desired_N = int(desired_duration / dt_real)

    # pad with zeros
    amps_padded = np.pad(amps, (0, desired_N - N), mode='constant', constant_values=0)
    N_padded = len(amps_padded)

    # calculate fft:
    fft_output = np.fft.rfft(amps_padded)
    fft_mag = np.abs(fft_output)
    # normalize magnitudes:
    fft_mag_volts = fft_mag * (2 / N)  # Still normalize by original N
    freqs = np.fft.rfftfreq(N_padded, dt_real)

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

    return freqs_khz, fft_mag_volts, main_freq_khz, main_mag, main_freq

def main():
    plt.figure(figsize=(14, 6), dpi=150)
    fft_for = fft(get_data('200_ms_moves_forward.csv'))
    fft_stat = fft(get_data('200_ms_static.csv'))
    fft_back = fft(get_data('200_ms_move_backward.csv'))

    plt.subplot(3, 1, 1)
    plt.plot(fft_for[0], fft_for[1], linewidth=0.5)
    if fft_for[3] > 0:
        plt.scatter(fft_for[2], fft_for[3], color='red', marker='o', s=100, zorder=5,
                    label=f'Main Peak: {fft_for[4]:.2f} Hz')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude(V)')
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(fft_stat[0], fft_stat[1], linewidth=0.5)
    if fft_stat[3] > 0:
        plt.scatter(fft_stat[2], fft_stat[3], color='red', marker='o', s=100, zorder=5,
                    label=f'Main Peak: {fft_stat[4]:.2f} Hz')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude(V)')
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(fft_back[0], fft_back[1], linewidth=0.5)
    if fft_back[3] > 0:
        plt.scatter(fft_back[2], fft_back[3], color='red', marker='o', s=100, zorder=5,
                    label=f'Main Peak: {fft_back[4]:.2f} Hz')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude(V)')
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

#main()


def plot_single_fft(data: tuple):
    data, df, time, amps, dt = data
    dt_real = dt
    N = len(amps)

    # Zero-padding to get better frequency resolution
    # Pad to make duration = 0.1 seconds for ~10 Hz resolution
    desired_duration = 1  # seconds
    desired_N = int(desired_duration / dt_real)

    # pad with zeros
    amps_padded = np.pad(amps, (0, desired_N - N), mode='constant', constant_values=0)
    N_padded = len(amps_padded)

    # calculate fft:
    fft_output = np.fft.rfft(amps_padded)
    fft_mag = np.abs(fft_output)
    # normalize magnitudes:
    fft_mag_volts = fft_mag * (2 / N)  # Still normalize by original N
    freqs = np.fft.rfftfreq(N_padded, dt_real)

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

    plt.figure(figsize=(14, 6), dpi=150)
    plt.plot(freqs_khz, fft_mag_volts, linewidth=0.5)
    if main_mag > 0:
        plt.scatter(main_freq_khz, main_mag, color='red', marker='o', s=100, zorder=5,
                    label=f'Main Peak: {main_freq:.2f} Hz')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Amplitude(V)')
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#plot_single_fft(get_data(''))



def lower_sample_rate(data: tuple):
    data, df, time, amps, dt = data
    new_time = []
    new_amps = []
    for i in range(len(time)):
        if i % 1000 == 1:
            new_time.append(time[i])
            new_amps.append(amps[i])
    dt *= 1000
    # shift to positive:
    if new_time[0] < 0:
        for j in range(len(new_time)):
            new_time[j] += new_time[0]
    final_t = np.array(new_time)
    final_amps = np.array(new_amps)
    # here data and df doesnt change, but we dont use them in fft so its ok
    return data, df, final_t, final_amps, dt

plot_single_fft(lower_sample_rate(get_data('final/static.csv')))
plot_single_fft(lower_sample_rate(get_data('final/foreward_with_trigger.csv')))
plot_single_fft(lower_sample_rate(get_data('final/backward_good.csv')))
plot_single_fft(lower_sample_rate(get_data('final/30_truepm5.csv')))
plot_single_fft(lower_sample_rate(get_data('final/20pm5.csv')))