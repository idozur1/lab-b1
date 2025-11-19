from matplotlib import pyplot as plt
import numpy as np
import pandas as pn
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.interpolate as interp

# helper functions:

def get_data1(filename: str) -> dict:
    ''' this function returns the data as a dict with keys the TAB NAMES and values the data as DF type'''
    all_sheet_dict = pn.read_excel(filename, sheet_name = None)
    for dist, data in all_sheet_dict.items():
        # sheet is a df
        all_sheet_dict[dist] = data.iloc[7:, :2]
    return all_sheet_dict

def get_data2(filename: str) -> dict:
    ''' this function returns the data as a dict with keys the TAB NAMES and values the data as DF type'''
    all_sheet_dict = pn.read_excel(filename, sheet_name = None)
    output = {}
    for tab, data in all_sheet_dict.items():
        data = data.iloc[10:,:2]
        data_dict = {}
        for i in range(len(data.iloc[:,1])):
            data_dict[data.iloc[i,0]] = data.iloc[i,1]
        output[tab] = data_dict
    return output

def get_time_interval(data:dict) -> float:
    ''' this function gets a measurment dict with keys (meas times) in seconds and returns delta t'''
    meas_times = np.array(list(data.keys()))
    time_interval = np.abs(np.average([meas_times[i+1]-meas_times[i] for i in range(len(meas_times)-1)]))
    return time_interval


def get_interest_freqs(fft: dict, min_freq, max_freq, with_0 = True) -> dict:
    interest_output = {}
    for freq, value in fft.items():
        if (freq < max_freq) and (freq > min_freq) and not((with_0 == False) & (freq == 0)):
            interest_output[freq] = value
    return interest_output

def get_residuals(data:dict, fit_func):

    residuals = {}

    for w, value in data.items():
        residuals[w] = value-fit_func(w)

    return residuals

# fitting functions:

def sinc(w:float,A:float,T:float):
    return A*T*np.sinc(w*T) #check Matap 2 Week 7 Page 12 for pituach

def sinc2(w:float,A:float,T:float):
    return sinc(w,A,T)**2

def tri_fourier_fit(x):
    # need to complete this
    ...

def fit_func(r, k, b):
        return k/r**2 + b

def get_fit_params(fit_func, x_data, y_data):

    params, covariance = curve_fit(fit_func, x_data, y_data)
    output = params, covariance

    return output

def fit(x_data, y_data):
    params = get_fit_params(x_data, y_data)[0]
    y_fit = [fit_func(i,params[0],params[1]) for i in x_data]

    return y_fit

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

# Analysis functions:

def fourier_trans(data: dict, with_0:bool, time_interval={'mode':'auto', 'value': None}, min_freq = -500, max_freq = 500) -> dict:
    #example usage: fourier_trans([1,8,1,0,0,5,4,2 5], 0.5) for a 4 sec measurement in 2 Hz (4*2=8 items in list)

    #setting variables

    meas_times = np.array(list(data.keys()))
    meas_values = np.array(list(data.values()))

    if time_interval['mode'] == 'auto':
        time_interval = get_time_interval(data)
    else:
        time_interval = time_interval['value']

    #calculate fft and save to dict

    fft = (np.fft.rfft(meas_values,
                            norm='forward'))  # norm = 'forward' divides results by n number of counts keeping normalization (inverse fourier requires no further division)
    output_freqs = np.fft.rfftfreq(len(meas_values), time_interval) #size is floor(len(fft)/2) because in rfft only positive values are calculated
    output_data = {}
    for i in range(len(output_freqs)):
        output_data[output_freqs[i]] = fft[i]

    output_data = get_interest_freqs(output_data, min_freq, max_freq, with_0)

    output = {'fourier_trans': output_data,
              'time_interval': time_interval}

    return output

def file_fourier_trans(dataset: str, with_0:bool, time_interval={'mode':'auto', 'value': None}, min_freq = -500, max_freq = 500) -> dict:
    ''' this function returns a dict with tab_name:df(fft:amp)'''
    output = {}

    dataset = get_data2(dataset)

    for tab, data in dataset.items():
        output[tab] = fourier_trans(data, with_0, time_interval, min_freq, max_freq)

    return output

def dataset_fourier_trans(dataset: dict, with_0: True, min_freq: float, max_freq: float, time_interval={'mode':'auto', 'value': None}) -> dict:

    results = {}

    for tab, data in dataset.items():

        output = fourier_trans(data, with_0, time_interval)['fourier_trans']

        #save results

        max_amp_freq = max(output, key=lambda k: np.abs(output[k]))
        max_amp = output[max_amp_freq]  # Get the complex value
        results[max_amp_freq] = max_amp

    return results

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

# noise reduction:

BG_NOISE_df = pn.read_excel("noise.xlsx")
BG_NOISE_df = BG_NOISE_df.iloc[10:, :2]
noise_data_dict = {}
for i in range(len(BG_NOISE_df.iloc[:,1])):
    noise_data_dict[BG_NOISE_df.iloc[i,0]] = BG_NOISE_df.iloc[i,1]
fft_noise_output = fourier_trans(noise_data_dict, with_0=True, time_interval={'mode':'manual', 'value': 0.0002})
fft_noise_dict = fft_noise_output['fourier_trans']
noise_freqs = sorted([f for f in fft_noise_dict.keys() if f>0])
noise_amps = [np.abs(fft_noise_dict[f]) for f in noise_freqs]
noise_amps_db = []
NOISE_INTERP = interp.interp1d(noise_freqs, noise_amps, bounds_error=False, fill_value=0)

def noise_reduction(data: dict) -> dict:
    # receives a dict of freq:volt values in frequency domain
    new_data = {}
    for freq, amp in data.items():
        if freq >= 0:
            estimated_noise = NOISE_INTERP(freq)
            amp_mag = np.abs(amp)
            new_data[freq] = max(0, amp_mag - estimated_noise)
        else:
            new_data[freq] = amp
    return new_data

# old function:
# old analyse_fourier func
# def analyse_fourier(dataset: dict) -> dict:

  #  results = {}

    #calculate fft and save to dict

    #for dist, data in dataset.items():
     #   volt_data = list(data.iloc[:,1])
      #  length = len(volt_data)
       # fft = np.abs(np.fft.fft(volt_data, norm = 'forward')) #normalization problem - amp too high in close ditsances
        #output_freqs = np.fft.fftfreq(length, 0.001) #since measurement is in 1/ms
        #output = {}
        #for i in range(len(output_freqs)):
        #    output[output_freqs[i]] = fft[i]
        #output = noise_reduction(output)
        #save result

       # amp = max(fft)
       # results[dist] = amp

        #trim dict to interest frequencies

       # interest_output = {}
       # for freq,value in output.items():
        #    if freq < 100 and freq > 0:
         #       interest_output[freq] = value

        #max_value = max(interest_output.values())
        #max_amp_freq, max_amp = [(key, value) for key, value in interest_output.items() if value == max_value][0]

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

    #return(results)





#results plot

# plot functions:

def get_fourier_plot(files,x_mistake, max_freq, min_freq):
    #plots the final plot of week 1, based on multiple Fourier analysises
    plt.title("Amplitude VS Distance - Fourier method")
    atten_results = {}
    for freq, filepath in files.items():
        dataset = get_data2(filepath)
        results = fourier_trans(get_data2(filepath),max_freq, min_freq)
        for key, value in results.items():
            atten_results[int(key[0:len(files.keys())])] = value-x_mistake

        x_data = list(atten_results.keys())
        y_data = list(atten_results.values())

        x_for_fit = [i for i in range(min_freq,max_freq)]
        params = get_fit_params(x_data,y_data)[0]
        y_for_fit = [fit_func(i, params[0],params[1]) for i in x_for_fit]

        plt.scatter(x_data, y_data, label = f"{freq} Hz Data")
        #plt.errorbar(x_data, y_data, xerr = 0.001, yerr = 0.001, fmt='o', capsize=3) #Errorbars are too small to see

        plt.plot(x_for_fit, y_for_fit, label = f"fit - {freq} Hz Data"),
        plt.legend()
    #plt.show()

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

def plot_fourier_transform(data:dict, title:str, xlabel:str, ylabel:str, fit_func = None, residuals_flag = False):

    #variabales

    freqs = [float(i) for i in data.keys()]
    values = [np.abs(i) for i in data.values()]

    #get fit
    if fit_func is not None:
        params = get_fit_params(fit_func,freqs,values)[0]

        def fit_func_with_params(w):
            return fit_func(w,params[0],params[1])

        fit_data = {}
        for w in freqs:
            fit_data[w] = fit_func_with_params(w) #assumes fit function with 2 fitting parameters
        plt.plot(fit_data.keys(),fit_data.values(), label = 'fit', color = 'orange')

    #get residuals
    if residuals_flag == True:
        residuals = get_residuals(data, fit_func_with_params)
        plt.scatter(residuals.keys(), residuals.values(), label = 'residuals')

    #plot

    plt.title(title)
    plt.plot(freqs, values, label='fft')
    # plt.annotate(f'Freq: {max_amp_freq:.2f}, Amplitude: {max_amp:.2f}',
    #              xy = (max_amp_freq, max_amp),
    #              xytext = (max_amp_freq, max_amp),
    #              arrowprops=dict(facecolor='black', shrink=0.05),  # Arrow properties
    #              fontsize=10,
    #              color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    #plt.show()

# Main Functions:

def get_rect_transform(fit_func, residuals_flag = False):
    dataset = file_fourier_trans('triangle.xlsx', min_freq=0, max_freq=100, with_0=True)
    for tab, output in dataset.items():
        peaks = {}

        for key, value in output['fourier_trans'].items():
            if np.abs(value) > 0.002:
                peaks[key] = value

        plot_fourier_transform(peaks, tab, residuals_flag=residuals_flag, xlabel='Frequency (Hz)', ylabel='Amplitude (Volts)', fit_func=fit_func)

#get_rect_transform(sinc2, residuals_flag = False)

def week1_main():
    files = {"2": "frequency2hz.xlsx",
             "75": "frequency75hz.xlsx"}
    x_mistake = 0.0097

    #get_fourier_plot(files, x_mistake)
    #get_min_max_plot(analyze_min_max(get_data1(files["2"]), 2), "2Hz")
    #get_min_max_plot(analyze_min_max(get_data1(files["75"]), 75), "75Hz")
    #plt.show()

# week 2:

# plot noise graph, change to db in the beggining of the code if needed

#plt.plot(noise_freqs, noise_amps)
#plt.xlim(0, 500)
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Amplitude (dB)")
#plt.grid(True)
#plt.show()

# Aliasing:

def alias_plot(filename):
    alias_data = get_data1(filename)
    for name, df in alias_data.items():
        time_vals = df.iloc[:,0].tolist()
        interval = time_vals[1] - time_vals[0]
        volt_vals = df.iloc[:,1].tolist()
        data_dict = dict(zip(time_vals, volt_vals))
        fft_output = fourier_trans(data_dict, with_0=True, time_interval = {'mode':'manual', 'value':interval})
        meas_spectrum = fft_output['fourier_trans']
        filtered_spectrum = noise_reduction(meas_spectrum)
        #plot:
        x_data = []
        y_data = []
        for freq, amp in filtered_spectrum.items(): #change meas_spectrum to filtered_spectrum to get noise reduction
            if freq >= 0:
                x_data.append(freq)
                y_data.append(amp)
        epsilon = 1e-12 # small epsilon to avoid log(0)
        #for i in range(len(y_data)): # convert V -> dB
          #  y_data[i] = 20 * np.log10(np.abs(y_data[i]) + epsilon)
        sorted_pairs = sorted(zip(x_data, y_data))
        freqs = [pair[0] for pair in sorted_pairs]
        amps = [pair[1] for pair in sorted_pairs]
        plt.figure(figsize = (10, 5))
        plt.plot(freqs, amps)
        plt.title(str(name))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (V)")
        plt.grid(True)
        plt.savefig(f"{name}.png")
        #plt.show()

#alias_plot("alias.xlsx")






# week 3:

def demodulate_am_wave(filename, carrier_freq) -> dict:
    # works only for cosine carrier waves. doesnt work for sine or a combination.
    # take the df of the wave in time space
    all_sheets = get_data1(filename)
    first_sheet_name = list(all_sheets.keys())[0]
    am_wave_df = all_sheets[first_sheet_name]
    # extract data:
    t_data = am_wave_df.iloc[:, 0]
    amps_data = am_wave_df.iloc[:, 1]
    t = pn.to_numeric(t_data, errors='coerce')
    amps = pn.to_numeric(amps_data, errors='coerce')

    # applying noise reduction here is complicated, start without and add later
    #original_data_dict = dict(zip(t, amps))
    # noise reduction:
    #filtered_freq_original = noise_reduction(fourier_trans(original_data_dict)['fourier_trans'])
    # how to apply ifft?

    # make local carrier cos(w0*t):
    loc_carrier = np.cos(2*np.pi*carrier_freq*t)
    # multiply with original signal:
    mod_wave_amps = amps * loc_carrier
    # calculate full fft:
    n_samples = len(t)
    dt = t.iloc[1] - t.iloc[0]
    full_fft_vals = np.fft.rfft(mod_wave_amps, norm ='forward')
    full_freqs = np.fft.rfftfreq(n_samples, dt)
    # apply low-pass filter:
    CUTOFF_FREQ = 50 #in hz
    lp_filtered_array = np.zeros_like(full_fft_vals) # zero padding with N/2 + 1 length
    for i, freq in enumerate(full_freqs):
        if (freq <= CUTOFF_FREQ):
            lp_filtered_array[i] = full_fft_vals[i]
        # else remains 0 (thats the low pass filter)
    # make inverse fft after LP filtering:
    recovered_data = np.fft.irfft(lp_filtered_array, n=n_samples)
    # Plotting:
    plt.figure(figsize=(10, 5))
    plt.plot(t, recovered_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.grid(True)
    plt.title("AM test")
    plt.show()

    return



demodulate_am_wave('AM tri single.xlsx', 100)
    # READ THIS!!
    # need to finish -
        # 1. take mod wave (now at freq domain) and filter it
        # understand how to filter -> we need to filter out a(2-2w0) and a(w+20)
        # ifft the filtered wave and get a(t)



    # original_data_signal = ifft(fft_am_wave)
     #return original_data_signal

# old demod function for documnetation, problem was with Fourier trans func cutting freqs higher than 500

#def demodulate_am_wave(filename, carrier_freq) -> dict:
    # works only for cosine carrier waves. doesnt work for sine or a combination.
    # take the df of the wave in time space
    # all_sheets = get_data1(filename)
    # first_sheet_name = list(all_sheets.keys())[0]
    # am_wave_df = all_sheets[first_sheet_name]
    # # extract data:
    # t_data = am_wave_df.iloc[:, 0]
    # amps_data = am_wave_df.iloc[:, 1]
    # t = pn.to_numeric(t_data, errors='coerce')
    # amps = pn.to_numeric(amps_data, errors='coerce')
    #
    # # applying noise reduction here is complicated, start without and add later
    # #original_data_dict = dict(zip(t, amps))
    # # noise reduction:
    # #filtered_freq_original = noise_reduction(fourier_trans(original_data_dict)['fourier_trans'])
    # # how to apply ifft?
    #
    # # make local carrier cos(w0*t):
    # loc_carrier = np.cos(2*np.pi*carrier_freq*t)
    # # multiply with original signal:
    # mod_wave_amps = amps * loc_carrier
    # #make data dict for fft:
    # mod_wave_data_dict = dict(zip(t, mod_wave_amps))
    # # fft:
    # fft_output = fourier_trans(mod_wave_data_dict, True, time_interval={'mode': 'auto'})
    # results_dict = fft_output['fourier_trans']
    # # apply low-pass filter:
    # CUTOFF_FREQ = 50 #in hz
    # n_samples = len(t)/2 + 1
    # lp_filtered = np.zeros(int(n_samples))
    # i = 0
    # for freq, val in results_dict.items():
    #     if freq <= CUTOFF_FREQ:
    #         lp_filtered[i] = val
    #         i += 1
    #     else:
    #         lp_filtered[i] = 0
    #         i += 1
    # # make inverse fft:
    # original_freqs = np.fft.irfft(lp_filtered, len(t))
    # # add a plot function.
    # plt.figure(figsize=(10, 5))
    # plt.plot(t,original_freqs)
    # plt.show()