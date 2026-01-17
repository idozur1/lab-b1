import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter

# files received are in the following format:
# first column - height
# second to last column - width per frame

LIST_OF_FILES = [f'week 2/High measurements/meas {i}.csv' for i in range(1,9)]
#LIST_OF_FILES = [f'results/meas {i}.csv' for i in range(1,6)]
THETA = np.deg2rad(79) # in radians
V0 = 0.411 # m/s, ampirically measured

def get_data(filename: str)-> pd.DataFrame:
    ''' a function that takes the csv and return a pd.DataFrame with the same format'''
    data = pd.read_csv(filename)
    num_cols = data.shape[1]
    data.columns = ['height'] + [str(i) for i in range(1, num_cols)]
    return data

def get_single_data(data: pd.DataFrame, frame_num: int)-> pd.DataFrame:
    ''' a function that takes the full df and return a pd.DataFrame of 1 frame only'''
    new_data = pd.DataFrame()
    new_data['height'] = data['height']
    new_data['width'] = data[str(frame_num)]
    return new_data

def add_all_videos(list_of_files) -> pd.DataFrame:
    ''' a function take takes the list of filenames and returns a pd.df of all of them.
        new df is 'height':'nums' where nums is the frame number in the TOTAL count'''
    result = pd.DataFrame()
    current_size = 1
    result['height'] = get_data(list_of_files[0])['height']
    for filename in list_of_files:
        data = get_data(filename)
        # get min rows:
        min_len = min(len(result), len(data))
        result = result.iloc[:min_len]
        data = data.iloc[:min_len]
        #continue adding:
        data_cols = data.shape[1] - 1
        new_col_names = [str(i) for i in range(current_size, current_size+data_cols)]
        result[new_col_names] = data.iloc[:, 1:].values
        current_size += data_cols
    return result

def get_average_data(data: pd.DataFrame)-> pd.DataFrame:
    ''' a function that takes the full df and return a new df with the average vals'''
    avg_data = pd.DataFrame()
    avg_data['height'] = data['height']
    avg_data['avg_width'] = data.iloc[:, 1:].mean(axis=1)
    avg_data['std_width'] = data.iloc[:, 1:].std(axis=1)
    return avg_data

def linear_model(x, a, b):
    return a * x + b

def plot_final(list_of_files):
    ''' plots height:avg_width for the entire list of filenames, adds linear curve_fit'''
    data = add_all_videos(list_of_files)
    avg_data = get_average_data(data).iloc[:400, :]
    # scale parameters
    x = avg_data['height']
    y = avg_data['avg_width']
    y_err = avg_data['std_width']
    # fit:
    popt, pcov = curve_fit(linear_model, x, y)
    slope, intercept = popt
    # get errors:
    perr = np.sqrt(np.diag(pcov))
    slope_err, intercept_err = perr
    # get fit line:
    y_fit = linear_model(x, slope, intercept)
    plt.figure(figsize=[10, 7])
    plt.errorbar(x, y, yerr=y_err, fmt='o', color='blue',
                 markersize=3, capsize=3, alpha=0.6, label='Measured Data')
    label_fit = (f'Fit: y = ax + b\n'
                  f'a = {slope:.4f} $\pm$ {slope_err:.2e}\n'
                  f'b = {intercept:.2e} $\pm$ {intercept_err:.2e}')
    plt.plot(x, y_fit, color='red', linestyle='-', linewidth=1.5, label=label_fit)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, which='both', linestyle='-', alpha=0.5)
    plt.xlabel('Height (m)')
    plt.ylabel('Average Width (m)')
    ticks_x = np.arange(0, 0.055, 0.005)
    plt.xticks(ticks_x)
    ticks_y = np.arange(0, 0.035, 0.005)
    plt.yticks(ticks_y)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(-3, -3), useMathText=True)
    #plt.ylim(0,0.02)
    #plt.xlim(0,0.04)
    plt.tight_layout()
    plt.show()

def final1():
    plot_final(LIST_OF_FILES)

# Week 2:
def normalize_cloud_width(data: pd.DataFrame, theta: float)-> pd.DataFrame:
    ''' a function that takes the *aveaged* data and returns the normalized data.
        normalization means subtracting the main cone shape cloud which comes from initial momentum
        using the formula: w_measured_mean - 2*h*cos(theta) = w_real'''
    new_data = pd.DataFrame()
    new_data['height'] = data['height']
    new_data['norm_width'] = data['avg_width']
    new_data['norm_width'] = new_data['norm_width'] - data['height'] * 2 * np.cos(theta)
    new_data['std_norm_width'] = data.iloc[:, 1:].std(axis=1)
    return new_data

def sqrt_fit(x, a, b):
    return a*np.sqrt(x) + b

def plot_normalized(list_of_files):
    data = add_all_videos(list_of_files)
    avg = get_average_data(data)
    norm = normalize_cloud_width(avg, THETA)

    x, y, y_err = norm['height'], norm['norm_width'], norm['std_norm_width']

    try:
        popt, pcov = curve_fit(sqrt_fit, x, y, p0=[1.0, 0.0])
        perr = np.sqrt(np.diag(pcov))
        label_fit = f'Fit: $a\sqrt{{x}}+b$\n$a={popt[0]:.4f}\pm{perr[0]:.2e}$'
        y_fit = sqrt_fit(x, *popt)
    except Exception:
        y_fit, label_fit = np.zeros_like(x), "Fit Failed"

    plt.figure(figsize=(10, 7))
    plt.errorbar(x, y, yerr=y_err, fmt='o', color='royalblue',
                 ms=3, capsize=3, alpha=0.6, label='Measured Data')
    plt.plot(x, y_fit, 'r-', lw=1.5, label=label_fit)

    plt.title("Normalized Cloud Width vs Height")
    plt.xlabel("Height (m)")
    plt.ylabel("Normalized Width")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='sci', scilimits=(-3, 3), useMathText=True)
    plt.show()

def h_to_t_function(h, h0, k):
    ''' a function that uses Fluid Entrainment formula for velocity to get t(h).
        final t(h) = (h^2 + 2h0h)/2k. returns t for inserted h'''
    return (h **2 + 2 * h0 * h) / (2 * k)

def get_h0_k(data: pd.DataFrame):
    ''' a function that takes the full averaged data, make a curve fit for w_measured/height
        and calculated h0 (which is the intersection of the fit with x axis) and k which is v0*h0'''
    heights = data['height']
    width = data['avg_width']
    popt, pcov = curve_fit(linear_model, heights, width)
    slope, intercept = popt
    h0 = np.abs(intercept / slope)
    k = V0 * h0
    return h0, k

def plot_norm_width_on_time(list_of_files):
    ''' a function that takes the raw data, than:
        1. calculates h0 and k (and prints them)
        2. normalizes the data
        3. translate height to time
        4. makes a sqrt fit
        5. plot normalized width to time
        '''
    data = add_all_videos(list_of_files)
    avg = get_average_data(data).iloc[:450, :]
    h0, k = get_h0_k(avg)
    norm = normalize_cloud_width(avg, THETA)
    norm['time'] = h_to_t_function(norm['height'], h0, k)
    x_axis = norm['time']
    y_axis = norm['norm_width']
    y_err = norm['std_norm_width']

    try:
        popt, pcov = curve_fit(sqrt_fit, x_axis, y_axis)
        perr = np.sqrt(np.diag(pcov))
        a_val, b_val = popt
        a_err = perr[0]

        y_fit = sqrt_fit(x_axis, *popt)
        label_fit = f'Sqrt Fit: $a\sqrt{{t}} + b$\n$a = {a_val:.4f} \pm {a_err:.2e}$'
    except Exception as e:
        print(f"Fit failed: {e}")
        y_fit = np.zeros_like(x_axis)
        label_fit = "Fit Failed"

    plt.figure(figsize=(10, 7))
    plt.errorbar(x_axis, y_axis, yerr=y_err, fmt='o', color='black', ecolor = '#6495ED',
                 ms=3, capsize=3, alpha=0.5, label='Normalized Width')

    plt.plot(x_axis, y_fit, 'r-', lw=2, label=label_fit)

    ax = plt.gca()  # Get current axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, -3))  # Forces 10^-3
    ax.yaxis.set_major_formatter(formatter)

    plt.title(f"Cloud Diffusion Expansion Over Time\n(Jet Model: $h_0={h0:.4f}m, k={k:.2e}$)")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Width (m)")
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    print(f"h0 = {h0:.5f}, k = {k:.5f}")
    print(f"Diffusion Coefficient (a): {a_val:.5f}")

def final2():
    plot_norm_width_on_time(LIST_OF_FILES)
if __name__ == '__main__':
    #final1()
    final2()

#old functions:
# def t_to_h_function(t):
#     ''' a function that takes t and returns h '''
#     v0 = 0.411
#     m = 4e-15
#     alpha = 1.5e-8
#     g = 9.81
#     theta = THETA
#     return v0 * (m/alpha) * (np.exp((-alpha*t)/m) - 1) - (m/alpha)*g*np.sin(theta)*t
# def h_to_t(data: pd.DataFrame, function):
#     ''' a function that makes a map of general h(t) using t_to_h func, and returns the new data
#         in an array of t/w'''
#     time_array = np.arange(0, 3, 0.0001)
#     h_per_t = function(time_array)
#     print(h_per_t)
#     sort_idx = np.argsort(h_per_t)
#     times_for_measured_heights = np.interp(data['height'],h_per_t[sort_idx],time_array[sort_idx])
#
#     new_data = pd.DataFrame()
#     new_data['time'] = times_for_measured_heights
#     new_data['width'] = data['avg_width'].values
#     new_data['width_std'] = data['std_width'].values
#     return new_data
# old function - 0.5 * ( (-1.5)+np.sqrt(2.25 + 2.25e5*h))