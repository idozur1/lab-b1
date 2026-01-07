import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# files received are in the following format:
# first column - height
# second to last column - width per frame

LIST_OF_FILES = [f'results/meas {i}.csv' for i in range(1,12)]
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
    avg_data = get_average_data(data)
    # scale parameters
    x = avg_data['height'] * 1e-3
    y = avg_data['avg_width'] * 1e-3
    y_err = avg_data['std_width'] * 1e-3
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
                 f'a = {slope:.4f} $\pm$ {slope_err:.4f}\n'
                 f'b = {intercept:.2e} $\pm$ {intercept_err:.2e}')
    plt.plot(x, y_fit, color='red', linestyle='-', linewidth=1.5, label=label_fit)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, which='both', linestyle='-', alpha=0.5)
    plt.xlabel('Height (m)')
    plt.ylabel('Average Width (m)')
    plt.tight_layout()
    plt.show()

def final():
    plot_final(LIST_OF_FILES)

if __name__ == '__main__':
    final()


