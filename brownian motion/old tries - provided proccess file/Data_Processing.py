import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FILE_DIR = 'outputs_3/progress.csv'
PARTICLE_NUM = 173 # change here
INTERVAL_LIST = [2] # insert intervals in number of frames
FRAMERATE = 1/15 # insert framerate in secs (eg 1/15)
PARTICLE_LIST = [] # insert numbers of needed paticles. insert particle NUMBER (int)

def get_data(filename: str) -> pd.DataFrame:
    ''' a function that takes the file name (csv) and returns data frame'''
    data = pd.read_csv(filename, engine = 'python')
    return data

def get_single_data(data: pd.DataFrame, particle_num: int) -> pd.DataFrame:
    '''a function that takes the full data and returns only the data about the given particle in the given time
       returns a df of time/x/y'''
    time_data = data.time
    x_data = data[f'Point-{particle_num} X']
    y_data = data[f'Point-{particle_num} Y']
    single_data_df = pd.DataFrame({'time': time_data, 'x': x_data, 'y': y_data}) # create single particle df
    # need to multiply by real distance - add later. currently calculating r^2 with pixels, not real distance
    return single_data_df

def get_mean_r2_single(data: pd.DataFrame, interval: int):
    ''' a function that receives the data then returns <r^2> for a single interval. interval is a number of frames'''
    r2_list = []
    idx = 0
    while (idx + interval) < len(data):
        curr_x0 = data.iloc[idx, 1]
        curr_y0 = data.iloc[idx, 2]
        dx = data.iloc[idx + interval, 1] - curr_x0
        dy = data.iloc[idx + interval, 2] - curr_y0
        curr_r2 = dx ** 2 + dy ** 2
        r2_list.append(curr_r2)
        idx += 1
    mean_r2 = np.mean(r2_list)
    return mean_r2

def get_mean_r2(data: pd.DataFrame, interval_list: list, framerate: float):
    ''' a function that receives the data, uses mean_r2_single for the entire interval list, and returns a final df of time/<r2>'''
    times = [i * framerate for i in interval_list]
    mean_r2_list =[]
    for interval in interval_list:
        curr_mean_r2 = get_mean_r2_single(data, interval)
        mean_r2_list.append(curr_mean_r2)
    new_df = pd.DataFrame({'time': times, 'mean_r2': mean_r2_list})
    return new_df

def plot_single_mean_r2_time(df: pd.DataFrame):
    ''' plot function for a single particle of <r^2>/time'''
    plt.figure(figsize = (12, 7))
    plt.plot(df.time, df.mean_r2)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Mean $r^2$') # need to insert units
    plt.grid(True, alpha = 0.3)
    plt.show()

def plot_final_mean_r2(df_list: list):
    ''' final plot function of several particles <r^2>/time. it receives a LIST OF DF!!!'''
    plt.figure(figsize=(12, 7))
    for df in df_list:
        plt.plot(df.time, df.mean_r2)
        # need to add legend of patricle size - complete later
    plt.xlabel('Time (s)')
    plt.ylabel(r'Mean $r^2$')  # need to insert units
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    data = get_data(FILE_DIR)
    particle_data = get_single_data(data, PARTICLE_NUM)
    plot_single_mean_r2_time(get_mean_r2(particle_data, INTERVAL_LIST, FRAMERATE))

def final():
    data = get_data(FILE_DIR)
    res_list = []
    for particle in PARTICLE_LIST:
        particle_data = get_single_data(data, particle)
        curr_mean_r2 = get_mean_r2(particle_data, INTERVAL_LIST, FRAMERATE)
        res_list.append(curr_mean_r2)
    plot_final_mean_r2(res_list)

if __name__ == '__main__':
    main()


