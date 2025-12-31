from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


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
                 'week2/25% - 2.817E-6.xlsx',
                 'week2/10% - 2.910E-6.xlsx',
                 'week2/7% - 2.278E-6.xlsx']


def get_data(filename: str) -> pd.DataFrame:
    ''' a function that takes the file name (csv) and returns data frame (also name for plotting)'''
    data = pd.read_excel(filename)
    data.columns = ['t', 'x', 'y']
    return data ,filename

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

def single_week1():
    for FILE_DIR in PARTICLE_LIST:
        data, filename = get_data(FILE_DIR)
        size = float(filename[12:-5])
        plot_single_mean_r2_time(get_mean_r2_no_drift(data, INTERVAL_LIST, FRAMERATE), size, filename) #change to with drift t find drift

def final_week1():
    res_list = []
    for particle in PARTICLE_LIST: #particle is a string of the filename
        particle_data, filename = get_data(particle)
        size = float(filename[12:-5])
        curr_mean_r2 = get_mean_r2_no_drift(particle_data, INTERVAL_LIST, FRAMERATE)
        res_list.append((curr_mean_r2, size))
    plot_final_mean_r2(res_list)

#if __name__ == '__main__':
 #   single_week1()
  #  final_week1()

# Nimrod's Functions for week 1:
def get_damping_const(dynamic_viscosity, radius) -> float:
    #get alpha, the damping constant
    damping_const = 6*np.pi*dynamic_viscosity*radius
    return damping_const
#Analysis

def get_NA(radii, r2_slopes):

    diffusions = r2_slopes / 4 #<r^2>(t)=4Dt
    inverse_radii = 1 / radii

    R = 8.314462
    T = 296 #Kelvin for 23 Celsius
    etha = 0.9321e-3 #water in 23 Celsius - Pascal/Sec

    fit = np.polynomial.polynomial.Polynomial.fit(inverse_radii, diffusions, deg=1).convert().coef
    b = fit[1] #slope
    print(b)
    c = fit[0] # + c

    NA = (R*T)/(6*np.pi*etha*b)

    plt.scatter(inverse_radii, diffusions, label = 'Diffusion as a function of the inverse radius')
    plt.plot([0, 10e5], [c, c+10e5*b], label = 'linear regression fit')
    plt.xlabel('1/Particle Radius (1/m)')
    plt.ylabel('Diffusion (m^2/t)')
    plt.legend()
    plt.show()

    return NA

def get_Boltzman_Const(radii, viscosities, temps, r2_slopes):
    #In week 2 we change the approximated variable to Boltzman constant hence calculations might be inconsistent between weeks

    diffusions = r2_slopes / 4  # <r^2>(t)=4D - this is diffusion relative to radius

    relative_diffusions = diffusions / radii
    params = temps/viscosities

    fit = np.polynomial.polynomial.Polynomial.fit(params, relative_diffusions, deg=1).convert().coef
    b = fit[1]  # slope
    c = fit[0]  # + c

    k = b / (6*np.pi) #We evaluated T/viscosity but k is the slope given by T/(visocity*6*pi)

    plt.scatter(params, relative_diffusions, label='Diffusion as a function of the inverse radius')
    plt.plot([0, 600], [c, c + 600 * b], label='linear regression fit')
    plt.xlabel('temp/viscosity (K/(Ns/m^2)')
    plt.ylabel('Relative Diffusion (m/t')
    plt.legend()
    plt.show()

    return k

#data After dbugging in Ido's code

radii = np.array([2.993e-6, 3.436e-6, 3.869e-6, 6.385e-6, 7.071e-6, 2.428e-6]) / 2 #turning diameter to radius
r2_slopes = np.array([2.27e-12, 1.94e-12, 4.27e-13, 2.55e-13, 1.29e-13, 2.12e-12])

k = get_Boltzman_Const(radii, viscosities, temps, r2_slopes)
print('Boltzman constant approximated as ', k)

## 3 methods are needed:
## 1 - calculate <r^2> - <r>^2 = 4Dt. that means plot var(<r>)/t and get 4Dt in linear fit
## 2 - assume v_drift = const. for each interval take dx,dy = dx-v_x*t, dy-v_y*t. get new <r^2> and plot like week 1.
## 3 - do a parabolic fit at^2+bt+c. a is v^2, b is 4D and c ~ 0.

def find_drift(df: pd.DataFrame, interval):
    drifts = df.iloc[:, [1, 2]].diff(periods=interval)
    means = drifts.mean()
    return means.iloc[0], means.iloc[1]

##################### Method 1 - Variance of r #############################

def get_mean_r_single(data: pd.DataFrame, interval:int):
    ''' a helper function that takes the data as df and returns <r> for a single interval'''
    x_list = []
    y_list = []
    idx = 0
    while (idx + interval) < len(data):
        curr_x0 = data.iloc[idx, 1]
        curr_y0 = data.iloc[idx, 2]
        dx = (data.iloc[idx + interval, 1]) - curr_x0
        x_list.append(dx)
        dy = (data.iloc[idx + interval, 2]) - curr_y0
        y_list.append(dy)
        idx += 1
    mean_r = np.sqrt(np.mean(x_list) ** 2 + np.mean(y_list) ** 2)
    return mean_r # calculated using the formula |<r>| = sqrt(<x>^2 + <y>^2)

def get_var_r_single(data: pd.DataFrame, interval:int):
    ''' a function that takes the data as df and a single interval and returns the variance of r for that interval'''
    mean_r2 = get_mean_r2_single_no_drift(data, interval)
    mean_r = get_mean_r_single(data,interval)
    return mean_r2 - (mean_r ** 2)
    # this returns for every interval the result of var r (we get and array of var_r / t)

def get_var_r(data: pd.DataFrame, interval_list: list, framerate:float):
    ''' a function that takes the data as df and the interval list and returns var(r)/t'''
    dt = framerate
    times = [i * dt for i in interval_list]
    var_list = []
    for interval in interval_list:
        curr_var = get_var_r_single(data, interval)
        var_list.append(curr_var)
    new_df = pd.DataFrame({'time': times, 'variance': var_list})
    return new_df

def plot_var_on_time_single(df: pd.DataFrame, filename):
    plt.figure(figsize=(12, 7))

    # fit:
    popt, pcov = curve_fit(linear_fit, df.time, df.variance)
    slope, intercept = popt
    slope_err = np.sqrt(np.diag(pcov))[0]

    # label here:

    scatter = plt.scatter(df.time, df.variance, label=filename)
    x_fit = np.linspace(df.time.min(), df.time.max(), 100)
    y_fit = linear_fit(x_fit, *popt)

    fit_label = f"Fit: slope={slope:.2e} $\pm$ {slope_err:.1e}"
    print(f"{filename}: {fit_label}")
    plt.plot(x_fit, y_fit, color='black', linestyle='-', label=fit_label)

    plt.title(filename)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Mean $\langle Var(r) \rangle$ ($m^2$)')
    plt.grid(True, alpha=0.3)
    plt.legend(title="Particle Diameter")  # Adds a title to the box
    plt.show()


def plot_var_on_time_final(df_list: list):
    '''
    df_list: list of tuples (dataframe, filename)
    '''
    plt.figure(figsize=(12, 8))

    # Color map to make sure points and lines have matching colors
    colors = plt.cm.inferno(np.linspace(0, 0.8, len(df_list)))

    for i, (df, filename) in enumerate(df_list):
        # 1. Linear Fit
        popt, pcov = curve_fit(linear_fit, df.time, df.variance)
        slope, intercept = popt

        # 2. Calculate D = Slope / 4
        D = slope / 4

        # 3. Extract Viscosity % from filename (e.g., "week2/40% - ..." -> "40%")
        # Adjust logic if your filename structure changes
        try:
            # Splits "week2/40% - ..." by " - " -> take first part -> split by "/" -> take last part
            viscosity_name = filename.split(' - ')[0].split('/')[-1]
        except:
            viscosity_name = filename  # Fallback

        label_text = f"{viscosity_name} | $D={D:.2e}$"

        # 4. Plot Scatter
        plt.scatter(df.time, df.variance, color=colors[i], label=label_text, s=30)

        # 5. Plot Fit Line
        x_fit = np.linspace(df.time.min(), df.time.max(), 100)
        y_fit = linear_fit(x_fit, *popt)
        plt.plot(x_fit, y_fit, color=colors[i], linestyle='--', alpha=0.7)

    plt.xlabel('Time (s)')
    plt.ylabel(r'Variance of $\vec{r}$')
    plt.title(r'Variance Method: Extraction of Diffusion Coefficient $D$')
    plt.grid(True, alpha=0.3)
    plt.legend(title="Viscosity | Diffusion Coeff ($m^2/s$)")
    plt.show()

def var_final():
    res_list = []
    for particle in PARTICLE_LIST:
        data, filename = get_data(particle)
        curr_var_df = get_var_r(data, INTERVAL_LIST, FRAMERATE)

        res_list.append((curr_var_df, filename))
        plot_var_on_time_final(res_list)


if __name__ == '__main__':
    var_final()
################################# Method 2: Parabolic Fit #########################################

def parabolic_fit(x, a, b, c):
    return a * (x ** 2) + b * x + c

def parabolic_fit_plot_single(df, filename):
    plt.figure(figsize=(12, 7))

    # fit:
    popt, pcov = curve_fit(parabolic_fit, df.time, df.mean_r2)
    a, b, c = popt
    perr = np.sqrt(np.diag(pcov))
    a_err, b_err, c_err = perr


    # f"{value:.2f}" rounds to 2 decimal places

    scatter = plt.scatter(df.time, df.mean_r2)
    x_fit = np.linspace(df.time.min(), df.time.max(), 100)
    y_fit = parabolic_fit(x_fit, *popt)

    fit_label = f"Fit: v={np.sqrt(a):.2e} $\pm$ {a_err:.1e}, 4D ={b} $\pm$ {b_err:.1e}, c={c} $\pm$ {c_err:.1e}"
    print(f"{filename}: {fit_label}")
    plt.plot(x_fit, y_fit, color='black', linestyle='-', label=fit_label)

    plt.title(filename)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Mean $\langle r^2 \rangle$ ($m^2$)')
    plt.grid(True, alpha=0.3)
    plt.legend(title="Particle Diameter")  # Adds a title to the box
    plt.show()


def parabolic_fit_final(df_list):
    '''
    df_list: list of tuples (dataframe, filename)
    '''
    plt.figure(figsize=(12, 8))

    # Use 'inferno' colormap, stopping at 0.8 to avoid bright yellow
    colors = plt.cm.inferno(np.linspace(0, 0.8, len(df_list)))

    for i, (df, filename) in enumerate(df_list):
        try:
            # 1. Fit Parabola: y = at^2 + bt + c
            # Parameters: a = v^2, b = 4D, c = offset
            popt, pcov = curve_fit(parabolic_fit, df.time, df.mean_r2)
            a, b, c = popt

            # 2. Calculate Physical Parameters
            v_drift = np.sqrt(np.abs(a))  # v = sqrt(a)
            D = b / 4  # D = b/4

            # 3. Extract Viscosity % from filename
            # e.g., "week2/40% - ..." -> "40%"
            try:
                viscosity_name = filename.split(' - ')[0].split('/')[-1]
            except:
                viscosity_name = filename

            # 4. Create Legend Label: % | D | v
            label_text = f"{viscosity_name} | D={D:.2e} | v={v_drift:.2e}"

            # 5. Plot
            plt.scatter(df.time, df.mean_r2, color=colors[i], s=30)

            x_fit = np.linspace(df.time.min(), df.time.max(), 100)
            y_fit = parabolic_fit(x_fit, *popt)
            plt.plot(x_fit, y_fit, color=colors[i], linestyle='-', label=label_text)

        except Exception as e:
            print(f"Could not fit {filename}: {e}")

    plt.xlabel('Time (s)')
    plt.ylabel(r'Mean Squared Displacement $\langle r^2 \rangle$ ($m^2$)')
    plt.title(r'Parabolic Fit Method: $\langle r^2 \rangle = v_{drift}^2 t^2 + 4Dt + c$')
    plt.grid(True, alpha=0.3)
    plt.legend(title="Viscosity | Diffusion ($m^2/s$) | Drift ($m/s$)")
    plt.show()


def run_parabolic_final():
    res_list = []
    for particle in PARTICLE_LIST:
        data, filename = get_data(particle)
        # Note: We use the normal MSD function (without drift subtraction)
        # because the parabolic fit models the drift itself!
        curr_msd_df = get_mean_r2_no_drift(data, INTERVAL_LIST, FRAMERATE)
        res_list.append((curr_msd_df, filename))

    parabolic_fit_final(res_list)


if __name__ == '__main__':
    run_parabolic_final()

##################################################################################################
def get_mean_r2_single_with_drift(data: pd.DataFrame, interval: int, drift: Tuple):
    # check this method really works as needed. add description
    r2_list = []
    idx = 0
    drift = find_drift(data, 1)
    while (idx + interval) < len(data):
        curr_x0 = data.iloc[idx, 1]
        curr_y0 = data.iloc[idx, 2]
        dx = (data.iloc[idx + interval, 1] - curr_x0) - drift[0] * interval
        dy = (data.iloc[idx + interval, 2] - curr_y0) - drift[1] * interval
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


    ...
    ## a function tha plots the original data of mean_r2, than fits with a parabolic fit,
    ## and returns the fit parameters for each of the files (v^2, 4D).

#def calculate_kb(D,r, viscosity, T):
#    Kb = (6*D*np.pi*viscosity*r)/T
 #   return Kb