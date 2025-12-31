import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

#helper functions

# def get_normalized_slope(r2_slope, radius):
#     return r2_slope / radius #since we want to calculate diffusion as a function of dynamic viscosity, we need to eliminate dependence in radius

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
r2_slopes = np.array([2.27e-12, 1.94e-12, 4.27e-13, 2.55e-13, 1.29e-13, 2.12e-12])  #the slopes fit gave from measurement - this is the slope of <r^2>(t)

#programs

#week1
# NA = get_NA(radii, r2_slopes)
# print(NA)

#week2
#dummy data
viscosities = np.ones(1)
temps = np.ones(6)*296

k = get_Boltzman_Const(radii, viscosities, temps, r2_slopes)
print('Boltzman constant approximated as ', k)
# old data

# radii = np.array([2.993e-6, 3.436e-6, 3.869e-6, 7.071e-6]) / 2 #turning diameter to radius
# r2_slopes = np.array([2.24e-12, 1.37e-12, 3.52e-13, 1.29e-13])  #the slopes fit gave from measurement - this is the slope of <r^2>(t)


# #analysis with overlapping sections
# radii_ov = np.array([2.993e-6, 3.436e-6, 3.869e-6, 6.385e-6, 7.071e-6, 2.428e-6]) / 2 #turning diameter to radius
# r2_slopes_ov = np.array([2.24e-12, 1.37e-12, 3.52e-13, 2.53e-13, 1.29e-13, 2.11e-12])  #the slopes fit gave from measurement - this is the slope of <r^2>(t)
#
# #analysis without overlapping sections (classic use of Ergodic assumption)
# radii_nov = np.array([6.385e-6, 7.071e-6, 3.869e-6, 3.436e-6, 2.993e-6, 2.428e-6]) / 2 #turning diameter to radius
# r2_slopes_nov = np.array([7.61e-14, 1.18e-13, 3.37e-13, 1.14e-12, 1.53e-12, 1.93e-12])  #the slopes fit gave from measurement - this is the slope of <r^2>(t)
#
# #analysis without overlapping sections (classic use of Ergodic assumption) & only good ones (manually filtered)
# radii_gnov = np.array([7.071e-6, 3.436e-6, 2.993e-6, 2.428e-6]) / 2 #turning diameter to radius
# r2_slopes_gnov = np.array([1.18e-13, 1.14e-12, 1.53e-12, 1.93e-12])  #the slopes fit gave from measurement - this is the slope of <r^2>(t)


# model = LinearRegression(fit_intercept=True)
# model.fit(inverse_radii.reshape(-1, 1), diffusions)
# b = model.coef_[0]  # coefficient