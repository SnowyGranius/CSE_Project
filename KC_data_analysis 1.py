import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.optimize import curve_fit

mainplots = True
auxplots = False

testshape = 'inverse square'

# Functions for analysis
def K_func(m0m1m2, shape = 'linear'):
    m0, m1, m2 = m0m1m2
    if shape == 'linear':
        Kval = m2
    elif shape == 'inverse square':
        Kval = 1/m2**2
    elif shape == 'inverse':
        Kval = 1/m2
    elif shape == 'mean inverse square':
        Kval = np.mean(1/m2**2)
    else:
        print('No/invalid model selected, defaulting to linear formulation')
        Kval = m2
    return Kval

def KC_model_old(m0m1m2, gamma, const = False):
    # Old KC model, with modularity for the functional representation of K
    m0, m1, m2 = m0m1m2
    # Setting to keep all M's outside of M0 constant
    if const:
        return gamma*(m0**3)/((1-m0)**2)
    else:
        return gamma*(m0**3)/((m1**2)*K_func(m0m1m2, shape = testshape)*((1-m0)**2))

    
def KC_model(m0m1m2, km):
    # New KC model based on Sijmen's formulation
    m0, m1, m2 = m0m1m2

    return (1/km)*(m0**3)/((m1**2)*((m1/(2*m2**0.5))+m0))

# Import data
data = pd.read_csv("Minkowskis_rectangle_1_porespy.csv")
#data = pd.read_csv("Minkowskis_summarized.csv")

M_0 = np.array(data["Porosity_mean"])
M_1 = np.array(data["Surface_mean"])
M_2 = np.array(data["Euler_mean_vol_mean"])
k = np.array(data["Permeability_mean"])
k_stdDev = np.sqrt(np.array(data["Permeability_variance"]))
energy = np.array(data["Energy_mean"])
energy_stdDev = np.sqrt(np.array(data["Energy_variance"]))

# Curve fitting
idx = 0

popt, cov = curve_fit(KC_model, (M_0[idx:], M_1[idx:], M_2[idx:]), k[idx:])
print(popt, cov)

km_vals = []
# Fitting at each datapoint
for i, sol in enumerate(k):
    kopt, kcov = curve_fit(KC_model, (M_0[i], M_1[i], M_2[i]), [sol])
    km_vals.append(kopt)

# Testing vals
M_0_hat = np.linspace(np.min(M_0), np.max(M_0), 100)
M_1_hat = np.linspace(np.min(M_1), np.max(M_1), 100)
M_2_hat = np.linspace(np.min(M_2), np.max(M_2), 100)

#k_hat = KC_model((M_0_hat, M_1_hat, M_2_hat), *popt)
k_hat = KC_model((M_0, M_1, M_2), *popt)
k_star = KC_model((M_0, M_1, M_2), 2.5)

# Plots
# Plot permeability data
if auxplots:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')

    ax2.scatter(M_0, M_1, k)
    ax2.set_xlabel('$M_0$')
    ax2.set_ylabel('$M_1$')
    ax2.set_zlabel('$\\kappa$')
    ax2.set_title('$\\kappa$ data')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot()

    ax3.scatter(M_0, M_2/M_1)
    ax3.set_xlabel('$M_0$')
    ax3.set_ylabel('$\\frac{M_2}{M_1}$')
    ax3.set_title('$M_0$ vs $\\frac{M_2}{M_1}$')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot()

    ax4.scatter(M_1, M_2)
    ax4.set_xlabel('$M_1$')
    ax4.set_ylabel('$M_2$')
    ax4.set_title('$M_1$ vs $M_2$')


### KEEP MAIN PLOTS DOWN HERE ###
if mainplots:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(M_0, M_1, k)
    #ax.plot(M_0_hat, M_1_hat, k_hat, label = 'Fitted')
    ax.plot(M_0, M_1, k_hat, label = 'Fitted')
    #ax.plot(M_0, M_1, k_star, label = 'Theoretical')
    ax.set_xlabel('$M_0$')
    ax.set_ylabel('$M_1$')
    ax.set_zlabel('$\\kappa$')
    ax.set_title('$\\kappa$ as a function of $M_0$, $M_1$, $M_2$')
    ax.legend()
    
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(M_0, k)
    #ax.plot(M_0_hat, M_1_hat, k_hat, label = 'Fitted')
    ax.semilogy(M_0, k_hat, label = 'Fitted')
    ax.semilogy(M_0, k_star, label = 'Theoretical')
    ax.set_xlabel('$M_0$')
    ax.set_ylabel('$\\kappa$')
    ax.set_title('$\\kappa$ as a function of $M_0$, $M_1$, $M_2$')
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(M_0, km_vals)
    ax.set_xlabel('$M_0$')
    ax.set_ylabel('$k_m$')
    ax.set_title('Fitted $k_m$ as a function of $M_0$, $M_1$, $M_2$')


plt.show()