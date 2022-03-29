'''
Auxiliary integrals needed for computing the likelihood update functions
for probit regression.
'''

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfc
from scipy.optimize import minimize_scalar, root_scalar, minimize
import scipy.stats as stats

from . import utility

int_bounds = 2
threshold = -20

def gaussian(x, mean=0, var=1):
    return np.exp(-.5 * (x-mean)**2/var) / np.sqrt(2*np.pi*var)

def cdf(x):
    return stats.norm.cdf(x)

def loss(z):
    # return - np.log(cdf(z))
    return np.log(2) - utility.logerfc(- z / np.sqrt(2.0))

def dloss(y, z):
    threshold = -20
    if z * y < threshold:
        ratio = - z * y
    else:
        ratio = (stats.norm.pdf(z) / stats.norm.cdf(z * y))
    return - y * ratio

def ddloss(y, z):
    threshold = -20
    if z * y < threshold:
            ratio = - z * y
    else:
        ratio = (stats.norm.pdf(z) / stats.norm.cdf(z * y))
    return y * (z * ratio + y * ratio**2)

def moreau_loss(x, y, omega,V):
    return (x-omega)**2/(2*V) + loss(y*x)

def moreau_prime(x, y, omega, V):
    return (x - omega) / V + dloss(y, x)

def rescaled_moreau_prime(x, y, omega, V):
    # take the opposite of the moreau loss (as done in ben's code + rescale by V)
    return omega - x - V * dloss(y, x)

def moreau_second(x, y, omega, V):
    return 1. / V + ddloss(y, x)

def proximal(y, omega, V):
    root = root_scalar(
                lambda x : rescaled_moreau_prime(x, y, omega, V), bracket=[-1e20, 1e20], xtol=1e-15).root
    return root

# Define helper functions (partition functions and so on ...)

def find_star(y, omega, V):
    z_star = proximal(y, omega, V)
    dz_star = 1 / (1 + V * ddloss(y, z_star))
    l_y_z = loss(y * z_star)
    return z_star, dz_star, l_y_z
        
def Zout_0(y, omega, V):
    # Noiseless case
    if y == 1:
        return 1 / 2 * (1 + erf(omega / np.sqrt(2 * V)))
    elif y == -1:
        return 1 / 2 * (1 - erf(omega / np.sqrt(2 * V)))
        
def fout_0(y, omega, V):
    Z_0 = Zout_0(y, omega, V)

    # Noiseless case
    if y == 1:
        return 1 / Z_0 * gaussian(omega, 0, V)
    elif y == -1:
        return - 1 / Z_0 * gaussian(omega, 0, V)

def fout(y, omega, V):
    z_star = find_star(y, omega, V)
    return 1 / V * (z_star - omega)

def fout_dfout(y, omega, V):
    z_star, dz_star, l_y_z = find_star(y, omega, V)
    ## Zout ##
    L = 1 / (2 * V) * (z_star - omega)**2 + l_y_z
    Zout = np.exp(- L) / \
        np.sqrt((2 * np.pi * V) * (2 * np.pi))
    ## fout ##
    fout = 1 / V * (z_star - omega)
    ## dfout ##
    dfout = 1 / V * (dz_star - 1)
    return Zout, fout, dfout

# new update functions for hat overlaps

def SP_m_hat(M, Q, V, Vstar):
    return np.sum([quad(SP_m_hat_arg, -int_bounds, int_bounds, args=(y, M, Q, V, Vstar))[0] for y in [-1, 1]])

def SP_m_hat_arg(xi, y, m, q, V, V_0):
    omega_0 = m / np.sqrt(q) * xi
    omega = np.sqrt(q) * xi

    _, fout, _ = fout_dfout(y, omega, V)
    res = gaussian(xi, 0, 1) * Zout_0(y, omega_0, V_0) * \
        fout * fout_0(y, omega_0, V_0)

    if np.isnan(res):
        res = 0
    return res

def SP_q_hat(M, Q, V, Vstar):
    # NOTE : added noise
    res = 0
    for y in [-1, 1]:
        res_tmp = quad(SP_q_hat_arg, -int_bounds, int_bounds, args=(y, M, Q, V, Vstar))
        res += res_tmp[0]
    return res

def SP_q_hat_arg(xi, y, M, Q, V, Vstar):
    omega_0 = M / np.sqrt(Q) * xi
    omega = np.sqrt(Q) * xi
    
    _, fout, _ = fout_dfout(y, omega, V)
    return gaussian(xi, 0, 1) * Zout_0(y, omega_0, Vstar) * fout**2

def SP_V_hat(M, Q, V, Vstar):
    return - np.sum([quad(SP_V_hat_arg, -int_bounds, int_bounds, args=(y, M, Q, V, Vstar))[0] for y in [-1, 1]])

def SP_V_hat_arg(xi, y, M, Q, V, Vstar):
    omega_0 = M / np.sqrt(Q) * xi
    omega = np.sqrt(Q) * xi
    
    _, _, dfout = fout_dfout(y, omega, V)
    return gaussian(xi, 0, 1) * Zout_0(y, omega_0, Vstar) * dfout

def Integrand_training_error_plus_probit(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
#     λstar_plus = np.float(mpmath.findroot(lambda λstar_plus: λstar_plus - ω - V/(1 + np.exp(np.float(λstar_plus))), 10e-10))
    λstar_plus = proximal(1, ω, V) 
    l_plus = loss(λstar_plus) 
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * l_plus

def Integrand_training_error_minus_probit(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    # λstar_minus = np.float(mpmath.findroot(lambda λstar_minus: λstar_minus - ω + V/(1 + np.exp(-np.float(λstar_minus))), 10e-10))
    λstar_minus = proximal(-1, ω, V) 
    l_minus = loss(-λstar_minus)
    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * l_minus

def traning_error_probit(M, Q, V, Vstar):
    I1 = quad(lambda ξ: Integrand_training_error_plus_probit(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    I2 = quad(lambda ξ: Integrand_training_error_minus_probit(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    return (1/2)*(I1 + I2)