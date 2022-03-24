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

def moreau_loss(x, y, omega,V):
    return (x-omega)**2/(2*V) + loss(y*x)

def moreau_prime(x, y, omega, V):
    if x * y < threshold:
        ratio = - x * y
    else:
        ratio = (gaussian(x) / cdf(x * y))
    return (x - omega) / V - y * ratio

def moreau_second(x, y, omega, V):
    if x * y < threshold:
        return 1. / V + 1
    else:
        ratio = (gaussian(x) / cdf(x * y))
    return 1. / V + y * ( x * ratio + y * ratio**2)

def proximal(y, omega, V):
    return root_scalar(
        lambda x : moreau_prime(x, y, omega, V),
        x0 = omega,
        fprime = lambda x : moreau_second(x, y, omega, V)
    ).root

def f_mhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = proximal(1, ω, V)
    return np.exp(-ωstar**2/(2*Vstar))*(λstar_plus - ω)

def f_mhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = λstar_plus = proximal(-1, ω, V)
    return np.exp(-ωstar**2/(2*Vstar))*(λstar_minus - ω)

def integrate_for_mhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_mhat_plus(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    I2 = quad(lambda ξ: f_mhat_minus(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    return (I1 - I2) * (1/np.sqrt(2*np.pi*Vstar))

# Vhat_x #
def f_Vhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = proximal(1, ω, V)
    return (1/(1/V + (1/4) * (1/np.cosh(λstar_plus/2)**2))) * (1 + erf(ωstar/np.sqrt(2*Vstar)))

def f_Vhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = proximal(-1, ω, V)
    return (1/(1/V + (1/4) * (1/np.cosh(-λstar_minus/2)**2))) * (1 - erf(ωstar/np.sqrt(2*Vstar)))
    
def integrate_for_Vhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_Vhat_plus(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    I2 = quad(lambda ξ: f_Vhat_minus(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    return (1/2) * (I1 + I2)

# Qhat_x#
def f_qhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = proximal(1, ω, V)
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * (λstar_plus - ω)**2

def f_qhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = proximal(-1, ω, V)
    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * (λstar_minus - ω)**2

def integrate_for_Qhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_qhat_plus(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    I2 = quad(lambda ξ: f_qhat_minus(ξ, M, Q, V, Vstar)* gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    return (1/2) * (I1 + I2)

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
#   λstar_minus = np.float(mpmath.findroot(lambda λstar_minus: λstar_minus - ω + V/(1 + np.exp(-np.float(λstar_minus))), 10e-10))
    λstar_minus = proximal(-1, ω, V)
    
    l_minus = loss(-λstar_minus)

    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * l_minus

def traning_error_probit(M, Q, V, Vstar):
    I1 = quad(lambda ξ: Integrand_training_error_plus_probit(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    I2 = quad(lambda ξ: Integrand_training_error_minus_probit(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    return (1/2)*(I1 + I2)

def traning_error_probit(M, Q, V, Vstar):
    I1 = quad(lambda ξ: Integrand_training_error_plus_probit(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    I2 = quad(lambda ξ: Integrand_training_error_minus_probit(ξ, M, Q, V, Vstar) * gaussian(ξ), -int_bounds  , int_bounds  , limit=500)[0]
    return (1/2)*(I1 + I2)
