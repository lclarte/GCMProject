'''
Auxiliary integrals needed for computing the likelihood update functions
for logistic regression.
'''

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import minimize_scalar
import scipy.stats as stats
from . import utility

def gaussian(x, mean=0, var=1):
    return np.exp(-.5 * (x-mean)**2/var) / np.sqrt(2*np.pi*var)

def loss(z):
    return np.log(1 + np.exp(-z))

def moreau_loss(x, y, omega,V):
    return (x-omega)**2/(2*V) + loss(y*x)

def f_mhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    return np.exp(-ωstar**2/(2*Vstar))*(λstar_plus - ω)

def f_mhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    return np.exp(-ωstar**2/(2*Vstar))*(λstar_minus - ω)

def integrate_for_mhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_mhat_plus(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
    I2 = quad(lambda ξ: f_mhat_minus(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
    return (I1 - I2)*(1/np.sqrt(2*np.pi*Vstar))

# Vhat_x #
def f_Vhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    return (1/(1/V + (1/4) * (1/np.cosh(λstar_plus/2)**2))) * (1 + erf(ωstar/np.sqrt(2*Vstar)))

def f_Vhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    return (1/(1/V + (1/4) * (1/np.cosh(-λstar_minus/2)**2))) * (1 - erf(ωstar/np.sqrt(2*Vstar)))
    
def integrate_for_Vhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_Vhat_plus(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
    I2 = quad(lambda ξ: f_Vhat_minus(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
    return (1/2) * (I1 + I2)

# Qhat_x#
def f_qhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * (λstar_plus - ω)**2

def f_qhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * (λstar_minus - ω)**2

def integrate_for_Qhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_qhat_plus(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
    I2 = quad(lambda ξ: f_qhat_minus(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
    return (1/2) * (I1 + I2)

def Integrand_training_error_plus_logistic(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
#   λstar_plus = np.float(mpmath.findroot(lambda λstar_plus: λstar_plus - ω - V/(1 + np.exp(np.float(λstar_plus))), 10e-10))
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    
    l_plus = loss(λstar_plus)
    
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * l_plus

def Integrand_training_error_minus_logistic(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
#   λstar_minus = np.float(mpmath.findroot(lambda λstar_minus: x - ω + V/(1 + np.exp(-np.float(λstar_minus))), 10e-10))
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    
    l_minus = loss(-λstar_minus)

    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * l_minus

def traning_error_logistic(M, Q, V, Vstar):
    I1 = quad(lambda ξ: Integrand_training_error_plus_logistic(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
    I2 = quad(lambda ξ: Integrand_training_error_minus_logistic(ξ, M, Q, V, Vstar) * gaussian(ξ), -10, 10, limit=500)[0]
    return (1/2)*(I1 + I2)

# == Compute the Hessian of the loss 

def integrand_training_hessian_plus(xi, m, q, v, vstar):
    w     = np.sqrt(q) * xi
    wstar = m / np.sqrt(q) * xi
    lambdastar_plus = minimize_scalar(lambda x: moreau_loss(x, +1, w, v))['x']

    return 0.5 * (1 + erf(wstar/np.sqrt(2*vstar))) * utility.sigmoid(lambdastar_plus) * (1.0 - utility.sigmoid(lambdastar_plus))

def integrand_training_hessian_minus(xi, m, q, v, vstar):
    w     = np.sqrt(q) * xi
    wstar = m / np.sqrt(q) * xi
    lambdastar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, w, v))['x']

    return 0.5 * (1 - erf( wstar / np.sqrt(2*vstar))) * utility.sigmoid(lambdastar_minus) * (1.0 - utility.sigmoid(lambdastar_minus))

def training_hessian_integral(m, q, v, vstar):
    # for y = 1
    I1 = quad(lambda xi : integrand_training_hessian_minus(xi, m, q, v, vstar) * gaussian(xi), -10.0, 10.0)[0]
    I2 = quad(lambda xi : integrand_training_hessian_plus(xi, m, q, v, vstar) * gaussian(xi), -10.0, 10.0)[0]

    return I1 + I2

def training_hessian(rho, m, q, V, effective_Delta, lambda_, tr_omega, tr_omega_2):
    Vstar = rho - m**2 / q + effective_Delta
    integral = training_hessian_integral(m, q, V, Vstar)
    return tr_omega_2 * integral + tr_omega * lambda_

### Integrals for finite temperature logistic 

def p_out(x):
    if x > -10:
        return np.log(1. + np.exp(- x))
    else:
        return -x

# ft stands for finite temperature
def ft_logistic_Z(y, w, V, beta, bound = 5.0):
    sqrtV = np.sqrt(V)
    return sqrtV * quad(lambda z : np.exp(-beta * p_out(y * (z * sqrtV + w))) * np.exp(- z**2 / 2), -bound, bound, limit=500)[0]

def ft_logistic_dwZ(y, w, V, beta, bound = 5.0):
    sqrtV = np.sqrt(V)
    return quad(lambda z : z *  np.exp(-beta * p_out(y * (z * sqrtV + w))) * np.exp(- z**2 / 2), -bound, bound, limit=500)[0]

def ft_logistic_ddwZ(y, w, V, beta, Zerm = None, bound = 5.0):
    Zerm = Zerm or ft_logistic_Z(y, w, V, beta, bound)
    sqrtV = np.sqrt(V)
    to_integrate = lambda z : z**2 * np.exp(-beta * p_out(y * (z * sqrtV + w))) * np.exp(-z**2 / 2.)
    return - Zerm / V + quad(to_integrate, -bound, bound, limit=500)[0] / sqrtV

def ft_logistic_ferm(y, w, V, beta, bound = 10.0):
    return ft_logistic_dwZ(y, w, V, beta, bound) / ft_logistic_Z(y, w, V, beta, bound)

def ft_logistic_dferm(y, w, V, beta, bound = 10.0):
    Zerm = ft_logistic_Z(y, w, V, beta, bound)
    dZerm = ft_logistic_dwZ(y, w, V, beta, bound) 
    ddZerm = ft_logistic_ddwZ(y, w, V, beta, Zerm, bound)
    return ddZerm / Zerm - (dZerm / Zerm)**2

def Z0(y, w, V):
    return utility.probit(y * w / np.sqrt(V))

def dZ0(y, w, V):
    return y * np.exp(- w**2 / (2 * V)) / np.sqrt(2 * np.pi * V)

def f0(y, w, V):
    return  dZ0(y, w, V) / Z0(y, w, V)

def ft_integrate_for_mhat(M, Q, V, Vstar, beta):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * ft_logistic_ferm(y, np.sqrt(Q)*xi, V, beta) * dZ0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

def ft_integrate_for_Qhat(M, Q, V, Vstar, beta):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * ft_logistic_ferm(y, np.sqrt(Q)*xi, V, beta)**2 * Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]  
    return somme

def ft_integrate_for_Vhat(M, Q, V, Vstar, beta):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * ft_logistic_dferm(y, np.sqrt(Q)*xi, V, beta) * Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

# trace of Omega**2

def spectrum_trace_Omega_squared(kappa1, kappastar, gamma):
    lambda_minus, lambda_plus = (1 - np.sqrt(gamma))**2, (1 + np.sqrt(gamma))**2
    kk1, kkstar = kappa1**2, kappastar**2
    # spectrum due to non-zero of MP
    trace = kk1**2 * quad(lambda x : np.sqrt((lambda_plus - x) * (x - lambda_minus)) * x / (2 * np.pi * gamma), lambda_minus, lambda_plus)[0]
    trace += 2 * kk1 * kkstar * quad(lambda x : np.sqrt((lambda_plus - x) * (x - lambda_minus)) / (2 * np.pi * gamma), lambda_minus, lambda_plus)[0]
    return trace + kkstar**2

def spectrum_trace_Omega(kappa1, kappastar, gamma):
    lambda_minus, lambda_plus = (1 - np.sqrt(gamma))**2, (1 + np.sqrt(gamma))**2
    kk1, kkstar = kappa1**2, kappastar**2
    # Spectrum due to non-zero of MP
    trace = kk1 * quad(lambda x : np.sqrt((lambda_plus - x) * (x - lambda_minus)) / (2 * np.pi * gamma), lambda_minus, lambda_plus)[0]
    # Spectrum due to the Dirac on 0
    return trace + kkstar
    
