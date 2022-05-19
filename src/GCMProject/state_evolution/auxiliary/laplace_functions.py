import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import minimize_scalar
from . import utility

# === Stuff for Laplace approximation

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
    
# Compute the Hessian 

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
    """
    Returns the expectation of sigma * (1 - sigma) with the proximal operator 
    """
    # for y = 1
    I1 = quad(lambda xi : integrand_training_hessian_minus(xi, m, q, v, vstar) * utility.gaussian(xi), -10.0, 10.0)[0]
    I2 = quad(lambda xi : integrand_training_hessian_plus(xi, m, q, v, vstar) * utility.gaussian(xi), -10.0, 10.0)[0]

    return I1 + I2

def training_hessian_omega_from_kappa(rho, m, q, V, Delta, alpha, lambda_, kappa1, kappastar, gamma):
    """
    returns tr(H Omega)
    """
    Vstar = rho + Delta - m**2 / q
    integral = training_hessian_integral(m, q, V, Vstar)

    tr_omega_squared = spectrum_trace_Omega_squared(kappa1, kappastar, gamma)
    tr_omega         = spectrum_trace_Omega(kappa1, kappastar, gamma)

    return alpha * integral * tr_omega_squared + lambda_ * tr_omega

def training_inverse_hessian_omega_from_kappa(rho, m, q, V, Delta, alpha, lambda_, kappa1, kappastar, gamma):
    """
    DOES NOT WORK !!!!! 
    Returns the trace of H^-1 Omega where H is the hessian of the empirical risk at w = w_erm
    H \sim d * (alpha * E(...) * Omega + d / \lambda * I) => H^{-1} \sim (1 / d) * ( ... )
    """
    Vstar    = rho + Delta - m**2 / q
    integral = training_hessian_integral(m, q, V, Vstar)

    lambda_minus, lambda_plus = (1. - np.sqrt(gamma))**2, (1. + np.sqrt(gamma))**2

    int_part = quad(lambda x : np.sqrt((lambda_plus - x) * (x - lambda_minus)) / (2 * np.pi * gamma * x) * (kappa1**2 * x + kappastar**2) / (alpha * integral * (kappa1**2 * x + kappastar**2) + lambda_), 
                     lambda_minus, lambda_plus)[0]
    
    if gamma > 1.0:
        zero_part = (1. - 1.0 / gamma) * (kappastar**2 / (alpha * integral * kappastar**2 + lambda_))
        return zero_part + int_part

    return int_part