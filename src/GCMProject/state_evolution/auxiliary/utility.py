from typing import List
import numpy as np
import scipy.optimize as opt4
from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import erf, erfc, erfcx
from scipy.linalg import sqrtm
from scipy.integrate import quad
from datetime import datetime

sigmoid = np.vectorize(lambda x : 1. / (1. + np.exp( -x )))
sigmoid_inv = np.vectorize(lambda y : np.log(y/(1-y)))

erf_prime = np.vectorize(lambda x : 2. / np.sqrt(np.pi) * np.exp(-x**2))
erfc_prime = np.vectorize(lambda x : -2. / np.sqrt(np.pi) * np.exp(-x**2))

def logerfc(x): 
    if x > 0.0:
        return np.log(erfcx(x)) - x**2
    else:
        return np.log(erfc(x))

bernoulli_variance = np.vectorize(lambda p : 4 * p * (1. - p))

@np.vectorize
def probit(x):
    return 0.5 * erfc(- x / np.sqrt(2))

def get_Phi_for_diagonal_covariances(Psi, Omega, Phi):
    Psi_inv          = np.linalg.inv(Psi)
    Psi_inv_sqrt     = np.real(sqrtm(Psi_inv))

    Omega_inv          = np.linalg.inv(Omega)
    Omega_inv_sqrt     = np.real(sqrtm(Omega_inv))
    return Psi_inv_sqrt @ Phi @ Omega_inv_sqrt

def get_diagonalized_covariance(Psi, Omega, Phi):
    """
    returns :
        - Psi, Omega, Phi after the transformation
    """
    p, d = Phi.shape
    return np.eye(p), np.eye(d), get_Phi_for_diagonal_covariances(Psi, Omega, Phi)

def get_additional_noise_from_kappas(kappa1, kappastar, gamma):
    """
    Returns the variance 
    """
    kk1, kkstar          = kappa1**2, kappastar**2
    lambda_minus, lambda_plus = (1. - np.sqrt(gamma))**2, (1. + np.sqrt(gamma))**2
    def to_integrate(lambda_, kk1, kkstar, lambda_minus, lambda_plus):
        return np.sqrt((lambda_plus - lambda_) * (lambda_ - lambda_minus)) / (kkstar + kk1 * lambda_)
    return 1.0 - kk1 * quad(lambda lambda_ : to_integrate(lambda_, kk1, kkstar, lambda_minus, lambda_plus), lambda_minus, lambda_plus)[0] / (2 * np.pi)