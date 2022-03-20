from typing import List
import numpy as np
import scipy.optimize as opt4
from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import erf, erfc
from scipy.linalg import sqrtm
from datetime import datetime

sigmoid = np.vectorize(lambda x : 1. / (1. + np.exp( -x )))
sigmoid_inv = np.vectorize(lambda y : np.log(y/(1-y)))

erf_prime = np.vectorize(lambda x : 2. / np.sqrt(np.pi) * np.exp(-x**2))
erfc_prime = np.vectorize(lambda x : -2. / np.sqrt(np.pi) * np.exp(-x**2))

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