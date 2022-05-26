import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import minimize_scalar
from . import utility

def omega_inv_hessian_trace_random_features(kappa1, kappastar, gamma, lambda_, Vhat):
    """
    Gives the asymptotic value of x . H^{-1} . x -> Tr(H^{-1} Omega) where the covariance of x is Omega
    """
    kk1 = kappa1**2
    kkstar = kappastar**2
    return utility.mp_integral(lambda x : (kk1 * x + kkstar) / (lambda_ + Vhat * (kk1 * x + kkstar)), gamma)

