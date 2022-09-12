from typing import List

import numpy as np
from scipy.special import erfc, erfcx
from scipy.linalg import sqrtm
from scipy.integrate import quad

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

def gaussian(x, mean=0, var=1):
    return np.exp(-.5 * (x-mean)**2/var) / np.sqrt(2*np.pi*var)

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

# For Marcenko Pastur integration

def mp_integral(f : callable, gamma):
    """
    integrates an arbitrary function against MP distriubtion
    """
    lambda_minus, lambda_plus = (1. - np.sqrt(gamma))**2, (1. + np.sqrt(gamma))**2
    integral = quad(lambda x : f(x) * np.sqrt((lambda_plus - x) * (x - lambda_minus)) / (2 * np.pi * gamma * x), lambda_minus, lambda_plus)[0]
    if gamma > 1.0:
        return integral + (1.0 - 1.0 / gamma) * f(0.0)
    return integral

### Data models 

# POURQUOI SIGMA APPARAIT PAS ??? -> apparait implicitement dans V 
class ProbitDataModel:
    @classmethod
    def Z0(self, y, w, V):
        return probit(y * w / np.sqrt(V))

    @classmethod
    def dZ0(self, y, w, V):
        return y * np.exp(- w**2 / (2 * V)) / np.sqrt(2 * np.pi * V)
    
    @classmethod
    def f0(self, y, w, V):
        return self.dZ0(y, w, V) / self.Z0(y, w, V)

class LogisticDataModel:
    @classmethod
    def Z0(self, y, w, V):
        sqrtV = np.sqrt(V)
        # With the sqrtV, it doesn't look normalized
        # return sqrtV * quad(lambda z : sigmoid(y * (z * sqrtV + w)) * np.exp(- z**2 / 2), -5.0, 5.0, limit=500)[0] / np.sqrt(2 * np.pi)
        # NOTE : Below, normalized partition function (so it defines an expectation)
        return quad(lambda z : sigmoid(y * (z * sqrtV + w)) * np.exp(- z**2 / 2), -10.0, 10.0, limit=500)[0] / np.sqrt(2 * np.pi)

    @classmethod
    def dZ0(self, y, w, V, V_threshold = 1e-10):
        if V > V_threshold:
            sqrtV = np.sqrt(V)
            # return quad(lambda z : z *  sigmoid(y * (z * sqrtV + w)) * np.exp(- z**2 / 2), -5.0, 5.0, limit=500)[0] / np.sqrt(2 * np.pi)
            return quad(lambda z : z *  sigmoid(y * (z * sqrtV + w)) * np.exp(- z**2 / 2), -10.0, 10.0, limit=500)[0] / np.sqrt(2 * np.pi * V)
        else:
            return sigmoid(y * w)

    @classmethod
    def f0(self, y, w, V):
        return self.dZ0(y, w, V) / self.Z0(y, w, V)

class PseudoBayesianDataModel:
    """
    The sign in p_out is wrong but it's ok because the sign in the likelihood is also wrong
    TODO : Fix this 
    """
    threshold_p = -10 
    threshold_l = -10

    @classmethod
    def p_out(self, x):
        if x > self.threshold_p:
            return np.log(1. + np.exp(- x))
        else:
            return -x

    @classmethod
    def likelihood(self, z, beta):
        if z > self.threshold_l:
            return np.exp(-beta * self.p_out(z))
        # maybe simplifies the expression when z is very small ? 
        return np.exp(beta * z)

    @classmethod
    def Z0(self, y, w, V, beta =   1.0, bound = 5.0, threshold = 1e-10):
        if V > threshold:
            sqrtV = np.sqrt(V)
            return quad(lambda z : self.likelihood(y * (z * sqrtV + w), beta) * np.exp(- z**2 / 2), -bound, bound, limit=100)[0] / np.sqrt(2 * np.pi)
        else:
            return self.likelihood(y * w, beta)

    @classmethod
    def dZ0(self, y, w, V, beta = 1.0, bound = 5.0):
        # derivative w.r.t w I think ? 
        sqrtV = np.sqrt(V)
        return quad(lambda z : z * self.likelihood(y * (z * sqrtV + w), beta) * np.exp(- z**2 / 2), -bound, bound, limit=100)[0] / np.sqrt(2 * np.pi * V)

    @classmethod
    def ddZ0(self, y, w, V, beta = 1.0, bound = 5.0, threshold = 1e-10, Z0 = None):
        Z0 = Z0 or self.Z0(y, w, V, beta, bound, threshold)
        sqrtV = np.sqrt(V)
        to_integrate = lambda z : z**2 * self.likelihood(y * (z * sqrtV + w), beta) * np.exp(-z**2 / 2.) / np.sqrt(2 * np.pi)
        # return - Zerm / V + quad(to_integrate, -bound, bound, limit=500)[0] / sqrtV
        return - Z0 / V + quad(to_integrate, -bound, bound, limit=500)[0] / V

    @classmethod
    def f0(self, y, w, V, beta = 1.0, bound = 5.0):
        """
        Z0 = mpmath.mp.mpf(self.Z0(y, w, V, beta, bound))
        print(Z0)
        dZ0 = mpmath.mp.mpf(self.dZ0(y, w, V, beta, bound))
        mp_result = mpmath.fdiv(dZ0, Z0)
        return float(mp_result)
        """
        z0 = self.Z0(y, w, V, beta, bound)
        dz0 = self.dZ0(y, w, V, beta, bound)
        return dz0 / z0

    @classmethod
    def df0(self, y, w, V, beta = 1.0, bound = 5.0):
        z0     = self.Z0(y, w, V, beta, bound)
        dz0    = self.dZ0(y, w, V, beta, bound)
        ddz0   = self.ddZ0(y, w, V, beta, bound, Z0 = z0)
        return ddz0 / z0 - (dz0 / z0)**2

    # Derivative w.r.t beta, to optimmize the hyper-parameters beta, lambda
    def dbetaZ0(self, y, w, V, beta = 1.0, bound = 5.0):
        sqrtV  = np.sqrt(V)
        return quad(lambda z : self.p_out(y * (z * sqrtV + w)) * self.likelihood(y * (z * sqrtV + w), beta) * np.exp(-z**2 / 2.), -bound, bound, limit=500)[0] / np.sqrt(2 * np.pi)

class NormalizedPseudoBayesianDataModel:
    @classmethod
    def p_out(self, x):
        return np.log( sigmoid(x) )

    @classmethod
    def likelihood(self, z, beta):
        return np.exp( beta * self.p_out(z) )

    @classmethod
    def normalized_likelihood(self, z, beta):
        # the below expression is correct but can be simplified
        # return self.likelihood(z, beta) / (self.likelihood(z, beta) + self.likelihood(-z, beta))
        return sigmoid(beta * np.log(sigmoid(z) / sigmoid(-z)) )

    @classmethod
    def Z0(self, y, w, V, beta = 1.0, bound = 10.0):
        sqrtV  = np.sqrt(V)
        return quad(lambda z : self.normalized_likelihood(y * (z * sqrtV + w) * np.exp(-z**2 / 2.), beta), -bound, bound, limit=500)[0] / np.sqrt(2 * np.pi)

    @classmethod
    def dZ0(self, y, w, V, beta = 1.0, bound = 10.0):
        sqrtV  = np.sqrt(V)
        return quad(lambda z : z * self.normalized_likelihood(y * (z * sqrtV + w), beta)* np.exp(-z**2 / 2.), -bound, bound, limit=500)[0] / np.sqrt(2 * np.pi * V)

    @classmethod
    def ddZ0(self, y, w, V, beta = 1.0, bound = 10.0, threshold = 1e-10, Z0 = None):
        Z0           = Z0 or self.Z0(y, w, V, beta, bound, threshold)
        sqrtV        = np.sqrt(V)
        to_integrate = lambda z : z**2 * self.normalized_likelihood(y * (z * sqrtV + w), beta) * np.exp(-z**2 / 2.) / np.sqrt(2 * np.pi)
        # return - Zerm / V + quad(to_integrate, -bound, bound, limit=500)[0] / sqrtV
        return - Z0 / V + quad(to_integrate, -bound, bound, limit=500)[0] / V

    @classmethod
    def f0(self, y, w, V, beta = 1.0, bound = 10.0):
        return self.dZ0(y, w, V, beta, bound) / self.Z0(y, w, V, beta, bound)

    @classmethod
    def df0(self, y, w, V, beta = 1.0, bound = 10.0):
        z0   = self.Z0(y, w, V, beta, bound)
        dz0  = self.dZ0(y, w, V, beta, bound)
        ddz0 = self.ddZ0(y, w, V, beta, bound, Z0 = z0)
        value = ddz0 / z0 - (dz0 / z0)**2
        if np.isnan(value):
            raise Exception(f'z0, dz0, ddz0 are {z0, dz0, ddz0} but df0 is Nan !')

    @classmethod
    def dbetaZ0(self, y, w, V, beta = 1.0, bound = 10.0, threshold = 1e-10):
        # Derivative w.r.t beta, to optimmize the hyper-parameters
        def to_integrate(z):
            lf         = z * sqrtV + w
            likelihood = self.normalized_likelihood(y * lf, beta)
            # the likel. is a sigmoid hence the shape of derivative 
            return np.log(sigmoid(y * lf) / sigmoid(- y * lf)) * likelihood * (1.0 - likelihood)

        if V > threshold:
            sqrtV  = np.sqrt(V)
            return quad(lambda z : to_integrate(z) * np.exp(- z**2 / 2.0), -bound, bound)[0] / np.sqrt(2 * np.pi)
        else:
            return to_integrate(0)