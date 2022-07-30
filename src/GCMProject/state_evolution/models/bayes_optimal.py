from random import sample
from re import S
import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from scipy.stats import norm
from scipy.linalg import sqrtm

from .base_model import Model
from ..auxiliary import utility

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_inv(y):
    return np.log(y/(1-y))

class BayesOptimal(Model):
    '''
    Implements updates for logistic regression task.
    See base_model for details on modules.
    NOTE : Assume for now that Omega is the identity covariance
    NOTE : Here, rho DOES NOT INCLUDE PSI => DOES NOT INCLUDE THE ADDITIONAL NOISE DUE TO THE MISMATCH OF THE MODEL

    '''
    def __init__(self, Delta = 0., *, sample_complexity):
        """
        Note : gamma = student_size / teacher_size
        arguments: 
            - Delta : variance of the noise 
        """
        super(BayesOptimal, self).__init__(sample_complexity=sample_complexity, Delta=Delta)

    def init_with_data_model(self, data_model):
        super().init_with_data_model(data_model)
        
        self.Omega_inv = np.linalg.inv(self.Omega)
        self.Omega_inv_sqrt = sqrtm(self.Omega_inv)

        # transpose data_model.Phi because it's transposed in the definition of the Custom data model class
        
        # New covariance of the teacher (granted the teacher has identity covariance)
        self.cov        = self.Omega_inv_sqrt @ self.Phi.T @ self.Phi @ self.Omega_inv_sqrt
        self.eigvals    = np.linalg.eigvalsh(self.cov)

    def get_info(self):
        info = {
            'model': 'bo_probit',
            'sample_complexity': self.alpha,
        }
        return info


    def _update_overlaps(self, Vhat, qhat, mhat):
        if self.matching:
            q = qhat / (1. + qhat)
            m = q
            # We use projected rho and not rho because of the prior on the weights ! 
            V = self.projected_rho - q
            return V, q, m

        if self.using_kappa == False:
            q = np.sum(qhat * self.eigvals**2 / (1. + qhat * self.eigvals)) / self.teacher_size

        else:
            # d eigenvalues and divide by p => multiply the integral by gamma
            kk1 = self.kappa1**2
            kkstar = self.kappastar**2
            q = self.gamma * qhat * utility.mp_integral(lambda x : (kk1 * x / (kk1 * x + kkstar))**2 / (1. + qhat * (kk1 * x / (kk1 * x + kkstar))), self.gamma)
    
        m = q
        # We use projected rho and not rho because of the prior on the weights ! 
        V = self.projected_rho - q

        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        if self.str_teacher_data_model == 'probit':
            int_lims = 20.0
            Delta = self.effective_Delta
            def integrand(z):
                # NOTE : In the noiseless case Delta = 0, we recover 2q + V + Delta = 1 + q
                return 1/(np.pi * np.sqrt(V + Delta)) * norm.pdf(np.sqrt(2*q + V + Delta) * z) * 1/ utility.probit(np.sqrt(q) * z)
            
            # Bayes optimal => all these quantities are always the same
            # NOTE : Here, we must multiply by n / d => if we keep the noation n / d, by alpha * gamma
            Vhat = mhat = qhat = (self.alpha * self.gamma) * quad(integrand, -int_lims, int_lims, epsabs=1e-10, epsrel=1e-10, limit=200)[0]
            return Vhat, qhat, mhat
        elif self.str_teacher_data_model == 'logit':
            Vstar = self.rho - m**2 / q
            somme = 0.0
            # not sure 
            for y in [-1.0, 1.0]:
                somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * utility.PseudoBayesianDataModel.f0(y, np.sqrt(q)*xi, V, beta = 1.0)**2  * utility.LogisticDataModel.Z0(y, m / np.sqrt(q) * xi, Vstar), -10.0, 10.0, limit=500)[0]  
            Vhat = mhat = qhat = (self.alpha * self.gamma) * somme
            return Vhat, qhat, mhat
        print(self.str_teacher_data_model)
        raise Exception()

    def update_se(self, V, q, m):
        if not self.initialized:
            raise Exception()
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)