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

class BayesOptimalProbit(Model):
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
        self.alpha = sample_complexity
        self.Delta = Delta
        self.initialized = False

    def get_info(self):
        info = {
            'model': 'bo_probit',
            'sample_complexity': self.alpha,
        }
        return info

    def init_with_spectrum(self, kappa1, kappastar, gamma):
        self.initialized = True
        self.using_kappa = True
        self.matching    = False

        self.kappa1 = kappa1
        self.kappastar = kappastar
        self.gamma = gamma

        self.add_noise = utility.get_additional_noise_from_kappas(kappa1, kappastar, gamma)
        self.effective_Delta = self.Delta + self.add_noise
        self.rho = 1.0
        self.projected_rho = self.rho - self.add_noise

    def init_with_data_model(self, data_model):    
        self.initialized = True
        self.using_kappa = False
        self.matching    = False

        self.data_model  = data_model

        # Do a transformation of the matrices to simplify
        self.teacher_size = len(data_model.Psi)
        self.student_size = len(data_model.Omega)
        self.gamma = self.student_size / self.teacher_size

        self.Psi       = data_model.Psi
        self.Omega     = data_model.Omega
        self.Omega_inv = np.linalg.inv(self.Omega)
        self.Omega_inv_sqrt = sqrtm(self.Omega_inv)

        # transpose data_model.Phi because it's transposed in the definition of the Custom data model class
        self.Phi       = data_model.Phi.T
        
        # New covariance of the teacher (granted the teacher has identity covariance)
        self.cov        = self.Omega_inv_sqrt @ self.Phi.T @ self.Phi @ self.Omega_inv_sqrt
        self.eigvals    = np.linalg.eigvalsh(self.cov)

        # NOTE : Here, rho DOES NOT INCLUDE PSI => DOES NOT INCLUDE THE ADDITIONAL NOISE DUE TO THE MISMATCH OF THE MODEL
        self.rho           =  data_model.get_rho()
        self.projected_rho =  data_model.get_projected_rho()

        # SETTING THE NOISE 
        
        # NOTE : Don't add Delta in the data_model because the noise is not a property of the data but of the teacher
        # Effective noise is due to the mismatch in the models.
        # Should appear only in the update of the hat overlaps normally
        self.mismatch_noise_var = data_model.get_rho() - data_model.get_projected_rho()
        self.effective_Delta = self.Delta + self.mismatch_noise_var

    def _update_overlaps(self, Vhat, qhat, mhat):
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
        int_lims = 20.0
        Delta = self.effective_Delta
        def integrand(z):
            # NOTE : In the noiseless case Delta = 0, we recover 2q + V + Delta = 1 + q
            return 1/(np.pi * np.sqrt(V + Delta)) * norm.pdf(np.sqrt(2*q + V + Delta) * z) * 1/ utility.probit(np.sqrt(q) * z)
        
        # Bayes optimal => all these quantities are always the same
        # NOTE : Here, we must multiply by n / d => if we keep the noation n / d, by alpha * gamma
        Vhat = mhat = qhat = (self.alpha * self.gamma) * quad(integrand, -int_lims, int_lims, epsabs=1e-10, epsrel=1e-10, limit=200)[0]
        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        if not self.initialized:
            raise Exception()
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)