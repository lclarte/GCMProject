import numpy as np
import scipy.stats as stats
from scipy.integrate import quad
from scipy.stats import norm

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
    def __init__(self, Delta = 0., *, sample_complexity, data_model, student_teacher_size_ratio):
        """
        arguments: 
            - Delta : variance of the noise 
        """
        self.alpha = sample_complexity
        self.gamma = student_teacher_size_ratio
    
        self.data_model = data_model

        # Do a transformation of the matrices to simplify
        self.teacher_size = len(data_model.Psi)
        self.student_size = len(data_model.Omega)

        self.Psi       = data_model.Psi
        self.Omega     = data_model.Omega

        # transpose data_model.Phi because it's transposed in the definition of the Custom data model class
        self.Phi       = data_model.Phi.T
        
        # New covariance of the teacher (granted the teacher has identity covariance)
        self.cov        = self.Phi.T @ self.Phi
        self.eigvals    = np.linalg.eigvalsh(self.cov)

        # NOTE : Here, rho DOES NOT INCLUDE PSI => DOES NOT INCLUDE THE ADDITIONAL NOISE DUE TO THE MISMATCH OF THE MODEL
        self.rho           = data_model.get_rho()
        self.projected_rho =  data_model.get_projected_rho()

        # SETTING THE NOISE 
        
        # NOTE : Don't add Delta in the data_model because the noise is not a property of the data but of the teacher
        self.Delta      = Delta
        # Effective noise is due to the mismatch in the models.
        # Should appear only in the update of the hat overlaps normally
        self.mismatch_noise_var = data_model.get_rho() - data_model.get_projected_rho()
        self.effective_Delta = Delta + self.mismatch_noise_var

    def get_info(self):
        info = {
            'model': 'bo_probit',
            'sample_complexity': self.alpha,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        q = np.sum(qhat * self.eigvals**2 / (1. + qhat * self.eigvals)) / self.teacher_size
        m = q
        V = self.projected_rho - q
        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        int_lims = 20.0
        Delta = self.effective_Delta
        def integrand(z):
            # NOTE : In the noiseless case Delta = 0, we recover 2q + V + Delta = 1 + q
            return 1/(np.pi * np.sqrt(V + Delta)) * norm.pdf(np.sqrt(2*q + V + Delta) * z) * 1/ utility.probit(np.sqrt(q) * z)
        
        # Bayes optimal => all these quantities are always the same
        # NOTE : Here, alpha = n / p ! and not n / d !! => if we keep the noation nn / d 
        Vhat = mhat = qhat = (self.alpha * self.gamma) * quad(integrand, -int_lims, int_lims, epsabs=1e-10, epsrel=1e-10, limit=200)[0]
        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)

    def get_test_error(self, q, m):
        # NOTE : Removed the noise to be like the GCM Project
        # We add the noise due to the mismatch because rho does not include it 
        return np.arccos(m/np.sqrt(q * self.rho))/np.pi

    def get_test_loss(self, q, m):
        return - 1.0

    def get_train_loss(self, V, q, m):
        return -1.0
    
    def get_calibration(self, q, m, p=0.75):
        """
        Bayes is always calibrated (normally ? TODO : Check this)
        """
        return 0.0