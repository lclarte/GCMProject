from math import erfc
from readline import read_history_file
import numpy as np
import scipy.stats as stats
from scipy.integrate import quad, nquad
from scipy.linalg import sqrtm
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
    TODO : Ajouter une option pour fixer manuellement le bruit du au mismath du modèle 
    (il faut probablement le calculer AVANT le changement de variable)
    NOTE : Assume for now that Omega is the identity covariance
    '''
    def __init__(self, Delta = 0., *, sample_complexity, data_model):
        """
        arguments: 
            - Delta : variance of the noise 
        """

        self.alpha = sample_complexity
    
        self.data_model = data_model

        # Do a transformation of the matrices to simplify
        self.teacher_size = len(data_model.Psi)
        self.student_size = len(data_model.Omega)

        self.Psi       = data_model.Psi
        self.Omega     = data_model.Omega

        if np.linalg.norm(data_model.Omega - np.eye(self.student_size)) > 1e-10:
            raise Exception()
        # transpose data_model.Phi because it's transposed in the definition of the Custom data model class
        self.Phi       = data_model.Phi.T
        
        # New covariance of the teacher (granted the teacher has identity covariance)
        self.cov       = self.Phi.T @ self.Phi

        # SETTING THE NOISE 
        
        # NOTE : Don't add Delta in the data_model because the noise is not a property of the data but of the teacher
        self.Delta      = Delta
        # Effective noise is due to the mismatch in the models.
        # Should appear only in the update of the hat overlaps normally
        self.mismatch_noise_var = np.trace(self.Psi - self.Phi @ self.Phi.T) / self.teacher_size
        self.effective_Delta = Delta + self.mismatch_noise_var

    def get_info(self):
        info = {
            'model': 'bo_probit',
            'sample_complexity': self.alpha,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        # should not be affected by the noise level
        # NOTE : Assumes the distribution of the input data is identity ! 
        woodbury_inverse = np.linalg.inv( (1. / qhat) * np.eye(self.student_size) + self.cov)
        q = (1. / self.teacher_size) * np.trace(
            qhat * self.cov @ (self.cov - self.cov @ woodbury_inverse @ self.cov)
        )
        m = q
        V = np.trace(self.cov) / self.teacher_size - q

        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        int_lims = 20.0
        Delta = self.effective_Delta
        def integrand(z):
            # NOTE : In the noiseless case Delta = 0, we recover 2q + V + Delta = 1 + q
            return 1/(np.pi * np.sqrt(V + Delta)) * norm.pdf(np.sqrt(2*q + V + Delta) * z) * 1/ utility.probit(np.sqrt(q) * z)
        
        # Bayes optimal => all these quantities are always the same
        Vhat = mhat = qhat = self.alpha * quad(integrand, -int_lims, int_lims, epsabs=1e-10, epsrel=1e-10, limit=200)[0]
        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)

    # NOTE : La forme de la test loss pour Bayes Optimal et la regression logistique devrait être la même
    def get_test_error(self, q, m):
        return np.arccos(m/np.sqrt(q * (self.data_model.rho + self.effective_Delta)))/np.pi

    def get_test_loss(self, q, m):
        return - 1.0

    def get_train_loss(self, V, q, m):
        return -1.0
    
    def get_calibration(self, q, m, p=0.75):
        """
        Bayes is always calibrated (normally ? TODO : Check this)
        """
        return 0.0