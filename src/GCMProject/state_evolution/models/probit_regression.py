from math import erfc
import numpy as np
import scipy.stats as stats
import scipy.integrate
from scipy.linalg import sqrtm
from .base_model import Model
from ..auxiliary.probit_integrals import SP_V_hat, SP_q_hat, SP_m_hat, traning_error_probit


def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_inv(y):
    return np.log(y/(1-y))

class ProbitRegression(Model):
    '''
    Implements updates for logistic regression task.
    See base_model for details on modules.
    '''
    def __init__(self, Delta = 0., *, sample_complexity, regularisation, data_model):
        self.alpha = sample_complexity
        self.lamb = regularisation
        self.data_model = data_model
        # effective_Delta should be useful ONLY for the computatino of the test error,
        # the additional noise due to the model mismatch is taken caren of indirectly in the state evolution

        # Do a transformation of the matrices to simplify
        self.teacher_size = data_model.k
        self.student_size = data_model.p

        self.Phi = data_model.Phi.T
        self.Psi = data_model.Psi
        self.Omega = data_model.Omega

        Omega_inv      = np.linalg.inv(data_model.Omega)

        # Effective noise is due to the mismatch in the models.
        # Should appear only in the update of the hat overlaps normally
        self.mismatch_noise_var = np.trace(self.Psi - self.Phi @ Omega_inv @ self.Phi.T) / self.teacher_size
        
        # NOTE : Don't add Delta in the data_model because the noise is not a property of the data but of the teacher
        # Delta = Sigma**2
        self.Delta = Delta
        self.effective_Delta = Delta + self.mismatch_noise_var
        # old
        # self.rho = self.data_model.rho
        self.rho = np.trace(self.Psi) / self.teacher_size
        
    def get_info(self):
        info = {
            'model': 'probit_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        """ 
        NOTE : For now, only works for the base model psi = omega = phi = identity
        """
        # should not be affected by the noise level
        V = np.mean(self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega))

        if self.data_model.commute:
            q = np.mean((self.data_model.spec_Omega**2 * qhat +
                                           mhat**2 * self.data_model.spec_Omega * self.data_model.spec_PhiPhit) /
                                          (self.lamb + Vhat*self.data_model.spec_Omega)**2)

            m = mhat / np.sqrt(self.data_model.gamma) * np.mean(self.data_model.spec_PhiPhit/(self.lamb + Vhat*self.data_model.spec_Omega))

        else:
            q = qhat * np.mean(self.data_model.spec_Omega**2 / (self.lamb + Vhat*self.data_model.spec_Omega)**2)
            q += mhat**2 * np.mean(self.data_model._UTPhiPhiTU * self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega)**2)

            m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model._UTPhiPhiTU/(self.lamb + Vhat * self.data_model.spec_Omega))

        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        Vstar = self.rho - m**2/q

        mhat = self.alpha * SP_m_hat(m, q, V, Vstar + self.Delta)
        qhat = self.alpha * SP_q_hat(m, q, V, Vstar + self.Delta)
        Vhat = self.alpha * SP_V_hat(m, q, V, Vstar + self.Delta)
    
        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        V, q, m = self._update_overlaps(Vhat, qhat, mhat)
        return V, q, m

    def get_test_error(self, q, m):
        # NOTE : Changed Delta -> effective_Delta to take into account the GCM
        return np.arccos(m/np.sqrt(q * (self.rho + self.effective_Delta)))/np.pi

    def get_test_loss(self, q, m):
        Sigma = np.array([
            [self.rho + self.effective_Delta, m],
            [m, q]
        ])

        def loss_integrand(lf_teacher, lf_erm):
            return np.log(1. + np.exp(- np.sign(lf_teacher) * lf_erm)) * stats.multivariate_normal.pdf([lf_teacher, lf_erm], mean=np.zeros(2), cov=Sigma)
        
        ranges = [(-10.0, 10.0), (-10.0, 10.0)]
        lossg_mle = scipy.integrate.nquad(loss_integrand, ranges)[0]
        return lossg_mle
    
    def get_calibration(self, q, m, p=0.75):
        return -1.0

    def get_train_loss(self, V, q, m):
        Vstar = self.rho - m**2/q
        return traning_error_probit(m, q, V, Vstar + self.effective_Delta)
