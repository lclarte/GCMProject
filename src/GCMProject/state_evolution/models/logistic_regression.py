from math import erfc
import numpy as np
import scipy.stats as stats
import scipy.integrate
from scipy.linalg import sqrtm
from .base_model import Model
from ..auxiliary.logistic_integrals import integrate_for_mhat, integrate_for_Vhat, integrate_for_Qhat, traning_error_logistic

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_inv(y):
    return np.log(y/(1-y))

class LogisticRegression(Model):
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
        self.rho = self.data_model.rho
        
    def get_info(self):
        info = {
            'model': 'logistic_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
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
        Vstar = self.data_model.rho - m**2/q

        Im = integrate_for_mhat(m, q, V, Vstar + self.Delta)
        Iv = integrate_for_Vhat(m, q, V, Vstar + self.Delta)
        Iq = integrate_for_Qhat(m, q, V, Vstar + self.Delta)

        mhat = self.alpha/np.sqrt(self.data_model.gamma) * Im/V
        Vhat = self.alpha * ((1/V) - (1/V**2) * Iv)
        qhat = self.alpha * Iq/V**2

        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)

    def get_test_error(self, q, m):
        # NOTE : Removed the noise to be like the GCM Project
        return np.arccos(m/np.sqrt(q * self.data_model.rho))/np.pi

    def get_test_loss(self, q, m):
        Sigma = np.array([
            [self.data_model.rho + self.effective_Delta, m],
            [m, q]
        ])

        def loss_integrand(lf_teacher, lf_erm):
            return np.log(1. + np.exp(- np.sign(lf_teacher) * lf_erm)) * stats.multivariate_normal.pdf([lf_teacher, lf_erm], mean=np.zeros(2), cov=Sigma)
        
        ranges = [(-10.0, 10.0), (-10.0, 10.0)]
        lossg_mle = scipy.integrate.nquad(loss_integrand, ranges)[0]
        return lossg_mle
    
    def get_calibration(self, q, m, p=0.75):
        # TODO : Adapt to covariate model 
        inv_p = sigmoid_inv(p)
        rho   = self.data_model.rho
        return p - 0.5 * erfc(- (m / q * inv_p) / np.sqrt(2*(rho - m**2 / q + self.effective_Delta)))

    def get_train_loss(self, V, q, m):
        # TODO : Remplacer effective_Delta par Delta ? 
        Vstar = self.data_model.rho - m**2/q
        return traning_error_logistic(m, q, V, Vstar + self.effective_Delta)

class LogisticRegressionKappa(Model):
    """
    NOTE : In this class, d = dimension = size of the teacher, while p = parameters = size of the student
    Assumes that the teacher covariance is identity
    """
    # NOTE : Ne marche pas 
    def __init__(self, Delta = 0., *, sample_complexity, gamma, regularisation, kappa_star, kappa1):
        self.alpha = sample_complexity
        self.gamma = gamma
        self.lambda_ = regularisation

        self.kappa_star = kappa_star
        self.kappa1     = kappa1
         
        self.Delta = Delta
        self.rho   = 1.0

    def get_info(self):
        info = {
            'model': 'logistic_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def integrate_for_qvm(self, qhat,mhat,vhat):
        alpha = self.alpha
        gamma = self.gamma
        sigma= self.kappa1
        kk = self.kappa_star**2
        alphap=(sigma*(1+np.sqrt(alpha)))**2
        alpham=(sigma*(1-np.sqrt(alpha)))**2
        lamb = self.lambda_

        if lamb==0:
            den =1+kk*vhat
            aux =np.sqrt(((alphap+kk)*vhat+1)*((alpham+kk)*vhat+1))
            aux2=np.sqrt(((alphap+kk)*vhat+1)/((alpham+kk)*vhat+1))
            IV = ((kk*vhat+1)*((alphap+alpham)*vhat+2)-2*kk*vhat**2*np.sqrt(alphap*alpham)-2*aux)/(4*alpha*vhat**2*(kk*vhat+1)*sigma**2)
            IV = IV + max(0,1-gamma)*kk/(1+vhat*kk)
            I1= (alphap*vhat*(-3*den+aux)+4*den*(-den+aux)+alpham*vhat*(-2*alphap*vhat-3*den+aux))/(4*alpha*vhat**3*sigma**2*aux)
            I2= (alphap*vhat+alpham*vhat*(1-2*aux2)+2*den*(1-aux2))/(4*alpha*vhat**2*aux*sigma**2)
            I3= (2*vhat*alphap*alpham+(alphap+alpham)*den-2*np.sqrt(alphap*alpham)*aux)/(4*alpha*den**2*sigma**2*aux)
            IQ = (qhat+mhat**2)*I1+(2*qhat+mhat**2)*kk*I2+qhat*kk**2*I3
            IQ = IQ + max(0,1-gamma)*qhat*kk**2/den**2
            IM = ((alpham+alphap+2*kk)*vhat+2-2*aux)/(4*alpha*vhat**2*sigma**2)
        else:
            den =lamb+kk*vhat
            aux =np.sqrt(((alphap+kk)*vhat+lamb)*((alpham+kk)*vhat+lamb))
            aux2=np.sqrt(((alphap+kk)*vhat+lamb)/((alpham+kk)*vhat+lamb))
            IV = ((kk*vhat+lamb)*((alphap+alpham)*vhat+2*lamb)-2*kk*vhat**2*np.sqrt(alphap*alpham)-2*lamb*aux)/(4*alpha*vhat**2*(kk*vhat+lamb)*sigma**2)
            IV = IV + max(0,1-gamma)*kk/(lamb+vhat*kk)
            I1= (alphap*vhat*(-3*den+aux)+4*den*(-den+aux)+alpham*vhat*(-2*alphap*vhat-3*den+aux))/(4*alpha*vhat**3*sigma**2*aux)
            I2= (alphap*vhat+alpham*vhat*(1-2*aux2)+2*den*(1-aux2))/(4*alpha*vhat**2*aux*sigma**2)
            I3= (2*vhat*alphap*alpham+(alphap+alpham)*den-2*np.sqrt(alphap*alpham)*aux)/(4*alpha*den**2*sigma**2*aux)
            IQ = (qhat+mhat**2)*I1+(2*qhat+mhat**2)*kk*I2+qhat*kk**2*I3
            IQ = IQ + max(0,1-gamma)*qhat*kk**2/den**2
            IM = ((alpham+alphap+2*kk)*vhat+2*lamb-2*aux)/(4*alpha*vhat**2*sigma**2)
        return IV, IQ, IM

    def update_hat(self, q, m, v):
        sigma=self.rho-m**2/q
        
        Im = integrate_for_mhat(m, q, v, sigma + self.Delta)
        Iv = integrate_for_Vhat(m, q, v, sigma + self.Delta)
        Iq = integrate_for_Qhat(m, q, v, sigma + self.Delta)
            
        mhat = self.alpha / np.sqrt(self.gamma*self.rho) * Im
        vhat = -self.alpha * Iv
        qhat = self.alpha * Iq
        
        return qhat, mhat, vhat

    def update_overlaps(self, qhat, mhat, vhat):
        IV, IQ, IM = self.integrate_for_qvm(qhat,mhat,vhat)
        v = IV
        m = mhat/np.sqrt(self.gamma)* IM
        q= IQ
        
        return q, m, v

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self.update_hat(q, m, V)
        return self.update_overlaps(qhat, mhat, Vhat)