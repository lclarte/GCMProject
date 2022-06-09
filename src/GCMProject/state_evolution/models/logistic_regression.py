from math import erfc
from random import sample
import numpy as np
import scipy.stats as stats
import scipy.integrate
from scipy.linalg import sqrtm
from .base_model import Model
from ..auxiliary.logistic_integrals import integrate_for_mhat, integrate_for_Vhat, integrate_for_Qhat
from ..auxiliary.logistic_data_logistic_integrals import logistic_integrate_for_mhat, logistic_integrate_for_Qhat, logistic_integrate_for_Vhat
from ..auxiliary import utility

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_inv(y):
    return np.log(y/(1-y))

class LogisticRegression(Model):
    '''
    Implements updates for logistic regression task.
    See base_model for details on modules.
    '''
    def __init__(self, Delta = 0., *, sample_complexity, regularisation):
        super().__init__(Delta=Delta, sample_complexity=sample_complexity)
        # self.alpha        = sample_complexity
        self.lamb         = regularisation
        # self.Delta        = Delta
        
    def get_info(self):
        info = {
            'model': 'logistic_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps_covariance(self, Vhat, qhat, mhat):
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

    def integrate_for_qvm(self, vhat, qhat, mhat):
        """
        NOTE : self.gamma = student_size / teacher_size, mais le gamma qu'on definit ci-dessous est juste une valeur 
        utilisee dans Marcenko-Pastur. De meme, alpha ici ne vaut pas n / d mais est seulement une constante liee a MP
        """
        alpha  = self.gamma
        gamma  = 1.0 / self.gamma
        
        sigma  = self.kappa1
        lamb   = self.lamb
        kk     = self.kappastar**2
        alphap = (sigma*(1 + np.sqrt(alpha)))**2
        alpham = (sigma*(1 - np.sqrt(alpha)))**2
        if lamb == 0:
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

    def _update_overlaps_spectrum(self, vhat, qhat, mhat):
        IV, IQ, IM = self.integrate_for_qvm(vhat, qhat, mhat)
        V = IV
        m = mhat * np.sqrt(self.gamma) * IM
        q = IQ
        return V, q, m

    def _update_overlaps_matching(self, vhat, qhat, mhat):
        V = 1. / (self.lamb + vhat)
        q = (mhat**2 + qhat) / (self.lamb + vhat)**2
        m = mhat / (self.lamb + vhat)
        return V, q, m

    def _update_overlaps(self, vhat, qhat, mhat):
        if self.matching:
            return self._update_overlaps_matching(vhat, qhat, mhat)
        elif self.using_kappa:
            return self._update_overlaps_spectrum(vhat, qhat, mhat)
        else:
            return self._update_overlaps_covariance(vhat, qhat, mhat)

    def _update_hatoverlaps(self, V, q, m):
        Vstar = self.rho - m**2/q

        # NOTE : Normally we don't use effective_Delta, it's (implicitely) 
        # taken into account in the integrals !
        # NOTE : Ici on multuplied par self.gamma = d / p
        # dans le code originel, gamma = p / d donc ils divisent par sqrt(gamma)
        if self.type_of_data_model == 'probit':
            Im = integrate_for_mhat(m, q, V, Vstar + self.Delta)
            Iv = integrate_for_Vhat(m, q, V, Vstar + self.Delta)
            Iq = integrate_for_Qhat(m, q, V, Vstar + self.Delta)
            mhat = self.alpha * np.sqrt(self.gamma) * Im/V
            Vhat = self.alpha * ((1/V) - (1/V**2) * Iv)
            qhat = self.alpha * Iq/V**2
            return Vhat, qhat, mhat
            
        elif self.type_of_data_model == 'logit':
            assert self.Delta == 0.0
            Im = logistic_integrate_for_mhat(m, q, V, Vstar)
            Iv = logistic_integrate_for_Vhat(m, q, V, Vstar)
            Iq = logistic_integrate_for_Qhat(m, q, V, Vstar)
            mhat = self.alpha * np.sqrt(self.gamma) * Im
            Vhat = - self.alpha * Iv
            qhat = self.alpha * Iq
            
            return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        if not self.initialized:
            raise Exception('Model not initialized !! ')
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)