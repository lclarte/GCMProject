from math import erfc
import numpy as np
import scipy.stats as stats
import scipy.integrate
from scipy.linalg import sqrtm
from .base_model import Model
from ..auxiliary.logistic_integrals import integrate_for_mhat, integrate_for_Vhat, integrate_for_Qhat, traning_error_logistic
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
        self.alpha        = sample_complexity
        self.lamb         = regularisation
        self.Delta        = Delta
        
    def get_info(self):
        info = {
            'model': 'logistic_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def init_with_data_model(self, data_model):
        self.initialized = True
        self.using_kappa = False
        self.matching    = False

        self.data_model  = data_model        
        
        # Do a transformation of the matrices to simplify
        self.teacher_size = data_model.k
        self.student_size = data_model.p
        self.gamma        = self.student_size / self.teacher_size

        self.Phi          = data_model.Phi.T
        self.Psi          = data_model.Psi
        self.Omega        = data_model.Omega

        # Effective noise is due to the mismatch in the models.
        # Should appear only in the update of the hat overlaps normally
        self.mismatch_noise_var = self.data_model.get_rho() - self.data_model.get_projected_rho()
        # NOTE : Don't add Delta in the data_model because the noise is not a property of the data but of the teacher
        # Delta = Sigma**2 
        # effective_Delta should be useful ONLY for the computatino of the test error,
        # the additional noise due to the model mismatch is taken caren of indirectly in the state evolution
        self.effective_Delta = self.Delta + self.mismatch_noise_var
        self.rho = self.data_model.get_projected_rho()

    def init_with_spectrum(self, kappa1, kappastar, gamma):
        self.initialized = True
        self.matching    = False
        self.using_kappa = True
        self.kappa1      = kappa1
        self.kappastar   = kappastar
        self.gamma       = gamma

        self.mismatch_noise_var = utility.get_additional_noise_from_kappas(kappa1, kappastar, gamma)
        self.effective_Delta = self.Delta + self.mismatch_noise_var
        self.rho         = 1.0

    def init_matching(self):
        self.initialized     = True
        self.matching        = True
        self.rho             = 1.0
        self.gamma           = 1.0
        self.effective_Delta = self.Delta

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

    def _update_overlaps_matching(self, vhat, qhat, mhat):
        V = 1. / (self.lamb + vhat)
        q = (mhat**2 + qhat) / (self.lamb + vhat)**2
        m = mhat / (self.lamb + vhat)
        return V, q, m

    def _update_overlaps_spectrum(self, vhat, qhat, mhat):
        IV, IQ, IM = self.integrate_for_qvm(vhat, qhat, mhat)
        V = IV
        m = mhat * np.sqrt(self.gamma) * IM
        q = IQ
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
        Im = integrate_for_mhat(m, q, V, Vstar + self.Delta)
        Iv = integrate_for_Vhat(m, q, V, Vstar + self.Delta)
        Iq = integrate_for_Qhat(m, q, V, Vstar + self.Delta)

        # NOTE : Ici on multuplied par self.gamma = d / p
        # dans le code originel, gamma = p / d donc ils divisent par sqrt(gamma)
        mhat = self.alpha * np.sqrt(self.gamma) * Im/V
        Vhat = self.alpha * ((1/V) - (1/V**2) * Iv)
        qhat = self.alpha * Iq/V**2

        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        if not self.initialized:
            raise Exception('Model not initialized !! ')
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)

    def get_test_error(self, q, m):
        # NOTE : Removed the noise to be like the GCM Project
        # We still include the noise due to the mismatch because this is incompressible
        return np.arccos(m/np.sqrt(q * self.rho))/np.pi

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
        inv_p = sigmoid_inv(p)
        rho   = self.rho
        return p - 0.5 * erfc(- (m / q * inv_p) / np.sqrt(2*(rho - m**2 / q + self.effective_Delta)))

    def get_train_loss(self, V, q, m):
        Vstar = self.rho - m**2/q
        return traning_error_logistic(m, q, V, Vstar + self.effective_Delta)