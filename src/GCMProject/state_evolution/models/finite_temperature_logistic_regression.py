from math import erfc

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .base_model import Model
from ..auxiliary.logistic_integrals import ft_integrate_for_mhat, ft_integrate_for_Vhat, ft_integrate_for_Qhat, lambda_objective, lambda_objective_prime
from ..auxiliary import utility

class FiniteTemperatureLogisticRegression(Model):
    '''
    We assume the covariance matrix of the teacher data (i.e Psi) is the identity
    We only need the kappa coefficients coming from the activation function of the student
    '''
    # NOTE : Code sale. On va faire un boolean qui si est mis a vrai, va override tous les autres parametres 
    # Dans le cas ou on est pas overparametrized
    def __init__(self, Delta = 0., *, sample_complexity, beta, regularisation):
        """
        arguments : 
            - beta : inv. temperature
        The energy function here is e^{beta * (loss + lambda / 2 * | w |^2)}
        """
        self.initialized= False
        self.alpha      = sample_complexity
        self.lamb       = regularisation
        # when we opt the lambda for evidence
        self.beta       = beta
        self.rho        = 1.0
        self.matching   = False

        # NOTE : Don't add Delta in the data_model because the noise is not a property of the data but of the teacher
        # Delta = Sigma**2 
        self.Delta = Delta

        self.update_lambda_bool = False
        self.lamb_opt = self.lamb
        self.lambda_min = 1e-5
        self.lambda_max = 1.0

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
        self.rho          = 1.0

    def init_matching(self):
        self.initialized     = True
        self.matching        = True
        self.rho             = 1.0
        self.gamma           = 1.0
        self.effective_Delta = self.Delta


    def get_info(self):
        info = {
            'model': 'logistic_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def integrate_for_qvm(self, vhat, qhat, mhat, lamb):
        """
        NOTE : self.gamma = student_size / teacher_size, mais le gamma qu'on definit ci-dessous est juste une valeur 
        utilisee dans Marcenko-Pastur. De meme, alpha ici ne vaut pas n / d mais est seulement une constante liee a MP
        """
        alpha  = self.gamma
        gamma  = 1.0 / self.gamma
        
        sigma  = self.kappa1
        kk     = self.kappastar**2
        alphap = (sigma*(1 + np.sqrt(alpha)))**2
        alpham = (sigma*(1 - np.sqrt(alpha)))**2
        if lamb == 0:
            den =1+kk*vhat
            aux =np.sqrt(((alphap+kk)*vhat+1)*((alpham+kk)*vhat+1))
            aux2=np.sqrt(((alphap+kk)*vhat+1)/((alpham+kk)*vhat+1))
            IV = ((kk*vhat+1)*((alphap+alpham)*vhat+2)-2*kk*vhat**2*np.sqrt(alphap*alpham)-2*aux)/(4*alpha*vhat**2*(kk*vhat+1)*sigma**2)
            IV = IV + max(0,1-gamma)*kk/(1+vhat*kk)
            I1 = (alphap * vhat*(-3*den+aux)+4*den*(-den+aux)+alpham*vhat*(-2*alphap*vhat-3*den+aux))/(4*alpha*vhat**3*sigma**2*aux)
            I2 = (alphap * vhat+alpham*vhat*(1-2*aux2)+2*den*(1-aux2))/(4*alpha*vhat**2*aux*sigma**2)
            I3 = (2*vhat * alphap*alpham+(alphap+alpham)*den-2*np.sqrt(alphap*alpham)*aux)/(4*alpha*den**2*sigma**2*aux)
            IQ = (qhat + mhat**2)*I1+(2*qhat+mhat**2)*kk*I2+qhat*kk**2*I3
            IQ = IQ + max(0,1-gamma)*qhat*kk**2/den**2
            IM = ((alpham + alphap+2*kk)*vhat+2-2*aux)/(4*alpha*vhat**2*sigma**2)
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

    def _update_overlaps_spectrum(self, vhat, qhat, mhat, lamb):    
        IV, IQ, IM = self.integrate_for_qvm(vhat, qhat, mhat, lamb)
        V = IV
        m = mhat * np.sqrt(self.gamma) * IM
        q = IQ
        return V, q, m

    def _update_overlaps_matching(self, vhat, qhat, mhat, lamb):
        V = 1. / (lamb + vhat)
        q = (mhat**2 + qhat) / (lamb + vhat)**2
        m = mhat / (lamb + vhat)
        return V, q, m

    def _update_overlaps_covariance(self, Vhat, qhat, mhat, lamb):
        # should not be affected by the noise level
        V = np.mean(self.data_model.spec_Omega/(lamb + Vhat * self.data_model.spec_Omega))

        if self.data_model.commute:
            q = np.mean((self.data_model.spec_Omega**2 * qhat +
                                           mhat**2 * self.data_model.spec_Omega * self.data_model.spec_PhiPhit) /
                                          (lamb + Vhat*self.data_model.spec_Omega)**2)

            m = mhat * np.sqrt(self.gamma) * np.mean(self.data_model.spec_PhiPhit/(lamb + Vhat*self.data_model.spec_Omega))

        else:
            q = qhat * np.mean(self.data_model.spec_Omega**2 / (lamb + Vhat*self.data_model.spec_Omega)**2)
            q += mhat**2 * np.mean(self.data_model._UTPhiPhiTU * self.data_model.spec_Omega/(lamb + Vhat * self.data_model.spec_Omega)**2)

            m = mhat * np.sqrt(self.gamma) * np.mean(self.data_model._UTPhiPhiTU/(lamb + Vhat * self.data_model.spec_Omega))

        return V, q, m

    def _update_overlaps(self, vhat, qhat, mhat):
        """
        Update of overlaps is NOT the same as logistic regression because here we have exp^{lambda ... }
        """
        lamb = self.lamb * self.beta
        
        if self.matching:
            return self._update_overlaps_matching(vhat, qhat, mhat, lamb)
        if self.using_kappa:
            return self._update_overlaps_spectrum(vhat, qhat, mhat, lamb)
        else:
            return self._update_overlaps_covariance(vhat, qhat, mhat, lamb)

    def _update_hatoverlaps(self, V, q, m):
        # the overparametrization does not change the hat overlap update so we don't have to change this 
        # since the noise level stays the same 
        sigma = self.rho - m**2/q + self.Delta
        
        Im = ft_integrate_for_mhat(m, q, V, sigma, self.beta)
        Iv = ft_integrate_for_Vhat(m, q, V, sigma, self.beta)
        Iq = ft_integrate_for_Qhat(m, q, V, sigma, self.beta)
            
        mhat = self.alpha * np.sqrt(self.gamma) * Im
        Vhat = - self.alpha * Iv
        qhat = self.alpha * Iq

        return Vhat, qhat, mhat

    def update_lambda(self, lambda_, Vhat, qhat, mhat):
        objective = lambda lambda_ : lambda_objective(lambda_, self.beta, self.kappa1, self.kappastar, self.gamma, qhat, mhat, Vhat)**2
        objective_prime = lambda lambda_ : 2.0 * lambda_objective_prime(lambda_, self.beta, self.kappa1, self.kappastar, self.gamma, qhat, mhat, Vhat) \
                                            * lambda_objective(lambda_, self.beta, self.kappa1, self.kappastar, self.gamma, qhat, mhat, Vhat)
        res = minimize(fun = lambda x : objective(x[0]), x0=[lambda_], bounds=[(self.lambda_min, self.lambda_max)], jac=objective_prime, tol=1e-8)
        return res.x[0]
        # res = minimize_scalar(objective, bounds=(self.lambda_min, self.lambda_max), tol=1e-8)
        # return res.x
        

    def update_se(self, V, q, m):
        if not self.initialized:
            raise Exception('Not initialized')
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        if self.update_lambda_bool:
            self.lamb_opt = self.update_lambda(self.lamb_opt, Vhat, qhat, mhat)
            print(self.lamb_opt)
        return self._update_overlaps(Vhat, qhat, mhat)

    def get_test_error(self, q, m):
        # NOTE : Removed the noise to be like the GCM Project
        # We still include the noise due to the mismatch because this is incompressible
        return np.arccos(m/np.sqrt(q * self.rho))/np.pi

    def get_test_loss(self, q, m):
        return -1
    
    def get_calibration(self, q, m, p=0.75):
        return -1

    def get_train_loss(self, V, q, m):
        return -1