# TODO : Adapter ce code pour avoir la likelihood normalisee 

from math import erfc
from random import sample

import numpy as np
from numba import jit
from scipy.integrate import quad

from .base_model import Model
from ..auxiliary.ft_logistic_integrals import ft_integrate_for_mhat, ft_integrate_for_Vhat, ft_integrate_for_Qhat, ft_integrate_derivative_beta, ft_integrate_from_mhat_pseudobayesiandatamodel_probit_teacher
from ..auxiliary import utility

class FiniteTemperatureLogisticRegression(Model):
    '''
    We assume the covariance matrix of the teacher data (i.e Psi) is the identity
    We only need the kappa coefficients coming from the activation function of the student
    '''
    # NOTE : Code sale. On va faire un boolean qui si est mis a vrai, va override tous les autres parametres 
    # Dans le cas ou on est pas overparametrized
    def __init__(self, *, sample_complexity, beta, regularisation, Delta):
        """
        arguments : 
            - beta : inv. temperature
        The energy function here is e^{beta * (loss + lambda / 2 * | w |^2)}
        """
        super(FiniteTemperatureLogisticRegression, self).__init__(Delta = Delta, sample_complexity = sample_complexity)
        self.lamb       = regularisation
        self.beta       = beta

        # record the original values if we optimize beta, lambda_
        self.original_lamb = self.lamb
        self.original_beta = self.beta

        self.rho        = 1.0
        # maximize lambda for the "evidence"
        self.optimize_lambda = False
        self.optimize_beta   = False
        self.set_bool_normalized_data_model(False)

        self.optimize_lambda_tolerance = 1e-4

    def set_optimize_lambda(self, val):
        # On garde ca mais on va changer la fonction qui optimise 
        self.optimize_lambda = val

    def set_optimize_beta(self, val):
        # On garde ca mais on va changer la fonction qui optimise 
        self.optimize_beta = val

    def set_bool_normalized_data_model(self, val : bool):
        if val:
            self.student_data_model = utility.NormalizedPseudoBayesianDataModel
        else:
            self.student_data_model = utility.PseudoBayesianDataModel

    def get_info(self):
        info = {
            'model': 'logistic_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    @staticmethod
    @jit(nopython=True)
    def aux_integrate_for_qvm(vhat : np.float32, qhat : np.float32, mhat : np.float32, lamb : np.float32, gamma : np.float32, kappa1 : np.float32, kappastar : np.float32):
        """
        NOTE : self.gamma = student_size / teacher_size, mais le gamma qu'on definit ci-dessous est juste une valeur 
        utilisee dans Marcenko-Pastur. De meme, alpha ici ne vaut pas n / d mais est seulement une constante liee a MP
        """
        alpha  = gamma
        gamma  = 1.0 / gamma
        
        sigma  = kappa1
        kk     = kappastar**2
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

    def integrate_for_qvm(self, vhat, qhat, mhat, lamb):
        return self.aux_integrate_for_qvm(vhat, qhat, mhat, self.lamb, self.gamma, self.kappa1, self.kappastar)

    def _update_overlaps_spectrum(self, vhat, qhat, mhat, lamb):    
        IV, IQ, IM = self.integrate_for_qvm(vhat, qhat, mhat, lamb)
        V = IV
        m = mhat * np.sqrt(self.gamma) * IM
        q = IQ
        return V, q, m

    def _update_overlaps_matching(self, vhat, qhat, mhat, lamb_beta):
        V = 1. / (lamb_beta + vhat)
        q = (mhat**2 + qhat) / (lamb_beta + vhat)**2
        m = mhat / (lamb_beta + vhat)
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
        lamb_beta = self.lamb * self.beta
        
        if self.matching:
            return self._update_overlaps_matching(vhat, qhat, mhat, lamb_beta)
        if self.using_kappa:
            return self._update_overlaps_spectrum(vhat, qhat, mhat, lamb_beta)
        else:
            return self._update_overlaps_covariance(vhat, qhat, mhat, lamb_beta)

    def _update_hatoverlaps(self, V, q, m):
        # the overparametrization does not change the hat overlap update so we don't have to change this 
        # since the noise level stays the same 
        sigma = self.rho - m**2/q + self.Delta
        
        # NOTE : Temporary, normally use ft_integrate_for_mhat/qhat/Vhat (m, q, V, sigma, self.beta, data_model=self.str_teacher_data_model)
        # Im = ft_integrate_from_mhat_pseudobayesiandatamodel_probit_teacher(m, q, V, sigma, self.beta)
        Im = ft_integrate_for_mhat(m, q, V, sigma, self.beta, data_model=self.str_teacher_data_model)
        Iv = ft_integrate_for_Vhat(m, q, V, sigma, self.beta, data_model=self.str_teacher_data_model)
        Iq = ft_integrate_for_Qhat(m, q, V, sigma, self.beta, data_model=self.str_teacher_data_model)
            
        mhat = self.alpha * np.sqrt(self.gamma) * Im
        Vhat = - self.alpha * Iv
        qhat = self.alpha * Iq

        return Vhat, qhat, mhat

    ### FUNCTIONS TO OPTIMIZE BETA, LAMBDA FOR EVIDENCE
    # We used the normalized likelihood to optimize the parameters of the NON-overparametrized likelihood

    def psi_w(self, Vhat, qhat, mhat, beta_lambda):
        if self.matching:
            return - 0.5 * np.log(beta_lambda + Vhat) + 0.5 * (mhat**2 + qhat) / (beta_lambda + Vhat)
        elif self.using_kappa:
            kk1, kkstar = self.kappa1**2, self.kappastar**2
            return - 0.5 * utility.mp_integral(lambda x : np.log(beta_lambda + Vhat * (kk1 * x + kkstar)), self.gamma) + 0.5 * utility.mp_integral(lambda x : (mhat * kk1 * x + qhat * (kk1 * x + kkstar)) / (beta_lambda + Vhat * (kk1 * x + kkstar)), self.gamma)
        else:
            # TODO : This
            return NotImplementedError

    def psi_y(self, V, q, m, beta):
        bound = 5.0
        teacher_data_model = {'logit' : utility.LogisticDataModel, 'probit' : utility.ProbitDataModel}[self.str_teacher_data_model]
        somme = 0.0
        Vstar = self.rho - m**2 / q + self.Delta
        for y in [-1.0, 1.0]:
            somme += quad(lambda xi : teacher_data_model.Z0(y, m / np.sqrt(q) * xi, Vstar) * np.log(self.student_data_model.Z0(y, np.sqrt(q) * xi, V, beta)) * np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi), -bound, bound)[0]
        return somme

    def get_log_partition(self, V, q, m, Vhat, qhat, mhat, beta, lambda_):
        """
        To get the evidence, take the (e.g. unnormalized) likelihood -> log_evidence + 0.5 * log(beta * lambda) 
        Verifie le 22/08 : L'expression semble bonne (au signe pres )
        """
        return self.psi_w(Vhat, qhat, mhat, beta * lambda_) + self.alpha * self.psi_y(V, q, m, self.beta) - np.sqrt(self.gamma) * m * mhat + 0.5 * (q * Vhat - qhat * V) + 0.5 * V * Vhat

    def derivative_psi_w_beta_lambda(self, Vhat, qhat, mhat, beta_lambda):
        """
        returns the derivative of Psi_w (the prior term in the free energy) w.r.t lambda_ * beta
        """
        if self.matching:
            return - 0.5 / (beta_lambda + Vhat) - 0.5 * (mhat**2 + qhat) / (beta_lambda + Vhat)**2
        elif self.using_kappa:
            kk1, kkstar = self.kappa1**2, self.kappastar**2
            return - 0.5 * utility.mp_integral(lambda x : 1.0 / (beta_lambda + Vhat * (kk1 * x + kkstar)), self.gamma) \
                   - 0.5 * utility.mp_integral(lambda x : kk1 * mhat**2 * x + qhat * (kk1 * x + kkstar) / (beta_lambda + Vhat * (kk1 * x + kkstar))**2 , self.gamma)
        else:
            # use the matrices
            # TODO : this
            raise NotImplementedError

    def _optimize_lambda_evidence(self, Vhat, qhat, mhat, beta, lambda_):
        # if self.student_data_model != utility.NormalizedPseudoBayesianDataModel:
        #     raise Exception()
        # reminder that Delta must be 0 for the logit data model ! 
        d_psi_w = self.derivative_psi_w_beta_lambda(Vhat, qhat, mhat, beta * lambda_)
        return - 0.5 / ( beta * d_psi_w )
        
    def _optimize_beta_evidence(self, V, q, m, Vhat, qhat, mhat, beta, lambda_, alpha):
        """
        NOTE : Does not work ATM
        Returns the new beta
        """
        if self.student_data_model != utility.NormalizedPseudoBayesianDataModel:
            raise Exception()
        
        Vstar   = self.rho - m**2 / q + self.Delta
        d_psi_y = ft_integrate_derivative_beta(V, q, m, Vstar, beta)
        d_psi_w = self.derivative_psi_w_beta_lambda(Vhat, qhat, mhat, beta * lambda_)
        return  - 0.5 / (lambda_ * d_psi_w + alpha * d_psi_y)

    #######################

    def update_se(self, V, q, m):
        # optimization of beta not done at the moment
        assert not self.optimize_beta
        if not self.initialized:
            raise Exception('Not initialized')
        
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)

        # I removed the optimisation of beta and lambda for this branch
        
        V, q, m = self._update_overlaps(Vhat, qhat, mhat)

        if np.isnan(V) or np.isnan(m) or np.isnan(q):
            raise Exception(f'The overlaps are NaN ! m, q, V, mhat, qhat, Vhat = {m, q, V, mhat, qhat, Vhat}')

        # record the overlaps for easier access 
        self.V, self.q, self.m, self.Vhat, self.qhat, self.mhat = V, q, m, Vhat, qhat, mhat

        return V, q, m