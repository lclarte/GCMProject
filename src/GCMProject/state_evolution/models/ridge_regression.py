import numpy as np
import scipy.integrate as integrate
from .base_model import Model


class RidgeRegression(Model):
    '''
    Implements updates for ridge regression task.
    See base_model for details on modules.
    '''
    def __init__(self, *, sample_complexity, regularisation, data_model):
        self.alpha = sample_complexity
        self.lamb = regularisation

        self.data_model = data_model
        self.rho = self.data_model.rho

    def get_info(self):
        info = {
            'model': 'ridge_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        V = np.mean(self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega))

        if self.data_model.commute:
            q = np.mean((self.data_model.spec_Omega**2 * qhat +
                        mhat**2 * self.data_model.spec_Omega * self.data_model.spec_PhiPhit) /
                        (self.lamb + Vhat*self.data_model.spec_Omega)**2)

            m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model.spec_PhiPhit /
                                                    (self.lamb + Vhat*self.data_model.spec_Omega))

        else:
            q = qhat * np.mean(self.data_model.spec_Omega**2 / (self.lamb + Vhat*self.data_model.spec_Omega)**2)
            q += mhat**2 * np.mean(self.data_model._UTPhiPhiTU *
                                   self.data_model.spec_Omega /
                                   (self.lamb + Vhat * self.data_model.spec_Omega)**2)

            m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model._UTPhiPhiTU/
                                                (self.lamb + Vhat * self.data_model.spec_Omega))


        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        Vhat = self.alpha * 1/(1+V)
        qhat = self.alpha * (self.data_model.rho + q - 2*m)/(1+V)**2
        mhat = self.alpha/np.sqrt(self.data_model.gamma) * 1/(1+V)

        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)


    def get_test_error(self, q, m):
        return self.data_model.rho + q - 2*m

    def get_train_loss(self, V, q, m):
        return (self.data_model.rho + q - 2*m) / (1+V)**2

class PoissonResampleRidgeRegression(Model):
    '''
    Implements updates for ridge regression task.
    See base_model for details on modules.
    '''
    def __init__(self, *, sample_complexity, regularisation, data_model):
        self.alpha = sample_complexity
        self.lamb = regularisation

        self.data_model = data_model
        self.rho = self.data_model.rho

    def get_info(self):
        info = {
            'model': 'ridge_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        V = np.mean(self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega))

        if self.data_model.commute:
            q = np.mean((self.data_model.spec_Omega**2 * qhat +
                        mhat**2 * self.data_model.spec_Omega * self.data_model.spec_PhiPhit) /
                        (self.lamb + Vhat*self.data_model.spec_Omega)**2)

            m = mhat / np.sqrt(self.data_model.gamma) * np.mean(self.data_model.spec_PhiPhit /
                                                    (self.lamb + Vhat*self.data_model.spec_Omega))

        else:
            q = qhat * np.mean(self.data_model.spec_Omega**2 / (self.lamb + Vhat*self.data_model.spec_Omega)**2)
            q += mhat**2 * np.mean(self.data_model._UTPhiPhiTU *
                                   self.data_model.spec_Omega /
                                   (self.lamb + Vhat * self.data_model.spec_Omega)**2)

            m = mhat/np.sqrt(self.data_model.gamma) * np.mean(self.data_model._UTPhiPhiTU/
                                                (self.lamb + Vhat * self.data_model.spec_Omega))


        return V, q, m

    def _update_hatoverlaps(self, V, q, m):
        # le facteur serait 1 / (1 + V) si on avait pas le resample
        facteur = integrate.quad(lambda t : t**(1.0 / V) * np.exp(t), 0.0, 1.0)[0]/ ( np.e * V)

        Vhat = self.alpha * facteur
        qhat = self.alpha * (self.data_model.rho + q - 2*m) * facteur**2
        mhat = self.alpha/np.sqrt(self.data_model.gamma) * facteur

        return Vhat, qhat, mhat

    def update_se(self, V, q, m):
        Vhat, qhat, mhat = self._update_hatoverlaps(V, q, m)
        return self._update_overlaps(Vhat, qhat, mhat)


    def get_test_error(self, q, m):
        return self.data_model.rho + q - 2*m

    def get_train_loss(self, V, q, m):
        return (self.data_model.rho + q - 2*m) / (1+V)**2