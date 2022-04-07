import numpy as np
import scipy.stats as stats

from scipy.stats import norm
from scipy.integrate import quad

from .base_model import Model
from ..auxiliary import utility

class BayesOptimalProbitSpectrum(Model):
    '''
    We assume the covariance matrix of the teacher data (i.e Psi) is the identity
    We only need the kappa coefficients coming from the activation function of the student
    '''
    # NOTE : Code sale. On va faire un boolean qui si est mis a vrai, va override tous les autres parametres 
    # Dans le cas ou on est pas overparametrized
    def __init__(self, Delta = 0., *, sample_complexity, gamma, kappa1, kappastar):
        self.alpha      = sample_complexity
        self.gamma      = gamma
        self.rho        = 1.0

        self.kappa1     = kappa1
        self.kappa_star = kappastar
        
        # NOTE : Don't add Delta in the data_model because the noise is not a property of the data but of the teacher
        # Delta = Sigma**2 
        self.Delta = Delta
        
    def get_info(self):
        info = {
            'model'             : 'logistic_regression',
            'sample_complexity' : self.alpha,
            'lambda'            : self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat):
        lambda_plus  = (1. + np.sqrt(self.gamma))**2
        lambda_minus = (1. - np.sqrt(self.gamma))**2
        
        kk1    = self.kappa1**2
        kkstar = self.kappa_star**2

        def to_integrate(lambda_):
            mp_density    = np.sqrt((lambda_ - lambda_minus) * (lambda_plus - lambda_))
            # change_of_var = np.abs((kk1 / (kkstar + kk1 * lambda_)) - (kk1**2 * lambda_ / (kkstar + kk1 * lambda_)**2) )
            change_of_var = 1.0
            return kk1**2 * lambda_ / ((kkstar + kk1 * lambda_) * (kkstar + kk1 * lambda_ *( 1. + qhat))) * mp_density * change_of_var

        q = qhat / (2 * np.pi) * quad(to_integrate, lambda_minus, lambda_plus)[0] 
        return self.rho - q, q, q

    def _update_hatoverlaps(self, V, q, m):
        int_lims = 20.0
        Delta = self.Delta

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
        # We still include the noise due to the mismatch because this is incompressible
        return np.arccos(m/np.sqrt(q * self.rho))/np.pi

    def get_test_loss(self, q, m):
        return -1
    
    def get_calibration(self, q, m, p=0.75):
        return -1

    def get_train_loss(self, V, q, m):
        return -1