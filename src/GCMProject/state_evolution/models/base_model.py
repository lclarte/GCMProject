import numpy as np
from scipy.linalg import sqrtm

from ..auxiliary import utility

class Model(object):
    '''
    Base class for a model.
    -- args --
    sample_complexity: sample complexity
    regularisation: ridge penalty coefficient
    data_model: data_model instance. See /data_model/
    '''

    def __init__(self, *, sample_complexity, Delta):
        self.alpha = sample_complexity
        self.Delta = Delta
        self.initialized = False
        self.using_kappa = False
        self.matching    = False
        self.use_probit_data_model()

    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'model': 'generic',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def update_se(self, V, q, m):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        raise NotImplementedError

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
        self.rho         = self.data_model.get_rho()
        self.projected_rho = self.data_model.get_projected_rho()

    def init_with_random_features(self, kappa1, kappastar, gamma):
        self.initialized = True
        self.matching    = False
        self.using_kappa = True
        self.kappa1      = kappa1
        self.kappastar   = kappastar
        self.gamma       = gamma

        self.mismatch_noise_var = utility.get_additional_noise_from_kappas(kappa1, kappastar, gamma)
        self.effective_Delta = self.Delta + self.mismatch_noise_var
        self.rho         = 1.0
        self.projected_rho = self.rho - self.mismatch_noise_var

    def init_matching(self):
        self.initialized     = True
        self.matching        = True
        self.using_kappa     = False
        self.rho             = 1.0
        self.projected_rho   = 1.0
        self.gamma           = 1.0
        self.effective_Delta = self.Delta

    def use_logistic_data_model(self):
        if self.Delta != 0:
            raise Exception()
        # no additional noise
        self.type_of_data_model = 'logit'

    def use_probit_data_model(self):
        self.type_of_data_model = 'probit'