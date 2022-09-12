"""
Integrals for finite temperature logistic
NOTE : For now, we only use the probit data model but we can easily extend to the logistic model
"""

import sys
# temporary solution
sys.path.append('/Users/clarte/opt/usr/local/lib')
import cython_functions

import numpy as np

from   ..auxiliary.utility import LogisticDataModel, ProbitDataModel, PseudoBayesianDataModel, NormalizedPseudoBayesianDataModel
from   scipy.integrate import quad
from   numba import jit


FT_QUAD_BOUND = 10.0

def modified_ft_f0(y, w, V, beta):
    try:
        result = PseudoBayesianDataModel.f0(y, w, V, beta)
    except:
        return 0.0
    if np.isfinite(result):
        return result
    return 0.0

def modified_df0(y, w, V, beta):
    try:
        result = PseudoBayesianDataModel.df0(y, w, V, beta)
    except:
        return 0.0
    if np.isfinite(result):
        return result
    return 0.0

def ft_integrate_for_mhat(M, Q, V, Vstar, beta, data_model = 'probit', student_data_model = PseudoBayesianDataModel):
    assert student_data_model in [PseudoBayesianDataModel, NormalizedPseudoBayesianDataModel]
    bound = FT_QUAD_BOUND
    somme = 0.0
    limit = 500
    # "current" = the one of the learner
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
        integrale = quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * modified_ft_f0(y, np.sqrt(Q)*xi, V, beta) * current_data_model.dZ0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=limit)[0]
        somme += integrale
    return somme

def ft_integrate_for_Qhat(M, Q, V, Vstar, beta, data_model = 'probit', student_data_model = PseudoBayesianDataModel):
    assert student_data_model in [PseudoBayesianDataModel, NormalizedPseudoBayesianDataModel]
    bound = FT_QUAD_BOUND
    somme = 0.0
    limit = 500
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
        integrale = quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * modified_ft_f0(y, np.sqrt(Q)*xi, V, beta)**2 * current_data_model.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=limit)[0]  
        somme += integrale
    return somme

def ft_integrate_for_Vhat(M, Q, V, Vstar, beta, data_model = 'probit', student_data_model = PseudoBayesianDataModel):
    # assert student_data_model in [PseudoBayesianDataModel, NormalizedPseudoBayesianDataModel]
    bound = FT_QUAD_BOUND
    somme = 0.0
    limit = 500
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
        integrale = quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * modified_df0(y, np.sqrt(Q)*xi, V, beta) * current_data_model.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=limit)[0]
        somme += integrale
    return somme

def ft_integrate_derivative_beta(V, q, m, Vstar, beta):
    """
    Integrate Z_0 \partial_{beta} Z_{erm} / Z_{erm} which appears in the derivative w.r.t beta of the evidence
    => useful for evidence maximization
    NOTE : for the probit model, Vstar must include sigma**2
    """
    bound              = FT_QUAD_BOUND
    somme              = 0.0
    limit              = 500
    teacher_data_model = LogisticDataModel
    for y in [-1, 1]:
        integrale = quad(lambda xi : NormalizedPseudoBayesianDataModel.dbetaZ0(y, np.sqrt(q) * xi, V, beta) / NormalizedPseudoBayesianDataModel.Z0(y, np.sqrt(q) * xi, V, beta) * teacher_data_model.Z0(y, m / np.sqrt(q) * xi, Vstar) * np.exp(- xi**2 / 2.0), -bound, bound, limit=limit)[0] / np.sqrt(2 * np.pi)
        somme += integrale
    return somme
