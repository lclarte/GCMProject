"""
Integrals for finite temperature logistic
NOTE : For now, we only use the probit data model but we can easily extend to the logistic model
"""

import numpy as np
from ..auxiliary.utility import LogisticDataModel, ProbitDataModel, PseudoBayesianDataModel, NormalizedPseudoBayesianDataModel
from scipy.integrate import quad

def p_out(x):
    if x > -10:
        return np.log(1. + np.exp(- x))
    else:
        return -x

def ft_integrate_for_mhat(M, Q, V, Vstar, beta, data_model = 'probit', student_data_model = PseudoBayesianDataModel):
    assert student_data_model in [PseudoBayesianDataModel, NormalizedPseudoBayesianDataModel]
    bound = 5.0
    somme = 0.0
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * student_data_model.f0(y, np.sqrt(Q)*xi, V, beta) * current_data_model.dZ0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

def ft_integrate_for_Qhat(M, Q, V, Vstar, beta, data_model = 'probit', student_data_model = PseudoBayesianDataModel):
    assert student_data_model in [PseudoBayesianDataModel, NormalizedPseudoBayesianDataModel]
    bound = 5.0
    somme = 0.0
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * student_data_model.f0(y, np.sqrt(Q)*xi, V, beta)**2 * current_data_model.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]  
    return somme

def ft_integrate_for_Vhat(M, Q, V, Vstar, beta, data_model = 'probit', student_data_model = PseudoBayesianDataModel):
    assert student_data_model in [PseudoBayesianDataModel, NormalizedPseudoBayesianDataModel]
    bound = 5.0
    somme = 0.0
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * student_data_model.df0(y, np.sqrt(Q)*xi, V, beta) * current_data_model.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

def ft_integrate_derivative_beta(V, q, m, Vstar, beta):
    """
    Integrate Z_0 \partial_{beta} Z_{erm} / Z_{erm} which appears in the derivative w.r.t beta of the evidence
    => useful for evidence maximization
    NOTE : for the probit model, Vstar must include sigma**2
    """
    bound              = 5.0
    somme              = 0.0
    teacher_data_model = LogisticDataModel
    for y in [-1, 1]:
        somme += quad(lambda xi : NormalizedPseudoBayesianDataModel.dbetaZ0(y, np.sqrt(q) * xi, V, beta) / NormalizedPseudoBayesianDataModel.Z0(y, np.sqrt(q) * xi, V, beta) * teacher_data_model.Z0(y, m / np.sqrt(q) * xi, Vstar) * np.exp(- xi**2 / 2.0), -bound, bound)[0] / np.sqrt(2 * np.pi)
    return somme
