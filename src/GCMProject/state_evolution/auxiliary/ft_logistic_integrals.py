"""
Integrals for finite temperature logistic
NOTE : For now, we only use the probit data model but we can easily extend to the logistic model
"""

import numpy as np
from ..auxiliary.utility import LogisticDataModel, ProbitDataModel
from scipy.integrate import quad

def p_out(x):
    if x > -10:
        return np.log(1. + np.exp(- x))
    else:
        return -x

def ft_logistic_Z(y, w, V, beta, bound = 5.0):
    sqrtV = np.sqrt(V)
    return sqrtV * quad(lambda z : np.exp(-beta * p_out(y * (z * sqrtV + w))) * np.exp(- z**2 / 2), -bound, bound, limit=500)[0]

def ft_logistic_dwZ(y, w, V, beta, bound = 5.0):
    sqrtV = np.sqrt(V)
    return quad(lambda z : z * np.exp(-beta * p_out(y * (z * sqrtV + w))) * np.exp(- z**2 / 2), -bound, bound, limit=500)[0] 

def ft_logistic_ddwZ(y, w, V, beta, Zerm = None, bound = 5.0):
    Zerm = Zerm or ft_logistic_Z(y, w, V, beta, bound)
    sqrtV = np.sqrt(V)
    to_integrate = lambda z : z**2 * np.exp(-beta * p_out(y * (z * sqrtV + w))) * np.exp(-z**2 / 2.)
    return - Zerm / V + quad(to_integrate, -bound, bound, limit=500)[0] / sqrtV

def ft_logistic_ferm(y, w, V, beta, bound = 5.0):
    return ft_logistic_dwZ(y, w, V, beta, bound) / ft_logistic_Z(y, w, V, beta, bound)

def ft_logistic_dferm(y, w, V, beta, bound = 5.0):
    Zerm = ft_logistic_Z(y, w, V, beta, bound)
    dZerm = ft_logistic_dwZ(y, w, V, beta, bound) 
    ddZerm = ft_logistic_ddwZ(y, w, V, beta, Zerm, bound)
    return ddZerm / Zerm - (dZerm / Zerm)**2

def ft_integrate_for_mhat(M, Q, V, Vstar, beta, data_model = 'probit'):
    bound = 5.0
    somme = 0.0
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
            somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * ft_logistic_ferm(y, np.sqrt(Q)*xi, V, beta) * current_data_model.dZ0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

def ft_integrate_for_Qhat(M, Q, V, Vstar, beta, data_model = 'probit'):
    bound = 5.0
    somme = 0.0
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * ft_logistic_ferm(y, np.sqrt(Q)*xi, V, beta)**2  * current_data_model.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]  
    return somme

def ft_integrate_for_Vhat(M, Q, V, Vstar, beta, data_model = 'probit'):
    bound = 5.0
    somme = 0.0
    current_data_model = ProbitDataModel
    if data_model == 'logit':
        current_data_model = LogisticDataModel
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * ft_logistic_dferm(y, np.sqrt(Q)*xi, V, beta) * current_data_model.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

### Using the logistic data model