'''
Auxiliary integrals needed for computing the likelihood update functions
for logistic regression.
'''

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import minimize_scalar

from . import utility

def loss(z):
    # = - log(sigmoid(z))
    return np.log(1 + np.exp(-z))

def moreau_loss(x, y, omega,V):
    return (x-omega)**2/(2*V) + loss(y*x)

# === Integrals for logistic regression === 

def f_mhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    return np.exp(-ωstar**2/(2*Vstar))*(λstar_plus - ω)

def f_mhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    return np.exp(-ωstar**2/(2*Vstar))*(λstar_minus - ω)

def integrate_for_mhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_mhat_plus(ξ, M, Q, V, Vstar) * utility.gaussian(ξ), -10, 10, limit=500)[0]
    I2 = quad(lambda ξ: f_mhat_minus(ξ, M, Q, V, Vstar) * utility.gaussian(ξ), -10, 10, limit=500)[0]
    return (I1 - I2)*(1/np.sqrt(2*np.pi*Vstar))

# Vhat_x #
def f_Vhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    return (1/(1/V + (1/4) * (1/np.cosh(λstar_plus/2)**2))) * (1 + erf(ωstar/np.sqrt(2*Vstar)))

def f_Vhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    return (1/(1/V + (1/4) * (1/np.cosh(-λstar_minus/2)**2))) * (1 - erf(ωstar/np.sqrt(2*Vstar)))
    
def integrate_for_Vhat(M, Q, V, Vstar, data_model = 'probit'):
    I1 = quad(lambda ξ: f_Vhat_plus(ξ, M, Q, V, Vstar) * utility.gaussian(ξ), -10, 10, limit=500)[0]
    I2 = quad(lambda ξ: f_Vhat_minus(ξ, M, Q, V, Vstar) * utility.gaussian(ξ), -10, 10, limit=500)[0]
    return (1/2) * (I1 + I2)

# Qhat_x#
def f_qhat_plus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * (λstar_plus - ω)**2

def f_qhat_minus(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * (λstar_minus - ω)**2

def integrate_for_Qhat(M, Q, V, Vstar):
    I1 = quad(lambda ξ: f_qhat_plus(ξ, M, Q, V, Vstar) * utility.gaussian(ξ), -10, 10, limit=500)[0]
    I2 = quad(lambda ξ: f_qhat_minus(ξ, M, Q, V, Vstar) * utility.gaussian(ξ), -10, 10, limit=500)[0]
    return (1/2) * (I1 + I2)

### === Integrals for training loss with logistic loss and probit data model

def Integrand_training_loss_plus_logistic(ξ, M, Q, V, Vstar):
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
#   λstar_plus = np.float(mpmath.findroot(lambda λstar_plus: λstar_plus - ω - V/(1 + np.exp(np.float(λstar_plus))), 10e-10))
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    
    l_plus = loss(λstar_plus)
    
    return (1 + erf(ωstar/np.sqrt(2*Vstar))) * l_plus

def Integrand_training_loss_minus_logistic(ξ, M, Q, V, Vstar):
    # !!! Only works for probit data model !!! 
    ω = np.sqrt(Q)*ξ
    ωstar = (M/np.sqrt(Q))*ξ
#   λstar_minus = np.float(mpmath.findroot(lambda λstar_minus: x - ω + V/(1 + np.exp(-np.float(λstar_minus))), 10e-10))
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    
    l_minus = loss(-λstar_minus)

    return (1 - erf(ωstar/np.sqrt(2*Vstar))) * l_minus

def training_loss_logistic(M, Q, V, Vstar):
    """
    Returns the training ERROR (and not the loss ...) for ERM
    """
    I1 = quad(lambda ξ: Integrand_training_loss_plus_logistic(ξ, M, Q, V, Vstar) * utility.gaussian(ξ), -10, 10, limit=500)[0]
    I2 = quad(lambda ξ: Integrand_training_loss_minus_logistic(ξ, M, Q, V, Vstar) * utility.gaussian(ξ), -10, 10, limit=500)[0]
    return (1/2)*(I1 + I2)


### ==== to optimize for lambda to maximize the evidence 

def lambda_objective(lambda_, beta, kappa1, kappastar, gamma, qhat, mhat, Vhat):
    """
    Function we want to cancel (might be positive or negative)
    """
    kk1, kkstar = kappa1**2, kappastar**2
    integrand = lambda x : 1. / (lambda_ * beta + Vhat * (kk1 * x + kkstar)) \
                            - (mhat**2 * kk1 * x + qhat * (kk1 * x + kkstar)) / (beta * lambda_ + Vhat * (kk1 * x + kkstar))**2
    return utility.mp_integral(integrand, gamma)

def lambda_objective_prime(lambda_, beta, kappa1, kappastar, gamma, qhat, mhat, Vhat):
    kk1, kkstar = kappa1**2, kappastar**2
    integrand = lambda x : - 1. / (lambda_ * beta + Vhat * (kk1 * x + kkstar))**2 \
                            + 2 * (mhat**2 * kk1 * x + qhat * (kk1 * x + kkstar)) / (beta * lambda_ + Vhat * (kk1 * x + kkstar))**3
    return beta * utility.mp_integral(integrand, gamma)
