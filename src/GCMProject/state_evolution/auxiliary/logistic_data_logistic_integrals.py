import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

from ..auxiliary import utility
from ..auxiliary.utility import LogisticDataModel

## functions to compute the proximal and its derivative 

def logistic_loss(y, z):
    return np.log(1 + np.exp(- y * z))

def logistic_loss_dl(y, z):
    if y * z > 0:
        x = np.exp(- y * z)
        return -y * x / (1. + x)
    else:
        return - y / (np.exp(y * z) + 1)

def logistic_loss_ddl(y, z):
    if np.abs(y * z) > 500:
        if y * z > 0:
            return y**2 / (4) * np.exp(-y * z)
        else:
            return y**2 / (4) * np.exp(y * z)
    else:
        return y**2 / (4 * np.cosh(y * z / 2)**2)

def find_star(y, omega, Vstar):
    root = root_scalar(
        z_star_arg, bracket=[-1e20, 1e20], args=(y, omega, Vstar), xtol=1e-15)

    ## Find Proximal ##
    z_star = root.root
    dz_star = 1 / (1 + Vstar * logistic_loss_ddl(y, z_star))
    l_y_z = logistic_loss(y, z_star)
    return z_star, dz_star, l_y_z

def z_star_arg(z, y, omega, Vstar):
    return omega - Vstar * logistic_loss_dl(y, z) - z
    # oppose : (z - omega) + V * loss.dl

def fout_dfout(y, omega, V):
        V_inf = V
        z_star, dz_star, l_y_z = find_star(y, omega, V_inf)
        ## Zout ##
        L = 1 / (2 * V_inf) * (z_star - omega)**2 + l_y_z
        Zout = np.exp(- L) / \
            np.sqrt((2 * np.pi * V) * (2 * np.pi))
        ## fout ##
        fout = 1 / V * (z_star - omega)
        ## dfout ##
        dfout = 1 / V * (dz_star - 1)
        return Zout, fout, dfout

def fout(y, w, V):
    z_star, _, _ = find_star(y, w, V)
    x = 1 / V * (z_star - w)
    return x

def dfout(y, w, V):
    _, dz_star, _ = find_star(y, w, V)
    return 1.0 / V * (dz_star - 1)

def logistic_integrate_for_mhat(M, Q, V, Vstar):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * fout(y, np.sqrt(Q) * xi, V) * LogisticDataModel.dZ0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

def logistic_integrate_for_Qhat(M, Q, V, Vstar):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * fout(y, np.sqrt(Q)*xi, V)**2 * LogisticDataModel.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme

def logistic_integrate_for_Vhat(M, Q, V, Vstar):
    bound = 10.0
    somme = 0.0
    for y in [-1, 1]:
        somme += quad(lambda xi : np.exp(- xi**2 / 2.0) / np.sqrt(2 * np.pi) * dfout(y, np.sqrt(Q)*xi, V) * LogisticDataModel.Z0(y, M / np.sqrt(Q) * xi, Vstar), -bound, bound, limit=500)[0]
    return somme