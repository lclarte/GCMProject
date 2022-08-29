from numba import jit, config
import numpy as np
from time import time

from state_evolution.models.finite_temperature_logistic_regression import FiniteTemperatureLogisticRegression

config.DISABLE_JIT = False

def test_speed_iteration_pseudo_bayes():
    """
    Test the speed (with/without yaml by changing the .numba_config.yaml file) of pseudo bayes iteration
    """
    alpha   = 1.0
    beta    = 1.0
    lambda_ = 1e-4
    sigma   = 0.5

    m0 = q0 = 0.01
    V0 = 0.99

    model = FiniteTemperatureLogisticRegression(sample_complexity=alpha, beta=beta, regularisation=lambda_, Delta=sigma**2)
    model.init_matching()

    model.update_se(V0, q0, m0)

    debut = time()
    for _ in range(10):
        V0, q0, m0 = model.update_se(V0, q0, m0)
    end = time()
    print('Elapsed time w/o compilation: ', end - debut)

x = np.arange(10000).reshape(100, 100)

if __name__ == '__main__':
    test_speed_iteration_pseudo_bayes()