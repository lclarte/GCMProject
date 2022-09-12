from GCMProject.state_evolution.auxiliary.utility import PseudoBayesianDataModel
from numba import jit, config
import numpy as np
import matplotlib.pyplot as plt
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

def plot_pseudo_bayes():
    # arbitrary parameters
    m = q = 0.1
    V = 0.01
    y = 1

    xis = [-1, 0, 1]

    zs = np.linspace(-5.0, 5.0)
    for xi in xis:
        fs = [ PseudoBayesianDataModel.Z0_quad_argument(z, y, np.sqrt(q) * xi, np.sqrt(V), beta = 10.0) for z in zs]
        plt.plot(zs, fs)
        plt.show()


x = np.arange(10000).reshape(100, 100)

if __name__ == '__main__':
    # test_speed_iteration_pseudo_bayes()
    plot_pseudo_bayes()