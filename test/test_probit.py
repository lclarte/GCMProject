# NOTE : Used for debugging the probit regression script

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

from state_evolution.auxiliary import probit_integrals, logistic_integrals
from state_evolution.models import probit_regression, logistic_regression
from state_evolution.data_models.custom import Custom

def compare_q_hat_probit_logistic():
    alpha   = 10.0
    sigma   = 0.0
    lambda_ = 0.1
    d = 512
    Omega = Psi = Phi = np.eye(d)

    theta = np.random.normal(0, 1, size = d)

    data_model = Custom(teacher_teacher_cov = Psi,
                    student_student_cov = Omega,
                    teacher_student_cov = Phi,
                    teacher_weights     = theta)

    pr_task = probit_regression.ProbitRegression(sample_complexity = alpha,
                                            regularisation    = lambda_,
                                            data_model        = data_model,
                                            Delta             = sigma**2)

    lr_task = logistic_regression.LogisticRegression(sample_complexity = alpha,
                                            regularisation    = lambda_,
                                            data_model        = data_model,
                                            Delta             = sigma**2)

    V, q, m = 100, 0.01, 0.01

    for step in range(20):
        print(f'Step = {step}, V, q, m = {V, q, m}')        

        Vhat_pr, qhat_pr, mhat_pr = pr_task._update_hatoverlaps(V, q, m)
        Vhat_lr, qhat_lr, mhat_lr = lr_task._update_hatoverlaps(V, q, m)

        print(f'For probit, hats V, q, m = {Vhat_pr, qhat_pr, mhat_pr}')
        print(f'For logistic, hats V, q, m = {Vhat_lr, qhat_lr, mhat_lr}')
        input()

        V, q, m = lr_task._update_overlaps(Vhat_lr, qhat_lr, mhat_lr)
        V_pr, q_pr, m_pr = pr_task._update_overlaps(Vhat_lr, qhat_lr, mhat_lr)

        print(f'For probit,  V, q, m = {V, q, m}')
        print(f'For logistic,V, q, m = {V_pr, q_pr, m_pr}')
        input()

        print('===========')
        
def compare_proximal():
    w_list, V_list = np.linspace(-10.0, 10.0, 100), np.linspace(0.1, 20.0, 100)
    diffs = np.zeros((100, 100))   
    prs = np.zeros((100, 100))   
    lrs = np.zeros((100, 100))   
    
    
    for i in range(100):
        for j in range(100):
            w, V = w_list[i], V_list[j]

            pr = minimize_scalar(lambda x: probit_integrals.moreau_loss(x, 1, w, V))['x']
            lr = minimize_scalar(lambda x: logistic_integrals.moreau_loss(x, 1, w, V))['x']

            # relative difference of the two
            diffs[i, j] = np.abs((pr - lr) / lr)
            prs[i, j] = pr
            lrs[i, j] = lr

    plt.imshow(diffs, origin='lower', extent=[0.1, 20.0, -10.0, 10.0])
    plt.colorbar()
    plt.xlabel('V')
    plt.ylabel('w')
    plt.show()

    plt.title('Proximal value for probit loss')
    plt.imshow(prs, origin='lower', extent=[0.1, 20.0, -10.0, 10.0])
    plt.colorbar()
    plt.xlabel('V')
    plt.ylabel('w')
    plt.show()

    plt.title('Proximal value for logistic loss')
    plt.imshow(lrs, origin='lower', extent=[0.1, 20.0, -10.0, 10.0])
    plt.colorbar()
    plt.xlabel('V')
    plt.ylabel('w')
    plt.show()
    return diffs, prs, lrs 
    
def compare_proximal_2():
    w_list = np.linspace(-100.0, 100.0, 1000)
    V = 3

    prs = np.zeros((1000, ))   
    lrs = np.zeros((1000, ))   
    
    for i in range(1000):
            w = w_list[i]

            # pr = minimize_scalar(lambda x: probit_integrals.moreau_loss(x, 1, w, V))['x']
            pr = probit_integrals.proximal(-1, w, V)
            lr = minimize_scalar(lambda x: logistic_integrals.moreau_loss(x, -1, w, V))['x']

            prs[i] = pr
            lrs[i] = lr
    plt.plot(w_list, prs, label='probit')
    plt.plot(w_list, lrs, label='logistic')
    
    plt.xlabel('omega')
    plt.legend()
    plt.show()

def plot_second_derivative():
    x_list = np.linspace(-100.0, 100.0, 1000)
    prs = np.zeros(1000)
    prs_1 = np.zeros(1000)
    w, V = 0.5, 100

    for i in range(1000):
        x = x_list[i]

        pr = probit_integrals.moreau_second(x, -1, w, V)
        prs[i] = pr

        prs_1[i] = probit_integrals.moreau_prime(x, -1, w, V)
    

    plt.plot(x_list, prs_1)
    plt.show()

    plt.plot(x_list, prs)
    plt.show()

compare_proximal_2()