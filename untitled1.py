

import numpy as np
from scipy.optimize import minimize, Bounds


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    """

    Args:
        x (array): массив данных.
        tau (TYPE): DESCRIPTION.
        mu1 (TYPE): DESCRIPTION.
        sigma1 (TYPE): DESCRIPTION.
        mu2 (TYPE): DESCRIPTION.
        sigma2 (TYPE): DESCRIPTION.
        rtol (TYPE, optional): DESCRIPTION. Defaults to 1e-3.

    Returns:
        None.

    """

    def p(x, mu, sigma):
        a = np.exp(-(x-mu)**2/(sigma**2))
        return a / ((2 * np.pi)**(1/2) * (sigma**2))

    def log_p(x, mu, sigma):
        return -(1/2)*(np.log(sigma**2)+1/sigma**2 * (x-mu)**2)

    def t(x, tau, mu1, sigma1, mu2, sigma22):
        tau0 = tau
        tau1 = 1 - tau0

        T0 = (tau0 / np.sqrt(2 * np.pi * sigma1**2)) * np.exp(
            -0.5 * (x - mu1)**2 / sigma1**2)
        T1 = (tau1 / np.sqrt(2 * np.pi * sigma2**2)) * np.exp(
            -0.5 * (x - mu2)**2 / sigma2**2)

        T_sum = T1 + T0
        T0 = np.divide(T0, T_sum, out=np.full_like(T0, 0.5),
                       where=T_sum != 0.0)
        T1 = np.divide(T1, T_sum, out=np.full_like(T1, 0.5),
                       where=T_sum != 0.0)
        return T0, T1

    def L_to_minimize(th, x):
        tau, mu1, sigma1, mu2, sigma2 = th
        th1 = (mu1, sigma1)
        th2 = (mu2, sigma2)
        T0, T1 = t(x, *th)
        L1 = np.sum(-np.log(T0 * p(x, *th1) + T1 * p(x, *th2)))
        return L1

    def L_jac(th, x):
        pass

    th0 = (tau, mu1, sigma1, mu2, sigma2)
    bds = Bounds(0, 1)
    result = minimize(L_to_minimize, th0, args=(x),
                      tol=rtol, bounds=bds)
    return result.x