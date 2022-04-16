#!/usr/bin/env python3
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

    N = x.shape[0]
    th0 = (tau, mu1, sigma1, mu2, sigma2)
    bds = Bounds(0, 1)
    result = minimize(L_to_minimize, th0, args=(x),
                      tol=rtol, bounds=bds)
    return result.x


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    def t(x, tau, mu1, mu2, sigma12, sigma22):
        tau0 = tau
        tau1 = 1 - tau0

        T0 = (tau0 / np.sqrt(2 * np.pi * sigma12)) * np.exp(
            -0.5 * (x-mu1)**2/sigma12)
        T1 = (tau1 / np.sqrt(2 * np.pi * sigma22)) * np.exp(
            -0.5 * (x-mu2)**2/sigma22)

        T_sum = T1 + T0
        T0 = np.divide(T0, T_sum, out=np.full_like(T0, 0.5),
                       where=T_sum != 0.0)
        T1 = np.divide(T1, T_sum, out=np.full_like(T1, 0.5),
                       where=T_sum != 0.0)
        return T0, T1

    def tht(x, tau, mu, mu2, sigma1, sigma2):
        t0, t1 = t(x, tau, mu1, mu2, sigma1, sigma2)
        t0_sum = np.sum(t0)
        t1_sum = np.sum(t1)
        tau_new = t0_sum / np.shape(x)[0]
        mu1_new = np.sum(t0*x) / t0_sum
        mu2_new = np.sum(t1*x) / t1_sum
        sigma1_new = np.sum(t0 * (x-mu1_new)**2) / t0_sum
        sigma2_new = np.sum(t1 * (x-mu2_new)**2) / t1_sum
        return (tau_new, mu1_new, mu2_new, sigma1_new, sigma2_new)

    th = (tau, mu1, mu2, sigma1, sigma2)
    theta = tht(x, *th)
    for i in range(int(1e3)):
        th_new = tht(x, *th)
        if np.linalg.norm(np.asarray(th_new)-np.asarray(th)) < rtol:
            break
        theta = th_new
    return theta


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02,
                      sigmax2, sigmav2, rtol=1e-5):
    


if __name__ == "__main__":
    pass
