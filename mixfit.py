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
    def p(x, mu, sigma):  # per one x
        a = np.exp(-(x-mu)**2/(2*sigma**2))
        return a*(np.pi)**(1/2)/sigma

    def log_p(x, mu, sigma):
        return -(1/2)*(np.log(sigma**2)+1/sigma**2 * (x-mu)**2)

    def t(x, tau, mu1, mu2, sigma12, sigma22):
        tau0 = tau
        tau1 = 1 - tau0
        
        T0 =(tau0/ np.sqrt(2*np.pi*sigma12))* np.exp(-0.5*(x-mu1)**2/sigma12)
        T1 =(tau1/ np.sqrt(2*np.pi*sigma22))* np.exp(-0.5*(x-mu2)**2/sigma22)
        
        T_sum = T1 +T0
        T0 = np.divide(T0, T_sum, out = np.full_like(T0, 0.5), where=T_sum!=0.0)
        T1 = np.divide(T1, T_sum, out = np.full_like(T1, 0.5), where=T_sum!=0.0)
        return T0, T1

    def L_to_minimize(th, x):
        tau, mu1, sigma1, mu2, sigma2 = th
        th1 = (tau, mu1, sigma1)
        th2 = (1 - tau, mu2, sigma2)
        T0, T1 = t(x, *th)
        L1 = np.sum(T0 * np.log(tau)) + np.sum(
            T0*log_p(x, mu1, sigma1))
        L2 = np.sum(T0 * np.log(1 - tau)) + np.sum(
            T1 *log_p(x, mu2, sigma2))
        return -L1-L2
    
    def jac_L(th, x):
        tau, mu1, sigma1, mu2, sigma2 = th
        th1 = (tau, mu1, sigma1)
        th2 = (1 - tau, mu2, sigma2)
        T0, T1 = t(x, *th)
        dtau = np.sum(T0 * (1/tau + np.log(tau)/tau)) + np.sum(
            T0 * log_p(x, mu1, sigma1) / tau) - np.sum(
                T1-np.log(1-tau))/(1-tau) - np.sum(
            T1*log_p(x, mu2, sigma2)) / (1-tau)
        dmu1 = np.sum(T0 * 1/sigma1**2 * (x-mu1))
        dsigma1 = np.sum(T0 * (- 1/sigma1 + 1/(sigma1**3) * (x-mu1)**2))
        dmu2 = np.sum(T1 * 1/sigma2**2 * (x - mu2))
        dsigma2 = np.sum(T1 * (- 1/sigma2 + 1/(sigma2**3) * (x-mu2)**2))
        return np.asarray([dtau, dmu1, dsigma1, dmu2, dsigma2])
    th0 = (tau, mu1, sigma1, mu2, sigma2)
    bds = Bounds(0, 1)
    result = minimize(L_to_minimize, th0, args=(x),
                      jac=jac_L, tol=rtol, bounds=bds)
    return result
    

def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    pass


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5):
    pass


if __name__ == "__main__":
    pass
