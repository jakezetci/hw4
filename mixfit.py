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

    def tht(x, tau, mu1, mu2, sigma1, sigma2):
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

    def p_normal(x, mu, sigma):
        x = np.atleast_1d(x)
        mu = np.atleast_1d(mu)
        n = x.shape[0]
        mu = np.broadcast_to(mu, (n, 4))
        sigmavec = [1/a for a in np.diag(sigma)]
        exp = np.exp(-1/2 * np.sum((x-mu) * ((x-mu) @ np.diag(sigmavec)),
                                   axis=1))
        return exp * (np.prod(
            sigmavec, where=np.nonzero(sigmavec))**1/2) / ((2 * np.pi)**(1/4))

    def p_normal_2d(x, mu, sigma):
        x = np.atleast_2d(x)
        mu = np.atleast_2d(mu)
        n = np.shape(x)[0]
        mu = np.broadcast_to(mu, (n, 2))
        sigmavec = [1/a for a in np.diag(sigma)]
        exp = np.exp(-1/2 * np.sum((x-mu) * ((x-mu) @ np.diag(sigmavec)),
                                   axis=1))
        return exp * (np.prod(
            sigmavec, where=np.nonzero(sigmavec))**1/2) / ((2 * np.pi)**(1/2))

    def T(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2):
        sigma = np.diag(np.array([sigmax2, sigmav2]).flat)
        sigma02 = np.diag(np.array(sigma02[2:4]).flat)
        tau3 = 1 - tau1 - tau2
        mu1vec = np.asarray([mu1, muv]).flatten()
        mu2vec = np.asarray([mu2, muv]).flatten()
        T1 = tau1 * p_normal(x, mu1vec, sigma)
        T2 = tau2 * p_normal(x, mu2vec, sigma)
        T3 = tau3 * p_normal_2d(x[:, 2:4], [0, 0], sigma02)
        T_sum = T1 + T2 + T3
        T1 = np.divide(T1, T_sum, out=np.full_like(T1, 0.33),
                       where=T_sum != 0.0)
        T2 = np.divide(T2, T_sum, out=np.full_like(T2, 0.33),
                       where=T_sum != 0.0)
        T3 = np.divide(T3, T_sum, out=np.full_like(T3, 0.33),
                       where=T_sum != 0.0)
        return T1, T2, T3

    def stdsum(y, mu):
        return (y[:, 0] - mu[:, 0])**2 + (y[:, 1] - mu[:, 1])**2

    def step(x, theta):
        N = x.shape[0]
        T1, T2, T3 = T(x, *theta)
        tau1 = np.sum(T1) / N
        tau2 = np.sum(T2) / N
        T1_2d = np.broadcast_to(T1, (2, N)).T
        T2_2d = np.broadcast_to(T2, (2, N)).T
        mu1 = np.sum(T1_2d * x[:, :2], axis=0) / (N * tau1)
        mu2 = np.sum(T2_2d * x[:, :2], axis=0) / (N * tau2)
        muv = np.sum(T1_2d * x[:, 2:4] +
                     T2_2d * x[:, 2:4], axis=0) / (N * (tau1+tau2))
        mu1_to_N = np.broadcast_to(mu1, (N, 2))
        mu2_to_N = np.broadcast_to(mu2, (N, 2))
        muv_to_N = np.broadcast_to(muv, (N, 2))
        sigmax2 = np.sum(T1 * stdsum(x, mu1_to_N) +
                         T2 * stdsum(x, mu2_to_N)) / ((np.sum(T1 + T2))*2)
        v = x[:, 2:4]
        sigmav2 = np.sum(T1 * stdsum(v, muv_to_N) +
                         T2 * stdsum(v, muv_to_N)) / ((
                             tau1+tau2)*N*2)
        sigma02 = np.sum(T3 * (v[:, 1]**2 + v[:, 0]**2)) / (2 * 2 * np.sum(T3))

        return (tau1, tau2, muv, mu1, mu2, [0.0, 0.0, sigma02, sigma02],
                [sigmax2, sigmax2], [sigmav2,  sigmav2])

    def L(x, theta):
        def Lj(x, tau, mu, sigma):
            n = x.shape[0]
            m = x.shape[1]
            mu = np.broadcast_to(mu, (n, m))
            sigmavec = [1/a for a in np.diag(sigma)]
            L1 = np.log(tau) - 0.5*np.log(
                np.prod(sigmavec, where=np.nonzero(sigmavec))) - 1/2 * np.sum((x-mu) *
                                               ((x-mu) @ np.diag(sigmavec)),
                                               axis=1)
            return L1
        T1, T2, T3 = T(x, *theta)
        tau1, tau2, muv, mu1, mu2, sigma0, sigmax, sigmav = theta
        sigma = np.diag(np.array([sigmax, sigmav]).flat)
        sigma02 = np.diag(np.array(sigma0[2:4]).flat)
        L = np.sum(T1 * Lj(x, tau1, np.asarray([mu1, mu2]).flatten(), np.asarray(sigma)) +
                   T2 * Lj(x, tau2, np.asarray([mu2, muv]).flatten(), np.asarray(sigma)) +
                   T3 * Lj(x[:, 2:4], (1-tau1-tau2), [0,0], np.asarray(sigma02)))
        return L

    theta0 = (tau1, tau2, muv, mu1, mu2, sigma02,
              sigmax2, sigmav2)
    th = step(x, theta0)
    L_old = L(x, theta0)
    for i in range(1000):
        th_new = step(x, theta0)
        L_new = L(x, th_new)
        if abs(L_new - L_old) < abs(rtol*L_old):
            break
        th = th_new
        L_old = L_new
    return th

def T(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2):
    def p_normal(x, mu, sigma):
        x = np.atleast_1d(x)
        mu = np.atleast_1d(mu)
        n = x.shape[0]
        mu = np.broadcast_to(mu, (n, 4))
        sigmavec = [1/a for a in np.diag(sigma)]
        exp = np.exp(-1/2 * np.sum((x-mu) * ((x-mu) @ np.diag(sigmavec)),
                                   axis=1))
        return exp * (np.prod(sigmavec, where=np.nonzero(sigmavec))**1/2)/ ((2 * np.pi)**(1/4))

    def p_normal_2d(x, mu, sigma):
        x = np.atleast_1d(x)
        mu = np.atleast_1d(mu)
        n = x.shape[0]
        mu = np.broadcast_to(mu, (n, 2))
        sigmavec = [1/a for a in np.diag(sigma)]
        exp = np.exp(-1/2 * np.sum((x-mu) * ((x-mu) @ np.diag(sigmavec)),
                                   axis=1))
        return exp * (np.prod(sigmavec, where=np.nonzero(sigmavec))**1/2) / ((2 * np.pi)**(1/2))

    sigma = np.diag(np.array([sigmax2, sigmav2]).flat)
    sigma02 = np.diag(np.array(sigma02[2:4]).flat)
    tau3 = 1 - tau1 - tau2
    mu1vec = np.asarray([mu1, muv]).flatten()
    mu2vec = np.asarray([mu2, muv]).flatten()
    T1 = tau1 * p_normal(x, mu1vec, sigma)
    T2 = tau2 * p_normal(x, mu2vec, sigma)
    T3 = tau3 * p_normal_2d(x[:, 2:4], [0, 0], sigma02)
    T_sum = T1 + T2 + T3
    T1 = np.divide(T1, T_sum, out=np.full_like(T1, 0.3),
                   where=T_sum != 0.0)
    T2 = np.divide(T2, T_sum, out=np.full_like(T2, 0.3),
                   where=T_sum != 0.0)
    T3 = np.divide(T3, T_sum, out=np.full_like(T3, 0.3),
                   where=T_sum != 0.0)
    return T1, T2, T3

if __name__ == "__main__":
    pass
