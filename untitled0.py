# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:13:28 2022

@author: cosbo
"""
from scipy.stats import multivariate_normal, uniform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from mixfit import max_likelihood, em_double_gauss, em_double_cluster


tau = 0.2
mu1 = 0.3
mu2 = 0.6
sigma12 = 0.02
sigma22 = 0.02
N = 100000
x1 = np.random.normal(mu1, sigma12, int(N*tau))
x2 = np.random.normal(mu2, sigma22, int(N*(1-tau)))
x = np.concatenate([x1, x2])
tau_init = 0.2
mu1_init = np.mean(x) - np.std(x)
mu2_init = np.mean(x) + np.std(x)
sigma12_init = 0.04
sigma22_init = 0.04
#theta = (tau, mu1, sigma12, mu2, sigma22)
#theta0 = (tau_init, mu1_init,
#                 sigma12_init, mu2_init, sigma22_init)
#result = max_likelihood(x, tau_init, mu1_init, sigma12_init, mu2_init, sigma22_init)
#print(assert_allclose(result, theta0, atol=0.01))
#k = em_double_gauss(x, tau_init, mu1_init, sigma12_init, mu2_init, sigma22_init)
tau1 = 0.2
tau2 = 0.15
tau3 = 1 - tau1 - tau2
mu1 = [35.53, 57.15]
mu2 = [35.22, 57.14]
muv = [-2.7, 5.93]
sigma02 = [0, 0, 13.024, 13.0244]
sigmav2 = [269.7, 269.7]
sigmax2 = [0.088, 0.088]
theta = (tau1, tau2, tau3, muv, mu1, mu2, sigma02, sigmax2, sigmav2)
theta0 = (0.2, 0.3, [-10, -2], [34.85, 57.17], [35.3, 57.17],
         [0, 0, 0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
N = 5000
x1 = multivariate_normal(mean=np.asarray([mu1, muv]).flat,
                         cov=np.asarray([sigmax2, sigmav2]).flat).rvs(size=int(N*tau1))
x2 = multivariate_normal(mean=np.asarray([mu2, muv]).flat,
                         cov=np.asarray([sigmax2, sigmav2]).flat).rvs(size=int(N*tau2))
x3 = uniform.rvs(loc=35.2, scale=0.5, size=int(N*tau3))
x4 = uniform.rvs(loc=57.1, scale=0.5, size=int(N*tau3))
v3 = multivariate_normal(mean=0, cov=sigma02[2]).rvs(size=(int(N*tau3)))
v4 = multivariate_normal(mean=0, cov=sigma02[3]).rvs(size=(int(N*tau3)))
x3 = np.vstack([x3, x4, v3, v4]).T
x = np.concatenate([x1, x2, x3])
result = em_double_cluster(x, *theta0)
result = np.asarray(result).flat