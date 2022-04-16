# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:13:28 2022

@author: cosbo
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mixfit import max_likelihood, em_double_gauss


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

r = max_likelihood(x, tau_init, mu1_init, sigma12_init, mu2_init, sigma22_init)
k = em_double_gauss(x, tau_init, mu1_init, sigma12_init, mu2_init, sigma22_init)
plt.figure()
plt.hist(x, bins = 200)