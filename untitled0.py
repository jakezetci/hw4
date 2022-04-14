# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:13:28 2022

@author: cosbo
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mixfit import max_likelihood

def t(x, tau, mu1, mu2, sigma12, sigma22):
    tau0 = tau
    tau1 = 1 - tau0
    
    T0 =(tau0/ np.sqrt(2*np.pi*sigma12))* np.exp(-0.5*(x-mu1)**2/sigma12)
    T1 =(tau1/ np.sqrt(2*np.pi*sigma22))* np.exp(-0.5*(x-mu2)**2/sigma22)
    
    T_sum = T1 +T0
    T0 = np.divide(T0, T_sum, out = np.full_like(T0, 0.5), where=T_sum!=0.0)
    T1 = np.divide(T1, T_sum, out = np.full_like(T1, 0.5), where=T_sum!=0.0)
    return T0, T1

def tht (x, tau, mu, mu2, sigma1, sigma2):
    t0, t1 = t(x, tau, mu1, mu2, sigma12, sigma22)
    t0_sum = np.sum(t0)
    t1_sum = np.sum(t1)
    tau_new = t0_sum / np.shape(x)[0]
    mu1_new = np.sum(t0*x) / t0_sum
    mu2_new = np.sum(t1*x) / t1_sum
    sigma1_new = np.sum(t0*(x-mu1_new)**2)/ t0_sum
    sigma2_new = np.sum(t1*(x-mu2_new)**2)/ t1_sum
    return (tau_new, mu1_new, mu2_new, sigma1_new, sigma2_new)


tau = 0.5
mu1 = 0.1
mu2 = 0.6
sigma12 = 0.08
sigma22 = 0.04
N = 100000
x1 = np.random.normal(mu1, sigma12, int(N*tau))
x2 = np.random.normal(mu2, sigma22, int(N*(1-tau)))
x = np.concatenate([x1, x2])
tau_init = 0.5
mu1_init = np.mean(x) - np.std(x)
mu2_init = np.mean(x) + np.std(x)
sigma12_init = 0.01
sigma22_init = 0.01

r = max_likelihood(x, tau_init, mu1_init, sigma12_init, mu2_init, sigma22_init)
plt.figure()
plt.hist(x, bins = 200)