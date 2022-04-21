import numpy as np
from mixfit import max_likelihood, em_double_gauss, em_double_cluster
from scipy.stats import multivariate_normal
import unittest
from scipy.stats import uniform

class MixFitTest(unittest.TestCase):
    def test_normal(self):
        tau = 0.2
        mu1 = 0.3
        mu2 = 0.6
        sigma1 = 0.02
        sigma2 = 0.02
        N = 100000
        x1 = np.random.normal(mu1, sigma1, int(N*tau))
        x2 = np.random.normal(mu2, sigma2, int(N*(1-tau)))
        x = np.concatenate([x1, x2])
        tau_init = 0.2
        mu1_init = np.mean(x) - np.std(x)
        mu2_init = np.mean(x) + np.std(x)
        sigma1_init = 0.04
        sigma2_init = 0.04
        theta0 = (tau_init, mu1_init,
                 sigma1_init, mu2_init, sigma2_init)
        theta = (tau, mu1, sigma1, mu2, sigma2)
        return x, theta0, theta
    
    def test_4d(self):
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
        return x, theta, theta0

    def test_likelihood(self):
        x, theta0, theta = self.test_normal()
        result = max_likelihood(x, *theta0)
        self.assertTrue(np.allclose(result, theta, atol=0.1))

    def test_em_double_gauss(self):
        x, theta0, theta = self.test_normal()
        result = em_double_gauss(x, *theta0)
        
        self.assertTrue(np.allclose(result, theta, atol=0.01))

    def test_em_double_cluster(self):
        x, theta, theta0 = self.test_4d()
        result = em_double_cluster(x, *theta0)
        print(result)
        print(theta)
        for i, param in enumerate(result):
            self.assertTrue(np.allclose(param, theta[i], atol=1e-2))

if __name__ == '__main__':
    unittest.main()
