"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n = X.shape[0]
    d = X.shape[1]
    k = mixture.p.shape[0]
    LL = 0
    post = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            post[i,j] = mixture.p[j]* 1/np.power((2*np.pi*mixture.var[j]),d/2)*np.exp(-1/(2*mixture.var[j])*np.power(np.linalg.norm(X[i]-mixture.mu[j]),2))
            sum = 0
            for n_gaussian in range(k):
                sum += mixture.p[n_gaussian] * 1/np.power((2*np.pi*mixture.var[n_gaussian]),d/2)*np.exp(-1/(2*mixture.var[n_gaussian])*np.power(np.linalg.norm(X[i]-mixture.mu[n_gaussian]),2))
                
            post[i,j] /= sum
        LL += np.log(sum)
    return post, LL


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    K = post.shape[1]
    n = X.shape[0]
    d = X.shape[1]
    n_hat = np.zeros(K)
    p_hat = np.zeros(K)
    mu_hat = np.zeros((K,d))
    sigma_hat = np.zeros(K)
    for j in range(K):
        for i in range(n):
            n_hat[j] += post[i,j]
        p_hat[j] = n_hat[j]/n
        sum_px = 0
        for k in range(n):
            sum_px += post[k,j] * X[k]
        mu_hat[j] = 1/n_hat[j] * sum_px
        sum_px = 0
        for k in range(n):
            sum_px += post[k,j] * np.power(np.linalg.norm(X[k] - mu_hat[j]),2)
        sigma_hat[j] = 1/(n_hat[j]*d) * sum_px
    return GaussianMixture(mu_hat, sigma_hat, p_hat)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    new_ll = None
    old_ll = None
    
    while new_ll == None or old_ll == None:
        old_ll = new_ll
        post, new_ll = estep(X, new_mixture)
        new_mixture = mstep(X, post)
    
    while new_ll - old_ll > 1e-6 * abs(new_ll):
        old_ll = new_ll
        post, new_ll = estep(X, new_mixture)
        new_mixture = mstep(X, post)

        
    return mixture, post, new_ll
    
