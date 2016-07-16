from __future__ import division, print_function
import numpy as np


def find_beta_WMCC(y, Phi, dy, max_inner_iterations=1, max_outer_iterations=100, stopping_tol=1.01, debug=False, full_output=False):
    """
    Find coefficient vector (beta in R^{Mx1}) that better fits data 
    vector (y in R^{Nx1}) to dictionary matrix (Phi in R^{NxM}) 
        y approx np.dot(Phi, beta)
    Objective function can be the Weighted Maximum Correntropy Criterion 
    (WMCC), Weighted Least Squares (WLS) or Ordinary Least Squares (OLS)
    
    max_inner_iterations -- Number of iterations for fixed point and 
    kernel size adaption routines
    max_outer_iterations -- Number of iterations for the whole routine
    stopping_tol -- Stopping criteria check for convergence
    
    Returns beta, cost function evolution vector, and kernel size 
    evolution vector. The latter two are only relevant for the WMCC
    
    TODO: Study kernel size normalizations for WMCC
    """

    N, M = Phi.shape
    beta = np.zeros(shape=(M,))
    dy2 = np.power(dy, 2.0)
    # Initial kernel size:
    iqr_y = np.percentile(y, 75) - np.percentile(y, 25)
    ks_silverman = 1.0592*np.minimum(np.std(y), iqr_y/1.34)*np.power(N, -0.2)
    # Optimize log(ks) instead of ks (positive constraint)
    log_ks = np.log(10.0*ks_silverman)  # Very conservative initial value
    cost_history = np.zeros(shape=(max_outer_iterations,))
    kernel_size_history = np.zeros(shape=(max_outer_iterations,))
    # ADADELTA parameters
    rho = 0.9
    eps = 1e-6 
    ks_grad2_hist = 0.0
    ks_dx2_hist = 0.01 # default should be zero but >0 improves convergence speed
    # Main routine
    kernel_size_history[0] = np.exp(log_ks)
    ks2 = dy2 + np.exp(2*log_ks)
    error = y - np.dot(Phi, beta)
    PhiT = np.transpose(Phi)
    for i in range(0, max_outer_iterations):
        # Fixed point routine to update beta
        ks = np.sqrt(ks2)
        for j in range(0, max_inner_iterations):
            cost = np.sum(np.divide(np.exp(-0.5*np.divide(np.power(error, 2.0), ks2)), ks))
            cost_history[i] = cost/(N*np.sqrt(2.0*np.pi))
            #cost_history[i] = 1.0 - cost/np.sum(1.0/ks)
            if i > 0 and cost_history[i] <= cost_history[i-1]*stopping_tol:
                break
            W = np.divide(cost, ks2)  # diagonal matrix
            beta = np.linalg.solve(np.dot(PhiT*W, Phi), np.dot(PhiT*W, y))
            """
            TODO: Inversion could be not feasible, check conditioning, add regularization...
            """
            error = y - np.dot(Phi, beta)
            #grad = np.dot(A.T, np.dot(W, e))/(N*np.sqrt(2.0*np.pi))
            #beta += lr*grad
        if i > 0 and cost_history[i] <= cost_history[i-1]*stopping_tol:  # More sophisticated stopping criteria could be tried
            if debug:
                print("Breaking at iteration %d" % i)
            break    
        # Gradient descent with ADADELTA to update the kernel size
        e2 = np.power(error, 2.0)
        for j in range(0, max_inner_iterations):
            ks = np.sqrt(ks2)
            cost = np.sum(np.divide(np.exp(-0.5*np.divide(e2, ks2)), ks))
            cost_history[i] = cost/(N*np.sqrt(2.0*np.pi))
            #den = np.sum(1.0/ks)
            #cost_history[i] = 1.0 - cost/den 
            if i > 0 and cost_history[i] <= cost_history[i-1]*stopping_tol:
                break
            W = np.divide(cost, ks2)
            grad = np.sum(np.multiply(W, np.divide(e2, ks2) - 1.0))/(N*np.sqrt(2.0*np.pi))
            #ks3 = np.sum(1.0/(ks2*ks))
            #grad = np.sum(np.multiply(W, np.divide(e2, ks2) - 1.0))/den + cost*ks3/den**2
            grad = grad*np.exp(2.0*log_ks)
            ks_grad2_hist = rho*ks_grad2_hist + (1.0-rho)*grad**2
            RMSgt2 = np.sqrt(ks_grad2_hist + eps)
            RMSdx = np.sqrt(ks_dx2_hist + eps)
            dlog_ks = (RMSdx/RMSgt2)*grad
            ks_dx2_hist = rho*ks_dx2_hist + (1.0-rho)*dlog_ks**2
            log_ks += dlog_ks
            kernel_size_history[i] = np.exp(log_ks)
            ks2 = dy2 + np.exp(2*log_ks)
    if full_output:
        return beta, cost_history[:i], kernel_size_history[:i]
    else:
        return beta, cost_history[:i][-1]
    

def find_beta_OLS(y, Phi, dy=None):
    """
    Ordinary least squares (OLS) regression for benchmark purposes
    """
    R = np.dot(Phi.T, Phi)
    P = np.dot(Phi.T, y)
    beta = np.linalg.solve(R, P)
    return beta, np.dot(P.T, beta)/(np.var(y)*len(y))


def find_beta_WLS(y, Phi, dy):
    """
    Weighted least squares (WLS) regression for benchmark purposes
    """
    W = np.power(dy, -2.0)
    #W_mat = np.diag(dy2)
    R = np.dot(Phi.T*W, Phi)
    P = np.dot(Phi.T*W, y)
    beta = np.linalg.solve(R, P)
    return beta, np.dot(P.T, beta)/np.sum(np.multiply(np.power(y, 2.0), W))
    

