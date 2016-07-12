import numpy as np

"""
Find coefficient vector (beta in R^{Mx1}) that better fits data vector (y in R^{Nx1}) to dictionary matrix (Phi in R^{NxM}) 
    y approx np.dot(Phi, beta)
Objective function is the Weighted Maximum Correntropy Criterion (WMCC).
Returns beta, cost function evolution vector, and kernel size evolution vector
"""

def find_beta_WMCC(y, Phi, dy, max_inner_iterations=1, max_outer_iterations=100, debug=False):
    N, M = Phi.shape
    beta = np.zeros(shape=(M,))
    dy2 = np.power(dy, 2.0)
    # Initial kernel size:
    iqr_y = np.percentile(y, 75) - np.percentile(y, 25)
    ks_silverman = 1.0592*np.minimum(np.std(y), iqr_y/1.34)*np.power(N, -0.2)
    # Optimize log(ks) instead of ks (positive constraint)
    log_ks = np.log(10.0*ks_silverman) 
    cost_history = np.zeros(shape=(max_outer_iterations,))
    kernel_size_history = np.zeros(shape=(max_outer_iterations,))
    # ADADELTA parameters
    rho = 0.9
    eps = 1e-6 
    ks_grad2_hist = 0.0
    ks_dx2_hist = 0.01 #This is relevant for convergence time
    stopping_tol = 1.01
    # Main routine
    kernel_size_history[0] = np.exp(log_ks)
    ks2 = dy2 + np.exp(2*log_ks)
    error = y - np.dot(Phi, beta)
    for i in range(0, max_outer_iterations):
        # Fixed point routine to update beta
        for j in range(0, max_inner_iterations):
            cost = np.divide(np.exp(-0.5*np.divide(np.power(error, 2.0), ks2)), np.sqrt(ks2))
            cost_history[i] = np.sum(cost)/(N*np.sqrt(2.0*np.pi))
            if i > 0 and cost_history[i] <= cost_history[i-1]*stopping_tol:
                break
            W_mat = np.diag(np.divide(cost_history[i], ks2))
            beta = np.linalg.solve(np.dot(np.dot(Phi.T, W_mat), Phi), np.dot(Phi.T, np.dot(W_mat, y)))
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
            cost = np.divide(np.exp(-0.5*np.divide(e2, ks2)), np.sqrt(ks2))
            cost_history[i] = np.sum(cost)/(N*np.sqrt(2.0*np.pi))
            if i > 0 and cost_history[i] <= cost_history[i-1]*stopping_tol:
                break
            W_mat = np.diag(np.divide(cost_history[i], ks2))
            grad = np.sum(np.dot(W_mat, np.divide(e2, ks2) - 1.0))*np.exp(2*log_ks)*2/(N*np.sqrt(2.0*np.pi))
            ks_grad2_hist = rho*ks_grad2_hist + (1.0-rho)*grad**2
            RMSgt2 = np.sqrt(ks_grad2_hist + eps)
            RMSdx = np.sqrt(ks_dx2_hist + eps)
            dlog_ks = (RMSdx/RMSgt2)*grad
            ks_dx2_hist = rho*ks_dx2_hist + (1.0-rho)*dlog_ks**2
            log_ks += dlog_ks
            kernel_size_history[i] = np.exp(log_ks)
            ks2 = dy2 + np.exp(2*log_ks)
            
    return beta, cost_history[:i], kernel_size_history[:i]
    
"""
Ordinary least squares (OLS) regression for benchmark purposes
"""
def find_beta_OLS(y, Phi):
    R = np.dot(Phi.T, Phi)
    P = np.dot(Phi.T, y)
    beta = np.linalg.solve(R, P)
    return beta, np.dot(P.T, beta)

"""
Weighted least squares (WLS) regression for benchmark purposes
"""
def find_beta_WLS(y, Phi, dy):
    dy2 = np.power(dy, -2.0)
    W_mat = np.diag(dy2)
    R = np.dot(np.dot(Phi.T, W_mat), Phi)
    P = np.dot(np.dot(Phi.T, W_mat), y)
    beta = np.linalg.solve(R, P)
    return beta, np.dot(P.T, beta)
