from __future__ import division, print_function
import numpy as np


def find_beta_WMCC(mag, Phi, err, max_iter=10, stopping_tol=1.01, full_output=False):
    N, M = Phi.shape
    PhiT = np.transpose(Phi)
    #beta = np.zeros(shape=(M,))
    beta, _ = find_beta_WLS(mag, Phi, err)
    w = 1.0/np.power(err, 2.0)
    w = w/np.sum(w)
    #e = mag.copy()
    e = mag - np.dot(Phi, beta)
    w_mean = np.average(e, weights=w)
    w_var = np.average(np.power(e - w_mean, 2.0), weights=w)
    s = 0.9*np.amin([np.sqrt(w_var), (np.percentile(e, 75) - np.percentile(e, 25))/1.34])*N**(-0.2)
    cost_history = np.zeros(shape=(max_iter,))
    kernel_size_history = np.zeros(shape=(max_iter,))
    early_stop = 0
    max_cost = 0.0
    best_beta = np.zeros(shape=beta.shape)
    for i in range(0, max_iter):
        Ge = np.exp(-0.5*np.power(e, 2.0)/s**2)
        W = np.multiply(w, Ge)
        cost_history[i] = np.mean(W/(s*np.sqrt(2.0*np.pi)))
        beta = np.linalg.solve(np.dot(PhiT*W, Phi), np.dot(PhiT*W, mag))
        if max_cost < cost_history[i]:
            max_cost = cost_history[i]
            best_beta = beta.copy()
        e = mag - np.dot(Phi, beta)
        w_mean = np.average(e, weights=w)
        w_var = np.average(np.power(e - w_mean, 2.0), weights=w)
        s = 0.9*np.amin([np.sqrt(w_var), (np.percentile(e, 75) - np.percentile(e, 25))/1.34])*N**(-0.2)
        kernel_size_history[i] = s
        if i > 1:
            if cost_history[i] < cost_history[i-1]*stopping_tol:
                early_stop += 1
            else:
                early_stop = 0
        if early_stop > 2:
            break
    if full_output:
        return best_beta, cost_history[:i], kernel_size_history[:i]
    else:
        return best_beta, cost_history[:i][-1]



def find_beta_WMEE(mag, Phi, err, max_iter=10, stopping_tol=1.01, batch_size=None, full_output=False):
    N, M = Phi.shape
    beta_wls, _ = find_beta_WLS(mag, Phi, err)
    w = 1.0/err**2
    w = w/np.sum(w)
    e = mag - np.dot(Phi, beta_wls) 
    w_mean = np.average(e, weights=w)
    w_var = np.average((e - w_mean)**2, weights=w)
    s = 0.9*np.amin([np.sqrt(w_var), (np.percentile(e, 75) - np.percentile(e, 25))/1.34])*N**(-0.2)
    early_stop = 0
    cost_history = np.zeros(shape=(max_iter,))
    kernel_size_history = np.zeros(shape=(max_iter,))
    max_cost = 0.0
    beta_best = np.zeros(shape=(M,))
    beta_best[0] = beta_wls[0]
    Phi = Phi[:, 1:] #discard the constant as MEE cannot fit it
    beta = beta_wls[1:]
    if batch_size == None:
        batch_size = N
    elif batch_size > N:
        batch_size = N
    for i in range(0, max_iter):
        rand_perm = np.random.permutation(N)
        magp = mag[rand_perm]
        wp = w[rand_perm]
        Phip = Phi[rand_perm, :]
        average_cost = 0.0
        beta_acum = np.zeros(shape=(M-1,))
        for NN, (start, end) in enumerate(zip(range(0, len(magp), batch_size), range(batch_size, len(magp)+batch_size, batch_size))):
            if end > N:
                end = N
            e = magp[start:end] - np.dot(Phip[start:end, :], beta) - beta_wls[0]
            de = np.tile(e, (e.shape[0], 1))
            dmag = np.tile(magp[start:end], (e.shape[0], 1))
            dw = np.tile(wp[start:end], (e.shape[0], 1))
            de = (de - np.transpose(de, axes=(1, 0))).ravel()
            dmag = (dmag - np.transpose(dmag, axes=(1, 0))).ravel()
            dw = np.multiply(dw, np.transpose(dw, axes=(1, 0))).ravel()
            dPhi = np.zeros(shape=(len(de), M-1))
            for k in range(M-1):
                dPhitmp = np.tile(Phip[start:end, k], (e.shape[0], 1))
                dPhi[:, k] = (dPhitmp - np.transpose(dPhitmp, axes=(1, 0))).ravel()
            Gde = np.multiply(dw, np.exp(-0.5*de**2/s**2))
            average_cost += np.mean(Gde/(s*np.sqrt(2.0*np.pi)))
            beta_acum += np.linalg.solve(np.dot(dPhi.T*Gde, dPhi), np.dot(dPhi.T*Gde, dmag))
        beta = beta_acum/(NN+1)
        cost_history[i] = average_cost/(NN+1)
        if max_cost < cost_history[i]:
            max_cost = cost_history[i]
            beta_best[1:] = beta
        e = mag - np.dot(Phi, beta) - beta_wls[0]
        w_mean = np.average(e, weights=w)
        w_var = np.average((e - w_mean)**2, weights=w)
        s = 0.9*np.amin([np.sqrt(w_var), (np.percentile(e, 75)-np.percentile(e, 25))/1.34])*N**(-0.2)
        if i > 1:
            if cost_history[i] < cost_history[i-1]*stopping_tol:
                early_stop += 1
            else:
                early_stop = 0
        if early_stop > 2:
            break
    if full_output:
        return beta_best, cost_history[:i], kernel_size_history[:i]
    else:
        return beta_best, cost_history[:i][-1]
            

def find_beta_WMCC_adaks(y, Phi, s, max_inner_iterations=1, max_outer_iterations=100, stopping_tol=0.999, full_output=False):
    """
    Finds coefficient vector (beta in R^{Mx1}) that better fits data 
    vector (y in R^{Nx1}) with uncertainties (s in R^{Nx1}) to dictionary 
    matrix (Phi in R^{NxM}) 

        y approx np.dot(Phi, beta)

    WMCC adapts the kernel size to take into account the heteroscedasticity 
    of the error (its properties change in time). In order to do that
    we set an individual kernel size for each sample as

        ksi**2 = ks**2 + alpha*si, i=1,...,N

    where si is the uncertainty associated to yi, ks>0 is the global kernel
    size and alpha>0 is a parameter that tunes the importance of si versus
    ks. alpha and ks are adapted using gradient descent with adadelta

    
    Parameters
    ----------
    max_inner_iterations: positive integer
        Number of iterations for fixed point and kernel size adaption 
        routines
    max_outer_iterations: positive integer
        Number of iterations for the main routine
    stopping_tol: float
        Stopping criteria check for convergence
    full_output: bool
        Whether to return the history of the cost function and kernel size
    
    Returns
    -------
    beta: ndarray
        Parameter vector resulting from fitting Phi to y
    cost_history: ndarray
        Vector containing the evolution of the cost function
    kernel_size: ndarray
        Vector containing the evolution of the kernel size

    
    TODO: Study kernel size normalizations for WMCC
    """

    N, M = Phi.shape
    beta = np.zeros(shape=(M,))
    s2 = np.power(s, 2.0)
    # Initial kernel size:
    iqr_y = np.percentile(y, 75) - np.percentile(y, 25)
    ks_silverman = 1.0592*np.minimum(np.std(y), iqr_y/1.34)*np.power(N, -0.2)
    # Optimize log(ks) instead of ks (positive constraint)
    log_ks = np.log(10.0*ks_silverman)  # Very conservative initial value
    # Initial alpha
    log_alpha = 0.0
    # adadelta learning rates for ks and alpha
    ks_ada = adadelta_updater()
    alpha_ada = adadelta_updater()
    # History of kernel size and WMCC
    cost_history = np.zeros(shape=(max_outer_iterations,))
    kernel_size_history = np.zeros(shape=(max_outer_iterations, 2))
    kernel_size_history[0, :] = [np.exp(log_ks), np.exp(log_alpha)]
    # Main routine
    ks2 = np.exp(2.0*log_ks) + np.exp(log_alpha)*s2
    error = y - np.dot(Phi, beta)
    PhiT = np.transpose(Phi)
    y2 = np.power(y, 2.0)
    for i in range(0, max_outer_iterations):
        # Fixed point routine to update beta
        ks = np.sqrt(ks2)
        for j in range(0, max_inner_iterations):
            invks = 1.0/ks
            cost = np.multiply(np.exp(-0.5*np.divide(np.power(error, 2.0), ks2)), invks)
            #cost_history[i] = np.sum(cost)/(N*np.sqrt(2.0*np.pi))
            cost_history[i] = np.sum(cost)/np.sum(invks)
            #cost_history[i] = (np.sum(invks) - np.sum(cost))/np.sum(np.multiply(1.0-np.exp(-0.5*y2/ks2), invks))
            if full_output:
                print("1 %f %d" %(cost_history[i], j))
            #if i > 0 and cost_history[i] <= cost_history[i-1]*stopping_tol:
            #    break
            W = np.divide(cost, ks2)  # diagonal matrix
            beta = np.linalg.solve(np.dot(PhiT*W, Phi), np.dot(PhiT*W, y))
            #beta = np.dot(np.linalg.pinv(np.dot(PhiT*W, Phi)), np.dot(PhiT*W, y))
            #TODO: Inversion could be not feasible, check conditioning, add regularization...
            error = y - np.dot(Phi, beta)
        #if i > 0 and cost_history[i] <= cost_history[i-1]*stopping_tol:  # More sophisticated stopping criteria could be tried
        #    break    
        # Gradient descent with ADADELTA to update the kernel size
        e2 = np.power(error, 2.0)
        for j in range(0, max_inner_iterations):
            ks = np.sqrt(ks2)
            invks = 1.0/ks
            eys = np.exp(-0.5*y2/ks2)
            ees = np.exp(-0.5*e2/ks2)
            cost = np.multiply(ees, invks)
            #cost_history[i] = np.sum(cost)/(N*np.sqrt(2.0*np.pi))
            cost_history[i] = np.sum(cost)/np.sum(invks)
            #cost_history[i] = (np.sum(cost))/(np.sum(invks) - np.sum(np.multiply(eys, invks)))
            if full_output:
                print("%f %d" %( cost_history[i], j))
            #if i > 0 and cost_history[i] >= cost_history[i-1]*stopping_tol:
            #    break
            W = np.divide(cost, ks2)
            #grad_v = np.multiply(W, np.divide(e2, ks2) - 1.0)/(N*np.sqrt(2.0*np.pi))
            grad_v = np.multiply(W, e2/ks2 - 1.0)/np.sum(invks) + np.sum(cost)*(invks**3)/np.sum(invks)**2
            #grad_v = -np.sum(cost)*np.sum(np.multiply(invks**3, np.multiply(eys, 1.0- y2/ks2)-1.0))/np.sum(np.multiply(invks, 1.0 - eys))**2
            #grad_v += np.sum(np.multiply(invks**3, np.multiply(ees, 1.0-e2/ks2)-1.0))/np.sum(np.multiply(invks, 1.0-eys))
            grad1 = np.sum(grad_v)*np.exp(log_ks)
            grad2 = np.sum(np.multiply(grad_v, 0.5*s2))
            #log_alpha -= alpha_ada.update(grad2)
            log_ks += ks_ada.update(grad1)
            kernel_size_history[i, :] = [np.exp(log_ks), np.exp(log_alpha)]
            ks2 = np.exp(2.0*log_ks) + np.exp(log_alpha)*s2
        """
        for j in range(0, max_inner_iterations):
            ks = np.sqrt(ks2)
            invks = 1.0/ks
            cost = np.multiply(1.0 - np.exp(-0.5*np.divide(e2, ks2)), invks)
            cost_history[i] = np.sum(cost)/np.sum(invks)
            W = np.divide(cost, ks2)
            grad_v = np.multiply(W, np.divide(e2, ks2) - 1.0) + (invks**3)*np.sum(invks)**(-2)
            #grad1 = np.sum(grad_v)*np.exp(log_ks)
            grad2 = np.sum(np.multiply(grad_v, np.exp(log_alpha)*s2))
            log_alpha -= alpha_ada.update(grad2)
            #log_ks += ks_ada.update(grad1)
            kernel_size_history[i, :] = [np.exp(log_ks), np.exp(log_alpha)]
            ks2 = np.exp(2.0*log_ks) + np.exp(2.0*log_alpha)*s2
        """
    if full_output:
        return beta, cost_history[:i], kernel_size_history[:i]
    else:
        return beta, cost_history[:i][-1]
    

class adadelta_updater:
    def __init__(self):
        self.rho = 0.9
        self.eps = 1e-6
        self.reset()

    def reset(self):
        self.param_dxsq_history = 0.01
        self.param_gradsq_history = 0.0

    def update(self, grad):
        self.param_gradsq_history = self.rho*self.param_gradsq_history + (1.0 - self.rho)*grad**2
        RMSgrad = np.sqrt(self.param_gradsq_history + self.eps)
        RMSdx = np.sqrt(self.param_dxsq_history + self.eps)
        dparam = (RMSdx/RMSgrad)*grad
        self.param_dxsq_history = self.rho*self.param_dxsq_history + (1.0 - self.rho)*dparam**2
        return dparam


def find_beta_OLS(y, Phi, s=None):
    """
    Ordinary least squares (OLS) regression for benchmark purposes
    """
    N = len(y)
    R = np.dot(Phi.T, Phi)
    P = np.dot(Phi.T, y)
    beta = np.linalg.solve(R, P)
    return beta, np.dot(P.T, beta)/(np.var(y)*N)


def find_beta_WLS(y, Phi, s):
    """
    Weighted least squares (WLS) regression for benchmark purposes
    """
    W = np.power(s, -2.0)
    W = W/np.sum(W)
    R = np.dot(Phi.T*W, Phi)
    P = np.dot(Phi.T*W, y)
    beta = np.linalg.solve(R, P)
    return beta, np.dot(P.T, beta)/np.sum(np.multiply(np.power(y, 2.0), W))
    
def weighted_IQR(y, w):
    """
    DOES NOT WORK; MAY YIELD NEGATIVE VALUES
    """
    N = len(y)
    I = np.argsort(y)
    S = np.zeros(shape=(N,))
    for k in range(N-1):
        S[k+1] = (k+1)*w[I[k+1]] + (N-1)*np.sum(w[I[:k]])
    I25 = np.argmin(np.absolute(S - 0.25))
    I75 = np.argmin(np.absolute(S - 0.75))
    return y[I75] - y[I25]

