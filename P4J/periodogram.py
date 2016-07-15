from __future__ import division, print_function
import numpy as np
#from scipy.stats import genextreme as gev
from scipy.stats import gumbel_r
from .regression import find_beta_WMCC, find_beta_OLS, find_beta_WLS
from .dictionary import harmonic_dictionary

class periodogram:
    def __init__(self, method='WMCC', M=1):
        """
        Class for multiharmonic periodogram computation
        M -- number of harmonic components used to fit the data
        method -- method used to perform the fit, options are OLS, WLS and WMCC
        """
        self.method = method
        self.M = M
        self.local_max_index = None
        self.freq = None
        self.per = None
        
    def fit(self, t, y, dy):
        self.t = t
        self.y = y #- np.mean(y)
        self.dy = dy
        self.T = t[-1] - t[0]
        if self.method == 'WLS':
            self.norm_constant = np.dot(y.T, np.dot(np.diag(np.power(dy, -2.0)), y))
        elif self.method == 'OLS':
            self.norm_constant = np.var(y)*len(y)*0.5
            
    def get_best_frequencies(self):
        """
        Returns the best n_local_max frequencies and their periodogram 
        values, sorted by per
        """
        return self.freq[self.local_max_index], self.per[self.local_max_index]
        
    def get_periodogram(self):
        return self.freq, self.per
        
    def grid_search(self, fmin=0.0, fmax=1.0, fres_coarse=1.0, fres_fine=0.1, n_local_max=10):
        """ 
        Computes self.method over a grid of frequencies specified by
        fmin -- starting frequency
        fmax -- stopping frequency
        fres_coarse -- step size in the frequency grid, note that the 
        actual frequency step is fres_coarse/self.T, where T is the 
        total time span of the time series
        
        Then it refines (fine-tune) the estimation for a given number of local maxima:
        n_local_max -- number of local maxima to refine
        fres_fine -- oversampling factor for the fine-tuning step
        """
        self.fres_coarse = fres_coarse
        freq = np.arange(np.amax([fmin, fres_coarse/self.T]), fmax, step=fres_coarse/self.T)
        Nf = len(freq)
        per = np.zeros(shape=(Nf,))
        for k in range(0, Nf):
            Phi = harmonic_dictionary(self.t, freq[k], self.M)
            if self.method == 'WMCC':
                beta, cost_history, _ = find_beta_WMCC(self.y, Phi, self.dy)
                per[k] = cost_history[-1]
            elif self.method == 'WLS':
                beta, cost =  find_beta_WLS(self.y, Phi, self.dy)
                per[k] = cost/self.norm_constant
            elif self.method == 'OLS':
                beta, cost = find_beta_OLS(self.y, Phi)
                per[k] = cost/self.norm_constant
        # Find the local minima and do analysis with finer frequency step
        local_max_index = []
        for k in range(1, Nf-1):
            if per[k-1] < per[k] and per[k+1] < per[k]:
                local_max_index.append(k)
        local_max_index = np.array(local_max_index)
        best_local_max = local_max_index[np.argsort(per[local_max_index])][::-1]
        #print(freq[best_local_max])
        # Do finetuning
        for j in range(0, n_local_max):
            freq_fine = freq[best_local_max[j]] - fres_coarse/self.T
            for k in range(0, int(2.0*fres_coarse/fres_fine)):
                Phi = harmonic_dictionary(self.t, freq_fine, self.M)
                if self.method == 'WMCC':
                    _, cost_history, _ = find_beta_WMCC(self.y, Phi, self.dy)
                    cost = cost_history[-1]
                elif self.method == 'WLS':
                    _, cost =  find_beta_WLS(self.y, Phi, self.dy)
                    cost = cost/self.norm_constant
                elif self.method == 'OLS':
                    _, cost = find_beta_OLS(self.y, Phi)
                    cost = cost/self.norm_constant
                if cost > per[best_local_max[j]]:
                    per[best_local_max[j]] = cost
                    freq[best_local_max[j]] = freq_fine
                freq_fine += fres_fine/self.T
        # Sort them
        idx = np.argmax(per[local_max_index])
        self.local_max_index = local_max_index[idx]
        self.freq = freq
        self.per = per
        return freq, per

    def get_confidence(self, per_value):
        """
        Computes the confidence for a given periodogram value
        """
        return gumbel_r.cdf(per_value, loc=self.param[0], scale=self.param[1])
    
    def get_FAP(self, p):
        """
        Computes the periodogram value associated to FAP=p
        """
        return self.param[0] - self.param[1]*np.log(-np.log(1.0-p))

    def fit_extreme_cdf(self, n_bootstrap=10, n_frequencies=10):
        """
        Perform false alarm probability (FAP) computation based on 
        generalized extreme value (gev) statistics. 
        
        n_bootstrap -- the number of bootstrap repetitions of time 
        series (t,y,dy)
        n_frequencies -- the number of frequencies to search for maxima, 
        it is a subset of self.freq
        
        Returns the maxima for each bootstrap repetition and the
        parameters resulting from the fit
        
        Reference:
        Süveges, M. "False Alarm Probability based on bootstrap and 
        extreme-value methods for periodogram peaks." ADA7-Seventh 
        Conference on Astronomical Data Analysis. Vol. 1. 2012.
        """
        
        y = self.y.copy()
        dy = self.dy.copy()
        #K = int(1.0/self.fres_coarse)  # oversampling factor 
        K = 1
        N = len(self.t)
        Nf = len(self.freq)
        # Sensible limits for the number of frequencies
        if n_frequencies > Nf: 
            n_frequencies = Nf
        if n_frequencies < 2*Nf/(K*N):
            n_frequencies = int(2*Nf/(K*N))
        idx = np.random.randint(0, N, (n_bootstrap, N))
        maxima_realization = np.zeros(shape=(n_bootstrap,))
        # Find the maxima
        for i in range(0, n_bootstrap):  # bootstrap
            random_freq = np.random.permutation(Nf)[:n_frequencies]
            per_gev = np.zeros(shape=(n_frequencies,))
            for k in range(0, n_frequencies):
                Phi = harmonic_dictionary(self.t, self.freq[random_freq[k]], self.M)
                if self.method == 'WMCC':
                    beta, cost_history, _ = find_beta_WMCC(y[idx[i]], Phi, dy[idx[i]])
                    per_gev[k] = cost_history[-1]
                elif self.method == 'WLS':
                    beta, cost =  find_beta_WLS(y[idx[i]], Phi, dy[idx[i]])
                    per_gev[k] = cost/self.norm_constant
                elif self.method == 'OLS':
                    beta, cost = find_beta_OLS(y[idx[i]], Phi)
                    per_gev[k] = cost/self.norm_constant
            maxima_realization[i] = np.amax(per_gev)
        # Fit the GEV parameters
        self.param = gumbel_r.fit(maxima_realization)
        #return self.param[0] - self.param[1]*np.log(-np.log(1.0-p))
        return maxima_realization, self.param
