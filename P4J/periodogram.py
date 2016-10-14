#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from scipy.stats import gumbel_r, genextreme
from .regression import find_beta_WMEE, find_beta_WMCC, find_beta_OLS, find_beta_WLS
from .dictionary import harmonic_dictionary
#from multiprocessing import Pool
#from functools import partial

class periodogram:
    def __init__(self, method='WMCC', M=1, n_jobs=1):
        """
        Class for multiharmonic periodogram computation
        
        The multiharmonic periodogram of a time series is obtained by 
        fitting a multiharmonic model to the time series at several different
        frequencies. The method used to perform the fitting defines the 
        properties of the periodogram, options are: Ordinary Least 
        Squares (OLS), Weighted Least Squares (WLS), Weighted
        Maximum Correntropy Criterion (WMCC) and Weighted Minimum
        Error Entropy (WMEE).
        
        Correntropy  is a generalized correlation for random variables 
        that is robust to outliers and non-Gaussian noise. For entropy Renyi's
        quadratic entropy is used. Entropy is estimated using Parzen windows.
        For more details on correntropy/entropy and the MCC/MEE we suggest
        
        J. C. Principe, "Information Theoretic Learning, Renyi's Entropy 
        and Kernel Perspectives", Springer, 2010, Chapters 2 and 10.
        
        
        Parameters
        ---------
        M: positive interger
            Number of harmonic components used to fit the data
        method: {'OLS', 'WLS', 'WMCC', 'WMEE'} 
            Method used to perform the fit
        n_jobs: positive integer 
            Number of parallel jobs for periodogram computation, NOT IMPLEMENTED
            
        Example
        -------
        
        This will compute a periodogram for a time series (t, y, dy), 
        where t, y and dy are ndarray vectors of length N, where y is 
        the variable of interest, t are the time instants where y was 
        sampled and dy are the uncertainties associated to y.
        
        >>> import P4J
        >>> my_per = P4J.periodogram(M=3, method='WMCC')
        >>> my_per.fit(t, y, dy)
        >>> my_per.grid_search(0.0, 5.0, 1.0, 0.1, n_local_max=10)
        >>> my_per.fit_extreme_cdf(n_bootstrap=50, n_frequencies=100)
        >>> fbest = my_per.get_best_frequency()
        >>> conf = my_per.get_confidence(fbest[1])
        >>> print("Best frequency is %0.5f" %(fbest[0]))
        >>> print("with confidence: %0.3f" %(conf))
        
        """
        if n_jobs < 1:
            raise ValueError("Number of jobs must be greater than 0")
        self.method = method
        self.M = M
        self.local_max_index = None
        self.freq = None
        self.per = None
        self.n_jobs = n_jobs
        if self.method == 'WMCC':
            self.get_cost = find_beta_WMCC
        elif self.method == 'WMEE':
            self.get_cost = find_beta_WMEE
        elif self.method == 'WLS':
            self.get_cost = find_beta_WLS
        elif self.method == 'OLS':
            self.get_cost = find_beta_OLS
        
    def fit(self, t, y, s, subtract_average=True):
        """
        Save the time series data, subtracts the weighted mean from y
        """
        self.t = t
        if subtract_average:
            w = np.power(s, -2.0)
            self.y = y - np.sum(w*y)/np.sum(w)
        else:
            self.y = y
        self.s = s
        self.T = t[-1] - t[0]
    
    def get_best_frequency(self):
        return self.freq[self.best_local_max[0]], self.per[self.best_local_max[0]]
        
    def get_best_frequencies(self):
        """
        Returns the best n_local_max frequencies and their periodogram 
        values, sorted by per
        """
        return self.freq[self.best_local_max], self.per[self.best_local_max]
        
    def get_periodogram(self):
        return self.freq, self.per
        
    #def compute_per_ordinate(self, f):
    #    Phi = harmonic_dictionary(self.t, f, self.M)
    #    beta, per = self.get_cost(self.y, Phi, self.s)
    #    if self.method == 'WMCC':
    #        return np.linalg.norm(beta)
    #    else: 
    #        return per
    
    def grid_search(self, fmin=0.0, fmax=1.0, fres_coarse=1.0, fres_fine=0.1, n_local_max=10):
        """ 
        Computes self.method (OLS, WLS, WMCC) over a grid of frequencies 
        with limits and resolution specified by the inputs. After that
        the best local maxima are evaluated over a finer frequency grid
        
        Parameters
        ---------
        fmin: float
            starting frequency
        fmax: float
            stopping frequency
        fres_coarse: float
            step size in the frequency grid, note that the actual 
            frequency step is fres_coarse/self.T, where T is the 
            total time span of the time series
        fres_fine: float
            step size in the frequency grid for frequency fine-tuning
        n_local_max: zero or positive integer
            Number of frequencies to be refined using fres_fine 
        
        Returns
        -------
        freq: ndarray
            frequency array that was sweeped to compute the periodogram
        
        per: ndarray
            periodogram evaluated at the frequencies given in freq
        """
        self.fres_coarse = fres_coarse
        freq = np.arange(np.amax([fmin, fres_coarse/self.T]), fmax, step=fres_coarse/self.T)
        Nf = len(freq)
        per = np.zeros(shape=(Nf,))      
        
        #partial_job = partial(self.compute_per_ordinate)
        #if self.n_jobs <= 1:
        #    m = map
        #else:
        #    pool = Pool(self.n_jobs)
        #    m = pool.map
        #per = list(m(self.compute_per_ordinate, freq))
        #if self.n_jobs > 1:
        #    pool.close()
        #    pool.join()
        #per = np.asarray(per, dtype=float)
              
        for k in range(0, Nf):
            #per[k] = self.compute_per_ordinate(freq[k], self.t, self.y, self.dy, self.M)
            #print(freq[k])
            Phi = harmonic_dictionary(self.t, freq[k], self.M)
            _, per[k] = self.get_cost(self.y, Phi, self.s)
        # Find the local minima and do analysis with finer frequency step
        local_max_index = []
        for k in range(1, Nf-1):
            if per[k-1] < per[k] and per[k+1] < per[k]:
                local_max_index.append(k)
        local_max_index = np.array(local_max_index)
        best_local_max = local_max_index[np.argsort(per[local_max_index])][::-1]
        if n_local_max > 0:
            best_local_max = best_local_max[:n_local_max]
        else:
            best_local_max = best_local_max[0]
        #print(freq[best_local_max])
        # Do finetuning
        for j in range(0, n_local_max):
            freq_fine = freq[best_local_max[j]] - fres_coarse/self.T
            for k in range(0, int(2.0*fres_coarse/fres_fine)):
                Phi = harmonic_dictionary(self.t, freq_fine, self.M)
                _, cost = self.get_cost(self.y, Phi, self.s)
                if cost > per[best_local_max[j]]:
                    per[best_local_max[j]] = cost
                    freq[best_local_max[j]] = freq_fine
                freq_fine += fres_fine/self.T
        # Sort them
        idx = np.argsort(per[best_local_max])[::-1]
        if n_local_max > 0:
            self.best_local_max= best_local_max[idx]
        else:
            self.best_local_max= best_local_max
        self.freq = freq
        self.per = per
        return freq, per

    def get_confidence(self, per_value):
        """
        Computes the confidence for a given periodogram value
        """
        # return gumbel_r.cdf(per_value, loc=self.param[0], scale=self.param[1])
        return genextreme.cdf(per_value, self.param[0], loc=self.param[1], scale=self.param[2])
    
    def get_FAP(self, p):
        """
        Computes the periodogram value associated to FAP=p
        """
        #return self.param[0] - self.param[1]*np.log(-np.log(1.0-p))
        mu = self.param[1]
        ss = self.param[2]
        chi = -self.param[0]
        return mu + (ss/chi)*((-np.log(1.0-p))**(-chi) -1.0)

    def fit_extreme_cdf(self, n_bootstrap=100, n_frequencies=10, rseed=None):
        """
        Perform false alarm probability (FAP) computation based on 
        generalized extreme value (gev) statistics and bootstrapping 
        
        Parameters
        --------
        n_bootstrap: positive integer
            The number of bootstrap repetitions of time series (t,y,dy)
        n_frequencies: positive integer
            The number of frequencies to search for maxima, it is a 
            subset of self.freq
        
        Returns
        -------
        maxima_realization: ndarray
            The maxima for each bootstrap repetition
        param: ndarray 
            The parameters resulting from GEV the fit
        
        Reference
        --------
        M. Süveges, "Extreme-value modelling for the significance 
        assessment of periodogram peaks", MNRAS, 2012.
        
        """
        np.random.seed(rseed)
        y = self.y.copy()
        s = self.s.copy()
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
            yr = y[idx[i]]
            sr = s[idx[i]]
            for k in range(0, n_frequencies):
                Phi = harmonic_dictionary(self.t, self.freq[random_freq[k]], self.M)
                _, per_gev[k] = self.get_cost(yr, Phi, sr)
            maxima_realization[i] = np.amax(per_gev)
        # Fit the GEV parameters
        #self.param = gumbel_r.fit(maxima_realization)
        s_init = np.std(maxima_realization)
        mu_init = np.mean(maxima_realization) - 0.5*s_init
        self.param = genextreme.fit(maxima_realization, 0.001, loc=mu_init, scale=s_init)
        return maxima_realization, self.param
