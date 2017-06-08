#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from .math import robust_center, robust_scale, wSTD
from .QMI import QMI
from .LKSL import LKSL
from .PDM import PDM
from .MHAOV import AOV

#from scipy.stats import gumbel_r, genextreme
#from .regression import find_beta_WMEE, find_beta_WMCC, find_beta_OLS, find_beta_WLS
#from .dictionary import harmonic_dictionary


class periodogram:
    def __init__(self, method='QMIEU', n_jobs=1, debug=False):
        """
        Class for light curve periodogram computation

        A light curve is a time series of stellar magnitude or flux. Some 
        light curves are periodic, i.e. their brightness change regularly in 
        time (Cepheids, RR Lyrae, Eclipsing Binaries). 
        
        The word periodogram is used in a broad sense to describe a function
        of the period (or frequency) and not the estimator of the power 
        spectral density. 

        Different criteria can be used to assess periodicity. In this package
        we focus on information theoretic based criteria. For more detailes see
       
        J. C. Principe, "Information Theoretic Learning, Renyi's Entropy 
        and Kernel Perspectives", Springer, 2010, Chapters 2 and 10.
        
        
        Parameters
        ---------
        method: {'PDM1', 'LKSL', 'MHAOV', 'QMICS', 'QMIEU} 
            Method used to perform the fit
            
            PDM: Phase Dispersion Minimization
            LKSL: Lafler Kinman statistic for string length
            MHAOV: Orthogonal multiharmonic AoV
            QMICS: Cauchy Schwarz Quadratic Mutual Information
            QMIEU: Euclidean Quadratic Mutual Information

        n_jobs: positive integer 
            Number of parallel jobs for periodogram computation, NOT IMPLEMENTED
            
        
        """
        if n_jobs < 1:
            raise ValueError("Number of jobs must be greater than 0")
        self.method = method
        self.local_optima_index = None
        self.freq = None
        self.per = None
        self.debug = debug
        self.n_jobs = n_jobs
        methods = ["QMICS", "QMIEU", "QME", "PDM1", "LKSL", "MHAOV"]
        if not method in methods:
            raise ValueError("Wrong method")
        
    def set_data(self, mjd, mag, err, whitten=False, **kwarg):
        """
        Saves the light curve data, where 
        mjd: modified julian date (time instants)
        mag: stellar magnitude 
        err: photometric error
        
        If whitten is True the data is normalized to have zero
        mean and unit standard deviation. Note that robust 
        estimators of these quantities are used

        TODO: verify that arrays are non empty, non constant, etc
        """
        
        self.mjd = mjd
        self.N = len(mjd)
        weights = np.power(err, -2.0)
        self.weights = weights/np.sum(weights)
        self.scale = robust_scale(mag, self.weights)
        if whitten:
            self.mag = (mag - robust_center(mag, self.weights))/self.scale
            self.err = err/self.scale
        else:
            self.mag = mag
            self.err = err
        self.mjd = self.mjd.astype('float32')
        self.mag = self.mag.astype('float32')
        self.err = self.err.astype('float32')
        if self.method == 'QMICS' or self.method == 'QMIEU' or self.method == 'QME':
            if whitten:
                hm = 0.9*self.N**(-0.2) # Silverman's rule, assuming data is whittened
            else:
                hm = 0.9*self.scale*self.N**(-0.2)
            if 'h_KDE_M' in kwarg:
                hm = hm*kwarg['h_KDE_M']
            hp = 1.0 # How to choose this more appropietly?
            if 'h_KDE_P' in kwarg:
                hp = hp*kwarg['h_KDE_P']
            kernel = 0  # Select the kernel for the magnitudes, 0 is safe
            if 'kernel' in kwarg:
                kernel = kwarg['kernel']
            if self.debug:
                print("Kernel bandwidths: %f , %f" %(hm, hp))
            self.my_QMI = QMI(self.mjd, self.mag, self.err, hm, hp, kernel)
        elif self.method == 'LKSL':  # Lafler-Kinman Minimum String Length
            self.my_SL = LKSL(self.mjd, self.mag, self.err)
        elif self.method == 'PDM1':  # Phase Dispersion Minimization
            Nbins = int(self.N/3)
            if 'Nbins' in kwarg:
                Nbins = kwarg['Nbins']
            self.my_PDM = PDM(self.mjd, self.mag, self.err, Nbins)
        elif self.method == 'MHAOV':  # Orthogonal Multiharmonics AoV periodogram
            Nharmonics = 1
            if 'Nharmonics' in kwarg:
                Nharmonics = kwarg["Nharmonics"]
            self.my_AOV = AOV(self.mjd, self.mag, self.err, Nharmonics)


   
    def get_best_frequency(self):
        return self.freq[self.best_local_optima[0]]
        
    def get_best_frequencies(self):
        """
        Returns the best n_local_max frequencies
        """
        return self.freq[self.best_local_optima], self.per[self.best_local_optima]
        
    def get_periodogram(self):
        return self.freq, self.per
        
   
    def finetune_best_frequencies(self, fresolution=1e-5, n_local_optima=10):
        """
        Computes the selected criterion over a grid of frequencies 
        around a specified amount of  local optima of the periodograms. This
        function is intended for additional fine tuning of the results obtained
        with grid_search
        """
        # Find the local optima
        local_optima_index = []
        for k in range(1, len(self.per)-1):
            if self.per[k-1] < self.per[k] and self.per[k+1] < self.per[k]:
                local_optima_index.append(k)
        local_optima_index = np.array(local_optima_index)
        if(len(local_optima_index) < n_local_optima):
            print("Warning: Not enough local maxima found in the periodogram, skipping finetuning")
            return
        # Keep only n_local_optima
        best_local_optima = local_optima_index[np.argsort(self.per[local_optima_index])][::-1]
        if n_local_optima > 0:
            best_local_optima = best_local_optima[:n_local_optima]
        else:
            best_local_optima = best_local_optima[0]
        # Do finetuning around each local optima
        for j in range(n_local_optima):
            freq_fine = self.freq[best_local_optima[j]] - self.fres_grid
            for k in range(0, int(2.0*self.fres_grid/fresolution)):
                cost = self.compute_metric(freq_fine)
                if cost > self.per[best_local_optima[j]]:
                    self.per[best_local_optima[j]] = cost
                    self.freq[best_local_optima[j]] = freq_fine
                freq_fine += fresolution
        # Sort them in descending order
        idx = np.argsort(self.per[best_local_optima])[::-1]
        if n_local_optima > 0:
            self.best_local_optima = best_local_optima[idx]
        else:
            self.best_local_optima = best_local_optima
 

    def frequency_grid_evaluation(self, fmin=0.0, fmax=1.0, fresolution=1e-4, n_local_max=10):
        """ 
        Computes the selected criterion over a grid of frequencies 
        with limits and resolution specified by the inputs. After that
        the best local maxima are evaluated over a finer frequency grid
        
        Parameters
        ---------
        fmin: float
            starting frequency
        fmax: float
            stopping frequency
        fresolution: float
            step size in the frequency grid
        
        """
        self.fres_grid = fresolution
        freq = np.arange(np.amax([fmin, fresolution]), fmax, step=fresolution).astype('float32')
        Nf = len(freq)
        per = np.zeros(shape=(Nf,)).astype('float32')     
              
        for k in range(0, Nf):
            per[k] = self.compute_metric(freq[k])
        self.freq = freq
        self.per = per

    def compute_metric(self, freq):
        if self.method == 'QMICS':
            return self.my_QMI.eval_frequency(freq, 0)
        elif self.method == 'QMIEU':
             return self.my_QMI.eval_frequency(freq, 1)
        elif self.method == 'QME':
              return self.my_QMI.eval_frequency(freq, 2)
        elif self.method == 'LKSL':
            return -self.my_SL.eval_frequency(freq)
        elif self.method == 'PDM1':
            return -self.my_PDM.eval_frequency(freq)
        elif self.method == 'MHAOV':
            return self.my_AOV.eval_frequency(freq)

