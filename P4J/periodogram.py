#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from .math import robust_center, robust_scale, wSTD
from .QMI import QMI
from .LKSL import LKSL
from .PDM import PDM
#from scipy.stats import gumbel_r, genextreme
#from .regression import find_beta_WMEE, find_beta_WMCC, find_beta_OLS, find_beta_WLS
#from .dictionary import harmonic_dictionary
#from multiprocessing import Pool
#from functools import partial



class periodogram:
    def __init__(self, method='QMI', n_jobs=1):
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
        method: {'PDM', 'LKSL', 'QMI'} 
            Method used to perform the fit
            
            PDM: Phase Dispersion Minimization
            LKSL: Lafler Kinman statistic for string length
            QMI: Quadratic Mutual Information

        n_jobs: positive integer 
            Number of parallel jobs for periodogram computation, NOT IMPLEMENTED
            
        
        """
        if n_jobs < 1:
            raise ValueError("Number of jobs must be greater than 0")
        self.method = method
        self.local_optima_index = None
        self.freq = None
        self.per = None
        self.n_jobs = n_jobs
        
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
        self.time_span = mjd[-1] - mjd[0]
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
        if self.method == 'QMI':
            if whitten:
                hm = 0.9*self.N**(-0.2) # Silverman's rule, assuming data is whittened
            else:
                hm = 0.9*self.scale*self.N**(-0.2)
            #print(hm)
            if 'h_KDE_M' in kwarg:
                hm = hm*kwarg['h_KDE_M']
            hp = 0.9/np.sqrt(12)*self.N**(-0.2)
            #print(hp)
            if 'h_KDE_P' in kwarg:
                hp = hp*kwarg['h_KDE_P']
            self.my_QMI = QMI(self.mjd, self.mag, self.err, hm, hp)
        elif self.method == 'LKSL':
            self.my_SL = LKSL(self.mjd, self.mag, self.err)
        elif self.method == 'PDM1':
            Nbins = self.N/3
            if 'Nbins' in kwarg:
                Nbins = kwarg['Nbins']
            self.my_PDM = PDM(self.mjd, self.mag, self.err, Nbins)
#            self.PDM_normalizer = wSTD(self.mag, self.weights)

   
    def get_best_frequency(self):
        return self.freq[self.best_local_optima[0]]
        
    def get_best_frequencies(self):
        """
        Returns the best n_local_max frequencies
        """
        return self.freq[self.best_local_optima]
        
    def get_periodogram(self):
        return self.freq, self.per
        
   
    def finetune_best_frequencies(self, fresolution=0.1, n_local_optima=10):
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
        # Keep only n_local_optima
        best_local_optima = local_optima_index[np.argsort(self.per[local_optima_index])][::-1]
        if n_local_optima > 0:
            best_local_optima = best_local_optima[:n_local_optima]
        else:
            best_local_optima = best_local_optima[0]
        # Do finetuning around each local optima
        for j in range(0, n_local_optima):
            freq_fine = self.freq[best_local_optima[j]] - self.fres_grid/self.time_span
            for k in range(0, int(2.0*self.fres_grid/fresolution)):
                cost = self.compute_metric(freq_fine)
                if cost > self.per[best_local_optima[j]]:
                    self.per[best_local_optima[j]] = cost
                    self.freq[best_local_optima[j]] = freq_fine
                freq_fine += fresolution/self.time_span
        # Sort them in descending order
        idx = np.argsort(self.per[best_local_optima])[::-1]
        if n_local_optima > 0:
            self.best_local_optima = best_local_optima[idx]
        else:
            self.best_local_optima = best_local_optima
 

    def frequency_grid_evaluation(self, fmin=0.0, fmax=1.0, fresolution=1.0, n_local_max=10):
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
            step size in the frequency grid, note that the actual 
            frequency step is fres_coarse/self.T, where T is the 
            total time span of the time series
        
        """
        self.fres_grid = fresolution
        freq = np.arange(np.amax([fmin, fresolution/self.time_span]), fmax, step=fresolution/self.time_span).astype('float32')
        Nf = len(freq)
        per = np.zeros(shape=(Nf,)).astype('float32')      
        
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
            per[k] = self.compute_metric(freq[k])
        self.freq = freq
        self.per = per

    def compute_metric(self, freq):
        if self.method == 'QMI':
            return self.my_QMI.eval_frequency(freq)
        elif self.method == 'LKSL':
            return -self.my_SL.eval_frequency(freq)
        elif self.method == 'PDM1':
            return -self.my_PDM.eval_frequency(freq)
        """
            phase = np.mod(self.mjd, 1.0/freq)*freq
            Nbins = 20.
            pdm_num = 0.0
            pdm_den = 0.0
            sorted_idx = np.argsort(phase)
            sorted_pha = phase[sorted_idx]
            sorted_mag = self.mag[sorted_idx]
            sorted_wei = self.weights[sorted_idx]
            idx_sta = 0
            idx_end = 0
            for m in range(int(Nbins)):
                for j in range(idx_sta+1, self.N):
                    if sorted_pha[j] - m/Nbins > 1.0/Nbins:
                        idx_end = j - 1
                        break
                N_in_bin = idx_end - idx_sta
                if N_in_bin < 3:
                    continue
                pdm_num += wSTD(sorted_mag[idx_sta:idx_end], sorted_wei[idx_sta:idx_end])*(N_in_bin-1)
                pdm_den += (N_in_bin-1)
                idx_sta = idx_end + 1 
            return -pdm_num/(pdm_den*self.PDM_normalizer)
        """

