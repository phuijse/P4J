#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from .math import robust_loc, robust_scale
from .QMI import QMI
from .LKSL import LKSL
from .PDM import PDM
from .AOV import AOV
from .MHAOV import MHAOV
#from joblib import Parallel, delayed

from collections import namedtuple
Stats = namedtuple('Stats', 'loc scale N')

class _Periodogram:
    
    def get_best_frequency(self):
        return self.freq[np.argmax(self.per)]
        
    def get_best_frequencies(self):
        """
        Returns the best n_local_max frequencies
        """
        
        return self.freq[self.best_local_optima], self.per[self.best_local_optima]
        
    def get_periodogram(self):
        return self.freq, self.per
    
    def find_local_maxima(self, n_local_optima=10):
        
        local_optima_index = 1+np.where((self.per[1:-1] > self.per[:-2]) & (self.per[1:-1] > self.per[2:]))[0]
        
        if(len(local_optima_index) < n_local_optima):
            print("Warning: Not enough local maxima found in the periodogram")
            return
        # Keep only n_local_optima
        best_local_optima = local_optima_index[np.argsort(self.per[local_optima_index])][::-1]
        if n_local_optima > 0:
            best_local_optima = best_local_optima[:n_local_optima]
        else:
            best_local_optima = best_local_optima[0]
            
        return best_local_optima

    
class MultiBandPeriodogram(_Periodogram):
    
    def __init__(self, method='MHAOV', **kwarg):
        
        #methods = ["MHAOV"]
        #if not method in methods:
        #    raise ValueError("Wrong method")
        self.method = method
        
        self.Nharmonics = 1
        if 'Nharmonics' in kwarg:
            self.Nharmonics = kwarg["Nharmonics"]
        
    def set_data(self, mjds, mags, errs, fids):
        
        self.filter_names = np.unique(fids)
        self.mjds = mjds.astype('float32')
        self.mags = mags.astype('float32')
        self.errs = errs.astype('float32')
        self.cython_per = {}
        self.lc_stats = {}
        for filter_name in self.filter_names:
            mask = fids == filter_name
            weights = np.power(self.errs[mask], -2.0)
            weights = weights/np.sum(weights)
            self.lc_stats.update({filter_name : Stats(loc=robust_loc(self.mags[mask], weights),
                                                      scale=robust_scale(self.mags[mask], weights),
                                                      N=len(self.mjds[mask]))})
                             
            self.cython_per.update({filter_name : MHAOV(self.mjds[mask], 
                                                       self.mags[mask], 
                                                       self.errs[mask], 1)})
            
            
    def frequency_grid_evaluation(self, fmin=0.0, fmax=1.0, fresolution=1e-4):
        
        freqs = np.arange(start=np.amax([fmin, fresolution]), stop=fmax, 
                          step=fresolution, dtype=np.float32)
        per_single_band = {}
        per_sum = np.zeros_like(freqs)  
        
        d1 = 2*self.Nharmonics*len(self.filter_names)
        d2_sum = 0.0
        wvar_sum = 0.0
        
        for filter_name in self.filter_names:
            per = np.zeros_like(freqs)  
            for k, freq in enumerate(freqs):
                per[k] = self.cython_per[filter_name].eval_frequency(freq)
            per_single_band.update({filter_name : per})
            
            d2 = float(self.lc_stats[filter_name].N - 2*self.Nharmonics - 1)  
            per_sum +=  d1*per*self.cython_per[filter_name].wvar/(d2 + d1*per)
            wvar_sum += self.cython_per[filter_name].wvar
            d2_sum += d2
            
        per_sum = d2_sum*per_sum/(d1*(wvar_sum - per_sum))
        self.freq = freqs
        self.per_single_band = per_single_band 
        self.per = per_sum
        
    def get_single_band_periodogram(self, fid):
        return self.freq, self.per_single_band[fid]
    
    
    

class periodogram(_Periodogram):
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
        method: {'PDM1', 'LKSL', 'AOV', 'MHAOV', 'QMICS', 'QMIEU'} 
            Method used to perform the fit
            
            PDM: Phase Dispersion Minimization
            LKSL: Lafler Kinman statistic for string length
            AOV: Analysis of Variance Periodo
            MHAOV: Orthogonal multiharmonic AoV
            QMICS: Cauchy Schwarz Quadratic Mutual Information
            QMIEU: Euclidean Quadratic Mutual Information

        n_jobs: positive integer 
            Number of parallel jobs for periodogram computation, NOT IMPLEMENTED
            
        
        """
        self.freq = None
        self.per = None
        self.debug = debug
        
        if not type(n_jobs) is int:
            raise TypeError("Number of jobs must be an integer")
        if n_jobs < 1:
            raise ValueError("Number of jobs must be greater than 0")
        self.n_jobs = n_jobs
        
        methods = ["QMICS", "QMIEU", "QME", "PDM1", "LKSL", "MHAOV", "AOV"]
        if not method in methods:
            raise ValueError("Wrong method")
        self.method = method
        
    def set_data(self, mjd, mag, err, standardize=False, **kwarg):
        """
        Saves the light curve data, where 
        mjd: modified julian date (time instants)
        mag: stellar magnitude 
        err: photometric error
        
        If standardize is True the data is transformed to have zero
        mean and unit standard deviation. Robust 
        estimators of these quantities are used

        TODO: verify that arrays are non empty, non constant, etc
        """
        
        self.mjd = mjd.astype('float32')
        self.mag = mag.astype('float32')
        self.err = err.astype('float32')
        
        # Nan Filter
        na_mask = np.isnan(self.mjd) | np.isnan(self.mag) | np.isnan(self.err)
        if np.sum(na_mask) > 0:
            print(f"Your data contain {np.sum(na_mask)} NaN values, cleaning")
            self.mjd = self.mjd[~na_mask]
            self.mag = self.mag[~na_mask]
            self.err = self.err[~na_mask]
        
        # Standardization
        weights = np.power(err, -2.0)
        weights = weights/np.sum(weights)
        self.lc_stats = {'loc' : robust_loc(mag, weights), 
                         'scale': robust_scale(mag, weights),
                         'N': len(mjd)}        
        
        if standardize:
            self.mag = (self.mag - self.lc_stats['loc'])/self.lc_stats['scale']
            self.err = self.err/self.lc_stats['scale']
        
        
        if self.method in ['QMICS', 'QMIEU', 'QME']:
            
            hm = 0.9*self.lc_stats['N']**(-0.2)
            if not standardize:
                hm = hm*self.lc_stats['scale']
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
            if self.method == 'QMICS':
                self.cython_per = QMI(self.mjd, self.mag, self.err, hm, hp, 0, kernel)
            elif self.method == 'QMIEU':
                self.cython_per = QMI(self.mjd, self.mag, self.err, hm, hp, 1, kernel)
            else:
                self.cython_per = QMI(self.mjd, self.mag, self.err, hm, hp, 2, kernel)
                
        elif self.method == 'LKSL':  # Lafler-Kinman Minimum String Length
            self.cython_per = LKSL(self.mjd, self.mag, self.err)
        elif self.method == 'PDM1':  # Phase Dispersion Minimization
            Nbins = 8
            if 'Nbins' in kwarg:
                Nbins = kwarg['Nbins']
            self.cython_per = PDM(self.mjd, self.mag, self.err, Nbins)
        elif self.method == 'AOV':  # Analysis of Variance periodogram
            Nbins = 8
            if 'Nbins' in kwarg:
                Nbins = kwarg['Nbins']
            use_errorbars = 1
            if 'use_errorbars' in kwarg:
                use_errorbars = kwarg['use_errorbars']
            self.cython_per = AOV(self.mjd, self.mag, self.err, Nbins, use_errorbars)
        elif self.method == 'MHAOV':  # Orthogonal Multiharmonics AoV periodogram
            Nharmonics = 1
            if 'Nharmonics' in kwarg:
                Nharmonics = kwarg["Nharmonics"]
            self.cython_per = MHAOV(self.mjd, self.mag, self.err, Nharmonics)  
        
   
    def finetune_best_frequencies(self, fresolution=1e-5, n_local_optima=10):
        """
        Computes the selected criterion over a grid of frequencies 
        around a specified amount of  local optima of the periodograms. This
        function is intended for additional fine tuning of the results obtained
        with grid_search
        """
        best_local_optima = self.find_local_maxima(n_local_optima)
        
        for j in range(n_local_optima):
            freq_fine = self.freq[best_local_optima[j]] - self.freq_step_coarse
            for k in range(0, int(2.0*self.freq_step_coarse/fresolution)):
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
        self.freq_step_coarse = fresolution
        freq = np.arange(np.amax([fmin, fresolution]), fmax, step=fresolution).astype('float32')
        Nf = len(freq)        
        if self.n_jobs == 1:
            per = np.zeros(shape=(Nf,)).astype('float32')
            for k in range(0, Nf):
                per[k] = self.compute_metric(freq[k])
        #else:
        #    per = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_metric)(freq_) for freq_ in freq)
        self.freq = freq
        self.per = per

    def compute_metric(self, freq):
        if self.method in ["PDM1", "LKSL"]:
            return -self.cython_per.eval_frequency(freq)
        else:
            return self.cython_per.eval_frequency(freq)
        
