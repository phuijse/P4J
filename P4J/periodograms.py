#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from .base_periodogram import BasePeriodogram
from .algorithms.mutual_information import QMI
from .algorithms.string_length import LKSL
from .algorithms.phase_dispersion_minimization import PDM
from .algorithms.analysis_of_variance import AOV
from .algorithms.multiharmonic_aov import MHAOV
from .math import robust_loc, robust_scale
#from joblib import Parallel, delayed

from collections import namedtuple
Stats = namedtuple('Stats', 'loc scale N')


class MultiBandPeriodogram(BasePeriodogram):
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
            self.lc_stats.update({
                filter_name:
                    Stats(loc=robust_loc(self.mags[mask], weights),
                          scale=robust_scale(self.mags[mask], weights),
                          N=len(self.mjds[mask]))})
                             
            self.cython_per.update({
                filter_name:
                    MHAOV(self.mjds[mask],
                          self.mags[mask] - self.lc_stats[filter_name].loc,
                          self.errs[mask], Nharmonics=1, mode=0)})

    def get_lc_time_length(self):
        return np.max(self.mjds) - np.min(self.mjds)
            
    def _compute_periodogram(self, freqs):        
        per_single_band = {}
        per_sum = np.zeros_like(freqs) 
        d1 = 2 * self.Nharmonics
        d2_sum = 0.0
        wvar_sum = 0.0
        
        for filter_name in self.filter_names:
            per = np.array(
                [self.cython_per[filter_name].eval_frequency(freq) for freq in freqs],
                dtype=np.float32)
            d2 = float(self.lc_stats[filter_name].N - 2*self.Nharmonics - 1)  
            per_single_band.update(
                {filter_name: (d2/d1)*per/(self.cython_per[filter_name].wvar-per)})
            per_sum += per
            wvar_sum += self.cython_per[filter_name].wvar
            d2_sum += d2

        return d2_sum*per_sum/(d1*(wvar_sum - per_sum)), per_single_band
    
    def _update_periodogram(self, replace_idx, freqs_fine, pers_fine):
        new_best = np.argmax(pers_fine[0])
        if pers_fine[0][new_best] > self.per[replace_idx]:
            self.freq[replace_idx] = freqs_fine[new_best]  # frequency
            self.per[replace_idx] = pers_fine[0][new_best]  # multiband
            for filter_name in self.filter_names:  # single band
                self.per_single_band[filter_name][replace_idx] = pers_fine[1][filter_name][new_best]
    

# TODO: rename as Periodogram in v2.0.0
class periodogram(BasePeriodogram):
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
        
        if type(n_jobs) is not int:
            raise TypeError("Number of jobs must be an integer")
        if n_jobs < 1:
            raise ValueError("Number of jobs must be greater than 0")
        self.n_jobs = n_jobs
        
        methods = ["QMICS", "QMIEU", "QME", "PDM1", "LKSL", "MHAOV", "AOV"]
        if method not in methods:
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
        weights = np.power(self.err, -2.0)
        weights = weights/np.sum(weights)
        self.lc_stats = {'loc': robust_loc(self.mag, weights),
                         'scale': robust_scale(self.mag, weights),
                         'N': len(self.mjd)}        
        
        if standardize:
            self.mag = (self.mag - self.lc_stats['loc'])/self.lc_stats['scale']
            self.err = self.err/self.lc_stats['scale']

        if self.method in ['QMICS', 'QMIEU', 'QME']:
            hm = 0.9*self.lc_stats['N']**(-0.2)
            if not standardize:
                hm = hm*self.lc_stats['scale']
            if 'h_KDE_M' in kwarg:
                hm = hm*kwarg['h_KDE_M']
            hp = 1.0  # How to choose this more appropietly?
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

    def get_lc_time_length(self):
        return np.max(self.mjd) - np.min(self.mjd)

    def _compute_periodogram(self, freqs):
        if self.n_jobs == 1:
            pers = np.array([self.cython_per.eval_frequency(freq) for freq in freqs], dtype=np.float32)
        #else:
        #    pers = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_metric)(freq) for freq in freqs)

        if self.method in ["PDM1", "LKSL"]:  # Minima are best
            pers = -pers
        return pers, None  # TODO: THIS IS UGLY!!
    
    def _update_periodogram(self, replace_idx, freqs_fine, pers_fine):
        new_best = np.argmax(pers_fine[0])
        if pers_fine[0][new_best] > self.per[replace_idx]:
            self.freq[replace_idx] = freqs_fine[new_best] 
            self.per[replace_idx] = pers_fine[0][new_best]
