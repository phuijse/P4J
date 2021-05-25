#!/usr/bin/python
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True, profile=False

cimport cython
from cython.operator cimport dereference
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from .utilities cimport argsort, mean, weighted_mean, unbiased_weighted_variance

ctypedef float DTYPE_t
ctypedef int ITYPE_t

cdef extern from "math.h":
    DTYPE_t sqrtf(DTYPE_t)
    DTYPE_t powf(DTYPE_t, DTYPE_t)
    DTYPE_t floorf(DTYPE_t)
    DTYPE_t fmodf(DTYPE_t, DTYPE_t)

"""

Analysis of Variance periodograma
Ref: http://adsabs.harvard.edu/full/1989MNRAS.241..153S

Please cite the paper above if using this code
"""

cdef class AOV:
    cdef Py_ssize_t N
    cdef DTYPE_t* phase
    cdef DTYPE_t* mjd
    cdef DTYPE_t* mag
    cdef DTYPE_t* err
    cdef DTYPE_t normalizer, barx
    cdef ITYPE_t Nbins
    cdef DTYPE_t* tmp_mag
    cdef DTYPE_t* tmp_err
    cdef ITYPE_t* tmp_sizes
    cdef int use_errorbars

    def __init__(self, DTYPE_t [::1] mjd, DTYPE_t [::1] mag, DTYPE_t [::1] err, ITYPE_t Nbins=10, use_errorbars=1):
        cdef Py_ssize_t i, j, mat_idx
        self.Nbins = Nbins
        self.use_errorbars = use_errorbars
        self.N = mag.shape[0]
        self.mjd = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.mag = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.err = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.phase = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        #self.tmp_mag = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        #self.tmp_err = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.tmp_mag = <DTYPE_t*>PyMem_Malloc(self.N*self.Nbins*sizeof(DTYPE_t))
        self.tmp_err = <DTYPE_t*>PyMem_Malloc(self.N*self.Nbins*sizeof(DTYPE_t))
        self.tmp_sizes = <ITYPE_t*>PyMem_Malloc(self.Nbins*sizeof(ITYPE_t))
        if not self.mjd:
            raise MemoryError()
        if not self.phase:
            raise MemoryError()
        if not self.mag:
            raise MemoryError()
        if not self.err:
            raise MemoryError()
        if not self.tmp_mag:
            raise MemoryError()
        if not self.tmp_err:
            raise MemoryError()
        if not self.tmp_sizes:
            raise MemoryError()
        for i in range(self.N):
            self.mjd[i] = mjd[i]
            self.mag[i] = mag[i]
            self.err[i] = err[i]
        self.normalizer = (self.N - Nbins)/(Nbins-1)
        if self.use_errorbars:
            self.barx = weighted_mean(self.mag, self.err, self.N)
        else:
            self.barx = mean(self.mag, self.N)

    def eval_frequency(self, DTYPE_t freq):
        # More memory consuming but almost twice as fast!
        cdef Py_ssize_t i, j
        cdef DTYPE_t one_float = 1.0
        for i in range(self.N):
            self.phase[i] = fmodf(self.mjd[i], one_float/freq)*freq  # output in [0.0, 1.0]
        cdef DTYPE_t num=0.0, den=0.0
        cdef ITYPE_t samples_in_bin
        cdef DTYPE_t barxi
        
        for j in range(self.Nbins):
            self.tmp_sizes[j] = 0
            
        for i in range(self.N):
            j = (int)(floorf(self.phase[i]*self.Nbins))
            self.tmp_mag[self.tmp_sizes[j] + self.N*j] = self.mag[i]
            self.tmp_err[self.tmp_sizes[j] + self.N*j] = self.err[i]
            self.tmp_sizes[j] += 1 
            
        for j in range(self.Nbins):
            if self.tmp_sizes[j] > 1:
                if self.use_errorbars:
                    barxi = weighted_mean((self.tmp_mag + self.N*j), 
                                          (self.tmp_err + self.N*j),
                                          self.tmp_sizes[j])
                else:
                    barxi = mean((self.tmp_mag + self.N*j), self.tmp_sizes[j])
                for i in range(self.tmp_sizes[j]):
                    den += (self.tmp_mag[i + self.N*j] - barxi)**2
                num += self.tmp_sizes[j]*(barxi - self.barx)**2
                
        return self.normalizer*num/den
    
    def eval_frequency_old(self, DTYPE_t freq):
        cdef Py_ssize_t i, j
        cdef DTYPE_t one_float = 1.0
        for i in range(self.N):
            self.phase[i] = fmodf(self.mjd[i], one_float/freq)*freq  # output in [0.0, 1.0]
        cdef DTYPE_t num=0.0, den=0.0 # These are the unormalized s1 and s2 in the paper
        cdef ITYPE_t samples_in_bin
        cdef DTYPE_t barxi
        for j in range(self.Nbins):
            samples_in_bin = 0
            for i in range(self.N):
                if floorf(self.phase[i]*self.Nbins) == j:
                    self.tmp_mag[samples_in_bin] = self.mag[i]
                    self.tmp_err[samples_in_bin] = self.err[i]
                    samples_in_bin += 1

            if samples_in_bin > 1:
                if self.use_errorbars:
                    barxi = weighted_mean(self.tmp_mag, self.tmp_err, samples_in_bin)
                else:
                    barxi = mean(self.tmp_mag, samples_in_bin)
                for i in range(samples_in_bin):
                    den += (self.tmp_mag[i] - barxi)**2
                num += samples_in_bin*(barxi - self.barx)**2
                
        return self.normalizer*num/den

    def __dealloc__(self):
        PyMem_Free(self.mjd)
        PyMem_Free(self.mag)
        PyMem_Free(self.err)
        PyMem_Free(self.phase)
        PyMem_Free(self.tmp_mag)
        PyMem_Free(self.tmp_err)
        PyMem_Free(self.tmp_sizes)
