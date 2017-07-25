#!/usr/bin/python
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True, profile=False

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from .utilities cimport argsort, unbiased_weighted_variance

ctypedef float DTYPE_t
ctypedef int ITYPE_t

cdef extern from "math.h":
    DTYPE_t sqrtf(DTYPE_t)
    DTYPE_t powf(DTYPE_t, DTYPE_t)
    DTYPE_t floorf(DTYPE_t)
    DTYPE_t fmodf(DTYPE_t, DTYPE_t)

"""

Phase Dispersion Minimization
Ref: http://adsabs.harvard.edu/abs/1978ApJ...224..953S

This implementation follows the reference except that variance calculations
are replaced by an unbiased weighted variance to include uncertainties.

There is also the PDM2 method which fits splines to the binned data
http://www.stellingwerf.com/rfs-bin/index.cgi?action=PageView&id=29
"""
cdef class PDM:
    cdef Py_ssize_t N
    cdef DTYPE_t* phase
    cdef DTYPE_t* mjd
    cdef DTYPE_t* mag
    cdef DTYPE_t* err2
    cdef DTYPE_t normalizer
    cdef ITYPE_t* sorted_idx
    cdef ITYPE_t Nbins
    cdef DTYPE_t* tmp_mag
    cdef DTYPE_t* tmp_err2
    def __init__(self, DTYPE_t [::1] mjd, DTYPE_t [::1] mag, DTYPE_t [::1] err, ITYPE_t Nbins=10):
        cdef Py_ssize_t i, j, mat_idx
        self.Nbins = Nbins
        self.N = mag.shape[0]
        self.mjd = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.mag = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.err2 = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.phase = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.sorted_idx = <ITYPE_t*>PyMem_Malloc(self.N*sizeof(ITYPE_t))
        self.tmp_mag = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.tmp_err2 = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        if not self.mjd:
            raise MemoryError()
        if not self.phase:
            raise MemoryError()
        if not self.mag:
            raise MemoryError()
        if not self.err2:
            raise MemoryError()
        if not self.sorted_idx:
            raise MemoryError()
        if not self.tmp_mag:
            raise MemoryError()
        if not self.tmp_err2:
            raise MemoryError()
        for i in range(self.N):
            self.mjd[i] = mjd[i]
            self.mag[i] = mag[i]
            self.err2[i] = powf(err[i], 2.0)
        self.normalizer = 1.0/unbiased_weighted_variance(self.mag, self.err2, self.N)

    def eval_frequency(self, DTYPE_t freq):
        cdef Py_ssize_t i, j
        for i in range(self.N):
            self.phase[i] = fmodf(self.mjd[i], 1.0/freq)*freq  # output in [0.0, 1.0]
        # argsort(self.phase, self.sorted_idx, self.N)
        cdef DTYPE_t PDM_num=0.0, PDM_den=0.0
        cdef ITYPE_t samples_in_bin
        cdef DTYPE_t V1, V2
        for j in range(self.Nbins):
            samples_in_bin = 0
            V1 = V2 = 0.0
            for i in range(self.N):
                if floorf(self.phase[i]*self.Nbins) == j:
                    self.tmp_mag[samples_in_bin] = self.mag[i]
                    self.tmp_err2[samples_in_bin] = self.err2[i]
                    samples_in_bin += 1
                    V1 += 1.0/self.err2[i]
                    V2 += 1.0/powf(self.err2[i], 2.0)
            if samples_in_bin > 2:
                PDM_num += unbiased_weighted_variance(self.tmp_mag, self.tmp_err2, samples_in_bin)*(V1 - V2/V1)
                PDM_den += (V1 - V2/V1)

        return PDM_num*self.normalizer/PDM_den

    def __dealloc__(self):
        PyMem_Free(self.mjd)
        PyMem_Free(self.mag)
        PyMem_Free(self.err2)
        PyMem_Free(self.phase)
        PyMem_Free(self.sorted_idx)
        PyMem_Free(self.tmp_mag)
        PyMem_Free(self.tmp_err2)


