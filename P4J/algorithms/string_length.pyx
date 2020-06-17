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
    DTYPE_t fmodf(DTYPE_t, DTYPE_t)

"""

Lafler Kinman String/Rope Length implementation
Ref: http://www.aanda.org/articles/aa/full/2002/17/aa2208/aa2208.html

The original method is modified to include uncertainties, the variance in
the denominator is replaced by the unbiased weighted variance, and the euclidean 
distance between sorted samples is replaced by a weighted version, i.e.
 sum[m[1:N] - m[0:N-1)**2/(e[1:N]**2 + e[0:N-1]**2)]/sum[1.0/(e[1:N]**2 + e[0:N-1]**2)]

Please cite the paper above if using this code
"""

cdef class LKSL:
    cdef Py_ssize_t N
    cdef DTYPE_t* phase
    cdef DTYPE_t* mjd
    cdef DTYPE_t* mag
    cdef DTYPE_t* err2
    cdef DTYPE_t normalizer
    cdef ITYPE_t* sorted_idx
    def __init__(self, DTYPE_t [::1] mjd, DTYPE_t [::1] mag, DTYPE_t [::1] err):
        cdef Py_ssize_t i, j, mat_idx
        self.N = mag.shape[0]
        self.mjd = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.mag = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.err2 = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.phase = <DTYPE_t*>PyMem_Malloc(self.N*sizeof(DTYPE_t))
        self.sorted_idx = <ITYPE_t*>PyMem_Malloc(self.N*sizeof(ITYPE_t))
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
        for i in range(self.N):
            self.mjd[i] = mjd[i]
            self.mag[i] = mag[i]
            self.err2[i] = powf(err[i], 2.0)
        self.normalizer = 1.0/unbiased_weighted_variance(self.mag, self.err2, self.N)

    def eval_frequency(self, DTYPE_t freq):
        cdef Py_ssize_t i, j
        for i in range(self.N):
            self.phase[i] = fmodf(self.mjd[i], 1.0/freq)*freq  # output in [0.0, 1.0]
        argsort(self.phase, self.sorted_idx, self.N)
        cdef DTYPE_t err2_err2 = self.err2[self.sorted_idx[0]] + self.err2[self.sorted_idx[self.N-1]]
        cdef DTYPE_t err2_acum = 1.0/err2_err2
        cdef DTYPE_t SL = powf(self.mag[self.sorted_idx[0]] - self.mag[self.sorted_idx[self.N-1]], 2.0)/err2_err2
        for i in range(1, self.N):
            err2_err2 = self.err2[self.sorted_idx[i-1]] + self.err2[self.sorted_idx[i]]
            err2_acum += 1.0/err2_err2
            SL += powf(self.mag[self.sorted_idx[i-1]] - self.mag[self.sorted_idx[i]], 2.0)/err2_err2
        return 0.5*SL*self.normalizer/err2_acum

    def __dealloc__(self):
        PyMem_Free(self.mjd)
        PyMem_Free(self.mag)
        PyMem_Free(self.err2)
        PyMem_Free(self.phase)
        PyMem_Free(self.sorted_idx)


