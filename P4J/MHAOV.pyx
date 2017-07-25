#!/usr/bin/python
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True, profile=False

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from .utilities cimport weighted_mean

ctypedef float DTYPE_t
ctypedef int ITYPE_t

cdef extern from "math.h":
    DTYPE_t sqrtf(DTYPE_t)
    DTYPE_t powf(DTYPE_t, DTYPE_t)
    DTYPE_t floorf(DTYPE_t)
    DTYPE_t cosf(DTYPE_t)
    DTYPE_t sinf(DTYPE_t)


"""

Multiharmonic weighted AoV Periodogram
Ref: http://adsabs.harvard.edu/abs/1996ApJ...460L.107S

This implementation follows the c code found at
http://users.camk.edu.pl/alex/soft/aovgui.tgz

Please cite the paper above if using this code
"""

cdef DTYPE_t M_PI = 3.1415926535897

cdef DTYPE_t* allocate_and_verify(Py_ssize_t N):
    cdef DTYPE_t* array = <DTYPE_t*>PyMem_Malloc(N*sizeof(DTYPE_t))
    if not array:
        raise MemoryError()
    return array

cdef class AOV:
    cdef Py_ssize_t N
    cdef DTYPE_t* mjd
    cdef DTYPE_t* mag
    cdef DTYPE_t* err
    cdef ITYPE_t Nharmonics
    cdef DTYPE_t d1, d2
    cdef DTYPE_t wmean, wvar
    cdef DTYPE_t* zr 
    cdef DTYPE_t* zi
    cdef DTYPE_t* znr
    cdef DTYPE_t* zni
    cdef DTYPE_t* pr
    cdef DTYPE_t* pi
    cdef DTYPE_t* cfr
    cdef DTYPE_t* cfi

    def __init__(self, DTYPE_t [::1] mjd, DTYPE_t [::1] mag, DTYPE_t [::1] err, ITYPE_t Nharmonics=1):
        cdef Py_ssize_t i
        if Nharmonics < 1:
            raise ValueError("Number of harmonics has to be greater or equal to 1")
        self.Nharmonics = Nharmonics
        self.N = mag.shape[0]
        self.mjd = allocate_and_verify(self.N)
        self.mag = allocate_and_verify(self.N)
        self.err = allocate_and_verify(self.N)
        self.zr = allocate_and_verify(self.N)
        self.zi = allocate_and_verify(self.N)
        self.znr = allocate_and_verify(self.N)
        self.zni = allocate_and_verify(self.N)
        self.pr = allocate_and_verify(self.N)
        self.pi = allocate_and_verify(self.N)
        self.cfr = allocate_and_verify(self.N)
        self.cfi = allocate_and_verify(self.N)
        
        for i in range(self.N):
            self.mjd[i] = mjd[i]
            self.mag[i] = mag[i]
            self.err[i] = err[i]
        
        self.wmean = weighted_mean(self.mag, self.err, self.N)
        self.d1 = Nharmonics*2.0
        self.d2 = self.N - Nharmonics*2 - 1
        self.wvar = 0.0
        for i in range(self.N):
            self.wvar += powf(self.mag[i] - self.wmean, 2.)/powf(self.err[i], 2.)
        

    def eval_frequency(self, DTYPE_t freq):
        cdef Py_ssize_t i, j
        cdef DTYPE_t sn, alr, ali, scr, sci
        cdef DTYPE_t aov=0.0
        cdef DTYPE_t arg, tmp, sr, si
        for i in range(self.N):
            arg = self.mjd[i]*freq
            arg = 2.0*M_PI*(arg - floorf(arg))
            self.zr[i] = cosf(arg)
            self.zi[i] = sinf(arg)
            self.znr[i] = 1.
            self.pr[i] = 1./self.err[i]
            self.zni[i] = 0.
            self.pi[i] = 0.
            self.cfr[i] = (self.mag[i] - self.wmean)*cosf(self.Nharmonics*arg)/self.err[i]
            self.cfi[i] = (self.mag[i] - self.wmean)*sinf(self.Nharmonics*arg)/self.err[i]
        for j in range(2*self.Nharmonics+1):
            sn = alr = ali = scr = sci = 0.0
            for i in range(self.N):
                sn += self.pr[i]**2 + self.pi[i]**2
                alr += (self.zr[i]*self.pr[i] - self.zi[i]*self.pi[i])/self.err[i]
                ali += (self.zr[i]*self.pi[i] + self.zi[i]*self.pr[i])/self.err[i]
                scr += self.pr[i]*self.cfr[i] + self.pi[i]*self.cfi[i]
                sci += self.pr[i]*self.cfi[i] - self.pi[i]*self.cfr[i]
            if sn > 1e-31:
                alr = alr/sn
                ali = ali/sn
            else:
                alr = alr/1e-31
                ali = ali/1e-31
            aov += (scr**2 + sci**2)/sn
            for i in range(self.N):
                sr = alr*self.znr[i] - ali*self.zni[i]
                si = alr*self.zni[i] + ali*self.znr[i]
                tmp = self.pr[i]*self.zr[i] - self.pi[i]*self.zi[i] - sr*self.pr[i] - si*self.pi[i]
                self.pi[i] = self.pr[i]*self.zi[i] + self.pi[i]*self.zr[i] + sr*self.pi[i] - si*self.pr[i]
                self.pr[i] = tmp
                tmp = self.znr[i]*self.zr[i] - self.zni[i]*self.zi[i]
                self.zni[i] = self.zni[i]*self.zr[i] + self.znr[i]*self.zi[i]
                self.znr[i] = tmp
        if self.wvar - aov > 1e-32:
            return self.d2*aov/(self.d1*(self.wvar - aov))
        else:
            return self.d2*aov/(self.d1*1e-32)

    def __dealloc__(self):
        PyMem_Free(self.mjd)
        PyMem_Free(self.mag)
        PyMem_Free(self.err)
        PyMem_Free(self.zr)
        PyMem_Free(self.zi)
        PyMem_Free(self.znr)
        PyMem_Free(self.zni)
        PyMem_Free(self.pr)
        PyMem_Free(self.pi)
        PyMem_Free(self.cfr)
        PyMem_Free(self.cfi)



