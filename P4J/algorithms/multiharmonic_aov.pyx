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

cdef class MHAOV:
    cdef Py_ssize_t N
    cdef DTYPE_t* mjd
    cdef DTYPE_t* mag_minus_wmean
    cdef DTYPE_t* err
    cdef ITYPE_t Nharmonics
    cdef ITYPE_t mode
    cdef DTYPE_t d1, d2
    cdef public DTYPE_t wmean, wvar
    cdef DTYPE_t* zr 
    cdef DTYPE_t* zi
    cdef DTYPE_t* znr
    cdef DTYPE_t* zni
    cdef DTYPE_t* pr
    cdef DTYPE_t* pi
    cdef DTYPE_t* cfr
    cdef DTYPE_t* cfi

    def __init__(self, DTYPE_t [::1] mjd, DTYPE_t [::1] mag, DTYPE_t [::1] err, ITYPE_t Nharmonics=1, ITYPE_t mode=1):
        cdef Py_ssize_t i
        if Nharmonics < 1:
            raise ValueError("Number of harmonics has to be greater or equal to 1")
        self.Nharmonics = Nharmonics
        self.N = mag.shape[0]
        self.mjd = allocate_and_verify(self.N)
        self.mag_minus_wmean = allocate_and_verify(self.N)
        self.err = allocate_and_verify(self.N)
        self.zr = allocate_and_verify(self.N)
        self.zi = allocate_and_verify(self.N)
        self.znr = allocate_and_verify(self.N)
        self.zni = allocate_and_verify(self.N)
        self.pr = allocate_and_verify(self.N)
        self.pi = allocate_and_verify(self.N)
        self.cfr = allocate_and_verify(self.N)
        self.cfi = allocate_and_verify(self.N)
        
        self.wmean = weighted_mean(&mag[0], &err[0], self.N)

        for i in range(self.N):
            self.mjd[i] = mjd[i]
            self.mag_minus_wmean[i] = mag[i] - self.wmean
            self.err[i] = err[i]

        self.d1 = Nharmonics*2.0
        self.d2 = self.N - Nharmonics*2 - 1
        self.wvar = 0.0
        for i in range(self.N):
            self.wvar += powf(self.mag_minus_wmean[i], 2.)/powf(self.err[i], 2.)
            
        self.mode = mode # 0: RAW, 1: F
        

    def eval_frequency(self, DTYPE_t freq):
        cdef Py_ssize_t i, j
        cdef DTYPE_t sn, alr, ali, scr, sci
        cdef DTYPE_t aov=0.0
        cdef DTYPE_t arg, tmp, sr, si
        cdef DTYPE_t two_float = 2.0
        cdef DTYPE_t one_float = 1.0
        
        for i in range(self.N):
            arg = self.mjd[i]*freq
            arg = two_float*M_PI*(arg - floorf(arg))

            # z = exp(j*arg), complex exp with f=freq eval at times mjd[i]
            self.zr[i] = cosf(arg)
            self.zi[i] = sinf(arg)

            # zn = 1, bias basis?
            self.znr[i] = 1.
            self.zni[i] = 0.

            # p = 1/err
            self.pr[i] = one_float/self.err[i]
            self.pi[i] = 0.

            # cf = (mag-wmean)*exp(j*n_harmonics*arg)/err
            self.cfr[i] = self.mag_minus_wmean[i]*cosf(self.Nharmonics*arg)/self.err[i]
            self.cfi[i] = self.mag_minus_wmean[i]*sinf(self.Nharmonics*arg)/self.err[i]
        for j in range(2*self.Nharmonics+1):
            sn = alr = ali = scr = sci = 0.0
            for i in range(self.N):
                # += |p|^2
                sn += self.pr[i]**2 + self.pi[i]**2

                # al += z*p/err
                alr += (self.zr[i]*self.pr[i] - self.zi[i]*self.pi[i])/self.err[i]
                ali += (self.zr[i]*self.pi[i] + self.zi[i]*self.pr[i])/self.err[i]

                # sc += conj(p)*cr
                scr += self.pr[i]*self.cfr[i] + self.pi[i]*self.cfi[i]
                sci += self.pr[i]*self.cfi[i] - self.pi[i]*self.cfr[i]
            sn = max(sn, 1e-9)

            # al = al/sn
            alr = alr/sn
            ali = ali/sn

            # aov += |sc|^2 / sn
            aov += (scr**2 + sci**2)/sn
            for i in range(self.N):
                # s = al*zn
                sr = alr*self.znr[i] - ali*self.zni[i]
                si = alr*self.zni[i] + ali*self.znr[i]


                # updating p = p*z - s*conj(p)
                # tmp = re(p*z)-re(s*conj(p))
                tmp = self.pr[i]*self.zr[i] - self.pi[i]*self.zi[i] - sr*self.pr[i] - si*self.pi[i]
                # im(p) = im(p*z)-im(s*conj(p))
                self.pi[i] = self.pr[i]*self.zi[i] + self.pi[i]*self.zr[i] + sr*self.pi[i] - si*self.pr[i]
                self.pr[i] = tmp

                # updating zn = zn * z
                tmp = self.znr[i]*self.zr[i] - self.zni[i]*self.zi[i]
                self.zni[i] = self.zni[i]*self.zr[i] + self.znr[i]*self.zi[i]
                self.znr[i] = tmp
        if self.mode == 0:
            return aov
        elif self.mode == 1:
            return (self.d2/self.d1)*aov/max(self.wvar - aov, 1e-9)
            
    def __dealloc__(self):
        PyMem_Free(self.mjd)
        PyMem_Free(self.mag_minus_wmean)
        PyMem_Free(self.err)
        PyMem_Free(self.zr)
        PyMem_Free(self.zi)
        PyMem_Free(self.znr)
        PyMem_Free(self.zni)
        PyMem_Free(self.pr)
        PyMem_Free(self.pi)
        PyMem_Free(self.cfr)
        PyMem_Free(self.cfi)



