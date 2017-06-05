#!/usr/bin/python
#cython: initializedcheck=False, boundscheck=False, wraparound=False, cdivision=True, profile=False

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "math.h":
    DTYPE_t powf(DTYPE_t, DTYPE_t)

"""

Computes an unbiased estimator of the weighted variance, where w_i = (1.0/e_i**2)

"""

cdef DTYPE_t weighted_mean(DTYPE_t* data, DTYPE_t* err, Py_ssize_t N):
    cdef DTYPE_t w_mean = 0.0
    cdef Py_ssize_t i
    cdef DTYPE_t w_sum = 0.0
    for i in range(N):
        w_sum += 1.0/powf(err[i], 2.0)
        w_mean += data[i]/powf(err[i], 2.0)
    return w_mean/w_sum

cdef DTYPE_t unbiased_weighted_variance(DTYPE_t* data, DTYPE_t* err2, Py_ssize_t N):
    cdef DTYPE_t w_mean = 0.0
    cdef DTYPE_t w_var = 0.0
    cdef DTYPE_t V1 = 0.0, V2 = 0.0
    cdef Py_ssize_t i
    for i in range(N):
        V1 += 1.0/err2[i]
        V2 += 1.0/powf(err2[i], 2.0)
        w_mean += data[i]/err2[i]
    for i in range(N):
        w_var += powf(data[i] - w_mean/V1, 2.0)/err2[i]
    return w_var/(V1 - V2/V1)


"""

Argsort implementation
Credit: https://github.com/jcrudy/cython-argsort/blob/master/cyargsort/argsort.pyx

"""
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct Sorter:
    Py_ssize_t index
    DTYPE_t value

cdef int _compare(const_void *a, const_void *b):
    cdef DTYPE_t v = ((<Sorter*>a)).value-((<Sorter*>b)).value
    if v < 0: return -1
    if v >= 0: return 1

cdef void cyargsort(DTYPE_t* data, Sorter * order, Py_ssize_t N):
    cdef Py_ssize_t i
    for i in range(N):
        order[i].index = i
        order[i].value = data[i]
    qsort(<void *> order, N, sizeof(Sorter), _compare)

cdef void argsort(DTYPE_t* data, ITYPE_t* order, Py_ssize_t N):
    cdef Py_ssize_t i
    cdef Sorter *order_struct = <Sorter *> PyMem_Malloc(N*sizeof(Sorter))
    cyargsort(data, order_struct, N)
    for i in range(N):
        order[i] = order_struct[i].index
    PyMem_Free(order_struct)

