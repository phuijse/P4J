
ctypedef float DTYPE_t
ctypedef int ITYPE_t

cdef void argsort(DTYPE_t*, ITYPE_t*, Py_ssize_t)
cdef DTYPE_t weighted_mean(DTYPE_t*, DTYPE_t*, Py_ssize_t)
cdef DTYPE_t unbiased_weighted_variance(DTYPE_t*, DTYPE_t*, Py_ssize_t)

