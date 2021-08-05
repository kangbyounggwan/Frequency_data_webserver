# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from sklearn.tree._tree cimport INT32_t          # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t         # Unsigned 32 bit integer

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._splitter cimport Splitter, SplitRecord

cdef class RobustRandomCutSplitter(Splitter):

    cdef const DTYPE_t[:, :] X

    cdef DTYPE_t* cumsum_feature_values

cdef class InsertPointSplitter(Splitter):

    cdef const DTYPE_t[:, :] X
    cdef const DTYPE_t[:] point

    cdef DTYPE_t* cumsum_feature_values

    cdef int point_reset(self, DTYPE_t[:] point) nogil except -1
