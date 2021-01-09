cimport cython
import numpy as np
import ctypes
from numpy import int32, float, double
from numpy cimport int32_t, float_t, double_t

# These cython declarations must be equivalent to C declarations and structures in src/cyInterface.h
cdef extern from "src/cyInterface.h":
    # Define cython data structure
    struct flat_array_2D:
        int NRows;
        int NCols;
        float *data_pt;  # Pointer to 1D contiguous array used by python

    # Define cython function
    int interface_matrix_multiplication( flat_array_2D *A, flat_array_2D *B, flat_array_2D *C );

    