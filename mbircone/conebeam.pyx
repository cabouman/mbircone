import numpy as np
import ctypes           # Import python package required to use cython
cimport cython          # Import cython package
cimport numpy as cnp    # Import specialized cython support for numpy

# This imports functions and data types from the matrices.pxd file in the same directory
from matrices cimport flat_array_2D, interface_matrix_multiplication


@cython.boundscheck(False)      # Deactivate bounds checking to increase speed
@cython.wraparound(False)       # Deactivate negative indexing to increase speed

def cython_matrix_multiplication(cnp.ndarray py_a, cnp.ndarray py_b):
    """
    Cython function that multiplies two single precision float matrices

    Args:
        py_a(float): 2D numpy float array with C continuous order, the left matrix A.
        py_b(float): 2D numpy float array with C continuous order, the right matrix B.

    Return:
        py_c: 2D numpy float array that is the product of A and B.
    """

    # Get shapes of A and B
    nrows_a, ncols_a = np.shape(py_a)
    nrows_b, ncols_b = np.shape(py_b)

    if (not py_a.flags["C_CONTIGUOUS"]) or (not py_b.flags["C_CONTIGUOUS"]):
        raise AttributeError("2D np.ndarrays must be C-contiguous")

    if (ncols_a != nrows_b):
        raise AttributeError("Matrix shapes are not compatible")

    # Set output matrix shape
    nrows_c = nrows_a
    ncols_c = ncols_b

    cdef cnp.ndarray[float, ndim=2, mode="c"] cy_a = py_a
    cdef cnp.ndarray[float, ndim=2, mode="c"] cy_b = py_b

    # Allocates memory, without initialization, for matrix to be passed back from C subroutine
    cdef cnp.ndarray[float, ndim=2, mode="c"] py_c = np.empty((nrows_a,ncols_b), dtype=ctypes.c_float)

    # Declare and initialize 3 matrices
    cdef flat_array_2D A     # Allocate C data structure matrix
    A.data_pt = &cy_a[0, 0]  # Assign pointer in C data structure
    A.NRows = nrows_a       # Set value of NRows in C data structure
    A.NCols = ncols_a       # Set value of NCols in C data structure

    cdef flat_array_2D B
    B.data_pt = &cy_b[0, 0]
    B.NRows = nrows_b
    B.NCols = ncols_b

    cdef flat_array_2D C
    C.data_pt = &py_c[0, 0]
    C.NRows = nrows_c
    C.NCols = ncols_c

    # Multiply matrices together by calling C subroutine
    interface_matrix_multiplication(&A, &B, &C)

    # Return cython ndarray
    return py_c
