import numpy as np
from cython_sandbox import cython_matrix_multiplication
import time

if __name__ == '__main__':
    # Generate random matrices A and B of compatible sizes
    A = np.random.randint(10, size=(1000, 500)).astype(np.float32)
    B = np.random.randint(10, size=(500, 10)).astype(np.float32)

    time1 = time.time()
    # Compute matrix multiplication using cython wrapper
    C1 = cython_matrix_multiplication(A, B)     # Requires that 2D np.ndarrays that are floats with C contiguous format
    time_diff1 = time.time()-time1
    print(f"The time for Cython execution is = {time_diff1}")
    print("Output from cython matrix multiplication:")
    print(C1)
    print(f"\n")

    time1 = time.time()
    # Compute matrix multiplication using numpy
    C2 = np.dot(A, B)
    time_diff1 = time.time()-time1
    print(f"The time for numpy execution is = {time_diff1}")
    print("Output from numpy matrix multiplication:")
    print(C2)
    print(f"\n")

    # Print error
    err = np.sum((C1 - C2) ** 2)
    print("L2 difference between cython and numpy.dot matrix product: %f" % err)