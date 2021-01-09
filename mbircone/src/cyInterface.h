#ifndef _CY_INTERFACE_H_
#define _CY_INTERFACE_H_

#include <stdio.h>

struct flat_array_2D
{
    int NRows;
    int NCols;
    float *data_pt;  /* Pointer to 1D contiguous array used by cython to pass array by reference */
};

int interface_matrix_multiplication(struct flat_array_2D *A_flat, struct flat_array_2D *B_flat , struct flat_array_2D *C_flat);
void allocate_matrix_from_flattened(struct matrix_float *mat, struct flat_array_2D *array);
void deallocate_matrix_to_flattened(struct matrix_float *mat);

#endif /* _CY_INTERFACE_H_ */