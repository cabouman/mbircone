
#include "allocate.h"
#include "interface.h"

int interface_matrix_multiplication(struct flat_array_2D *A_flat, struct flat_array_2D *B_flat , struct flat_array_2D *C_flat)
{

	struct matrix_float A,B,C;

	/* Allocate 2D multi-alloc pointer array from cython flattened array */
    allocate_matrix_from_flattened(&A, A_flat);
    allocate_matrix_from_flattened(&B, B_flat);
    allocate_matrix_from_flattened(&C, C_flat);

    /* Original c code for matrix multiplication that uses multi-alloc array*/

    /* Deallocate multi-alloc pointer array */
    deallocate_matrix_to_flattened(&A);
    deallocate_matrix_to_flattened(&B);
    deallocate_matrix_to_flattened(&C);

    return 0;
}


void allocate_matrix_from_flattened(struct matrix_float *A, struct flat_array_2D *array)
{
    int i;

    A->NRows = array->NRows;
    A->NCols = array->NCols;

    /* Allocate and set array of pointers for multialloc array */
    A->mat = get_spc(sizeof(float *), A->NRows);
    for (i = 0; i < A->NRows ; i ++ ) {
        A->mat[i] = array->data_pt + i*(A->NCols);
    }
}


void deallocate_matrix_to_flattened(struct matrix_float *A)
{
    free((void **)A->mat);
}