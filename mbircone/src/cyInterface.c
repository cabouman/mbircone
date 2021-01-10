
#include "allocate.h"
#include "cyInterface.h"


int writeSysMatrix(char *fName, double *angles, struct SinoParams *sinoParams, struct ImageParams *imgParams)
{
    struct SysMatrix A;
    struct ViewAngleList viewAngleList;

    computeSysMatrix(sinoParams, &imgParams, &A, &viewAngleList);
    
    printSysMatrixParams(&A);
    writeSysMatrix(pathNames.sysMatrix, &sino.params, &img.params, &A);

    return 0;
}


void allocate_3darray_from_flattened(struct matrix_float *A, struct flat_array_2D *array)
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


void deallocate_3darray_to_flattened(struct matrix_float *A)
{
    free((void **)A->mat);
}