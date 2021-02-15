
#include "allocate.h"
#include "MBIRModularUtilities3D.h"
#include "cyInterface.h"


void AmatrixComputeToFile(double *angles, struct SinoParams sinoParams, struct ImageParams imgParams, char *Amatrix_fname, char verbose)
{
    struct SysMatrix A;
    struct ViewAngleList viewAngleList;

    viewAngleList.beta = angles;

    computeSysMatrix(&sinoParams, &imgParams, &A, &viewAngleList);

    printf("verbose: %d\n",verbose );
    
    if(verbose){
    	printSysMatrixParams(&A);
    }

    writeSysMatrix(Amatrix_fname, &sinoParams, &imgParams, &A);

}


// void allocate_3darray_from_flattened(struct matrix_float *A, struct flat_array_2D *array)
// {
//     int i;

//     A->NRows = array->NRows;
//     A->NCols = array->NCols;

//     /* Allocate and set array of pointers for multialloc array */
//     A->mat = get_spc(sizeof(float *), A->NRows);
//     for (i = 0; i < A->NRows ; i ++ ) {
//         A->mat[i] = array->data_pt + i*(A->NCols);
//     }
// }


// void deallocate_3darray_to_flattened(struct matrix_float *A)
// {
//     free((void **)A->mat);
// }
