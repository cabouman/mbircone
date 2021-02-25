
#include "allocate.h"
#include "MBIRModularUtilities3D.h"
#include "cyInterface.h"


void AmatrixComputeToFile(double *angles, 
	struct SinoParams sinoParams, struct ImageParams imgParams, 
	char *Amatrix_fname, char verbose)
{
    struct SysMatrix A;
    struct ViewAngleList viewAngleList;

    viewAngleList.beta = angles;

    computeSysMatrix(&sinoParams, &imgParams, &A, &viewAngleList);
    
    if(verbose){
    	printSysMatrixParams(&A);
    }

    writeSysMatrix(Amatrix_fname, &sinoParams, &imgParams, &A);

}

void recon(float *x, float *sino, float *wght, float *x_init, float *proxmap_input,
	struct SinoParams sinoParams, struct ImageParams imgParams, struct ReconParams reconParams, 
	char *Amatrix_fname)
{
	/*float ***img;

	img = (float ***)mem_alloc_float3D_from_flat(x, ImageParams.N_x, ImageParams.N_y, ImageParams.N_x)

	mem_free_2D((void**)img);*/
	printf("initReconMode: %s\n",reconParams.initReconMode);
}

void ***mem_alloc_float3D_from_flat(float *dataArray, size_t N1, size_t N2, size_t N3)
{
	void ***topTree;
	long int i1, i2;

	topTree = (void***)mem_alloc_2D(N1, N2, sizeof(void*));

	for(i1 = 0; i1 < N1; i1++)
	for(i2 = 0; i2 < N2; i2++)
	{
		topTree[i1][i2] = dataArray + (i1*N2 + i2)*N3 * sizeof(float);
	}

	return (topTree);	
}

