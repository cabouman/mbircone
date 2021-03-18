
#include "allocate.h"
#include "MBIRModularUtilities3D.h"
#include "cyInterface.h"
#include "computeSysMatrix.h"


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
	float ***img;
	int i_x, i_y, i_z, i;


	for(i=0; i<imgParams.N_x*imgParams.N_y*imgParams.N_z; i++){
		x[i] = x_init[i];
	}


	// for(i_x=0; i_x<imgParams.N_x; i_x++){
	// 	for(i_y=0; i_y<imgParams.N_x; i_y++){
	// 		for(i_z=0; i_z<imgParams.N_x; i_z++){
	// 			img[i_x][i_y][i_z] = img_init[i_x][i_y][i_z];
	// 		}
	// 	}
	// }

	// memcpy(x, x_init, imgParams.N_x*imgParams.N_y*imgParams.N_z*sizeof(float*));

	img = (float ***)mem_alloc_float3D_from_flat(x, imgParams.N_x, imgParams.N_y, imgParams.N_z);

	printf("C: \n");
	printf("x_init: %f\n", x_init[0]);
	printf("x: %f\n", x[0]);

	mem_free_2D((void**)img);
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

