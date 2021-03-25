
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
	struct Sino sino;
    struct Image img;
	int i_x, i_y, i_z, i;

	/* Initial value */
	for(i=0; i<imgParams.N_x*imgParams.N_y*imgParams.N_z; i++){
		x[i] = x_init[i];
	}

	
	sino.estimateSino = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
    sino.e = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));

    /**
     *      Allocate space for image
     */
    img.wghtRecon = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
    img.proxMapInput = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
    img.lastChange = (float***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(float));
    img.timeToChange = (unsigned char***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(unsigned char));


	img.vox = (float ***)mem_alloc_float3D_from_flat(x, imgParams.N_x, imgParams.N_y, imgParams.N_z);

	mem_free_2D((void**)img.vox);

	mem_free_3D((void***)img.proxMapInput);
    mem_free_3D((void***)img.lastChange);
    mem_free_3D((void***)img.timeToChange);

    mem_free_3D((void***)sino.e);
    mem_free_3D((void***)sino.estimateSino);
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

