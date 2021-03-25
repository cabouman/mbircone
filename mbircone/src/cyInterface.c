
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

void recon(float *x, float *y, float *wght, float *x_init, float *proxmap_input,
	struct SinoParams sinoParams, struct ImageParams imgParams, struct ReconParams reconParams, 
	char *Amatrix_fname)
{
	struct Sino sino;
    struct Image img;
    struct SysMatrix A;
	int i_x, i_y, i_z, i;
	

	/* Set img and sino params inside data structure */
	copySinoParams(&imgParams, &img.params);
	copySinoParams(&sinoParams, &sino.params);


	/* Allocate 3D image from 1D vector */
	img.vox = (float ***)mem_alloc_float3D_from_flat(x, imgParams.N_x, imgParams.N_y, imgParams.N_z);

	/* Allocate error sinogram */
    sino.e = (float***)allocateSinoData3DCone(&sino.params, sizeof(float));

	/* Allocate other image data */
    img.wghtRecon = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
    img.proxMapInput = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
    img.lastChange = (float***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(float));
    img.timeToChange = (unsigned char***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(unsigned char));


    /* Initialize image */
    for(i=0; i<imgParams.N_x*imgParams.N_y*imgParams.N_z; i++){
		x[i] = x_init[i];
	}
	applyMask(img.vox, img.params.N_x, img.params.N_y, img.params.N_z);

     /* Initialize error sinogram e = y - Ax */
    forwardProject3DCone( sino.e, img.vox, &img.params, &A, &sino.params); /* e = Ax */
    floatArray_z_equals_aX_plus_bY(&sino.e[0][0][0], 1.0, &sino.vox[0][0][0], -1.0, &sino.e[0][0][0], sino.params.N_beta*sino.params.N_dv*sino.params.N_dw); /* e = 1.0 * y + (-1.0) * e */

    /* Initialize other image data */
    initializeWghtRecon(&A, &sino, &img, &reconParams);
    setFloatArray2Value(&img.proxMapInput[0][0][0], img.params.N_x*img.params.N_y*img.params.N_z, 0.0);
    setFloatArray2Value(&img.lastChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0.0);
    setUCharArray2Value(&img.timeToChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0);


	
	/* Free 2D pointer array for 3D image */
	mem_free_2D((void**)img.vox);

	/* Free allocated data */
	mem_free_3D((void***)img.proxMapInput);
    mem_free_3D((void***)img.lastChange);
    mem_free_3D((void***)img.timeToChange);
    mem_free_3D((void***)sino.e);
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

