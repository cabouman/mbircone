
#include "allocate.h"
#include "MBIRModularUtilities3D.h"
#include "interface.h"
#include "computeSysMatrix.h"
#include "recon3DCone.h"


void AmatrixComputeToFile(float *angles, 
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

    freeSysMatrix(&A);

}

void recon(float *x, float *y, float *wght, float *proxmap_input,
	struct SinoParams sinoParams, struct ImageParams imgParams, struct ReconParams reconParams, 
	char *Amatrix_fname)
{
	struct Sino sino;
    struct Image img;
    struct SysMatrix A;
	int i_x, i_y, i_z, i;
	

	/* Set img and sino params inside data structure */
	copyImgParams(&imgParams, &img.params);
	copySinoParams(&sinoParams, &sino.params);

	/* Perform normalizations on parameters*/
	computeSecondaryReconParams(&reconParams, &img.params);

	/* Read system matrix from disk */
	readSysMatrix(Amatrix_fname, &sino.params, &img.params, &A);
    char dummy_var;
    fprintf(stdout, "Ready to allocate memory in C. Press any char to continue ... \n");
    scanf("%c",&dummy_var);	
	img.vox = x;
	img.proxMapInput = proxmap_input;

	sino.vox = y;
    //sino.wgt = (float ***)mem_alloc_float3D_from_flat(wght, sinoParams.N_beta, sinoParams.N_dv, sinoParams.N_dw);
    sino.wgt = wght;
    /* Allocate error sinogram */
    //sino.e = (float***)allocateSinoData3DCone(&sino.params, sizeof(float));
    sino.e = (float*)allocateSinoData3DCone(&sino.params, sizeof(float));

	/* Allocate other image data */
    img.lastChange = (float***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(float));
    img.timeToChange = (unsigned char***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(unsigned char));

    fprintf(stdout, "Done allocating memory in C. Press any char to continue ... \n");
    scanf("%c",&dummy_var);	

	applyMask(img.vox, img.params.N_x, img.params.N_y, img.params.N_z);
    fprintf(stdout, "Done applying mask. Press any char to continue ... \n");
    scanf("%c",&dummy_var);	

     /* Initialize error sinogram e = y - Ax */
    forwardProject3DCone( sino.e, img.vox, &img.params, &A, &sino.params); /* e = Ax */
    fprintf(stdout, "Done forwardProject3DCone\n");
    floatArray_z_equals_aX_plus_bY(&sino.e[0], 1.0, &sino.vox[0], -1.0, &sino.e[0], sino.params.N_beta*sino.params.N_dv*sino.params.N_dw); /* e = 1.0 * y + (-1.0) * e */
    fprintf(stdout, "Done initializing error sinogram. Press any char to continue ... \n");
    scanf("%c",&dummy_var);	

    /* Initialize other image data */
    setFloatArray2Value(&img.lastChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0.0);
    setUCharArray2Value(&img.timeToChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0);
    fprintf(stdout, "Done initializing other image data. Ready to enter MBIR3DCone. Press any char to continue ... \n");
    scanf("%c",&dummy_var);	

    /* 
    Reconstruct 
    */
    MBIR3DCone(&img, &sino, &reconParams, &A);
    freeSysMatrix(&A);
	
	/* Free 2D pointer array for 3D data */
	// printf("Done free_2D\n");

	/* Free allocated data */
    mem_free_3D((void***)img.lastChange);
    mem_free_3D((void***)img.timeToChange);
    mem_free_1D((void*)sino.e);
    // printf("Done free_3D\n");

}

void forwardProject(float *y, float *x, 
	struct SinoParams sinoParams, struct ImageParams imgParams, 
	char *Amatrix_fname)
{
	
    struct SysMatrix A;

	/* Read system matrix from disk */
	readSysMatrix(Amatrix_fname, &sinoParams, &imgParams, &A);

	forwardProject3DCone(y, x, &imgParams, &A, &sinoParams);

	freeSysMatrix(&A);

	// printf("Done free_2D\n");

}

void ***mem_alloc_float3D_from_flat(float *dataArray, size_t N1, size_t N2, size_t N3)
{
	float ***topTree;
	long int i1, i2;

	topTree = (float***)mem_alloc_2D(N1, N2, sizeof(void*));

	for(i1 = 0; i1 < N1; i1++)
	for(i2 = 0; i2 < N2; i2++)
	{
		topTree[i1][i2] = dataArray + (i1*N2 + i2)*N3;	
		// topTree[i1][i2] = &dataArray[i1*N2*N3+i2*N3];
	}

	return (topTree);	
}
