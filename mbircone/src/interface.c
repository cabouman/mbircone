
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
    char dummy_var;

	/* Set img and sino params inside data structure */
	copyImgParams(&imgParams, &img.params);
	copySinoParams(&sinoParams, &sino.params);

	/* Perform normalizations on parameters*/
	computeSecondaryReconParams(&reconParams, &img.params);
    
    fprintf(stdout, "Ready to allocate system matrix. Press any char to continue ... \n");
    scanf("%c",&dummy_var);	

	/* Read system matrix from disk */
	readSysMatrix(Amatrix_fname, &sino.params, &img.params, &A);
    
    fprintf(stdout, "Done reading system matrix. Ready to allocate memory in C. Press any char to continue ... \n");
    scanf("%c",&dummy_var);	
	img.vox = x;
	img.proxMapInput = proxmap_input;

	sino.vox = y;
    sino.wgt = wght;
    /* Allocate error sinogram */
    sino.e = (float*)allocateSinoData3DCone(&sino.params, sizeof(float));
    
    fprintf(stdout, "Ready to allocate memory for lastChange and timeToChange. Press any char to continue ... \n");
    scanf("%c",&dummy_var);	

	/* Allocate other image data */
    img.lastChange = (float***) get_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(float));
    img.timeToChange = (unsigned char***) get_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(unsigned char));
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
    free_3D((void***)img.lastChange);
    free_3D((void***)img.timeToChange);
    free((void*)sino.e);
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
