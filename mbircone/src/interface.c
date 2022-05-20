
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
/*
 * This function initializes C variables related to qGGMRF reconstruction, read sysmatrix from disk, and invoke MBIR3DCone() function to perform qGGMRF recon or prox map estimation in place.
 * This function is invoked by recon_cy() function in interface_cy.pyx.
 * 
 * Input Variables:
 * x: pointer to the 1D initial image array as well as the recon image array. This array will be modified in-place in ICD iterations.
 * y: pointer to 1D sinogram array. This array will not be modified by C code.
 * wght: pointer to 1D sinogram weight array. This array will not be modified by C code.
 * proxmap_input: pointer to 1D proximal map input array. Will only be accessed when imgParams->prox_mode is True.
 * sinoParams: struct to store sinogram params. See MBIRModularUtilities3D.h for struct definition.
 * imgParams: struct to store recon image params. See MBIRModularUtilities3D.h for struct definition.
 * reconParams: struct to store reconstruction related hyperparams. See MBIRModularUtilities3D.h for struct definition.
 * Amatrix_fname: pointer to sysmatrix filename string.
 *
 * Return Variables: None.
 */
void recon(float *x, float *y, float *wght, float *proxmap_input,
    struct SinoParams sinoParams, struct ImageParams imgParams, struct ReconParams reconParams, 
    char *Amatrix_fname)
{
    struct Sino sino;
    struct Image img;
    struct SysMatrix A;
    int i;

    /* Set img and sino params inside data structure */
    copyImgParams(&imgParams, &img.params);
    copySinoParams(&sinoParams, &sino.params);

    /* Perform normalizations on parameters*/
    computeSecondaryReconParams(&reconParams, &img.params);
    
    /* Read system matrix from disk */
    readSysMatrix(Amatrix_fname, &sino.params, &img.params, &A);
    
    /* 'x' is reconstructed in place, so if proximal map is the same array, make a local copy */
    if(proxmap_input == x)
    {
        img.proxMapInput = (float *) mget_spc((size_t)img.params.N_x*img.params.N_y*img.params.N_z,sizeof(float));
        for(i=0; i<(size_t)img.params.N_x*img.params.N_y*img.params.N_z; i++)
            img.proxMapInput[i] = proxmap_input[i];
    }
    else
        img.proxMapInput = proxmap_input;
    
    img.vox = x;
    sino.vox = y;
    sino.wgt = wght;
    /* Allocate error sinogram */
    sino.e = (float*)allocateSinoData3DCone(&sino.params, sizeof(float));
    

    /* Allocate other image data */
    img.lastChange = (float***) multialloc(sizeof(float), 3, img.params.N_x, img.params.N_y, reconParams.numZiplines);
    img.timeToChange = (unsigned char***) multialloc(sizeof(unsigned char), 3, img.params.N_x, img.params.N_y, reconParams.numZiplines);

    applyMask(img.vox, img.params.N_x, img.params.N_y, img.params.N_z);

     /* Initialize error sinogram e = y - Ax */
    forwardProject3DCone( sino.e, img.vox, &img.params, &A, &sino.params); /* e = Ax */
    floatArray_z_equals_aX_plus_bY(&sino.e[0], 1.0, &sino.vox[0], -1.0, &sino.e[0], sino.params.N_beta*sino.params.N_dv*sino.params.N_dw); /* e = 1.0 * y + (-1.0) * e */

    /* Initialize other image data */
    setFloatArray2Value(&img.lastChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0.0);
    setUCharArray2Value(&img.timeToChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0);

    /* 
    Reconstruct 
    */
    MBIR3DCone(&img, &sino, &reconParams, &A);
    freeSysMatrix(&A);
    
    /* Free 2D pointer array for 3D data */
    // printf("Done free_2D\n");

    /* Free allocated data */
    multifree((void***)img.lastChange, 3);
    multifree((void***)img.timeToChange, 3);
    free((void*)sino.e);
    // printf("Done mem_free_3D\n");

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
