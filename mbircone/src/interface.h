#ifndef _CY_INTERFACE_H_
#define _CY_INTERFACE_H_

void AmatrixComputeToFile(float *angles, 
    struct SinoParams sinoParams, struct ImageParams imgParams, 
    char *Amatrix_fname, char verbose);

void denoise(float *x,
    struct ImageParams imgParams, struct ReconParams reconParams);

void recon(float *x, float *y, float *wght, float *proxmap_input,
    struct SinoParams sinoParams, struct ImageParams imgParams, struct ReconParams reconParams, 
    char *Amatrix_fname);

void forwardProject(float *y, float *x, 
    struct SinoParams sinoParams, struct ImageParams imgParams, 
    char *Amatrix_fname);

#endif /* _CY_INTERFACE_H_ */
