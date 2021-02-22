#ifndef _CY_INTERFACE_H_
#define _CY_INTERFACE_H_

void AmatrixComputeToFile(double *angles, 
	struct SinoParams sinoParams, struct ImageParams imgParams, 
	char *Amatrix_fname, char verbose);

void recon(float *x, float *sino, float *wght, float *x_init, float *proxmap_input,
	struct SinoParams sinoParams, struct ImageParams imgParams, struct ReconParams reconParams, 
	char *Amatrix_fname);

void ***mem_alloc_float3D_from_flat(float *dataArray, size_t N1, size_t N2, size_t N3);

#endif /* _CY_INTERFACE_H_ */