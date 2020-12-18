
Strategy:

1. Define target C subroutine interface
2. Modify C code to conformed to target interface (make wrapper + other changes)
3. Write cython wrapper to call C subroutines
4. Write python wrapper to using cython subroutines

```
void MBIR3DCone(float *x, 
	float *sino, 
	struct ReconParams *reconParams, 
	char *SysMatrix_fname);

void forwardProject(float *proj, 
	float *x, 
	struct ImageParams *imgParams, 
	struct SinoParams *sinoInfo, 
	char *SysMatrix_fname);

writeSysMatrix(char *fName, 
	struct SinoParams *sinoParams, 
	struct ImageParams *imgParams);
	
```

Example of documentation for a C subroutine call.
> 
> ```
> void MBIRReconstruct3D(
>     float *sino,   // sino[i_slice*Nchannels*Ntheta +i_theta*Nchannels + i_channel]
>     float *weight,
>     float *e,      // error sinogram; if NULL compute internally
>     float *image,  // image[i_slice*Nrows*Ncols+i_row*Ncols+i_col]
>     struct SinoParams3DParallel sinoparams,
>     struct ImageParams3D imgparams,
>     struct ReconParams reconparams,
>     char *Amatrix_fname,    // if NULL, compute internally
>     char verboseLevel);
>  
> void forwardProject3D(
>     float *image,
>     float *proj,
>     struct SinoParams3DParallel sinoparams,
>     struct ImageParams3D imgparams,
>     char *Amatrix_fname,    // if NULL, compute internally
>     char verboseLevel);
> 
> void AmatrixComputeToFile(
>     struct SinoParams3DParallel sinoparams,
>     struct ImageParams3D imgparams,
>     char *Amatrix_fname,
>     char verboseLevel);
> ```

