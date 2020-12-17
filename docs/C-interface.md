
Strategy:

1. Define target C subroutine interface
2. Modify C code to conformed to target interface (wrapper + other changes)
3. Write cython wrapper to call C subroutines
4. Write python wrapper to using cython subroutines



Example of documentation for a C subroutine call.


> Here are preliminary prototypes for c-routines I think we'll need. We'll need to think out the various ways these will be used by python so we can implement this properly (for example, supplying the initial error sinogram vs. not). It'll be slightly tricky if we're hiding the A matrix structure from python.
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

