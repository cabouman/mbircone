
Strategy:

1. Define target C subroutine interface
2. Modify C code to conformed to target interface (make wrapper + other changes)
3. Write cython wrapper to call C subroutines
4. Write python wrapper to using cython subroutines


```
struct SinoParams
{
    /* Number of detectors in v-direction and w-direction. */
    long int N_dv;
    long int N_dw;

    /* Detector width and spacing in v-direction and w-direction. */
    double Delta_dv;
    double Delta_dw;

    /* Number of discrete view angles. */
    long int N_beta;
    
    /* The source location on the u-axis. Assume u_s < 0. */
    double u_s;
    
    /* The location (u_r, v_r) of the center of rotation. */
    double u_r;
    double v_r;
    
    /* The location (u,v,w) of the corner of the first detector corresponding to
     (i_v, i_w) = (0, 0). All points on the detector have u = u_d0. */
    double u_d0;
    double v_d0;
    double w_d0;
    
    /* Noise variance estimation */
    double weightScaler_value;       /* Weight_true = Weight / weightScaler_value */
};

struct ImageParams
{
    /* Location of the corner of the first voxel corresponding to
     (j_x, j_y, j_z) = (0, 0, 0). */
    double x_0;
    double y_0;
    double z_0;
    
    /* Number of voxels in x, y, z direction. */
    long int N_x;
    long int N_y;
    long int N_z;

    /* Dimensions of a voxel */
    double Delta_xy;
    double Delta_z;
    
    /**
     *      Region of Interest (roi) parameters
     */
    /* indices of the first voxels in the roi */
    long int j_xstart_roi;
    long int j_ystart_roi;
    long int j_zstart_roi;

    /* indices of the last voxesl in the roi */
    long int j_xstop_roi;
    long int j_ystop_roi;
    long int j_zstop_roi;

    long int N_x_roi;
    long int N_y_roi;
    long int N_z_roi;
};
```

```
void MBIR3DCone(float *x, 
	float *sino, 
	struct ReconParams *reconParams, 
	char *Amatrix_fname);

void forwardProject(float *proj, 
	float *x, 
	struct ImageParams *imgParams, 
	struct SinoParams *sinoInfo, 
	char *Amatrix_fname);

writeSysMatrix(double *angles, 
	struct SinoParams *sinoParams, 
	struct ImageParams *imgParams,
    char *Amatrix_fname);
	
```
