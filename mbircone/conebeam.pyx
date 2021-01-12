
import numpy as np
import ctypes           # Import python package required to use cython
cimport cython          # Import cython package
cimport numpy as cnp    # Import specialized cython support for numpy

# Import c data structure
cdef extern from "./src/MBIRModularUtilities3D.h":
     
    struct SinoParams:
    
        long int N_dv;
        long int N_dw; 
        double Delta_dv;
        double Delta_dw;
       
        long int N_beta;
        
        double u_s;
        double u_r;
        double v_r;
        double u_d0;
        double v_d0;
        double w_d0;
        
        double weightScaler_value; 


    struct ImageParams:

        double x_0;
        double y_0;
        double z_0;

        long int N_x;
        long int N_y;
        long int N_z;

        double Delta_xy;
        double Delta_z;
        
        long int j_xstart_roi;
        long int j_ystart_roi;
        long int j_zstart_roi;
        long int j_xstop_roi;
        long int j_ystop_roi;
        long int j_zstop_roi;

        long int N_x_roi;
        long int N_y_roi;
        long int N_z_roi;


# Import a c function to compute A matrix.
cdef extern from "./src/cyInterface.h":
    void AmatrixComputeToFile(double *angles, SinoParams sinoParams, ImageParams imgParams, 
        char *fName);


cdef map_py2c_sinoparams(SinoParams* sinoparams_c, sinoparams_py):
    
    sinoparams_c.N_dv = sinoparams_py['N_dv'];
    sinoparams_c.N_dw; = sinoparams_py['N_dw;'] 
    sinoparams_c.Delta_dv = sinoparams_py['Delta_dv'];
    sinoparams_c.Delta_dw = sinoparams_py['Delta_dw'];
    sinoparams_c.N_beta = sinoparams_py['N_beta'];   
    sinoparams_c.u_s = sinoparams_py['u_s'];
    sinoparams_c.u_r = sinoparams_py['u_r'];
    sinoparams_c.v_r = sinoparams_py['v_r'];
    sinoparams_c.u_d0 = sinoparams_py['u_d0'];
    sinoparams_c.v_d0 = sinoparams_py['v_d0'];
    sinoparams_c.w_d0 = sinoparams_py['w_d0'];
    sinoparams_c.weightScaler_value; = sinoparams_py['weightScaler_value;'] 


cdef map_py2c_imgparams(ImageParams* imgparams_c, imgparams_py):

    imgparams_c.x_0 = imgparams_py['x_0'];
    imgparams_c.y_0 = imgparams_py['y_0'];
    imgparams_c.z_0 = imgparams_py['z_0'];
    imgparams_c.N_x = imgparams_py['N_x'];
    imgparams_c.N_y = imgparams_py['N_y'];
    imgparams_c.N_z = imgparams_py['N_z'];
    imgparams_c.Delta_xy = imgparams_py['Delta_xy'];
    imgparams_c.Delta_z = imgparams_py['Delta_z'];
    imgparams_c.j_xstart_roi = imgparams_py['j_xstart_roi'];
    imgparams_c.j_ystart_roi = imgparams_py['j_ystart_roi'];
    imgparams_c.j_zstart_roi = imgparams_py['j_zstart_roi'];
    imgparams_c.j_xstop_roi = imgparams_py['j_xstop_roi'];
    imgparams_c.j_ystop_roi = imgparams_py['j_ystop_roi'];
    imgparams_c.j_zstop_roi = imgparams_py['j_zstop_roi'];
    imgparams_c.N_x_roi = imgparams_py['N_x_roi'];
    imgparams_c.N_y_roi = imgparams_py['N_y_roi'];
    imgparams_c.N_z_roi = imgparams_py['N_z_roi'];


def AmatrixComputeToFile_cy(angles, imgparams, sinoparams, char[:] Amatrix_fname):

    # Declare image and sinogram Parameter structures
    cdef ImageParams3D imgparams_c
    cdef SinoParams3DParallel sinoparams_c

    # Get pointer to 1D array of angles
    cdef double *angle_arr = 

    map_py2c_sinoparams(&sinoparams_c, sinoparams)
    map_py2c_imgparams(&imgparams_c, imgparams)

    AmatrixComputeToFile(imgparams, sinoparams, &Amatrix_fname[0])

