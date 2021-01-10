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
    void writeSysMatrix(double *angles,
        SinoParams sinoParams,
        ImageParams imgParams,
        char *fName);


def cy_AmatrixComputeToFile(py_imageparams,
                            py_sinoparams,
                            char[:] Amatrix_fname,
                            char verboseLevel):
    '''
    Cython wrapper that calls c code to compute A matrix to file.
    Args:
        py_imageparams: python dictionary stores image parameters
        py_sinoparams: python dictionary stores sinogram parameters
        Amatrix_fname: path to store computed A matrix.
        verboseLevel: Possible values are {0,1,2}, where 0 is quiet, 1 prints minimal reconstruction progress information, and 2 prints the full information.
    Returns:
    '''
    # Declare image and sinogram Parameter structures.
    cdef ImageParams3D imgparams
    cdef SinoParams3DParallel sinoparams

    # Write parameter to c structures based on given py parameter List.
    write_ImageParams3D(&imgparams, py_imageparams)
    write_SinoParams3D(&sinoparams, py_sinoparams, py_sinoparams['ViewAngles'])

    # Compute A matrix.
    AmatrixComputeToFile(imgparams,sinoparams,&Amatrix_fname[0],verboseLevel)

