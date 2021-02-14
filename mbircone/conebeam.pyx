
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
    void AmatrixComputeToFile(double *angles, SinoParams c_sinoparams, ImageParams c_imgparams, 
        char *Amatrix_fname);


cdef map_py2c_sinoparams(SinoParams* c_sinoparams, sinoparams):
    
    c_sinoparams.N_dv = sinoparams['N_dv']
    c_sinoparams.N_dw = sinoparams['N_dw'] 
    c_sinoparams.Delta_dv = sinoparams['Delta_dv']
    c_sinoparams.Delta_dw = sinoparams['Delta_dw']
    c_sinoparams.N_beta = sinoparams['N_beta']   
    c_sinoparams.u_s = sinoparams['u_s']
    c_sinoparams.u_r = sinoparams['u_r']
    c_sinoparams.v_r = sinoparams['v_r']
    c_sinoparams.u_d0 = sinoparams['u_d0']
    c_sinoparams.v_d0 = sinoparams['v_d0']
    c_sinoparams.w_d0 = sinoparams['w_d0']
    c_sinoparams.weightScaler_value = sinoparams['weightScaler_value']


cdef map_py2c_imgparams(ImageParams* c_imgparams, imgparams):

    c_imgparams.x_0 = imgparams['x_0']
    c_imgparams.y_0 = imgparams['y_0']
    c_imgparams.z_0 = imgparams['z_0']
    c_imgparams.N_x = imgparams['N_x']
    c_imgparams.N_y = imgparams['N_y']
    c_imgparams.N_z = imgparams['N_z']
    c_imgparams.Delta_xy = imgparams['Delta_xy']
    c_imgparams.Delta_z = imgparams['Delta_z']
    c_imgparams.j_xstart_roi = imgparams['j_xstart_roi']
    c_imgparams.j_ystart_roi = imgparams['j_ystart_roi']
    c_imgparams.j_zstart_roi = imgparams['j_zstart_roi']
    c_imgparams.j_xstop_roi = imgparams['j_xstop_roi']
    c_imgparams.j_ystop_roi = imgparams['j_ystop_roi']
    c_imgparams.j_zstop_roi = imgparams['j_zstop_roi']
    c_imgparams.N_x_roi = imgparams['N_x_roi']
    c_imgparams.N_y_roi = imgparams['N_y_roi']
    c_imgparams.N_z_roi = imgparams['N_z_roi']

def string_to_char_array(input_str):
    """
    Args:
        input_str:  python string
    Returns:
        0-terminated array of unsigned byte with ascii representation of input_str
    """
    # Get the input length to prepare output space
    len_str = len(input_str)  
    
    # Create output array - note the len_str+1 to give 0-terminated array
    output_char_array = np.zeros(len_str + 1, dtype=np.ubyte)  
    
    # Fill in the output array with the input string
    output_char_array[:len_str] = bytearray(input_str.encode('ascii'))  

    return output_char_array


def AmatrixComputeToFile_cy(angles, sinoparams, imgparams, Amatrix_fname):

    # Declare image and sinogram Parameter structures
    cdef SinoParams c_sinoparams
    cdef ImageParams c_imgparams

    # Get pointer to 1D array of angles
    cdef cnp.ndarray[double, ndim=1, mode="c"] c_angles = angles

    map_py2c_sinoparams(&c_sinoparams, sinoparams)
    map_py2c_imgparams(&c_imgparams, imgparams)

    c_Amatrix_fname = string_to_char_array(Amatrix_fname)

    AmatrixComputeToFile(&c_angles[0], c_sinoparams, c_imgparams, &c_Amatrix_fname[0])

