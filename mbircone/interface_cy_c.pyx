
import numpy as np
import ctypes           # Import python package required to use cython
cimport cython          # Import cython package
cimport numpy as cnp    # Import specialized cython support for numpy
from libc.string cimport memset,strcpy


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



    struct ReconParams:
    
        double priorWeight_QGGMRF;                  # Prior mode: (0: off, 1: QGGMRF, 2: proximal mapping) 
        double priorWeight_proxMap;                  # Prior mode: (0: off, 1: QGGMRF, 2: proximal mapping) 
        
        # QGGMRF 
        double q;                   # q: QGGMRF parameter (q>1, typical choice q=2) 
        double p;                   # p: QGGMRF parameter (1<=p<q) 
        double T;                   # T: QGGMRF parameter 
        double sigmaX;              # sigmaX: QGGMRF parameter 
        double bFace;               # bFace: relative neighbor weight: cube faces 
        double bEdge;               # bEdge: relative neighbor weight: cube edges 
        double bVertex;             # bVertex: relative neighbor weight: cube vertices 
        # Proximal Mapping 
        double sigma_lambda;        # sigma_lambda: Proximal mapping scalar 
        int is_positivity_constraint;
        

         # Stopping Conditions

        double stopThresholdChange_pct;           # stop threshold (%) 
        double stopThesholdRWFE_pct;
        double stopThesholdRUFE_pct;
        int MaxIterations;              # maximum number of iterations 
        char relativeChangeMode[200];
        double relativeChangeScaler;
        double relativeChangePercentile;
    
    
         # Zipline Stuff

        int N_G;                # Number of groups for group ICD 
        int zipLineMode;                # Zipline mode: (0: off, 1: conventional Zipline, 2: randomized Zipline) 
        int numVoxelsPerZiplineMax;
        int numVoxelsPerZipline;
        int numZiplines;
    
        # Weight scaler stuff

        char weightScaler_estimateMode[200];     # Estimate weight scaler? 1: Yes. 0: Use user specified value 
        char weightScaler_domain[200];     
        double weightScaler_value;            # User specified weight scaler 
    
    
        # NHICD stuff 
        char NHICD_Mode[200];
        double NHICD_ThresholdAllVoxels_ErrorPercent;
        double NHICD_percentage;
        double NHICD_random;
    
        # Misc 
        int verbosity;
        int isComputeCost;


# Import a c function to compute A matrix.
cdef extern from "./src/interface.h":
    void AmatrixComputeToFile(double *angles, SinoParams c_sinoparams, ImageParams c_imgparams, 
        char *Amatrix_fname, char verbose);

    void recon(float *x, float *sino, float *wght, float *x_init, float *proxmap_input,
	SinoParams c_sinoparams, ImageParams c_imgparams, ReconParams c_reconparams,
	char *Amatrix_fname);

    void forwardProject(float *y, float *x, 
    SinoParams sinoParams, ImageParams imgParams, 
    char *Amatrix_fname)


cdef convert_py2c_SinoParams3D(SinoParams* c_sinoparams, sinoparams):
    
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


cdef convert_py2c_ImageParams3D(ImageParams* c_imgparams, imgparams):

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


cdef map_py2c_reconparams(ReconParams* c_reconparams,
                          reconparams,
                          const char* cy_relativeChangeMode,
                          const char* cy_weightScaler_estimateMode,
                          const char* cy_weightScaler_domain,
                          const char* cy_NHICD_Mode):

        c_reconparams.priorWeight_QGGMRF = reconparams['priorWeight_QGGMRF']                  # Prior mode: (0: off, 1: QGGMRF, 2: proximal mapping)
        c_reconparams.priorWeight_proxMap = reconparams['priorWeight_proxMap']                  # Prior mode: (0: off, 1: QGGMRF, 2: proximal mapping)

        # QGGMRF
        c_reconparams.q = reconparams['q']                   # q: QGGMRF parameter (q>1, typical choice q=2)
        c_reconparams.p = reconparams['p']                   # p: QGGMRF parameter (1<=p<q)
        c_reconparams.T = reconparams['T']                   # T: QGGMRF parameter
        c_reconparams.sigmaX = reconparams['sigmaX']              # sigmaX: QGGMRF parameter
        c_reconparams.bFace = reconparams['bFace']               # bFace: relative neighbor weight: cube faces
        c_reconparams.bEdge = reconparams['bEdge']               # bEdge: relative neighbor weight: cube edges
        c_reconparams.bVertex = reconparams['bVertex']             # bVertex: relative neighbor weight: cube vertices
        # Proximal Mapping
        c_reconparams.sigma_lambda = reconparams['sigma_lambda']        # sigma_lambda: Proximal mapping scalar
        c_reconparams.is_positivity_constraint = reconparams['is_positivity_constraint']


         # Stopping Conditions

        c_reconparams.stopThresholdChange_pct = reconparams['stopThresholdChange_pct']           # stop threshold (%)
        c_reconparams.stopThesholdRWFE_pct = reconparams['stopThesholdRWFE_pct']
        c_reconparams.stopThesholdRUFE_pct = reconparams['stopThesholdRUFE_pct']
        c_reconparams.MaxIterations = reconparams['MaxIterations']              # maximum number of iterations
        memset(c_reconparams.relativeChangeMode, '\0', sizeof(c_reconparams.relativeChangeMode))
        strcpy(c_reconparams.relativeChangeMode, cy_relativeChangeMode)
        c_reconparams.relativeChangeScaler = reconparams['relativeChangeScaler']
        c_reconparams.relativeChangePercentile = reconparams['relativeChangePercentile']


         # Zipline Stuff

        c_reconparams.N_G = reconparams['N_G']                # Number of groups for group ICD
        c_reconparams.zipLineMode = reconparams['zipLineMode']                # Zipline mode: (0: off, 1: conventional Zipline, 2: randomized Zipline)
        c_reconparams.numVoxelsPerZiplineMax = reconparams['numVoxelsPerZiplineMax']
        c_reconparams.numVoxelsPerZipline = reconparams['numVoxelsPerZipline']
        c_reconparams.numZiplines = reconparams['numZiplines']

        # Weight scaler stuff

        # Estimate weight scaler? 1: Yes. 0: Use user specified value
        memset(c_reconparams.weightScaler_estimateMode, '\0', sizeof(c_reconparams.weightScaler_estimateMode))
        strcpy(c_reconparams.weightScaler_estimateMode, cy_weightScaler_estimateMode)
        memset(c_reconparams.weightScaler_domain, '\0', sizeof(c_reconparams.weightScaler_domain))
        strcpy(c_reconparams.weightScaler_domain, cy_weightScaler_domain)

        c_reconparams.weightScaler_value = reconparams['weightScaler_value']            # User specified weight scaler


        # NHICD stuff
        memset(c_reconparams.NHICD_Mode, '\0', sizeof(c_reconparams.NHICD_Mode))
        strcpy(c_reconparams.NHICD_Mode, cy_NHICD_Mode)
        c_reconparams.NHICD_ThresholdAllVoxels_ErrorPercent = reconparams['NHICD_ThresholdAllVoxels_ErrorPercent']
        c_reconparams.NHICD_percentage = reconparams['NHICD_percentage']
        c_reconparams.NHICD_random = reconparams['NHICD_random']

        # Misc
        c_reconparams.verbosity = reconparams['verbosity']
        c_reconparams.isComputeCost = reconparams['isComputeCost']



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


def AmatrixComputeToFile_cy(angles, sinoparams, imgparams, Amatrix_fname, verbose=1):

    # Declare image and sinogram Parameter structures
    cdef SinoParams c_sinoparams
    cdef ImageParams c_imgparams

    # Get pointer to 1D char array of Amatrix 
    cdef cnp.ndarray[char, ndim=1, mode="c"] c_Amatrix_fname
    # Get pointer to 1D array of angles
    cdef cnp.ndarray[double, ndim=1, mode="c"] c_angles = angles

    convert_py2c_SinoParams3D(&c_sinoparams, sinoparams)
    convert_py2c_ImageParams3D(&c_imgparams, imgparams)

    c_Amatrix_fname = string_to_char_array(Amatrix_fname)

    AmatrixComputeToFile(&c_angles[0], c_sinoparams, c_imgparams, &c_Amatrix_fname[0], verbose)


def recon_cy(sino, wght, x_init, proxmap_input,
             sinoparams, imgparams, reconparams, py_Amatrix_fname):
    # sino, wght shape : views x slices x channels
    # recon shape: N_x N_y N_z (source-detector-line, channels, slices)

    cdef cnp.ndarray[float, ndim=3, mode="c"] py_x
    py_x = np.zeros((imgparams['N_x'],imgparams['N_y'],imgparams['N_z']), dtype=ctypes.c_float)
    py_sino = np.ascontiguousarray(sino, dtype=np.single)
    py_wght = np.ascontiguousarray(wght, dtype=np.single)
    py_x_init = np.ascontiguousarray(x_init, dtype=np.single)
    py_proxmap_input = np.ascontiguousarray(proxmap_input, dtype=np.single)

    cdef cnp.ndarray[float, ndim=3, mode="c"] cy_sino = py_sino
    cdef cnp.ndarray[float, ndim=3, mode="c"] cy_wght = py_wght
    cdef cnp.ndarray[float, ndim=3, mode="c"] cy_x_init = py_x_init
    cdef cnp.ndarray[float, ndim=3, mode="c"] cy_proxmap_input = py_proxmap_input
    cdef cnp.ndarray[char, ndim=1, mode="c"] c_Amatrix_fname = string_to_char_array(py_Amatrix_fname)
    cdef cnp.ndarray[char, ndim=1, mode="c"] cy_relativeChangeMode = string_to_char_array(reconparams["relativeChangeMode"])
    cdef cnp.ndarray[char, ndim=1, mode="c"] cy_weightScaler_estimateMode = string_to_char_array(reconparams["weightScaler_estimateMode"])
    cdef cnp.ndarray[char, ndim=1, mode="c"] cy_weightScaler_domain = string_to_char_array(reconparams["weightScaler_domain"])
    cdef cnp.ndarray[char, ndim=1, mode="c"] cy_NHICD_Mode = string_to_char_array(reconparams["NHICD_Mode"])

    cdef ImageParams c_imgparams
    cdef SinoParams c_sinoparams
    cdef ReconParams c_reconparams

    convert_py2c_SinoParams3D(&c_sinoparams, sinoparams)
    convert_py2c_ImageParams3D(&c_imgparams, imgparams)
    map_py2c_reconparams(&c_reconparams,
                          reconparams,
                          &cy_relativeChangeMode[0],
                          &cy_weightScaler_estimateMode[0],
                          &cy_weightScaler_domain[0],
                          &cy_NHICD_Mode[0])

    recon(&py_x[0,0,0],
          &cy_sino[0,0,0],
          &cy_wght[0,0,0],
          &cy_x_init[0,0,0],
          &cy_proxmap_input[0,0,0],
          c_sinoparams,
          c_imgparams,
          c_reconparams,
	      &c_Amatrix_fname[0])

    # print("Cython done")

    return py_x


def project(image, settings):
    """Forward projection function used by mbircone.project().

    Args:
        image (ndarray): 3D Image of shape (num_img_cols, num_img_rows, num_img_slices) to be projected.
        settings (dict): Dictionary containing projection settings.

    Returns:
        ndarray: 3D numpy array containing projection with shape (num_views, num_det_channels, num_det_rows).
    """

    imgparams = settings['imgparams']
    sinoparams = settings['sinoparams']
    sysmatrix_fname = settings['sysmatrix_fname']

    # Get shapes of projection
    num_views = sinoparams['N_beta']
    num_det_rows = sinoparams['N_dv']
    num_det_channels = sinoparams['N_dw']

    # Ensure image memory is aligned properly
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image, dtype=np.single)
    else:
        image = image.astype(np.single, copy=False)

    cdef cnp.ndarray[float, ndim=3, mode="c"] cy_image = image

    # Allocates memory, without initialization, for matrix to be passed back from C subroutine
    cdef cnp.ndarray[float, ndim=3, mode="c"] proj = np.empty((num_views, num_det_rows, num_det_channels), dtype=ctypes.c_float)

    # Write parameter to c structures based on given py parameter List.
    cdef ImageParams c_imgparams
    cdef SinoParams c_sinoparams
    convert_py2c_SinoParams3D(&c_sinoparams, sinoparams)
    convert_py2c_ImageParams3D(&c_imgparams, imgparams)

    cdef cnp.ndarray[char, ndim=1, mode="c"] Amatrix_fname = string_to_char_array(sysmatrix_fname)

    # Forward projection by calling C subroutine
    forwardProject(&proj[0,0,0],
                    &cy_image[0,0,0],
                    c_sinoparams,
                    c_imgparams,
                    &Amatrix_fname[0])

    # print("Cython done")

    return proj
