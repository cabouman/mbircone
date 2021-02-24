
import numpy as np
import ctypes           # Import python package required to use cython
cimport cython          # Import cython package
cimport numpy as cnp    # Import specialized cython support for numpy
from libc.string cimport strncpy


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
        double InitVal_recon;                  # Initialization value InitVal_proxMapInput (mm-1) 
        char initReconMode[200];
    
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
    

        # Parallel Stuff
        int numThreads;                 # numThreads: Number of threads 
    
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
        char backprojlike_type[200]; 


# Import a c function to compute A matrix.
cdef extern from "./src/cyInterface.h":
    void AmatrixComputeToFile(double *angles, SinoParams c_sinoparams, ImageParams c_imgparams, 
        char *Amatrix_fname, char verbose);

    void recon(float *x, float *sino, float *wght, float *x_init, float *proxmap_input,
	SinoParams c_sinoparams, ImageParams c_imgparams, ReconParams c_reconParams,
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


cdef map_py2c_reconparams(ReconParams* c_reconparams, reconparams):
        c_reconparams.InitVal_recon = c_reconparams['InitVal_recon']                  # Initialization value InitVal_proxMapInput (mm-1)
        char initReconMode[200];

        c_reconparams.priorWeight_QGGMRF = c_reconparams['priorWeight_QGGMRF']                  # Prior mode: (0: off, 1: QGGMRF, 2: proximal mapping)
        c_reconparams.priorWeight_proxMap = c_reconparams['priorWeight_proxMap']                  # Prior mode: (0: off, 1: QGGMRF, 2: proximal mapping)

        # QGGMRF
        c_reconparams.q = c_reconparams['q']                   # q: QGGMRF parameter (q>1, typical choice q=2)
        c_reconparams.p = c_reconparams['p']                   # p: QGGMRF parameter (1<=p<q)
        c_reconparams.T = c_reconparams['T']                   # T: QGGMRF parameter
        c_reconparams.sigmaX = c_reconparams['sigmaX']              # sigmaX: QGGMRF parameter
        c_reconparams.bFace = c_reconparams['bFace']               # bFace: relative neighbor weight: cube faces
        c_reconparams.bEdge = c_reconparams['bEdge']               # bEdge: relative neighbor weight: cube edges
        c_reconparams.bVertex = c_reconparams['bVertex']             # bVertex: relative neighbor weight: cube vertices
        # Proximal Mapping
        c_reconparams.sigma_lambda = c_reconparams['sigma_lambda']        # sigma_lambda: Proximal mapping scalar
        c_reconparams.is_positivity_constraint = c_reconparams['is_positivity_constraint']


         # Stopping Conditions

        c_reconparams.stopThresholdChange_pct = c_reconparams['stopThresholdChange_pct']           # stop threshold (%)
        c_reconparams.stopThesholdRWFE_pct = c_reconparams['stopThesholdRWFE_pct']
        c_reconparams.stopThesholdRUFE_pct = c_reconparams['stopThesholdRUFE_pct']
        c_reconparams.MaxIterations = c_reconparams['MaxIterations']              # maximum number of iterations
        char relativeChangeMode[200];
        c_reconparams.relativeChangeScaler = c_reconparams['relativeChangeScaler']
        c_reconparams.relativeChangePercentile = c_reconparams['relativeChangePercentile']


         # Zipline Stuff

        c_reconparams.N_G = c_reconparams['N_G']                # Number of groups for group ICD
        c_reconparams.zipLineMode = c_reconparams['zipLineMode']                # Zipline mode: (0: off, 1: conventional Zipline, 2: randomized Zipline)
        c_reconparams.numVoxelsPerZiplineMax = c_reconparams['numVoxelsPerZiplineMax']
        c_reconparams.numVoxelsPerZipline = c_reconparams['numVoxelsPerZipline']
        c_reconparams.numZiplines = c_reconparams['numZiplines']


        # Parallel Stuff
        c_reconparams.numThreads = c_reconparams['numThreads']                 # numThreads: Number of threads

        # Weight scaler stuff

        char weightScaler_estimateMode[200];     # Estimate weight scaler? 1: Yes. 0: Use user specified value
        char weightScaler_domain[200];
        c_reconparams.weightScaler_value = c_reconparams['weightScaler_value']            # User specified weight scaler


        # NHICD stuff
        char NHICD_Mode[200];
        c_reconparams.NHICD_ThresholdAllVoxels_ErrorPercent = c_reconparams['NHICD_ThresholdAllVoxels_ErrorPercent']
        c_reconparams.NHICD_percentage = c_reconparams['NHICD_percentage']
        c_reconparams.NHICD_random = c_reconparams['NHICD_random']

        # Misc
        c_reconparams.verbosity = c_reconparams['verbosity']
        c_reconparams.isComputeCost = c_reconparams['isComputeCost']
        char backprojlike_type[200];



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

    map_py2c_sinoparams(&c_sinoparams, sinoparams)
    map_py2c_imgparams(&c_imgparams, imgparams)

    c_Amatrix_fname = string_to_char_array(Amatrix_fname)

    AmatrixComputeToFile(&c_angles[0], c_sinoparams, c_imgparams, &c_Amatrix_fname[0], verbose)

