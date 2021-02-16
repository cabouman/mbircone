#ifndef MBIR_MODULAR_UTILITIES_3D_H
#define MBIR_MODULAR_UTILITIES_3D_H


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "allocate.h"

#define LOG_ICDLOOP "log_ICDLoop.txt"

#define OUTPUT_REFRESH_TIME 1.0

#define _MIN_(a, b) ((a)<(b) ? (a) : (b))
#define _MAX_(a, b) ((a)>(b) ? (a) : (b))
#define _ABS_(a)    ((a)>0 ? (a) : (-a))
#define _SWAP_(a, b, temp) (temp)=(a); (a)=(b); (b)=(temp)



#define PI 3.1415926535897932384

/* WEIGHTCHANGE */
#define WEIGHTDATATYPE float
#define WEIGHTDATATYPE_string "float"
/*#define WEIGHTDATATYPE unsigned char
#define WEIGHTDATATYPE_string "unsigned char"*/

#define AMATRIX_RHO 4.0/* System Matrix parameter rho: choose to be about 10<rho<100 */

/* AMATRIXCHANGE */
#define ISBIJCOMPRESSED 1       /* 1: used compressed mode, 0: use uncompressed mode */
#if ISBIJCOMPRESSED == 1
    #define BIJDATATYPE unsigned char
    #define BIJDATATYPE_string "unsigned char"
#else
    #define BIJDATATYPE float
    #define BIJDATATYPE_string "float"
#endif

#define ISCIJCOMPRESSED 1       /* 1: used compressed mode, 0: use uncompressed mode */
#if ISCIJCOMPRESSED == 1
    #define CIJDATATYPE unsigned char
    #define CIJDATATYPE_string "unsigned char"
#else
    #define CIJDATATYPE float
    #define CIJDATATYPE_string "float"
#endif

/* MATRIXINDEXCHANGE */
#define INDEXSTARTSTOPDATATYPE unsigned short
#define INDEXSTARTSTOPDATATYPE_string "unsigned short"

#define INDEXJUDATATYPE unsigned short
#define INDEXJUDATATYPE_string "unsigned short"

/* INDEXSTRIDE */
#define INDEXSTRIDEDATATYPE unsigned char
#define INDEXSTRIDEDATATYPE_string "unsigned char"



struct RandomZiplineAux
{
    /* Group index   */
    int k_G;

    /* Number of groups */
    int N_G;

    /* Number of members of in the zip line */
    int N_M;
    int N_M_max;

    /**
     *      orderXY[N_x*N_y]
     *      Order in which the indices (j_x,j_y) are in.
     *      Shuffled after group index is incremented
     */
    int *orderXY;

    /** 
     *      groupIndex[N_x][N_y][N_z]
     *      groupIndex[j_x][j_y][j_z] \in {0,...,N_G-1} is the group index of 
     *      voxel (j_x, j_y, j_z)
     */
    unsigned char ***groupIndex;           
};

struct RandomAux
{
    /**
     *      orderXY[N_x*N_y]
     *      Order in which the indices (j_x,j_y) are in.
     *      Shuffled after group index is incremented
     */
    long int *orderXYZ;          
};

struct PathNames
{
    char masterFile[1000];
    char plainParamsFile[1000];

    char sino[1000];
    char driftSino[1000];
    char origSino[1000];
    char wght[1000];
    char errSino[1000];
    char recon[1000];
    char reconROI[1000];
    char proxMapInput[1000];
    char lastChange[1000];
    char timeToChange[1000];
    char phantom[1000];
    char sysMatrix[1000];
    char wghtRecon[1000];
    char projInput[1000];
    char projOutput[1000];
    char backprojlikeInput[1000];
    char backprojlikeOutput[1000];
    char estimateSino[1000];
    char consensusRecon[1000];
    char jigMeasurementsSino[1000];
    
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


struct Image
{
    struct ImageParams params;
    float ***vox;           /* [N_x][N_y][N_z] */
    float ***wghtRecon;     /* [N_x][N_y][N_z] */
    float ***vox_roi;       /* [N_x_roi][N_y_roi][N_z_roi] */
    float ***proxMapInput;  /* input, v, to the proximal operator prox_f(.)*/
                            /*    prox_f(v) = argmin_x{ f(x) + 1/2 ||x-v||^2 } */
    float ***lastChange;
    unsigned char ***timeToChange;
    float ***phantom;
    float ***projInput;
    float ***backprojlikeOutput;
    struct RandomZiplineAux randomZiplineAux;
    struct RandomAux randomAux;
};


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


struct Sino
{
    struct SinoParams params;
    float ***vox;       /* [N_beta][N_dv][N_dw] */
    WEIGHTDATATYPE ***wgt;
    float ***e;
    float ***projOutput;
    float ***backprojlikeInput;
    float ***estimateSino;


};

struct ViewAngleList
{
    long int N_beta;
    double *beta;
};


struct ReconParams
{
    /**
     *      Miscellaneous
     */
    double InitVal_recon;                  /* Initialization value InitVal_proxMapInput (mm-1) */
    char initReconMode[200];

    /**
     *     Prior
     */
    double priorWeight_QGGMRF;                  /* Prior mode: (0: off, 1: QGGMRF, 2: proximal mapping) */
    double priorWeight_proxMap;                  /* Prior mode: (0: off, 1: QGGMRF, 2: proximal mapping) */
    
    /* QGGMRF */
        double q;                   /* q: QGGMRF parameter (q>1, typical choice q=2) */
        double p;                   /* p: QGGMRF parameter (1<=p<q) */
        double T;                   /* T: QGGMRF parameter */
        double sigmaX;              /* sigmaX: QGGMRF parameter */
        double bFace;               /* bFace: relative neighbor weight: cube faces */
        double bEdge;               /* bEdge: relative neighbor weight: cube edges */
        double bVertex;             /* bVertex: relative neighbor weight: cube vertices */
    /* Proximal Mapping */
        double sigma_lambda;        /* sigma_lambda: Proximal mapping scalar */
        int is_positivity_constraint;
        int isTGGMRF;
    
    /**
     *      Stopping Conditions
     */
    double stopThresholdChange_pct;           /* stop threshold (%) */
    double stopThesholdRWFE_pct;
    double stopThesholdRUFE_pct;
    double stopThesholdRRMSE_pct;
    int MaxIterations;              /* maximum number of iterations */
    char relativeChangeMode[200];
    double relativeChangeScaler;
    double relativeChangePercentile;

    /**
     *      Intermediate Saving
     */
    int downsampleFactorSino; 
    int downsampleFactorRecon;             
    char downsampleFNamePrefix[1000]; 

    /**
     *      Zipline Stuff
     */
    int N_G;      /*(*)*/           /* Number of groups for group ICD */
    int zipLineMode;                /* Zipline mode: (0: off, 1: conventional Zipline, 2: randomized Zipline) */
    int numVoxelsPerZiplineMax;
    int numVoxelsPerZipline;
    int numZiplines;

    /**
     *      Parallel Stuff
     */
    int numThreads;                 /* numThreads: Number of threads */

    /**
     *      Weight scaler stuff
     */
    char weightScaler_estimateMode[200];     /* Estimate weight scaler? 1: Yes. 0: Use user specified value */
    char weightScaler_domain[200];     
    double weightScaler_value;            /* User specified weight scaler */


    /* NHICD stuff */
    char NHICD_Mode[200];
    double NHICD_ThresholdAllVoxels_ErrorPercent;
    double NHICD_percentage;
    double NHICD_random;

    /* Misc */
    int verbosity;
    int isComputeCost;
    int isPhantomReconReference;
    char backprojlike_type[200]; 
};

struct SysMatrix
{ 
    /* Stored in the file in the following order: */
    long int i_vstride_max;        /* max_{i,j}(i_vstride) */
    long int i_wstride_max;        /* max_{i,j}(i_wstride) */
    long int N_u;
    double B_ij_max;
    double C_ij_max;
    double B_ij_scaler;  /* B_ij_true = B_ij * B_ij_scaler*/
    double C_ij_scaler;  /* C_ij_true = C_ij * C_ij_scaler*/

    double Delta_u;     /* = Delta_xy / rho      */
    double u_0;
    double u_1;

    BIJDATATYPE ***B;                   /* [N_x][N_y][N_beta*i_vstride_max]  */
    INDEXSTARTSTOPDATATYPE ***i_vstart; /* [N_x][N_y][N_beta]           */
    INDEXSTRIDEDATATYPE ***i_vstride;   /* [N_x][N_y][N_beta]           */
    INDEXJUDATATYPE ***j_u;             /* [N_x][N_y][N_beta]           */

    CIJDATATYPE **C;                    /* [N_u][N_z*i_wstride_max]          */
    INDEXSTARTSTOPDATATYPE **i_wstart;  /* [N_u][N_z]                   */
    INDEXSTRIDEDATATYPE **i_wstride;    /* [N_u][N_z]                   */


};


struct ICDInfo3DCone
{
    /**
     *      The following needs to be computed BEFORE "ICDStep3DCone" is called
     */
    long int j_x ; /* Index j_x of Pixel being updated */
    long int j_y ; /* Index j_y of Pixel being updated */
    long int j_z ; /* Index j_z of Pixel being updated */

    float neighborsFace[6], neighborsEdge[12], neighborsVertex[8];
    float lastChange;
    double old_xj; /* current pixel value */
    double wghtRecon_j;

    /**
     *      The following is arbitrary when "ICDStep3DCone" is called
     *      and will be used as as variables inside "ICDStep3DCone".
     */
    double Delta_xj;          /* Delta_xj = x_j^new - x_j^old */
    double proxMapInput_j;    /* jth voxel of the proxMapInput */

    double theta1_f;
    double theta2_f;
    double theta1_p_QGGMRF;
    double theta2_p_QGGMRF;
    double theta1_p_proxMap;
    double theta2_p_proxMap;

};

struct PartialTheta
{
    double t1;
    double t2;
};

struct ParallelAux
{
    int numThreads;
    int N_M_max;
    struct PartialTheta **partialTheta;     /* [numThreads][N_M_max] */
    long int *j_u;
    long int *i_v;
    double *B_ij;
    long int *k_M;
    long int *j_z;
    long int *i_w;
    double *A_ij;
};

struct SpeedAuxICD
{
    long int numberUpdatedVoxels;
    double tic;
    double toc;
    double voxelsPerSecond;    
};

struct IterationStatistics
{
    double cost;
    double relUpdate;
    double weightScaler_value;
    double voxelsPerSecond;
    double ticToc_iteration;
};


struct ReconAux
{ 
    int NHICD_isPartialUpdateActive;
    long int *NHICD_numUpdatedVoxels;
    double *NHICD_totalValueChange;
    int *NHICD_isPartialZiplineHot;
    
    double lastChangeThreshold;
    int N_M_max;
    double totalEquits;

    double relativeWeightedForwardError;
    double relativeUnweightedForwardError;

    double TotalValueChange;
    double TotalVoxelValue;
    long int NumUpdatedVoxels;

    float NHICD_neighborFilter[3][3];

};




void writeSysMatrix(char *fName, struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A);

void readSysMatrix(char *fName, struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A);


void forwardProject3DCone( float ***Ax, float ***x, struct ImageParams *imgParams, struct SysMatrix *A, struct SinoParams *sinoInfo);

void backProjectlike3DCone( float ***x_out, float ***y_in, struct ImageParams *imgParams, struct SysMatrix *A, struct SinoParams *sinoParams, struct ReconParams *reconParams);

void initializeWghtRecon(struct SysMatrix *A, struct Sino *sino, struct Image *img, struct ReconParams *reconParams);

double computeAvgWghtRecon(struct Image *img);
    
void computeSecondaryReconParams(struct ReconParams *reconParams, struct ImageParams *imgParams);

void invertDoubleMatrix(double **A, double ** A_inv, int size);

double computeNormSquaredFloatArray(float *arr, long int len);

double computeRelativeRMSEFloatArray(float *arr1, float *arr2, long int len);

double computeSinogramWeightedNormSquared(struct Sino *sino, float ***arr);

char isInsideMask(long int i_1, long int i_2, long int N1, long int N2);

long int computeNumVoxelsInImageMask(struct Image *img);

void copyImage2ROI(struct Image *img);

void applyMask(float ***arr, long int N1, long int N2, long int N3);

void floatArray_z_equals_aX_plus_bY(float *Z, double a, float *X, double b, float *Y, long int len);

void setFloatArray2Value(float *arr, long int len, float value);

void setUCharArray2Value(unsigned char *arr, long int len, unsigned char value);

void*** allocateSinoData3DCone(struct SinoParams *params, int dataTypeSize);

void*** allocateImageData3DCone( struct ImageParams *params, int dataTypeSize, int isROI);

void allocateSysMatrix(struct SysMatrix *A, long int N_x, long int N_y, long int N_z, long int N_beta, long int i_vstride_max, long int i_wstride_max, long int N_u);


void freeSysMatrix(struct SysMatrix *A);

void freeViewAngleList(struct ViewAngleList *list);


/**************************************** stuff for random update ****************************************/
void RandomZiplineAux_allocate(struct RandomZiplineAux *aux, struct ImageParams *imgParams, struct ReconParams *reconParams);

void RandomZiplineAux_Initialize(struct RandomZiplineAux *aux, struct ImageParams *imgParams, struct ReconParams *reconParams, int N_M_max);

void RandomAux_allocate(struct RandomAux *aux, struct ImageParams *imgParams);

void RandomAux_Initialize(struct RandomAux *aux, struct ImageParams *imgParams);

void RandomZiplineAux_free(struct RandomZiplineAux *aux);

void RandomAux_free(struct RandomAux *aux);


void RandomZiplineAux_ShuffleGroupIndices(struct RandomZiplineAux *aux, struct ImageParams *imgParams);

void RandomZiplineAux_ShuffleGroupIndices_FixedDistance(struct RandomZiplineAux *aux, struct ImageParams *imgParams);

void RandomZiplineAux_shuffleOrderXY(struct RandomZiplineAux *aux, struct ImageParams *imgParams);


void indexExtraction2D(long int j_xy, long int *j_x, long int N_x, long int *j_y, long int N_y);

void shuffleIntArray(int *arr, long int len);

void shuffleLongIntArray(long int *arr, long int len);

int bernoulli(double p);

long int uniformIntegerRV(long int l, long int h);

long int almostUniformIntegerRV(double mean, int sigma);



/**************************************** tic toc ****************************************/
void tic(double *ticToc);

void toc(double *ticToc);

void ticTocDisp(double ticToc, char *ticTocName);

/**************************************** timer ****************************************/

void timer_reset(double *timer);

int timer_hasPassed(double *timer, double time_passed);


/**************************************** percentile stuff ****************************************/

long int partition(float arr[], long int left, long int right);

float kthSmallest(float arr[], long int l, long int r, long int k);

float prctile(float arr[], long int len, float p);

float prctile_copyFast(float arr[], long int len, float p, int subsampleFactor);

/**************************************** IO ****************************************/

long int keepWritingToBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName);

long int keepReadingFromBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName);


void printFileIOInfo( char* message, char* fName, long int size, char mode);

void printProgressOfLoop( long int indexOfLoop, long int NumIterations);



void logAndDisp_message(char *fName, char* message);

void log_message(char *fName, char* message);

void resetFile(char *fName);


void printSinoParams(struct SinoParams *params);

void printImgParams(struct ImageParams *params);

void printReconParams(struct ReconParams *params);

void printSysMatrixParams(struct SysMatrix *A);

#endif /* MBIR_MODULAR_UTILITIES_3D_H */
