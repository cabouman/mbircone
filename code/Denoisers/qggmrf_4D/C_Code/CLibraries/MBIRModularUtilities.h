#ifndef MBIR_MODULAR_UTILITIES_H
#define MBIR_MODULAR_UTILITIES_H


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "allocate.h"
#include <omp.h>

#define LOG_PROGRESS "log_progress.txt"
#define LOG_STATS "log_stats.m"
#define LOG_ICDLOOP "log_ICDLoop.txt"
#define LOG_TIME "log_time.txt"

#define OUTPUT_REFRESH_TIME 1.0

#define _MIN_(a, b) ((a)<(b) ? (a) : (b))
#define _MAX_(a, b) ((a)>(b) ? (a) : (b))
#define _ABS_(a)    ((a)>0 ? (a) : (-a))


#define PI 3.1415926535897932384

#define STRLEN 500

#define DEBUG_FLAG 0


struct PathNames
{
    char masterFile[STRLEN];
    char plainParamsFile[STRLEN];

    char noisyBinaryFName_timeList[STRLEN];
    char denoisedBinaryFName_timeList[STRLEN];

    int N_t;
    char **noisyImageNames;
    char **denoisedImageNames;
    
};


struct RandomAux
{
   int *orderTXYZ;          
};


struct ImageParams
{    
    /* Number of voxels in t, x, y, z direction. */
    int N_t;
    int N_x;
    int N_y;
    int N_z;
};


struct Image
{
    struct ImageParams params;
    float ****noisy;                /* [N_t][N_x][N_y][N_z] */
    float ****denoised;             /* [N_t][N_x][N_y][N_z] */
    struct RandomAux randomAux;
};


struct Params
{
    int is_positivity_constraint;

    /* QGGMRF */
        double q;                   /* q: QGGMRF parameter (q>1, typical choice q=2) */
        double p;                   /* p: QGGMRF parameter (1<=p<q) */
        double T_s;                   /* T: QGGMRF parameter */
        double T_t;                   /* T: QGGMRF parameter */
        double sigma_s;              /* sigma_s: QGGMRF parameter */
        double sigma_t;              /* sigma_t: QGGMRF parameter */

    double sigma;                   /* Noise standard deviation */

    int spacePriorMode;
    int isTimePrior;
    
    double stopThreshold;
    int MaxIterations;

    int isSaveImage;                /* isSaveImage: Save the image to file after each iteration (TRUE: 1, FALSE: 0) */
    int verbose;

};

struct NeighborhoodInfo
{

    int numNeighbors;
    int numNeighbors_s;
    int numNeighbors_t;

    int *j_t_arr;
    int *j_x_arr;
    int *j_y_arr;
    int *j_z_arr;

    double *neighborWts;
};

struct ICDInfo
{
    int j_t ; /* Index j_t of Pixel being updated */
    int j_x ; /* Index j_x of Pixel being updated */
    int j_y ; /* Index j_y of Pixel being updated */
    int j_z ; /* Index j_z of Pixel being updated */

    struct NeighborhoodInfo *neighborhoodInfo;

    double old_xj; /* current pixel value */

    /**
     *      The following is arbitrary when "ICDStep3DCone" is called
     *      and will be used as as variables inside "ICDStep3DCone".
     */
    double Delta_xj;          /* Delta_xj = x_j^new - x_j^old */

    double theta1, theta2;
    double theta1_F, theta2_F; /* theta forward */
    double theta1_P, theta2_P; /* theta prior */

};

struct SpeedAuxICD
{
    int numberUpdatedVoxels;
    double tic;
    double toc;
    double voxelsPerSecond;    
};

struct IterationStatistics
{
    double cost;
    double relUpdatePercent;
    double weightScaler;
    double voxelsPerSecond;
    double ticToc_iteration;
};


void allocateImageData( struct Image *img);

void free_imageData( struct Image *img);

void initializeDenoisedImg_scalar( struct Image *img, float val );

void writeImageData(struct PathNames *pathNames, struct Image *img);

void readImageData(struct PathNames *pathNames, struct Image *img);


/**************************************** stuff for random update ****************************************/

void shuffleIntArray(int *arr, int len);

/**************************************** tic toc ****************************************/
void tic(double *ticToc);

void toc(double *ticToc);

void ticToc_logAndDisp(double ticToc, char *ticTocName);

/**************************************** timer ****************************************/

void timer_reset(double *timer);

int timer_hasPassed(double *timer, double time_passed);

/* ----- RMSE Measurement ---- */
double computeL2NormSquared(float ****arr, int N_t, int N_x, int N_y, int N_z);
double computeRMS(float ****arr, int N_t, int N_x, int N_y, int N_z);

/* ----- Others ---- */
int isWithinVol(int j_t, int j_x, int j_y, int j_z, struct Image *img);


#endif /* MBIR_MODULAR_UTILITIES_H */
