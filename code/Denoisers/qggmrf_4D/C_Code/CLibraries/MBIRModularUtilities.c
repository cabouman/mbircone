

#include "MBIRModularUtilities.h"
#include "io4d.h"



void allocateImageData( struct Image *img)
{
    img->noisy = (float****)mem_alloc_4D(img->params.N_t, img->params.N_x, img->params.N_y, img->params.N_z, sizeof(float));
    img->denoised = (float****)mem_alloc_4D(img->params.N_t, img->params.N_x, img->params.N_y, img->params.N_z, sizeof(float));
}

void free_imageData( struct Image *img)
{
    mem_free_4D((void****)(img->noisy));
    mem_free_4D((void****)(img->denoised));
}

void initializeDenoisedImg_scalar( struct Image *img, float val )
{
    int j_t, j_x, j_y, j_z;

    for (j_t = 0; j_t < img->params.N_t; ++j_t){
        for (j_x = 0; j_x < img->params.N_x; ++j_x){
            for (j_y = 0; j_y < img->params.N_y; ++j_y){
                for (j_z = 0; j_z < img->params.N_z; ++j_z){
                    img->denoised[j_t][j_x][j_y][j_z] = val;
                }
            }
        }
    }
}

void initializeDenoisedImg_fromNoisy( struct Image *img )
{
    int j_t, j_x, j_y, j_z;

    for (j_t = 0; j_t < img->params.N_t; ++j_t){
        for (j_x = 0; j_x < img->params.N_x; ++j_x){
            for (j_y = 0; j_y < img->params.N_y; ++j_y){
                for (j_z = 0; j_z < img->params.N_z; ++j_z){
                    img->denoised[j_t][j_x][j_y][j_z] = img->noisy[j_t][j_x][j_y][j_z];
                }
            }
        }
    }

}

void writeImageData(struct PathNames *pathNames, struct Image *img)
{

    int j_t, N_t;

    N_t = img->params.N_t;
    
    if( N_t != pathNames->N_t ){
        fprintf(stderr, "writeImageData: N_t mismatch\n");
        exit(-1);
    }

    for(j_t=0; j_t<N_t; j_t++){
        /* Write a 3D volume for each time point */
        write3DData(pathNames->noisyImageNames[j_t], (void***)(img->noisy[j_t]), img->params.N_x, img->params.N_y, img->params.N_z, "float");
        write3DData(pathNames->denoisedImageNames[j_t], (void***)(img->denoised[j_t]), img->params.N_x, img->params.N_y, img->params.N_z, "float");
    }

    
}

void readImageData(struct PathNames *pathNames, struct Image *img)
{

    int j_t, N_t;
    FILE* fp;
    int isDenoisedOnDisk;

    N_t = img->params.N_t;
    
    if( N_t != pathNames->N_t ){
        fprintf(stderr, "writeImageData: N_t mismatch\n");
        exit(-1);
    }

    for(j_t=0; j_t<N_t; j_t++){
        /* Read a 3D volume for each time point */
        read3DData(pathNames->noisyImageNames[j_t], (void***)(img->noisy[j_t]), img->params.N_x, img->params.N_y, img->params.N_z, "float");
    }

    /* See if denoised images exist on disk*/
    isDenoisedOnDisk = 1; 
    for(j_t=0; j_t<N_t && isDenoisedOnDisk==1 ; j_t++){
        fp = fopen(pathNames->denoisedImageNames[j_t], "r" );
        if(fp == NULL){
            isDenoisedOnDisk = 0;
        }
    }

    /* If denoised files don't exist then initialize by noisy */
    if( isDenoisedOnDisk == 1 ){
        for(j_t=0; j_t<N_t; j_t++){
            /* Read a 3D volume for each time point */
            read3DData(pathNames->denoisedImageNames[j_t], (void***)(img->denoised[j_t]), img->params.N_x, img->params.N_y, img->params.N_z, "float");
        }
    }
    else{
        printf("Denoised Image not on disk\n");
        for(j_t=0; j_t<N_t; j_t++){
            initializeDenoisedImg_fromNoisy( img );
        }
    }

}

/**************************************** stuff for random update ****************************************/

void RandomAux_allocate(struct RandomAux *aux, struct ImageParams *imgParams)
{
    aux->orderTXYZ = mem_alloc_1D(imgParams->N_t * imgParams->N_x * imgParams->N_y * imgParams->N_z, sizeof(int));
}

void RandomAux_Initialize(struct RandomAux *aux, struct ImageParams *imgParams)
{

    int N_t, N_x, N_y, N_z;
    int j_txyz;

    N_t = imgParams->N_t;
    N_x = imgParams->N_x;
    N_y = imgParams->N_y;
    N_z = imgParams->N_z;

    /**
     *      Initialize orderXY
     */
    for (j_txyz = 0; j_txyz < N_t*N_x*N_y*N_z; ++j_txyz)
    {
        aux->orderTXYZ[j_txyz] = j_txyz;
    }

}

void RandomAux_free(struct RandomAux *aux)
{
    mem_free_1D((void*)aux->orderTXYZ);
}


void shuffleIntArray(int *arr, int len)
{
    int target_idx, candidate_idx, target, candidate;

    /*srand(time(NULL));*/
    
    for (target_idx = 0; target_idx < len-1; target_idx++)
    {
        candidate_idx = target_idx + (rand() % (len-target_idx));

        /* Swap target and candidate */
        candidate = arr[candidate_idx];
        target = arr[target_idx];
        arr[candidate_idx] = target;
        arr[target_idx] = candidate;
    }
}

/**************************************** tic toc ****************************************/
void tic(double *ticToc)
{
    (*ticToc) = -omp_get_wtime();
}

void toc(double *ticToc)
{
    (*ticToc) += omp_get_wtime();
}

void ticToc_logAndDisp(double ticToc, char *ticTocName)
{
    char str[1000];

    sprintf(str, "\n[ticToc] %s = %e s\n", ticTocName, ticToc);
    logAndDisp_message(LOG_TIME, str);

}

/**************************************** timer ****************************************/
void timer_reset(double *timer)
{
    (*timer) = -omp_get_wtime();
}

int timer_hasPassed(double *timer, double time_passed)
{
    double time_now;
    time_now = omp_get_wtime();
    if ((*timer) + time_now > time_passed )
    {
        (*timer) = -time_now; /* reset timer */
        return 1;
    }
    else
    {
        return 0;
    }
}


/* ################################################################################ */
/* ################################################################################ */
/* ################################################################################ */

/* ---- Different RMSE Measures -------*/

double computeL2NormSquared(float ****arr, int N_t, int N_x, int N_y, int N_z)
{
    /**
     *      computes || arr ||^2
     */
    int j_t, j_x, j_y, j_z;
    double normSquared = 0;


    for (j_t = 0; j_t < N_t; ++j_t){
        for (j_x = 0; j_x < N_x; ++j_x){
            for (j_y = 0; j_y < N_y; ++j_y){
                for (j_z = 0; j_z < N_z; ++j_z){
                    normSquared += pow( arr[j_t][j_x][j_y][j_z] , 2);
                }
            }
        }
    }

    return normSquared;
}

double computeRMS(float ****arr, int N_t, int N_x, int N_y, int N_z)
{
    /**
     *      rmse(x) = sqrt(    (1/N) || x ||^2    )
     */
    long long int N;
    N = N_t * N_x * N_y * N_z;
    return sqrt( computeL2NormSquared(arr, N_t, N_x, N_y, N_z) /  N );
}

int isWithinVol(int j_t, int j_x, int j_y, int j_z, struct Image *img)
{
    if( j_t >= 0 &&  j_x >= 0 && j_y >= 0 && j_z >= 0 && j_t < img->params.N_t &&  j_x < img->params.N_x && j_y < img->params.N_y && j_z < img->params.N_z )
    {
        return 1;
    }
    else
    {
        return 0;
    }

}





