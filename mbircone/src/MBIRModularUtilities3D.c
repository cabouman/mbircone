#include "MBIRModularUtilities3D.h"

void forwardProject3DCone( float *Ax, float *x, struct ImageParams *imgParams, struct SysMatrix *A, struct SinoParams *sinoParams)
{
    long int j_u, j_x, j_y, i_beta, i_v, j_z, i_w;
    float B_ij, B_ij_times_x_j;

    setFloatArray2Value( &Ax[0], sinoParams->N_beta*sinoParams->N_dv*sinoParams->N_dw, 0);


    #pragma omp parallel for private(j_x, j_y, j_u, i_v, B_ij, j_z, B_ij_times_x_j, i_w)
    for (i_beta = 0; i_beta <= sinoParams->N_beta-1; ++i_beta)
    {

        for (j_x = 0; j_x <= imgParams->N_x-1; ++j_x)
        {
            for (j_y = 0; j_y <= imgParams->N_y-1; ++j_y)
            {
                j_u = A->j_u[j_x][j_y][i_beta];
                for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v < A->i_vstart[j_x][j_y][i_beta]+A->i_vstride[j_x][j_y][i_beta] ; ++i_v)
                {
                    B_ij = A->B_ij_scaler * A->B[j_x][j_y][i_beta*A->i_vstride_max + i_v-A->i_vstart[j_x][j_y][i_beta]];
                    for (j_z = 0; j_z <= imgParams->N_z-1; ++j_z)
                    {
                        B_ij_times_x_j = B_ij * x[index_3D(j_x,j_y,j_z,imgParams->N_y,imgParams->N_z)];    
                        for (i_w = A->i_wstart[j_u][j_z]; i_w < A->i_wstart[j_u][j_z]+A->i_wstride[j_u][j_z]; ++i_w)
                        {
                            Ax[index_3D(i_beta,i_v,i_w,sinoParams->N_dv,sinoParams->N_dw)] += B_ij_times_x_j * A->C_ij_scaler * A->C[j_u][j_z*A->i_wstride_max + i_w-A->i_wstart[j_u][j_z]];
                        }
                    }
                }
            }
        }
    }
}

void backProject3DCone( float *Ax, float *x, struct ImageParams *imgParams, struct SysMatrix *A, struct SinoParams *sinoParams)
{

    long int j_u, j_x, j_y, i_beta, i_v, j_z, i_w;
    float B_ij, A_ij;
    double ticToc;

    tic(&ticToc);
    
    // initialize image array to zero
    setFloatArray2Value( &x[0], imgParams->N_x*imgParams->N_y*imgParams->N_z, 0);    
   
    // compute x = A^t y 
    #pragma omp parallel for private(j_x, j_y, j_u, i_v, B_ij, j_z, i_w, A_ij)
    for (i_beta = 0; i_beta <= sinoParams->N_beta-1; ++i_beta)
    {

        for (j_x = 0; j_x <= imgParams->N_x-1; ++j_x)
        {
            for (j_y = 0; j_y <= imgParams->N_y-1; ++j_y)
            {
                if(isInsideMask(j_x, j_y, imgParams->N_x, imgParams->N_y))
                {
                    j_u = A->j_u[j_x][j_y][i_beta];
                    for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v < A->i_vstart[j_x][j_y][i_beta]+A->i_vstride[j_x][j_y][i_beta] ; ++i_v)
                    {
                        B_ij = A->B_ij_scaler * A->B[j_x][j_y][i_beta*A->i_vstride_max + i_v-A->i_vstart[j_x][j_y][i_beta]];
                        for (j_z = 0; j_z <= imgParams->N_z-1; ++j_z)
                        {
                            for (i_w = A->i_wstart[j_u][j_z]; i_w < A->i_wstart[j_u][j_z]+A->i_wstride[j_u][j_z]; ++i_w)
                            {
                                A_ij = B_ij * A->C_ij_scaler * A->C[j_u][j_z*A->i_wstride_max + i_w-A->i_wstart[j_u][j_z]];
                                

                                /* normal backprojection */
				x[index_3D(j_x,j_y,j_z,imgParams->N_y,imgParams->N_z)] += A_ij * Ax[index_3D(i_beta,i_v,i_w,sinoParams->N_dv,sinoParams->N_dw)];
                            }
                        }
                    }
                }
            }
        }
    }


    toc(&ticToc);
    ticTocDisp(ticToc, "backProject3DCone");

}


void computeSecondaryReconParams(struct ReconParams *reconParams, struct ImageParams *imgParams)
{
    float sum;
    int N_max, N_z;

    sum = 0;
    sum += reconParams->bFace>=0 ? 6*reconParams->bFace : 0;
    sum += reconParams->bEdge>=0 ? 12*reconParams->bEdge : 0;
    sum += reconParams->bVertex>=0 ? 8*reconParams->bVertex : 0;
    if (sum<=0)
    {
        fprintf(stderr, "Error in computeSecondaryReconParams: at least one neighbor weight needs to be positive\n");
        exit(-1);
    }

    if(reconParams->bFace>=0)
        reconParams->bFace /= sum;

    if(reconParams->bEdge>=0)
        reconParams->bEdge /= sum;

    if(reconParams->bVertex>=0)
        reconParams->bVertex /= sum;


    N_z = imgParams->N_z;
    N_max = reconParams->numVoxelsPerZiplineMax;
    reconParams->numVoxelsPerZipline = ceil((float)N_z / ceil((float)N_z/N_max));

    reconParams->numZiplines = ceil((float)N_z / reconParams->numVoxelsPerZipline);

}

void invertDoubleMatrix(float **A, float ** A_inv, int size)
{
    float det;
    if(size == 1)
    {
        A_inv[0][0] = 1.0 / A[0][0];
        return;
    }
    if(size == 2)
    {
        /**
         *          [ a b ]                     1    [  d  -b ]
         *     A =  [ c d ]   then    A^-1 = ------- [ -c   a ]
         *                                   ad - bc
         */
        det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
        A_inv[0][0] =   A[1][1] / det;
        A_inv[0][1] = - A[0][1] / det;
        A_inv[1][0] = - A[1][0] / det;
        A_inv[1][1] =   A[0][0] / det;
        return;
    }
    if(size>2)
    {
        fprintf(stderr, "Error in invertDoubleMatrix: only works for sizes 1 to 2\n");
        exit(-1);
    }
}

float computeNormSquaredFloatArray(float *arr, long int len)
{
    /* out = ||x||^2*/
    long int i;
    float out = 0;
    for (i = 0; i < len; ++i)
    {
        out += (arr[i]*arr[i]);
    }
    return out;
}

float computeRelativeRMSEFloatArray(float *arr1, float *arr2, long int len)
{
    /**
     *      out = sqrt(numerator/denominator)
     *
     *      numerator = ||x1-x2||^2     denominator = ||max(x1,x2)||^2
     */
    long int i;
    float numerator = 0, denominator = 0, m;
    for (i = 0; i < len; ++i)
    {
        numerator += (arr1[i]-arr2[i])*(arr1[i]-arr2[i]);
        m = _MAX_(arr1[i], arr2[i]);
        denominator += m*m;
    }
    return sqrt(numerator/denominator);
}

float computeImageWeightedNormSquared(struct Image *img, float *arr)
{
    /**
     *                      1  ||     ||2   
     *      normError    = --- || arr ||  
     *                      M  ||     ||L 
     *
     *      normError = weightScaler_value
     * 
     *      Weight_true = Weight / weightScaler_value
     */
    long int i_x, i_y, i_z;
    long int num_mask;
    float normError = 0;

    for (i_x = 0; i_x < img->params.N_x; ++i_x)
    for (i_y = 0; i_y < img->params.N_y; ++i_y)
    for (i_z = 0; i_z < img->params.N_z; ++i_z)
    {
        normError += arr[index_3D(i_x,i_y,i_z,img->params.N_y,img->params.N_z)] * arr[index_3D(i_x,i_y,i_z,img->params.N_y,img->params.N_z)];
    }

    num_mask = img->params.N_x * img->params.N_y * img->params.N_z;
    
    normError /= num_mask;

    return normError;
}



float computeSinogramWeightedNormSquared(struct Sino *sino, float *arr)
{
    /**
     *                      1  ||     ||2   
     *      normError    = --- || arr ||  
     *                      M  ||     ||L 
     *
     *      normError = weightScaler_value
     * 
     *      Weight_true = Weight / weightScaler_value
     */
    long int i_beta, i_v, i_w;
    long int num_mask;
    float normError = 0;

    for (i_beta = 0; i_beta < sino->params.N_beta; ++i_beta)
    for (i_v = 0; i_v < sino->params.N_dv; ++i_v)
    for (i_w = 0; i_w < sino->params.N_dw; ++i_w)
    {
        normError += arr[index_3D(i_beta,i_v,i_w,sino->params.N_dv,sino->params.N_dw)] * sino->wgt[index_3D(i_beta,i_v,i_w,sino->params.N_dv,sino->params.N_dw)] * arr[index_3D(i_beta,i_v,i_w,sino->params.N_dv,sino->params.N_dw)];
    }

    num_mask = sino->params.N_beta * sino->params.N_dv * sino->params.N_dw;
    
    normError /= num_mask;

    return normError;
}


char isInsideMask(long int i_1, long int i_2, long int N1, long int N2)
{
    /**
     *      returns 1 iff pixel is inside the circle whose diameter corresponds to the largest dimension of the rectangle
     */
    float center_1, center_2;
    float radius;
    float reldistance;

    center_1 = (N1-1.0)/2.0;
    center_2 = (N2-1.0)/2.0;

    radius = _MAX_(N1/2.0, N2/2.0);

    reldistance = pow((i_1-center_1)/radius, 2) + pow((i_2-center_2)/radius, 2);
    return (reldistance<1 ? 1 : 0);
}

long int computeNumVoxelsInImageMask(struct Image *img)
{
    long int j_x, j_y;
    long int count = 0;

    for (j_x = 0; j_x < img->params.N_x; ++j_x)
    for (j_y = 0; j_y < img->params.N_y; ++j_y)
    {
        count = count + 1*isInsideMask(j_x, j_y, img->params.N_x, img->params.N_y);
    }

    count = count*img->params.N_z;
    return count;
}


void copyImage2ROI(struct Image *img)
{
    long int j_x, j_y, j_z;
    long int j_xstart, j_xstop, j_ystart, j_ystop, j_zstart, j_zstop;
    long int N_x_roi, N_y_roi;


    j_xstart = img->params.j_xstart_roi;
    j_xstop = img->params.j_xstop_roi;
    j_ystart = img->params.j_ystart_roi;
    j_ystop = img->params.j_ystop_roi;
    j_zstart = img->params.j_zstart_roi;
    j_zstop = img->params.j_zstop_roi;

    N_x_roi = img->params.j_xstop_roi - img->params.j_xstart_roi + 1;
    N_y_roi = img->params.j_ystop_roi - img->params.j_ystart_roi + 1;

    
    
    for (j_x = j_xstart; j_x <= j_xstop; ++j_x)
    {
        for (j_y = j_ystart; j_y <= j_ystop; ++j_y)
        {
            for (j_z = j_zstart; j_z <= j_zstop; ++j_z)
            {
                img->vox_roi[j_x-j_xstart][j_y-j_ystart][j_z-j_zstart] = 
                          img->vox[index_3D(j_x,j_y,j_z,img->params.N_y, img->params.N_z)]
                        * isInsideMask(j_x-img->params.j_xstart_roi, j_y-img->params.j_ystart_roi, N_x_roi, N_y_roi);
            }
        }
    }
}

void applyMask3D(float ***arr, long int N1, long int N2, long int N3)
{
    long int i1, i2, i3;
    int b;

    for (i1 = 0; i1 < N1; ++i1)
    {
        for (i2 = 0; i2 < N2; ++i2)
        {
            b = isInsideMask(i1, i2, N1, N2);
            for (i3 = 0; i3 < N3; ++i3)
            {
                arr[i1][i2][i3] *= b;
            }

        }
    }
}

void applyMask(float *arr, long int N1, long int N2, long int N3)
{
    long int i1, i2, i3;
    int b;

    for (i1 = 0; i1 < N1; ++i1)
    {
        for (i2 = 0; i2 < N2; ++i2)
        {
            b = isInsideMask(i1, i2, N1, N2);
            for (i3 = 0; i3 < N3; ++i3)
            {
                arr[index_3D(i1,i2,i3,N2,N3)] *= b;
            }

        }
    }
}

void floatArray_z_equals_aX_plus_bY(float *Z, float a, float *X, float b, float *Y, long int len)
{
    long int i;

    for (i = 0; i < len; ++i)
    {
        Z[i] = a*X[i] + b*Y[i];
    }
}

void setFloatArray2Value(float *arr, long int len, float value)
{
    long int i;

    for (i = 0; i < len; ++i)
    {
        arr[i] = value;
    }
}

void setUCharArray2Value(unsigned char *arr, long int len, unsigned char value)
{
    long int i;

    for (i = 0; i < len; ++i)
    {
        arr[i] = value;
    }
}

void* allocateSinoData3DCone(struct SinoParams *params, int dataTypeSize)
{
    return mget_spc(params->N_beta*params->N_dv*params->N_dw, dataTypeSize);
}

void*** allocateImageData3DCone( struct ImageParams *params, int dataTypeSize, int isROI)
{
    long int N_x_roi, N_y_roi, N_z_roi;

    N_x_roi = params->j_xstop_roi - params->j_xstart_roi + 1;
    N_y_roi = params->j_ystop_roi - params->j_ystart_roi + 1;
    N_z_roi = params->j_zstop_roi - params->j_zstart_roi + 1;

    if (isROI)
    {
        return multialloc(dataTypeSize, 3, (int)N_x_roi, (int)N_y_roi, (int)N_z_roi);
    }
    else
    {
        return multialloc(dataTypeSize, 3, (int)params->N_x, (int)params->N_y, (int)params->N_z);
    }


}


void freeViewAngleList(struct ViewAngleList *list)
{
    free((void*)list->beta);
}


/**************************************** stuff for random update ****************************************/
void RandomZiplineAux_allocate(struct RandomZiplineAux *aux, struct ImageParams *imgParams, struct ReconParams *reconParams)
{
    long int N_x, N_y, N_z;

    N_x = imgParams->N_x;
    N_y = imgParams->N_y;
    N_z = imgParams->N_z;

    /**
     *      Initialize orderXY
     */
    aux->orderXY = mget_spc(N_x * N_y, sizeof(int));

    /**
     *      Initialize groupIndex
     */
    aux->groupIndex = (unsigned char***) multialloc(sizeof(unsigned char***), 3, (int)N_x, (int)N_y, (int)N_z);
}

void RandomZiplineAux_Initialize(struct RandomZiplineAux *aux, struct ImageParams *imgParams, struct ReconParams *reconParams, int N_M_max)
{
    long int N_x, N_y;
    long int j_xy;

    /**
     *      Initialize N_G
     */
    aux->N_G = reconParams->N_G;
    aux->N_M_max = N_M_max;

    N_x = imgParams->N_x;
    N_y = imgParams->N_y;

    /**
     *      Initialize orderXY
     */
    for (j_xy = 0; j_xy < N_x*N_y; ++j_xy)
    {
        aux->orderXY[j_xy] = j_xy;
    }
}

void RandomAux_allocate(struct RandomAux *aux, struct ImageParams *imgParams)
{
    aux->orderXYZ = mget_spc(imgParams->N_x * imgParams->N_y * imgParams->N_z, sizeof(long ));
}

void RandomAux_Initialize(struct RandomAux *aux, struct ImageParams *imgParams)
{

    long int N_x, N_y, N_z;
    long int j_xyz;

    N_x = imgParams->N_x;
    N_y = imgParams->N_y;
    N_z = imgParams->N_z;

    /**
     *      Initialize orderXY
     */
    for (j_xyz = 0; j_xyz < N_x*N_y*N_z; ++j_xyz)
    {
        aux->orderXYZ[j_xyz] = j_xyz;
    }

}

void RandomZiplineAux_free(struct RandomZiplineAux *aux)
{
    free((void*)aux->orderXY);
    multifree((void***)aux->groupIndex, 3);
}

void RandomAux_free(struct RandomAux *aux)
{
    free((void*)aux->orderXYZ);
}



void RandomZiplineAux_ShuffleGroupIndices(struct RandomZiplineAux *aux, struct ImageParams *imgParams)
{
    long int j_x, j_y, j_z, N_G, r;


    N_G = aux->N_G;
/*    srand(time(NULL));    
*/ 

    for(j_x = 0; j_x < imgParams->N_x; j_x++)
    {
        for (j_y = 0; j_y < imgParams->N_y; ++j_y)
        {


            /* random[1,N_G-1]*/
            aux->groupIndex[j_x][j_y][0] = rand() % N_G;

            for (j_z = 1; j_z < imgParams->N_z; ++j_z) 
            {
                /* r \in [1, ..., N_G-1] */
                r = 1 + (rand() % (N_G-1)); 
                /* next index is any of the other N_G-1 indices (uniformly random) */
                aux->groupIndex[j_x][j_y][j_z] = (aux->groupIndex[j_x][j_y][j_z-1] + r) % N_G;
                
            }
        }
    }
}

void RandomZiplineAux_ShuffleGroupIndices_FixedDistance(struct RandomZiplineAux *aux, struct ImageParams *imgParams)
{
    long int j_x, j_y, j_z, N_G, i;
    int *first_N_G_members;

    N_G = aux->N_G;
/*    srand(time(NULL));
*/    first_N_G_members = mget_spc(N_G, sizeof(int));

    /* Initialize first_N_G_members with 0, 1, ..., N_G-1 */
    for (i = 0; i < N_G; ++i)
    {
        first_N_G_members[i] = i;
    }


    for(j_x = 0; j_x < imgParams->N_x; j_x++)
    {
        for (j_y = 0; j_y < imgParams->N_y; ++j_y)
        {
            shuffleIntArray(first_N_G_members, N_G);

            for (j_z = 0; j_z < imgParams->N_z; ++j_z)
            {
                /* output array has the first N_G members repeated */
                aux->groupIndex[j_x][j_y][j_z] = first_N_G_members[j_z % N_G];
            }
        }
    }
}

void RandomZiplineAux_shuffleOrderXY(struct RandomZiplineAux *aux, struct ImageParams *imgParams)
{
    shuffleIntArray(aux->orderXY, imgParams->N_x*imgParams->N_y);
}



void indexExtraction2D(long int j_xy, long int *j_x, long int N_x, long int *j_y, long int N_y)
{
    /* j_xy = j_y + N_y j_x */

    *j_y = j_xy % N_y;
    *j_x = (j_xy - *j_y) / N_y;

    return;
}


void shuffleIntArray(int *arr, long int len)
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

void shuffleLongIntArray(long int *arr, long int len)
{
    long int target_idx, candidate_idx, target, candidate;

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

/**
 *                  { 1     with probability p
 *      bernoulli = { 
 *                  { 0     with probability 1-p
 *
 *      bernoulli(P/100)==1 is true with probability P[%]
 */     
        
int bernoulli(float p)
{
    float r;
    if(p==0)
        return 0;

    if(p==1)
        return 1;

    r = ((float) rand() / (RAND_MAX));
    if(r<p)
        return 1;
    else
        return 0;
}

long int uniformIntegerRV(long int l, long int h)
{
    return l+rand()%(h-l+1);
}

long int almostUniformIntegerRV(float mean, int sigma)
{
    /* creates random integer, Z, variable that is approx uniform in [mean-sigma, mean+sigma] */
    /* "mean" corresponds to the real expectation of Z*/
    /* range(Z) = 2*simga + 1 - delta(mean-ceil(mean)) */
    float mean_low, mean_high;
    float X_low, X_high;
    int b;

    mean_low = floor(mean);
    mean_high = ceil(mean);
    X_low = uniformIntegerRV(mean_low-sigma, mean_low+sigma);
    X_high = uniformIntegerRV(mean_high-sigma, mean_high+sigma);

    b = bernoulli(mean_high-mean);

    return b*X_low + (1-b)*X_high;
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

void ticTocDisp(double ticToc, char *ticTocName)
{
    printf("[ticToc] %s = %e s\n", ticTocName, ticToc);
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

/**************************************** misc ****************************************/

/*
Standard partition process of QuickSort().
It considers the mid element as pivot
and moves all smaller element to left of
it and greater elements to right
*/
long int partition(float arr[], long int left, long int right)
{
    float pivot, temp = 0;
    long int i,j ;
    pivot = arr[right];
    /*printf("(left, right) = (%d, %d)\n", left, right);*/
    i = left; /* everything < i is <= pivot*/
    for (j = left; j <= right - 1; j++)
    {
        if (arr[j] <= pivot)
        {
            _SWAP_(arr[i], arr[j], temp);
            i++;
        }
    }
    _SWAP_(arr[i], arr[right], temp);
    return i; /* end pivot position */
}

/*
This function returns k'th smallest 
element in arr[l..r] using QuickSort 
based method.
WARNING: it will also mix the array around
*/
float kthSmallest(float arr[], long int l, long int r, long int k)
{
    long int pivotIndex;
    /*
    If k is smaller than number of 
    elements in array
    */ 
    if (k > 0 && k <= r - l + 1) {
 
        /*
        Partition the array around last 
        element and get position of pivot 
        element in sorted array
        */ 
        pivotIndex = partition(arr, l, r);
 
        if (pivotIndex - l == k - 1)
            return arr[pivotIndex];

 
        if (pivotIndex - l > k - 1) 
        {
            return kthSmallest(arr, l, pivotIndex - 1, k);
        }
        else
        {
            return kthSmallest(arr, pivotIndex + 1, r, k - pivotIndex + l - 1);
        }
    }
    else
    {
        printf("ERROR in kthSmallest: k = %ld not in [0,...,r - l + 1]=[0,...,%ld]\n", k, r-l+1);
        exit(1);
    } 
}

/* Returns p-th percentile p \in 0 to 100 
WARNING: it will also mix the array around
*/
float prctile(float arr[], long int len, float p)
{
    long int k;
    k = p*(len-1)/100;
    return kthSmallest(arr, 0, len-1, k);
}

/* Returns approximately p-th percentile p \in 0 to 100 */
/* Uses arr(0:subsampleFacor:end-1) */
/* will leave original array unchanged */
float prctile_copyFast(float arr[], long int len, float p, int subsampleFactor)
{
    long int i,len_sub;
    float *arr_sub;
    float result;

    len_sub = len/subsampleFactor;
    arr_sub = mget_spc(len_sub, sizeof(float));

    for (i = 0; i < len_sub; ++i)
    {
        arr_sub[i] = arr[i*subsampleFactor];
    }

    result = prctile(arr_sub, len_sub, p);

    free(arr_sub);
    return result;
}


/* IO routines */



long int keepWritingToBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName)
{
    /* Return number of bytes written */
    long int numElsWritten;

    numElsWritten = fwrite(var, elSize, numEls, fp);
    if(numElsWritten != numEls)
    {
        fprintf(stderr, "ERROR in keepWritingToBinaryFile: file \"%s\" terminated early.\n", fName);
        fprintf(stderr, "Tried to write %li elements of size %d Bytes. Wrote %li elements.\n", numEls, elSize, numElsWritten);

        fclose(fp);
        exit(-1);
    }
    return (long int) numEls * elSize;
}

long int keepReadingFromBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName)
{
    /* Return number of bytes read */
    long int numElsRead;

    numElsRead = fread(var, elSize, numEls, fp);
    if(numElsRead != numEls)
    {
        fprintf(stderr, "ERROR in keepReadingFromBinaryFile: file \"%s\" terminated early.\n", fName);
        fprintf(stderr, "Tried to read %li elements of size %d Bytes. Read %li elements.\n", numEls, elSize, numElsRead);
        fclose(fp);
        exit(-1);
    }
    return (long int) numEls * elSize;
}


void printFileIOInfo( char* functionName, char* fName, long int size, char mode)
{
    char readwrite[200];    /* puts the word "Read" or "Write" into the output */
    switch(mode)
    {
        case 'r':   strcpy(readwrite, "Read "); break;
        case 'w':   strcpy(readwrite, "Write"); break;
        default:    printf("Error in printFileIOInfo: Use mode 'r' or 'w'\n");
                    exit(-1);
    }
    printf("\n");
    printf("    ************** FILE ACCESS ********************************\n");
    printf(" ****  File access in: %s\n", functionName);
    printf("*****  File name     : %s\n", fName);
    printf("*****  %-14s: %-15ld bytes\n", readwrite, size);
    printf("*****                = %-15e kB\n", (float) size*1e-3);
    printf(" ****                = %-15e MB\n", (float) size*1e-6);
    printf("    ***********************************************************\n");
}

void printProgressOfLoop( long int indexOfLoop, long int NumIterations)
{
    float percent;

    percent = (float) (1+indexOfLoop) / (float) NumIterations * 100;
    printf("\r[%.1e%%]", percent );
    fflush(stdout); 

}

void logAndDisp_message(char *fName, char* message)
{
    log_message(fName, message);
    printf("%s", message);
}

void log_message(char *fName, char* message)
{
    FILE *fp;

    fp = fopen(fName, "a");
    if (fp != NULL)
    {
        fprintf(fp, "%s", message);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "WARNING: In log_message: Could not open file %s\n", fName);
    }

}


void resetFile(char *fName)
{
    FILE *filePointer;

    filePointer = fopen(fName, "w");
    fclose(filePointer);
        
}


void copySinoParams(struct SinoParams *params_src, struct SinoParams *params_dest)
{
    params_dest->N_dv = params_src->N_dv;
    params_dest->N_dw = params_src->N_dw;
    params_dest->Delta_dv = params_src->Delta_dv;
    params_dest->Delta_dw = params_src->Delta_dw;
    params_dest->N_beta = params_src->N_beta;
    params_dest->u_s = params_src->u_s;
    params_dest->u_r = params_src->u_r;
    params_dest->v_r = params_src->v_r;
    params_dest->u_d0 = params_src->u_d0;
    params_dest->v_d0 = params_src->v_d0;
    params_dest->w_d0 = params_src->w_d0;
    params_dest->weightScaler_value = params_src->weightScaler_value;
}


void copyImgParams(struct ImageParams *params_src, struct ImageParams *params_dest)
{
    params_dest->x_0 = params_src->x_0;
    params_dest->y_0 = params_src->y_0;
    params_dest->z_0 = params_src->z_0;
    params_dest->N_x = params_src->N_x;
    params_dest->N_y = params_src->N_y;
    params_dest->N_z = params_src->N_z;
    params_dest->Delta_xy = params_src->Delta_xy;
    params_dest->Delta_z = params_src->Delta_z;
    params_dest->j_xstart_roi = params_src->j_xstart_roi;
    params_dest->j_ystart_roi = params_src->j_ystart_roi;
    params_dest->j_zstart_roi = params_src->j_zstart_roi;
    params_dest->j_xstop_roi = params_src->j_xstop_roi;
    params_dest->j_ystop_roi = params_src->j_ystop_roi;
    params_dest->j_zstop_roi = params_src->j_zstop_roi;
}


void printSinoParams(struct SinoParams *params)
{
    printf("\nSinogram parameters read:\n");

    printf("\tN_dv = %ld,\n", params->N_dv);
    printf("\tN_dw = %ld,\n", params->N_dw);
    printf("\tDelta_dv = %e,\n", params->Delta_dv);
    printf("\tDelta_dw = %e,\n", params->Delta_dw);
    printf("\tN_beta = %ld,\n", params->N_beta);
    printf("\tu_s = %e,\n", params->u_s);
    printf("\tu_r = %e,\n", params->u_r);
    printf("\tv_r = %e,\n", params->v_r);
    printf("\tu_d0 = %e,\n", params->u_d0);
    printf("\tv_d0 = %e,\n", params->v_d0);
    printf("\tw_d0 = %e,\n", params->w_d0);
    printf("\t(potentially uninitialized:)\n");
    printf("\tweightScaler_value = %e,\n", params->weightScaler_value);

}

void printImgParams(struct ImageParams *params)
{
    printf("\nImage parameters read:\n");

    printf("\tx_0 = %e \n", params->x_0);
    printf("\ty_0 = %e \n", params->y_0);
    printf("\tz_0 = %e \n", params->z_0);
    printf("\tN_x = %ld \n", params->N_x);
    printf("\tN_y = %ld \n", params->N_y);
    printf("\tN_z = %ld \n", params->N_z);
    printf("\tDelta_xy = %e \n", params->Delta_xy);
    printf("\tDelta_z = %e \n", params->Delta_z);
    printf("\tj_xstart_roi = %ld \n", params->j_xstart_roi);
    printf("\tj_ystart_roi = %ld \n", params->j_ystart_roi);
    printf("\tj_zstart_roi = %ld \n", params->j_zstart_roi);
    printf("\tj_xstop_roi = %ld \n", params->j_xstop_roi);
    printf("\tj_ystop_roi = %ld \n", params->j_ystop_roi);
    printf("\tj_zstop_roi = %ld \n", params->j_zstop_roi);

}


void printReconParams(struct ReconParams *params)
{

    printf("\nReconstruction parameters read:\n");
    printf("\tproximal map mode = %d \n", params->prox_mode);
    printf("\tq = %e \n", params->q);
    printf("\tp = %e \n", params->p);
    printf("\tT = %e \n", params->T);
    printf("\tsigmaX = %e \n", params->sigmaX);
    printf("\tbFace = %e \n", params->bFace);
    printf("\tbEdge = %e \n", params->bEdge);
    printf("\tbVertex = %e \n", params->bVertex);
    printf("\tsigma_lambda = %e \n", params->sigma_lambda);
    printf("\tis_positivity_constraint = %d \n", params->is_positivity_constraint);

    
    printf("\tstopThresholdChange_pct = %e \n", params->stopThresholdChange_pct);
    printf("\tstopThesholdRWFE_pct = %e \n", params->stopThesholdRWFE_pct);
    printf("\tstopThesholdRUFE_pct = %e \n", params->stopThesholdRUFE_pct);
    printf("\tMaxIterations = %d \n", params->MaxIterations);
    printf("\trelativeChangeMode = %s \n", params->relativeChangeMode);
    printf("\trelativeChangeScaler = %e \n", params->relativeChangeScaler);
    printf("\trelativeChangePercentile = %e \n", params->relativeChangePercentile);

    printf("\tN_G = %d \n", params->N_G);
    printf("\tzipLineMode = %d \n", params->zipLineMode);
    printf("\tnumVoxelsPerZiplineMax = %d \n", params->numVoxelsPerZiplineMax);
    printf("\tnumVoxelsPerZipline = %d \n", params->numVoxelsPerZipline);
    printf("\tnumZiplines = %d \n", params->numZiplines);
    printf("\tweightScaler_estimateMode = %s \n", params->weightScaler_estimateMode);
    printf("\tweightScaler_domain = %s \n", params->weightScaler_domain);
    printf("\tweightScaler_value = %e \n", params->weightScaler_value);

    printf("\tNHICD_Mode = %s \n", params->NHICD_Mode);
    printf("\tNHICD_ThresholdAllVoxels_ErrorPercent = %e \n", params->NHICD_ThresholdAllVoxels_ErrorPercent);
    printf("\tNHICD_percentage = %e \n", params->NHICD_percentage);
    printf("\tNHICD_random = %e \n", params->NHICD_random);

    printf("\tverbosity = %d \n", params->verbosity);
    printf("\tisComputeCost = %d \n", params->isComputeCost);

}


void printDenoiseParams(struct ImageParams *img_params, struct ReconParams *denoise_params)
{
    printf("\nImage parameters read:\n");

    printf("\tN_x = %ld \n", img_params->N_x);
    printf("\tN_y = %ld \n", img_params->N_y);
    printf("\tN_z = %ld \n", img_params->N_z);

    printf("\nDenoiser parameters read:\n");
    printf("\tq = %e \n", denoise_params->q);
    printf("\tp = %e \n", denoise_params->p);
    printf("\tT = %e \n", denoise_params->T);
    printf("\tsigmaX = %e \n", denoise_params->sigmaX);
    printf("\tbFace = %e \n", denoise_params->bFace);
    printf("\tbEdge = %e \n", denoise_params->bEdge);
    printf("\tbVertex = %e \n", denoise_params->bVertex);
    printf("\tsigma_lambda = %e \n", denoise_params->sigma_lambda);
    printf("\tis_positivity_constraint = %d \n", denoise_params->is_positivity_constraint);

    printf("\tstopThresholdChange_pct = %e \n", denoise_params->stopThresholdChange_pct);
    printf("\tMaxIterations = %d \n", denoise_params->MaxIterations);
    printf("\trelativeChangeMode = %s \n", denoise_params->relativeChangeMode);
    printf("\tweightScaler_value = %e \n", denoise_params->weightScaler_value);

    printf("\tverbosity = %d \n", denoise_params->verbosity);
}


void printSysMatrixParams(struct SysMatrix *A)
{

    printf("\nSystemMatrix parameters:\n");

    printf("\ti_vstride_max = %ld \n", A->i_vstride_max);
    printf("\ti_wstride_max = %ld \n", A->i_wstride_max);
    printf("\tN_u = %ld \n", A->N_u);
    printf("\tDelta_u = %e \n", A->Delta_u);
    printf("\tu_0 = %e \n", A->u_0);
    printf("\tu_1 = %e \n", A->u_1);
    printf("\tB_ij_max = %e \n", A->B_ij_max);
    printf("\tC_ij_max = %e \n", A->C_ij_max);
    printf("\tB_ij_scaler = %e \n", A->B_ij_scaler);
    printf("\tC_ij_scaler = %e \n", A->C_ij_scaler);

}
