

#include "MBIRModularUtilities3D.h"
#include "io3d.h"

/* write the System matrix to hard drive */
void writeSysMatrix(char *fName, struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A)
{
    FILE *fp;
    long int totsize = 0;
    long int N_x, N_y, N_z, N_beta, i_vstride_max, i_wstride_max, N_u;
    
    logAndDisp_message(LOG_PROGRESS, "\nWrite System Matrix ... \n");
    
    fp = fopen(fName, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "ERROR in WriteSysMatrix: can't open file %s.\n", fName);
        exit(-1);
    }
    
    /**
     *      Writing simple variables
     *      i_vstride_max, i_wstride_max, N_u, Delta_u, u_0 and u_1
     *      to file
     */

    totsize += keepWritingToBinaryFile(fp, &(A->i_vstride_max),     1, sizeof(long int), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->i_wstride_max),     1, sizeof(long int), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->N_u),               1, sizeof(long int), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->B_ij_max),          1, sizeof(double), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->C_ij_max),          1, sizeof(double), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->B_ij_scaler),       1, sizeof(double), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->C_ij_scaler),       1, sizeof(double), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->Delta_u),           1, sizeof(double), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->u_0),               1, sizeof(double), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->u_1),               1, sizeof(double), fName);

    /**
     *      Writing array variables
     *      B, i_vstart, i_vstride, j_u, C, i_wstart and i_wstride
     *      to file
     */
    N_x = imgParams->N_x;
    N_y = imgParams->N_y;
    N_z = imgParams->N_z;
    N_beta = sinoParams->N_beta;
    i_vstride_max = A->i_vstride_max;
    i_wstride_max = A->i_wstride_max;
    N_u = A->N_u;

    totsize += keepWritingToBinaryFile(fp, &(A->B[0][0][0]),        N_x*N_y*N_beta*i_vstride_max,   sizeof(BIJDATATYPE), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->i_vstart[0][0][0]), N_x*N_y*N_beta,                 sizeof(INDEXSTARTSTOPDATATYPE),   fName);
    totsize += keepWritingToBinaryFile(fp, &(A->i_vstride[0][0][0]),N_x*N_y*N_beta,                 sizeof(INDEXSTRIDEDATATYPE),   fName);
    totsize += keepWritingToBinaryFile(fp, &(A->j_u[0][0][0]),      N_x*N_y*N_beta,                 sizeof(INDEXJUDATATYPE),   fName);

    totsize += keepWritingToBinaryFile(fp, &(A->C[0][0]),           N_u*N_z*i_wstride_max,          sizeof(CIJDATATYPE), fName);
    totsize += keepWritingToBinaryFile(fp, &(A->i_wstart[0][0]),    N_u*N_z,                        sizeof(INDEXSTARTSTOPDATATYPE),   fName);
    totsize += keepWritingToBinaryFile(fp, &(A->i_wstride[0][0]),   N_u*N_z,                        sizeof(INDEXSTRIDEDATATYPE),   fName);
    
    printf("Total size written = %e GB\n", totsize/1e9);

    fclose(fp);
 
}



/* read the System matrix to hard drive */
/* Utility for reading the Sparse System Matrix */
/* Returns 0 if no error occurs */
void readSysMatrix(char *fName, struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A)
{

    FILE *fp;
    long int totsize = 0;
    long int N_x, N_y, N_z, N_beta, i_vstride_max, i_wstride_max, N_u;


    fp = fopen(fName, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "ERROR in WriteSysMatrix: can't open file %s.\n", fName);
        exit(-1);
    }
    
    /**
     *      Reading simple variables
     *      i_vstride_max, i_wstride_max, N_u, Delta_u, u_0 and u_1
     *      from file
     */
    
    totsize += keepReadingFromBinaryFile(fp, &(A->i_vstride_max),   1, sizeof(long int), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->i_wstride_max),   1, sizeof(long int), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->N_u),             1, sizeof(long int), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->B_ij_max),        1, sizeof(double), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->C_ij_max),        1, sizeof(double), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->B_ij_scaler),     1, sizeof(double), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->C_ij_scaler),     1, sizeof(double), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->Delta_u),         1, sizeof(double), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->u_0),             1, sizeof(double), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->u_1),             1, sizeof(double), fName);

    /**
     *          Note: Allocation has to happen here (after reading part of the file).
     *          This is because i_vstride_max, i_wstride_max and N_u are unknown before SysMatrix file is read
     *          but they are required to determine the array dimensions
     */
    N_x = imgParams->N_x;
    N_y = imgParams->N_y;
    N_z = imgParams->N_z;
    N_beta = sinoParams->N_beta;
    i_vstride_max = A->i_vstride_max;
    i_wstride_max = A->i_wstride_max;
    N_u = A->N_u;

    allocateSysMatrix(A, N_x, N_y, N_z, N_beta, i_vstride_max, i_wstride_max, N_u);

    /**
     *      Reading array variables
     *      B, i_vstart, i_vstride, j_u, C, i_wstart and i_wstride
     *      from file
     */
    totsize += keepReadingFromBinaryFile(fp, &(A->B[0][0][0]),     N_x*N_y*N_beta*i_vstride_max, sizeof(BIJDATATYPE), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->i_vstart[0][0][0]), N_x*N_y*N_beta,         sizeof(INDEXSTARTSTOPDATATYPE),   fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->i_vstride[0][0][0]),N_x*N_y*N_beta,         sizeof(INDEXSTRIDEDATATYPE),   fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->j_u[0][0][0]),      N_x*N_y*N_beta,         sizeof(INDEXJUDATATYPE),   fName);

    totsize += keepReadingFromBinaryFile(fp, &(A->C[0][0]),        N_u*N_z*i_wstride_max,        sizeof(CIJDATATYPE), fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->i_wstart[0][0]),    N_u*N_z,                sizeof(INDEXSTARTSTOPDATATYPE),   fName);
    totsize += keepReadingFromBinaryFile(fp, &(A->i_wstride[0][0]),   N_u*N_z,                sizeof(INDEXSTRIDEDATATYPE),   fName);
    
    /*printf("Total size read = %e GB\n", totsize/1e9);*/

    fclose(fp);
    
}



void forwardProject3DCone( float ***Ax, float ***x, struct ImageParams *imgParams, struct SysMatrix *A, struct SinoParams *sinoInfo)
{
    long int j_u, j_x, j_y, i_beta, i_v, j_z, i_w;
    double B_ij, B_ij_times_x_j;

    setFloatArray2Value( &Ax[0][0][0], sinoInfo->N_beta*sinoInfo->N_dv*sinoInfo->N_dw, 0);


    #pragma omp parallel for private(j_x, j_y, j_u, i_v, B_ij, j_z, B_ij_times_x_j, i_w)
    for (i_beta = 0; i_beta <= sinoInfo->N_beta-1; ++i_beta)
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
                        B_ij_times_x_j = B_ij * x[j_x][j_y][j_z];    
                        for (i_w = A->i_wstart[j_u][j_z]; i_w < A->i_wstart[j_u][j_z]+A->i_wstride[j_u][j_z]; ++i_w)
                        {
                            Ax[i_beta][i_v][i_w] += B_ij_times_x_j * A->C_ij_scaler * A->C[j_u][j_z*A->i_wstride_max + i_w-A->i_wstart[j_u][j_z]];
                        }
                    }
                }
            }
        }
    }
}

void backProjectlike3DCone( float ***x_out, float ***y_in, struct ImageParams *imgParams, struct SysMatrix *A, struct SinoParams *sinoParams, struct ReconParams *reconParams)
{

    long int j_u, j_x, j_y, i_beta, i_v, j_z, i_w;
    double B_ij, A_ij;
    double ticToc;
    char mode;
    float ***normalization, val, val2;

    if (strcmp(reconParams->backprojlike_type,"proj") == 0)
    {
        mode = 0;
    }
    else if (strcmp(reconParams->backprojlike_type,"entropy") == 0)
    {
        mode = 1;
        normalization = (float***) allocateImageData3DCone( imgParams, sizeof(float), 0);
    }
    else if (strcmp(reconParams->backprojlike_type,"kappa") == 0)
    {
        mode = 2;
    }
    else
    {
        fprintf(stderr, "ERROR in backProjectlike3DCone: can't recongnize backprojlike_type.\n");
        exit(-1);
    }

    logAndDisp_message(LOG_PROGRESS, "\n Computing backProjectlike ...\n");


    tic(&ticToc);
    #pragma omp parallel for private(j_y, j_z)
    for (j_x = 0; j_x <= imgParams->N_x-1; ++j_x)
    {
        for (j_y = 0; j_y <= imgParams->N_y-1 ; ++j_y)
        {
            for (j_z = 0; j_z <= imgParams->N_z-1; ++j_z)
            {
                x_out[j_x][j_y][j_z] = 0;
            }
        }
    }

    printf("mode: %d\n", mode);

    #pragma omp parallel for private(j_x, j_y, j_u, i_v, B_ij, j_z, i_w, A_ij, val)
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
                                

                                if(mode==0){
                                    x_out[j_x][j_y][j_z] += A_ij * y_in[i_beta][i_v][i_w] ;
                                }
                                else if(mode==1){
                                    val = A_ij * y_in[i_beta][i_v][i_w];
                                    if (val!=0){
                                        x_out[j_x][j_y][j_z] += val * log(val) ;
                                    }
                                    normalization[j_x][j_y][j_z] += A_ij * y_in[i_beta][i_v][i_w] ;
                                }
                                else if(mode==2){
                                    x_out[j_x][j_y][j_z] += A_ij * y_in[i_beta][i_v][i_w] * A_ij ;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    printf("mode: %d\n", mode);

    if(mode==1)
    {
        /* compute entropy in bits after normalization */
        #pragma omp parallel for private(j_y, j_z, val, val2)
        for (j_x = 0; j_x <= imgParams->N_x-1; ++j_x)
        {
            for (j_y = 0; j_y <= imgParams->N_y-1; ++j_y)
            {
                for (j_z = 0; j_z <= imgParams->N_z-1; ++j_z)
                {
                    val = x_out[j_x][j_y][j_z];
                    val2 = normalization[j_x][j_y][j_z];
                    if (val2==0){
                        x_out[j_x][j_y][j_z] = 0;
                    }
                    else{
                        x_out[j_x][j_y][j_z] = (log(val2) - val/val2)/log(2);
                    }
                }
            }   
        }
        mem_free_3D((void***)normalization);
    }


    toc(&ticToc);
    ticToc_logAndDisp(ticToc, "backProjectlike3DCone");

}


void initializeWghtRecon(struct SysMatrix *A, struct Sino *sino, struct Image *img, struct ReconParams *reconParams)
{
    long int j_u, j_x, j_y, i_beta, i_v, j_z, i_w;
    double B_ij, A_ij;
    double ticToc, avg;

    logAndDisp_message(LOG_PROGRESS, "\nInitialize WghtRecon ...\n");


    tic(&ticToc);
    #pragma omp parallel for private(j_y, j_z)
    for (j_x = 0; j_x <= img->params.N_x-1; ++j_x)
    {
        for (j_y = 0; j_y <= img->params.N_y-1 ; ++j_y)
        {
            for (j_z = 0; j_z <= img->params.N_z-1; ++j_z)
            {
                img->wghtRecon[j_x][j_y][j_z] = 0;
            }
        }
    }

    #pragma omp parallel for private(j_x, j_y, j_u, i_v, B_ij, j_z, i_w, A_ij)
    for (i_beta = 0; i_beta <= sino->params.N_beta-1; ++i_beta)
    {

        for (j_x = 0; j_x <= img->params.N_x-1; ++j_x)
        {
            for (j_y = 0; j_y <= img->params.N_y-1; ++j_y)
            {
                if(isInsideMask(j_x, j_y, img->params.N_x, img->params.N_y))
                {
                    j_u = A->j_u[j_x][j_y][i_beta];
                    for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v < A->i_vstart[j_x][j_y][i_beta]+A->i_vstride[j_x][j_y][i_beta] ; ++i_v)
                    {
                        B_ij = A->B_ij_scaler * A->B[j_x][j_y][i_beta*A->i_vstride_max + i_v-A->i_vstart[j_x][j_y][i_beta]];
                        for (j_z = 0; j_z <= img->params.N_z-1; ++j_z)
                        {
                            for (i_w = A->i_wstart[j_u][j_z]; i_w < A->i_wstart[j_u][j_z]+A->i_wstride[j_u][j_z]; ++i_w)
                            {
                                A_ij = B_ij * A->C_ij_scaler * A->C[j_u][j_z*A->i_wstride_max + i_w-A->i_wstart[j_u][j_z]];
                                img->wghtRecon[j_x][j_y][j_z] += 0.5 * A_ij * sino->wgt[i_beta][i_v][i_w] * A_ij;
                            }
                        }
                    }
                }
            }
        }
    }

    avg = computeAvgWghtRecon(img);

    toc(&ticToc);
    ticToc_logAndDisp(ticToc, "initializeWghtRecon");

    printf("\n -> Average Weight Scaler = %e\n", avg);


}

double computeAvgWghtRecon(struct Image *img)
{
    long int j_x, j_y, j_z;

    double sum = 0;
    long int num = 0;

    #pragma omp parallel for private(j_y, j_z) reduction(+:sum,num)
    for (j_x = 0; j_x <= img->params.N_x-1; ++j_x)
    {
        for (j_y = 0; j_y <= img->params.N_y-1 ; ++j_y)
        {
            if(isInsideMask(j_x, j_y, img->params.N_x, img->params.N_y))
                for (j_z = 0; j_z <= img->params.N_z-1; ++j_z)
                {
                    sum += img->wghtRecon[j_x][j_y][j_z];
                    num += 1;
                }
        }
    }
    return sum/num;

}


void computeSecondaryReconParams(struct ReconParams *reconParams, struct ImageParams *imgParams)
{
    double sum;
    int N_max, N_z;
    double s, eps, p, q, T, sigmaX;

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
    reconParams->numVoxelsPerZipline = ceil((double)N_z / ceil((double)N_z/N_max));

    reconParams->numZiplines = ceil((double)N_z / reconParams->numVoxelsPerZipline);

    p = reconParams->p;
    q = reconParams->q;

    if (reconParams->isTGGMRF == 1)
    {
        eps = reconParams->T;
        s = reconParams->sigmaX;

        T = pow( (p/q * pow(eps/s, q)) , 1/p);
        sigmaX = eps / T;

        reconParams->T = T;
        reconParams->sigmaX = sigmaX;
    }
    else if (reconParams->isTGGMRF == 0)
    {
        sigmaX = reconParams->sigmaX;
        T = reconParams->T;

        eps = sigmaX * T;
        s = sigmaX * T * pow(p / (pow(T,p) * q), 1/q);
    }
    else
    {
        fprintf(stderr, "ERROR in computeSecondaryReconParams: Unknown isTGGMRF mode\n");
        exit(-1);
    }

    printf("--- sigmaX = %e\n", sigmaX);
    printf("--- T = %e\n", T);
    printf("--- eps = %e\n", eps);
    printf("--- s = %e\n", s);


}

void invertDoubleMatrix(double **A, double ** A_inv, int size)
{
    double det;
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

double computeNormSquaredFloatArray(float *arr, long int len)
{
    /* out = ||x||^2*/
    long int i;
    double out = 0;
    for (i = 0; i < len; ++i)
    {
        out += (arr[i]*arr[i]);
    }
    return out;
}

double computeRelativeRMSEFloatArray(float *arr1, float *arr2, long int len)
{
    /**
     *      out = sqrt(numerator/denominator)
     *
     *      numerator = ||x1-x2||^2     denominator = ||max(x1,x2)||^2
     */
    long int i;
    double numerator = 0, denominator = 0, m;
    for (i = 0; i < len; ++i)
    {
        numerator += (arr1[i]-arr2[i])*(arr1[i]-arr2[i]);
        m = _MAX_(arr1[i], arr2[i]);
        denominator += m*m;
    }
    return sqrt(numerator/denominator);
}


double computeSinogramWeightedNormSquared(struct Sino *sino, float ***arr)
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
    double normError = 0;

    for (i_beta = 0; i_beta < sino->params.N_beta; ++i_beta)
    for (i_v = 0; i_v < sino->params.N_dv; ++i_v)
    for (i_w = 0; i_w < sino->params.N_dw; ++i_w)
    {
        normError += arr[i_beta][i_v][i_w] * sino->wgt[i_beta][i_v][i_w] * arr[i_beta][i_v][i_w];
    }

    num_mask = sino->params.N_beta * sino->params.N_dv * sino->params.N_dw;
    
    normError /= num_mask;

    return normError;
}


char isInsideMask(long int i_1, long int i_2, long int N1, long int N2)
{
    /**
     *      returns 1 iff pixel is inside the ellipse that fits in the rectangle
     */
    double center_1, center_2;
    double radius_1, radius_2;
    double reldistance;

    center_1 = (N1-1.0)/2.0;
    center_2 = (N2-1.0)/2.0;

    radius_1 = N1/2.0;
    radius_2 = N2/2.0;

    reldistance = pow((i_1-center_1)/radius_1, 2) + pow((i_2-center_2)/radius_2, 2);
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
                          img->vox[j_x][j_y][j_z]
                        * isInsideMask(j_x-img->params.j_xstart_roi, j_y-img->params.j_ystart_roi, N_x_roi, N_y_roi);
            }
        }
    }
}

void applyMask(float ***arr, long int N1, long int N2, long int N3)
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

void floatArray_z_equals_aX_plus_bY(float *Z, double a, float *X, double b, float *Y, long int len)
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

void*** allocateSinoData3DCone(struct SinoParams *params, int dataTypeSize)
{
    return mem_alloc_3D(params->N_beta, params->N_dv, params->N_dw, dataTypeSize);
}


void*** allocateImageData3DCone( struct ImageParams *params, int dataTypeSize, int isROI)
{
    long int N_x_roi, N_y_roi, N_z_roi;

    N_x_roi = params->j_xstop_roi - params->j_xstart_roi + 1;
    N_y_roi = params->j_ystop_roi - params->j_ystart_roi + 1;
    N_z_roi = params->j_zstop_roi - params->j_zstart_roi + 1;

    if (isROI)
    {
        return mem_alloc_3D(N_x_roi, N_y_roi, N_z_roi, dataTypeSize);
    }
    else
    {
        return mem_alloc_3D(params->N_x, params->N_y, params->N_z, dataTypeSize);
    }


}

void allocateSysMatrix(struct SysMatrix *A, long int N_x, long int N_y, long int N_z, long int N_beta, long int i_vstride_max, long int i_wstride_max, long int N_u)
{
    /*double totSizeGB;*/
    /*logAndDisp_message(LOG_PROGRESS, "\nAllocate Space for A-matrix...\n");*/

    /*
    totSizeGB =\
    (\
    N_x * N_y * N_beta * i_vstride_max * sizeof(BIJDATATYPE) + \
    N_x * N_y * N_beta * sizeof(INDEXSTARTSTOPDATATYPE) + \
    N_x * N_y * N_beta * sizeof(INDEXSTRIDEDATATYPE) + \
    N_x * N_y * N_beta * sizeof(INDEXJUDATATYPE) + \
    N_u * N_z * i_wstride_max * sizeof(CIJDATATYPE) + \
    N_u * N_z * sizeof(INDEXSTARTSTOPDATATYPE) + \
    N_u * N_z * sizeof(INDEXSTRIDEDATATYPE)\
    )\
    /1e9;*/
   /* printf("\tAllocating %e GB ...\n", totSizeGB);*/


    A->B =          (BIJDATATYPE***)                mem_alloc_3D(N_x, N_y, N_beta*i_vstride_max,    sizeof(BIJDATATYPE));
    A->i_vstart =   (INDEXSTARTSTOPDATATYPE***)     mem_alloc_3D(N_x, N_y, N_beta,                  sizeof(INDEXSTARTSTOPDATATYPE));
    A->i_vstride =    (INDEXSTRIDEDATATYPE***)      mem_alloc_3D(N_x, N_y, N_beta,                  sizeof(INDEXSTRIDEDATATYPE));
    A->j_u =        (INDEXJUDATATYPE***)            mem_alloc_3D(N_x, N_y, N_beta,                  sizeof(INDEXJUDATATYPE));

    A->C =          (CIJDATATYPE**)                mem_alloc_2D(N_u, N_z*i_wstride_max,            sizeof(CIJDATATYPE));
    A->i_wstart =   (INDEXSTARTSTOPDATATYPE**)      mem_alloc_2D(N_u, N_z,                          sizeof(INDEXSTARTSTOPDATATYPE));
    A->i_wstride =    (INDEXSTRIDEDATATYPE**)       mem_alloc_2D(N_u, N_z,                          sizeof(INDEXSTRIDEDATATYPE));
}

void freeSysMatrix(struct SysMatrix *A)
{
    mem_free_3D((void***)A->B);
    mem_free_3D((void***)A->i_vstart);
    mem_free_3D((void***)A->i_vstride);
    mem_free_3D((void***)A->j_u);
    mem_free_2D((void**)A->C);
    mem_free_2D((void**)A->i_wstart);
    mem_free_2D((void**)A->i_wstride);
}

void freeViewAngleList(struct ViewAngleList *list)
{
    mem_free_1D((void*)list->beta);
}

void writeSinoData3DCone(char *fName, void ***sino, struct SinoParams *sinoParams, char *dataType)
{

    write3DData(fName, (void***)sino, sinoParams->N_beta, sinoParams->N_dv, sinoParams->N_dw, dataType);
}

void readSinoData3DCone(char *fName, void ***sino, struct SinoParams *sinoParams, char *dataType)
{
    read3DData(fName, (void***)sino, sinoParams->N_beta, sinoParams->N_dv, sinoParams->N_dw, dataType);
}

void writeImageData3DCone(char *fName, void ***arr, struct ImageParams *params, int isROI, char *dataType)
{
    if (isROI) 
    {
        write3DData(fName, (void***)arr, params->N_x_roi, params->N_y_roi, params->N_z_roi, dataType);
    }
    else
    {
        write3DData(fName, (void***)arr, params->N_x, params->N_y, params->N_z, dataType);
    }
}

void readImageData3DCone(char *fName, void ***arr, struct ImageParams *params, int isROI, char *dataType)
{
    if (isROI) 
    {
        read3DData(fName, (void***)arr, params->N_x_roi, params->N_y_roi, params->N_z_roi, dataType);
    }
    else
    {
        read3DData(fName, (void***)arr, params->N_x, params->N_y, params->N_z, dataType);
    }
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
    aux->orderXY = mem_alloc_1D(N_x * N_y, sizeof(int));

    /**
     *      Initialize groupIndex
     */
    aux->groupIndex = (unsigned char***) mem_alloc_3D(N_x, N_y, N_z, sizeof(unsigned char***));
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
    aux->orderXYZ = mem_alloc_1D(imgParams->N_x * imgParams->N_y * imgParams->N_z, sizeof(long ));
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
    mem_free_1D((void*)aux->orderXY);
    mem_free_3D((void***)aux->groupIndex);
}

void RandomAux_free(struct RandomAux *aux)
{
    mem_free_1D((void*)aux->orderXYZ);
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
*/    first_N_G_members = mem_alloc_1D(N_G, sizeof(int));

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
        
int bernoulli(double p)
{
    double r;
    if(p==0)
        return 0;

    if(p==1)
        return 1;

    r = ((double) rand() / (RAND_MAX));
    if(r<p)
        return 1;
    else
        return 0;
}

long int uniformIntegerRV(long int l, long int h)
{
    return l+rand()%(h-l+1);
}

long int almostUniformIntegerRV(double mean, int sigma)
{
    /* creates random integer, Z, variable that is approx uniform in [mean-sigma, mean+sigma] */
    /* "mean" corresponds to the real expectation of Z*/
    /* range(Z) = 2*simga + 1 - delta(mean-ceil(mean)) */
    double mean_low, mean_high;
    double X_low, X_high;
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

void ticToc_logAndDisp(double ticToc, char *ticTocName)
{
    char str[1000];

    sprintf(str, "[ticToc] %s = %e s\n", ticTocName, ticToc);
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
    arr_sub = malloc(len_sub*sizeof(float));

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
    printf("*****                = %-15e kB\n", (double) size*1e-3);
    printf(" ****                = %-15e MB\n", (double) size*1e-6);
    printf("    ***********************************************************\n");
}

void printProgressOfLoop( long int indexOfLoop, long int NumIterations)
{
    double percent;

    percent = (double) (1+indexOfLoop) / (double) NumIterations * 100;
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


