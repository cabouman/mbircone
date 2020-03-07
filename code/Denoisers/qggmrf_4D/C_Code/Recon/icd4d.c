
#include "icd4d.h"
#include <math.h>
#include <stdio.h>


#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "../CLibraries/allocate.h"
#include "../CLibraries/io4d.h"


void ICDStep3DCone(struct Image *img, struct ICDInfo *icdInfo, struct Params *params)
{
    /**
     *      Updates one voxel. Voxel change is stored in icdInfo->Delta_xj. 
     */

    /* debug */
    /*if( icdInfo->j_x == 576 && icdInfo->j_y == 536 && icdInfo->j_z == 49 ){*/
    /*if( icdInfo->j_x == 0 && icdInfo->j_y == 0 && icdInfo->j_z == 0 ){
        printICDinfo(icdInfo);
    }*/

    /**
     *          Compute forward model term of theta1 and theta2:
     *      
     *      theta1_F = -e{j} W{j,j}
     *      theta2_F =  W{j,j}
     */
    computeTheta1Theta2ForwardTerm(img, icdInfo, params);
    
    /**
     *          Compute prior model term of theta1 and theta2:
     *      
     */
    computeTheta1Theta2PriorTermQGGMRF(icdInfo, params, img);

    /**
     *          Compute final theta1 and theta2:
     *      
     */
    computeTheta1Theta2(icdInfo);

    computeDeltaXjAndUpdate(icdInfo, params, img);
    
}

void initialize_NeighborhoodInfo(struct NeighborhoodInfo *neighborhoodInfo, struct Image *img, struct Params *params)
{
    int j_t, j_x, j_y, j_z;
    double dist, sqrd_dist;
    double sumWts_space=0;
    int neighborID=0;


    neighborhoodInfo->numNeighbors_s = findNumNeighbors_s(params);
    neighborhoodInfo->numNeighbors_t = findNumNeighbors_t(params);
    neighborhoodInfo->numNeighbors = neighborhoodInfo->numNeighbors_s + neighborhoodInfo->numNeighbors_t;

    neighborhoodInfo->neighborWts = (double *)mem_alloc_1D( neighborhoodInfo->numNeighbors, sizeof(double) );
    neighborhoodInfo->j_t_arr = (int *)mem_alloc_1D( neighborhoodInfo->numNeighbors, sizeof(int) );
    neighborhoodInfo->j_x_arr = (int *)mem_alloc_1D( neighborhoodInfo->numNeighbors, sizeof(int) );
    neighborhoodInfo->j_y_arr = (int *)mem_alloc_1D( neighborhoodInfo->numNeighbors, sizeof(int) );
    neighborhoodInfo->j_z_arr = (int *)mem_alloc_1D( neighborhoodInfo->numNeighbors, sizeof(int) );
    
    /* Iterate through space neighborhood */
    j_t = 0;
    for (j_x = -1; j_x <= 1; ++j_x){
        for (j_y = -1; j_y <= 1; ++j_y){
            for (j_z = -1; j_z <= 1; ++j_z){

                if( isNeighbor(j_t, j_x, j_y, j_z, params) ){
                    neighborhoodInfo->j_t_arr[neighborID] = j_t;
                    neighborhoodInfo->j_x_arr[neighborID] = j_x;
                    neighborhoodInfo->j_y_arr[neighborID] = j_y;
                    neighborhoodInfo->j_z_arr[neighborID] = j_z;

                    sqrd_dist =  pow( j_t , 2 ) + 
                                 pow( j_x , 2 ) + 
                                 pow( j_y , 2 ) + 
                                 pow( j_z , 2 );

                    dist = sqrt( sqrd_dist );

                    neighborhoodInfo->neighborWts[neighborID] = 1 / dist ;

                    sumWts_space += neighborhoodInfo->neighborWts[neighborID];
                    neighborID++;

                }
            }
        }
    }

    if(neighborID != neighborhoodInfo->numNeighbors_s){
        fprintf(stderr, "Error: num space neighbors doesnt match: %d != %d \n", neighborID, neighborhoodInfo->numNeighbors_s);
    }
    else{
        printf("numNeighbors_s: %d == %d \n", neighborID, neighborhoodInfo->numNeighbors_s);
    }

    for(neighborID=0; neighborID < neighborhoodInfo->numNeighbors_s; neighborID++){
        neighborhoodInfo->neighborWts[neighborID] /= sumWts_space;
    }

    /* Iterate through time neighborhood */
    j_x = 0;
    j_y = 0;
    j_z = 0;
    for (j_t = -1; j_t <= 1; ++j_t){
        if( isNeighbor(j_t, j_x, j_y, j_z, params) ){

            neighborhoodInfo->j_t_arr[neighborID] = j_t;
            neighborhoodInfo->j_x_arr[neighborID] = j_x;
            neighborhoodInfo->j_y_arr[neighborID] = j_y;
            neighborhoodInfo->j_z_arr[neighborID] = j_z;

            neighborhoodInfo->neighborWts[neighborID] = 0.5 ;



            neighborID++;

        }
    }
    
    if(neighborID != neighborhoodInfo->numNeighbors){
        fprintf(stderr, "Error: num neighbores doesnt match: %d != %d \n", neighborID, neighborhoodInfo->numNeighbors);
    }
    else{
        printf("numNeighbors %d == %d \n", neighborID, neighborhoodInfo->numNeighbors);
    }


}

int findNumNeighbors_t(struct Params *params)
{
    int j_t, j_x, j_y, j_z;
    int numNeighbors = 0;

    j_x = 0;
    j_y = 0;
    j_z = 0;
    for (j_t = -1; j_t <= 1; ++j_t){
        if( isNeighbor(j_t, j_x, j_y, j_z, params) ){
            numNeighbors++;
        }
    }

    return numNeighbors;
}

int findNumNeighbors_s(struct Params *params)
{
    int j_t, j_x, j_y, j_z;
    int numNeighbors = 0;

    j_t = 0;
    for (j_x = -1; j_x <= 1; ++j_x){
        for (j_y = -1; j_y <= 1; ++j_y){
            for (j_z = -1; j_z <= 1; ++j_z){
                if( isNeighbor(j_t, j_x, j_y, j_z, params) ){
                    numNeighbors++;
                }
            }
        } 
    }

    return numNeighbors;
}

int isNeighbor(int j_t, int j_x, int j_y, int j_z, struct Params *params)
{
    double sqrd_dist, dist, threshold;

    sqrd_dist =  pow( j_t , 2 ) + 
                 pow( j_x , 2 ) + 
                 pow( j_y , 2 ) + 
                 pow( j_z , 2 );
    dist = sqrt( sqrd_dist );

    if(j_t==0){

        switch(params->spacePriorMode){
            case 0:
                threshold = -1;
                break;
            case 1:
                threshold = 1.01;
                break;
            case 2: 
                threshold = 1.5;
                break;
            case 3:
                threshold = 2;
        }

    }
    else{
        if(params->isTimePrior==1){
            threshold = 1.01;
        }
        else{
            threshold = -1;
        }
    }


    if( j_t==0 && j_x==0 && j_y==0 && j_z==0 ){
        threshold = -1;
    }


    if(dist>threshold){
        return 0;
    }
    else{
        return 1;
    }

}

void free_NeighborhoodInfo(struct NeighborhoodInfo *neighborhoodInfo)
{
    mem_free_1D((void *)neighborhoodInfo->neighborWts);

    mem_free_1D((void *)neighborhoodInfo->j_t_arr);
    mem_free_1D((void *)neighborhoodInfo->j_x_arr);
    mem_free_1D((void *)neighborhoodInfo->j_y_arr);
    mem_free_1D((void *)neighborhoodInfo->j_z_arr);
}


void prepareICDInfo(int j_t, int j_x, int j_y, int j_z, struct ICDInfo *icdInfo, struct NeighborhoodInfo *neighborhoodInfo, struct Image *img)
{
    icdInfo->old_xj = img->denoised[j_t][j_x][j_y][j_z];
    icdInfo->j_t = j_t;
    icdInfo->j_x = j_x;
    icdInfo->j_y = j_y;
    icdInfo->j_z = j_z;
    icdInfo->theta1 = 0;
    icdInfo->theta2 = 0;
    icdInfo->theta1_F = 0;
    icdInfo->theta2_F = 0;
    icdInfo->theta1_P = 0;
    icdInfo->theta2_P = 0;

    icdInfo->neighborhoodInfo = neighborhoodInfo;
}


void computeTheta1Theta2(struct ICDInfo *icdInfo)
{
    icdInfo->theta1 = icdInfo->theta1_F + icdInfo->theta1_P ;
    icdInfo->theta2 = icdInfo->theta2_F + icdInfo->theta2_P ;
}

void computeTheta1Theta2ForwardTerm(struct Image *img, struct ICDInfo *icdInfo, struct Params *params)
{
    /**
     *      Compute forward model term of theta1 and theta2:
     *      
     *      theta1_F = -e^t W A_{*,j} = -e{j}/sigma^2
     *      theta2_F = A_{*,j}^t W A _{*,j} = 1/sigma^2
     *
     *
     *
     */
    float err;

    err = img->noisy[icdInfo->j_t][icdInfo->j_x][icdInfo->j_y][icdInfo->j_z] - img->denoised[icdInfo->j_t][icdInfo->j_x][icdInfo->j_y][icdInfo->j_z];
    icdInfo->theta1_F = - err / pow(params->sigma,2);
    icdInfo->theta2_F = 1 / pow(params->sigma,2);

}

void computeTheta1Theta2PriorTermQGGMRF(struct ICDInfo *icdInfo, struct Params *params, struct Image *img)
{
    /**
     *          Compute prior model term of theta1 and theta2:
     *      
     *      theta1_P =    sum       2 b_{j,r} * surrCoeff(x_j - x_r) * (x_j - x_r)
     *                  {r E ∂j}
     *                  
     *      theta2_P =    sum       2 b_{j,r} * surrCoeff(x_j - x_r)
     *                  {r E ∂j}
     */

    double delta, surrogateCoeff, neighborVal;
    double sum1=0;
    double sum2=0;
    double sumWtsNeighbor=0;

    int j_t, j_x, j_y, j_z;
    int neighbor_j_t, neighbor_j_x, neighbor_j_y, neighbor_j_z;
    int indx_t, indx_x, indx_y, indx_z;
    int neighborID;

    /* Iterate through neighborhood */
    for(neighborID=0; neighborID < icdInfo->neighborhoodInfo->numNeighbors; neighborID++){

        j_t = icdInfo->neighborhoodInfo->j_t_arr[neighborID] ;
        j_x = icdInfo->neighborhoodInfo->j_x_arr[neighborID] ;
        j_y = icdInfo->neighborhoodInfo->j_y_arr[neighborID] ;
        j_z = icdInfo->neighborhoodInfo->j_z_arr[neighborID] ;

        neighbor_j_t = j_t + icdInfo->j_t;
        neighbor_j_x = j_x + icdInfo->j_x;
        neighbor_j_y = j_y + icdInfo->j_y;
        neighbor_j_z = j_z + icdInfo->j_z;

        if( isWithinVol( neighbor_j_t, neighbor_j_x, neighbor_j_y, neighbor_j_z, img) ){
            
            neighborVal = img->denoised[neighbor_j_t][neighbor_j_x][neighbor_j_y][neighbor_j_z];

            delta = icdInfo->old_xj - neighborVal;
            if( j_t==0 ){
                surrogateCoeff = surrogateCoeffQGGMRF(delta, params->p, params->q, params->T_s, params->sigma_s);
            }
            else{
                surrogateCoeff = surrogateCoeffQGGMRF(delta, params->p, params->q, params->T_t, params->sigma_t);
            }

            sum1 += 2 * icdInfo->neighborhoodInfo->neighborWts[neighborID] * surrogateCoeff * delta;
            sum2 += 2 * icdInfo->neighborhoodInfo->neighborWts[neighborID] * surrogateCoeff;

            sumWtsNeighbor += icdInfo->neighborhoodInfo->neighborWts[neighborID];
            
       
        }
    }

    icdInfo->theta1_P = sum1 ;
    icdInfo->theta2_P = sum2 ;

}

double surrogateCoeffQGGMRF(double Delta, double p, double q, double T, double sigmaX)
{
    /**
     *                       /  rho'(Delta) / (2 Delta)             if Delta != 0
     *   surrCoeff(Delta) = {
     *                       \  rho''(0) / 2                        if Delta = 0
     */
    double qmp;
    double num, denom, temp;
    
    qmp = q - p;
    
    if(Delta == 0.0)
    {
        /**
         *      rho''(0)           1
         *      -------- = -----------------
         *         2       p sigmaX^q T^(q-p)
         */
        return 1.0 / ( p * pow(sigmaX, q) * pow(T, qmp) );
    }
    else /* Delta != 0 */
    {
        /**
         *      rho'(Delta)   |Delta|^(p-2)  # (q/p + #)
         *      ----------- = ------------- ------------
         *         2 Delta     2 sigmaX^p    (1 + #)^2
         *
         *      where          | Delta  |^(q-p)
         *                #  = |--------|
         *                     |T sigmaX|
         */
        temp = pow(fabs(Delta / (T*sigmaX)), qmp);  /* this is the # from above */
        num = pow(fabs(Delta), p-2) * temp * (q/p + temp);
        denom = 2 * pow(sigmaX,p) * (1.0 + temp) * (1.0 + temp);
        
        return num / denom;
    }

}

void updateIterationStats(double *TotalValueChange, double *TotalVoxelValue, int *NumUpdatedVoxels, struct ICDInfo *icdInfo, struct Image *img)
{
    *TotalValueChange += fabs(icdInfo->Delta_xj);
    *TotalVoxelValue += _MAX_(img->denoised[icdInfo->j_t][icdInfo->j_x][icdInfo->j_y][icdInfo->j_z], icdInfo->old_xj);
    (*NumUpdatedVoxels)++;
}

void resetIterationStats(double *TotalValueChange, double *TotalVoxelValue, int *NumUpdatedVoxels)
{
    *TotalValueChange = 0;
    *TotalVoxelValue = 0;
    *NumUpdatedVoxels = 0;
}



void RandomAux_ShuffleorderTXYZ(struct RandomAux *aux, struct ImageParams *params)
{
    shuffleIntArray(aux->orderTXYZ, params->N_t * params->N_x * params->N_y * params->N_z);
}

void indexExtraction4D(int j_txyz, int *j_t, int N_t, int *j_x, int N_x, int *j_y, int N_y, int *j_z, int N_z)
{
    /* j_txyz = j_z + N_z j_y + N_z N_y j_x + N_z N_y N_x j_t */

    int j_temp, j_temp2;

    j_temp = j_txyz;                /* Now, j_temp = j_z + N_z j_y + N_z N_y j_x + N_z N_y N_x j_t */

    *j_z = j_temp % N_z;
    j_temp = (j_temp-*j_z) / N_z;   /* Now, j_temp = j_y + N_y j_x + N_y N_x j_t */

    *j_y = j_temp % N_y;
    j_temp = (j_temp-*j_y) / N_y;   /* Now, j_temp = j_x + N_x j_t */

    *j_x = j_temp % N_x;
    j_temp = (j_temp-*j_x) / N_x;   /* Now, j_temp = j_t */

    *j_t = j_temp;

    j_temp2 = (*j_z) + N_z * (*j_y) + N_z * N_y * (*j_x) + N_z * N_y * N_x * (*j_t) ;

    if( j_txyz != j_temp2){
        printf("\nError in indexExtraction4D\n");
        exit(-1);
    }

    return;
}


double MAPCost4D(struct Image *img, struct Params *params, struct NeighborhoodInfo *neighborhoodInfo)
{
    /**
     *      Cost =    1/2 ||e||^{2}_{W}
     *      
     *              + sum       b_{s,r} rho(x_s-x_r)
     *               {s,r} E P
     */
    double cost = 0;
    double costF = 0;
    double costP = 0;
    int num_mask;


    costF = MAPCostForward(img, params);

    costP = MAPCostPrior_QGGMRF(img, params, neighborhoodInfo);

    cost = costF + costP;

    if(DEBUG_FLAG) printf("Cost=%f, (F=%f P=%f)\n",cost,costF,costP );

    return cost;
}

double MAPCostForward(struct Image *img, struct Params *params)
{
    /**
     *      ForwardCost =  1/2 ||e||^{2}_{W}
     */

    double cost = 0;
    float err;
    int j_t, j_x, j_y, j_z;

    for (j_t = 0; j_t < img->params.N_t; ++j_t){
        for (j_x = 0; j_x < img->params.N_x; ++j_x){
            for (j_y = 0; j_y < img->params.N_y; ++j_y){
                for (j_z = 0; j_z < img->params.N_z; ++j_z){
                    err = img->noisy[j_t][j_x][j_y][j_z] - img->denoised[j_t][j_x][j_y][j_z] ;
                    cost += pow( err / params->sigma , 2 );
                }
            }
        }
    }

    cost = cost / 2.0 ;


    return cost ;
}

double MAPCostPrior_QGGMRF(struct Image *img, struct Params *params, struct NeighborhoodInfo *neighborhoodInfo)
{
    /**
     *  cost = sum     b_{s,r}  rho(x_s-x_r)
     *        {s,r} E P
     */
    
    int j_t, j_x, j_y, j_z;
    struct ICDInfo icdInfo;
    double cost = 0;
    double temp;

    for (j_t = 0; j_t < img->params.N_t; ++j_t)
    {
        for (j_x = 0; j_x < img->params.N_x; ++j_x)
        {
            for (j_y = 0; j_y < img->params.N_y; ++j_y)
            {
                for (j_z = 0; j_z < img->params.N_z; ++j_z)
                {
                    /**
                     *      Prepare icdInfo
                     */
                    prepareICDInfo(j_t, j_x, j_y, j_z, &icdInfo, neighborhoodInfo, img);

                    temp = MAPCostPrior_QGGMRFSingleVoxel_HalfNeighborhood(img, &icdInfo, params);
                    cost += temp;
                }
            }
        }
    }
    return cost;
}

double MAPCostPrior_QGGMRFSingleVoxel_HalfNeighborhood(struct Image *img, struct ICDInfo *icdInfo, struct Params *params)
{
    /**
     *          Compute prior model term of theta1 and theta2:
     *      
     *      cost +=       sum          b_{j,r} * rho(x_j - x_r)
     *                  {r E ∂j^half}
     *              
     */

    double delta, neighborVal;
    double sum1=0;
    double sumWtsNeighbor=0;
    double cost;
    double potentialVal;

    int j_t, j_x, j_y, j_z;
    int neighbor_j_t, neighbor_j_x, neighbor_j_y, neighbor_j_z;
    int neighborID;

    /* Iterate through neighborhood */
    for(neighborID=0; neighborID < icdInfo->neighborhoodInfo->numNeighbors; neighborID++){

        j_t = icdInfo->neighborhoodInfo->j_t_arr[neighborID] ;
        j_x = icdInfo->neighborhoodInfo->j_x_arr[neighborID] ;
        j_y = icdInfo->neighborhoodInfo->j_y_arr[neighborID] ;
        j_z = icdInfo->neighborhoodInfo->j_z_arr[neighborID] ;


        neighbor_j_t = j_t + icdInfo->j_t;
        neighbor_j_x = j_x + icdInfo->j_x;
        neighbor_j_y = j_y + icdInfo->j_y;
        neighbor_j_z = j_z + icdInfo->j_z;

        if( isWithinVol( neighbor_j_t, neighbor_j_x, neighbor_j_y, neighbor_j_z, img) ){
            
            neighborVal = img->denoised[neighbor_j_t][neighbor_j_x][neighbor_j_y][neighbor_j_z];

            delta = icdInfo->old_xj - neighborVal;

            if(j_t == 0){
                potentialVal = QGGMRFPotential(delta, params->p, params->q, params->T_s, params->sigma_s);
            }
            else{
                potentialVal = QGGMRFPotential(delta, params->p, params->q, params->T_t, params->sigma_t);
            }
            sum1 += icdInfo->neighborhoodInfo->neighborWts[neighborID] * potentialVal;

            sumWtsNeighbor += icdInfo->neighborhoodInfo->neighborWts[neighborID];
            
        }
    }

    /*debug*/
    /*if( icdInfo->j_t == 0 && icdInfo->j_x == 4 && icdInfo->j_y == 0 && icdInfo->j_z == 0 ){
        printf("costHalf(0 4 0 0 )=%f sum1 : %f sumWtsNeighbor=%f\n", sum1/sumWtsNeighbor, sum1, sumWtsNeighbor );
        printNeighborsInImg(icdInfo, img);
    }
    if( icdInfo->j_t == 0 && icdInfo->j_x == 3 && icdInfo->j_y == 0 && icdInfo->j_z == 0 ){
        printf("costHalf(0 3 0 0 )=%f sum1 : %f sumWtsNeighbor=%f\n", sum1/sumWtsNeighbor, sum1, sumWtsNeighbor );
        printNeighborsInImg(icdInfo, img);
    }*/

    cost = sum1 ;

    return cost/2; /* since we add each clique twice */
}



/* the potential function of the QGGMRF prior model.  p << q <= 2 */
double QGGMRFPotential(double delta, double p, double q, double T, double sigmaX)
{
    double temp, GGMRF_Pot;
    
    GGMRF_Pot = pow(fabs(delta),p)/(p*pow(sigmaX,p));
    temp = pow(fabs(delta/(T*sigmaX)), q-p);
    return ( GGMRF_Pot * temp/(1.0+temp) );
}


void computeDeltaXjAndUpdate(struct ICDInfo *icdInfo, struct Params *params, struct Image *img)
{
    /**
     *          Compute voxel increment Delta_xj.
     *          Delta_xj >= -x_j accomplishes positivity constraint:
     *      
     *      Delta_xj = clip{   -theta1/theta2, [-x_j, inf)   }
     */

    icdInfo->Delta_xj = -icdInfo->theta1/icdInfo->theta2;

    if(params->is_positivity_constraint){
        icdInfo->Delta_xj = (icdInfo->Delta_xj > -icdInfo->old_xj ? icdInfo->Delta_xj : -icdInfo->old_xj);
    }

    /**
     *          Update voxel:
     *      
     *      x_j <- x_j + Delta_xj
     */

    img->denoised[icdInfo->j_t][icdInfo->j_x][icdInfo->j_y][icdInfo->j_z] += icdInfo->Delta_xj;
}



void dispAndLog_iterationInfo(int itNumber, int MaxIterations, double cost, double relUpdatePercent, double voxelsPerSecond, double ticToc_iteration, double normError)
{
    char str[2000];

    /**
     *      progress file
     */
    sprintf(str, "\n");
    sprintf(str, "%s ************************** Iteration %-2d (max. %d) **************************\n", str, itNumber, MaxIterations);
    sprintf(str, "%s *  Cost                       = %-15e\n", str, cost);
    sprintf(str, "%s *  Rel. Update                = %-e %%\n", str, relUpdatePercent);
    sprintf(str, "%s *  normError  = 1/N ||e||^2_L = %-e = 1/%f\n", str, normError, 1/normError);
    sprintf(str, "%s *  voxelsPerSecond            = %-e \n", str, voxelsPerSecond);
    sprintf(str, "%s *  time icd update            = %-e s\n", str, ticToc_iteration);
    sprintf(str, "%s ******************************************************************************\n", str);
    sprintf(str, "%s\n", str);
    logAndDisp_message(LOG_PROGRESS, str);

    /**
     *      stats file
     */
    sprintf(str, "\n");
    sprintf(str, "%scost(%d)   = %.12e;\n", str, itNumber+1, cost);
    sprintf(str, "%schange(%d) = %.12e;\n", str, itNumber+1, relUpdatePercent);
    log_message(LOG_STATS, str);
}

double computeRelUpdate(int NumUpdatedVoxels, double TotalValueChange, double TotalVoxelValue)
{
    double relUpdatePercent;
    double AvgValueChange, AvgVoxelValue;

    if(NumUpdatedVoxels>0)
    {
        AvgValueChange = TotalValueChange /  NumUpdatedVoxels;
        AvgVoxelValue = TotalVoxelValue /  NumUpdatedVoxels;
    }
    else
    {
        AvgValueChange = 0;
        AvgVoxelValue = 0;
    }

    if(AvgVoxelValue>0)
    {
        relUpdatePercent = AvgValueChange / AvgVoxelValue * 100.0;
    }
    else
    {
        relUpdatePercent = 0;
    }

    return relUpdatePercent;
}

void writeICDLoopStatus2File(char *fName, int index, int MaxIndex, int itNumber, double voxelsPerSecond)
{
    FILE *filePointer;
    double percentage;


    filePointer = fopen(fName, "a");
    percentage = ((double)index / (double)MaxIndex) * 100.0;
    fprintf(filePointer, "(%2d) Loop Progress = %e %%  voxelsPerSecond = %e\n", itNumber, percentage, voxelsPerSecond);
     printf(           "\r(%2d) Loop Progress = %e %%  voxelsPerSecond = %e", itNumber, percentage, voxelsPerSecond);
     fflush(stdout);
    fclose(filePointer);
        
    
}



/* * * * * * * * * * * * time aux ICD * * * * * * * * * * * * **/

void speedAuxICD_reset(struct SpeedAuxICD *speedAuxICD)
{
    speedAuxICD->numberUpdatedVoxels = 0;
    speedAuxICD->tic = omp_get_wtime();
    speedAuxICD->toc = -1.0;
    speedAuxICD->voxelsPerSecond = -1.0;
}

void speedAuxICD_update(struct SpeedAuxICD *speedAuxICD, int incrementNumber)
{
    speedAuxICD->numberUpdatedVoxels += incrementNumber;

}
 
void speedAuxICD_computeSpeed(struct SpeedAuxICD *speedAuxICD)
{
    if (speedAuxICD->numberUpdatedVoxels > 0)
    {
        speedAuxICD->toc = omp_get_wtime();
        speedAuxICD->voxelsPerSecond = ((double)speedAuxICD->numberUpdatedVoxels) / (speedAuxICD->toc - speedAuxICD->tic);
    }
    else
    {
        speedAuxICD->voxelsPerSecond = 0;
    }
}














