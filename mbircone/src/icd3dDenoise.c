/*[1] Balke, Thilo, et al. "Separable Models for cone-beam MBIR Reconstruction." Electronic Imaging 2018.15 (2018): 181-1.*/
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "icd3d.h"
#include "icd3dDenoise.h"
#include "allocate.h"

void ICDStep3DDenoise(struct Image *image, struct Image *err_image, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams)
{
    float theta1, theta2; 
    /**
     *         Updates one voxel. Voxel change is stored in icdInfo->Delta_xj. 
     */

     /* Compute forward model term of theta1 and theta2 */
    computeTheta1Theta2ForwardTermDenoise(err_image, icdInfo, reconParams);
    /* Compute prior model term of theta1 and theta2 */
    computeTheta1Theta2PriorTermQGGMRF(icdInfo, reconParams);
    
    /* Update Voxel */
    theta1 = icdInfo->theta1_f + icdInfo->theta1_p_QGGMRF;
    theta2 = icdInfo->theta2_f + icdInfo->theta2_p_QGGMRF;
    //theta1 = icdInfo->theta1_f;
    //theta2 = icdInfo->theta2_f;
    // fprintf(stdout, "theta1 = %.5f, theta2 = %.5f, old_xj = %.5f, ", theta1, theta2, icdInfo->old_xj);
    /* Clip to 0 */
    if (theta2 != 0)
    {
        icdInfo->Delta_xj = -theta1/theta2;

        if(reconParams->is_positivity_constraint)
            icdInfo->Delta_xj = _MAX_(icdInfo->Delta_xj, -icdInfo->old_xj);
    }
    else
    {
        icdInfo->Delta_xj = _MAX_(icdInfo->old_xj, 0);
    }    
    // fprintf(stdout, "Delta_xj = %.5f\n\n", icdInfo->Delta_xj);
    /* Update voxel: x_j <- x_j + Delta_xj */
    image->vox[index_3D(icdInfo->j_x,icdInfo->j_y,icdInfo->j_z,image->params.N_y,image->params.N_z)] += icdInfo->Delta_xj;    
    
    err_image->vox[index_3D(icdInfo->j_x, icdInfo->j_y, icdInfo->j_z, err_image->params.N_y, err_image->params.N_z)] -= icdInfo->Delta_xj; 
}

/*[1]: Algorithm 2 on page 181-5*/
void computeTheta1Theta2ForwardTermDenoise(struct Image *err_image, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams)
{
    /**
     *             Compute forward model term of theta1 and theta2:
     *         
     *       theta1_f = -e^t W A_{*,j}
     *         theta2_f = A_{*,j}^t W A _{*,j}
     */
    int j_x, j_y, j_z;
    j_x = icdInfo->j_x;
    j_y = icdInfo->j_y;
    j_z = icdInfo->j_z;
    icdInfo->theta1_f = -err_image->vox[index_3D(j_x, j_y, j_z, err_image->params.N_y, err_image->params.N_z)]/reconParams->weightScaler_value; 
    icdInfo->theta2_f = 1/reconParams->weightScaler_value;
 
}

float MAPCost3DDenoise(struct Image *image, struct Image *err_image, struct ReconParams *reconParams)
{
    /**
     *      Computes MAP cost function
     */
    float cost;
    
    // Initialize cost with forward model cost    
    cost = MAPCostForwardDenoise(err_image, reconParams);
    cost += MAPCostPrior_QGGMRF(image, reconParams);
    return cost;
}

float MAPCostForwardDenoise(struct Image *err_image, struct ReconParams *reconParams)
{
    /**
     *         ForwardCost =  1/2 ||e||^{2}_{W}
     */
    long int i_x, i_y, i_z;
    float cost;

    cost = 0;
    for (i_x = 0; i_x < err_image->params.N_x; ++i_x)
    {
        for (i_y = 0; i_y < err_image->params.N_y; ++i_y)
        {
            for (i_z = 0; i_z < err_image->params.N_z; ++i_z)
            {
                cost +=   err_image->vox[index_3D(i_x,i_y,i_z,err_image->params.N_y,err_image->params.N_z)]
                        * err_image->vox[index_3D(i_x,i_y,i_z,err_image->params.N_y,err_image->params.N_z)];
            }
        }
    }
    return cost / (2.0 * reconParams->weightScaler_value);
}


void disp_iterationInfo_denoise(int itNumber, int MaxIterations, float cost, float relUpdate, float stopThresholdChange, float weightScaler_value, double ticToc_iteration)
{
    printf("************************** Iteration %-2d (max. %d) **************************\n", itNumber, MaxIterations);
    printf("*  Cost                   = %-10.10e\n", cost);
    printf("*  Rel. Update            = %-10.10e %% (threshold = %-10.10e %%)\n", relUpdate*100, stopThresholdChange*100);
    printf("* ----------------------------------------------------------------------------\n");
    printf("*  weightScaler_value     = %-10.10e = 1/%-10.10f\n", weightScaler_value, 1/weightScaler_value);
    printf("* ----------------------------------------------------------------------------\n");
    printf("*  time icd update        = %-10.10e s\n", ticToc_iteration);
    printf("******************************************************************************\n\n");
}










